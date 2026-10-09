package com.arability.chinese_chess_coach_phone_version.chess

import android.graphics.Bitmap
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

/**
 * 数据飞轮「采集触发」：把识别出的「难帧」落盘，供离线 SAM 标注 → 重训 YOLO → OTA。
 *
 * 难帧判定（[shouldCollect]）：
 *  1. 任一逐子检测置信度 < [CONF_COLLECT]（模型不确定）；
 *  2. 局面非法（[isIllegalPosition]：双帅/缺帅、子越界、每种子数超上限、总数 > 32）；
 *  3. 局面与上一张已采帧不同（覆盖「模型自信但判错」的多样位置）。
 *
 * 节流：距上次采集 < [MIN_INTERVAL_MS] 直接跳过，避免同一静态局面重复落盘。
 * 落盘：每帧一张 JPEG + 一行 JSONL 元数据（四角 + FEN + YOLO 原始框），存到 `filesDir/flywheel/`。
 */
class FlywheelCollector(private val dir: File) {

    private var counter = 0
    private var lastCollectMs = 0L
    private var lastFen: String? = null

    init {
        dir.mkdirs()
        counter = dir.listFiles { f -> f.name.endsWith(JPEG_EXT) }?.size ?: 0
    }

    /** 决定是否采集并落盘。返回是否真正落盘。 */
    fun maybeCollect(
        bitmap: Bitmap,
        info: BoardInfo,
        board: Board,
        detections: List<PieceDetection>
    ): Boolean {
        val fen = board.toFen()
        if (!shouldCollect(fen, board, detections)) return false

        val now = System.currentTimeMillis()
        if (now - lastCollectMs < MIN_INTERVAL_MS) return false

        if (counter >= MAX_FRAMES) {
            if (counter == MAX_FRAMES) Log.w(TAG, "已达采集上限 $MAX_FRAMES，停止采集")
            return false
        }

        val saved = save(bitmap, info, fen, detections)
        if (saved) {
            lastCollectMs = now
            lastFen = fen
            counter++
        }
        return saved
    }

    private fun shouldCollect(fen: String, board: Board, detections: List<PieceDetection>): Boolean {
        if (detections.any { it.confidence < CONF_COLLECT }) return true
        if (isIllegalPosition(board)) return true
        return fen != lastFen
    }

    private fun save(
        bitmap: Bitmap,
        info: BoardInfo,
        fen: String,
        detections: List<PieceDetection>
    ): Boolean {
        try {
            val name = "frame_%05d".format(counter)
            val jpeg = File(dir, "$name$JPEG_EXT")
            FileOutputStream(jpeg).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
            }

            val corners = JSONArray()
            for (v in info.corners) corners.put(v.toDouble())

            val detArr = JSONArray()
            for (d in detections) {
                detArr.put(
                    JSONObject()
                        .put("cx", d.cx.toDouble())
                        .put("cy", d.cy.toDouble())
                        .put("label", d.label)
                        .put("conf", d.confidence.toDouble())
                )
            }

            val meta = JSONObject()
                .put("file", name)
                .put("ts", System.currentTimeMillis())
                .put("w", bitmap.width)
                .put("h", bitmap.height)
                .put("corners", corners)
                .put("fen", fen)
                .put("detections", detArr)

            FileOutputStream(metaFile(), true).use { out ->
                out.write((meta.toString() + "\n").toByteArray(Charsets.UTF_8))
            }
            return true
        } catch (t: Throwable) {
            Log.w(TAG, "采集帧落盘失败", t)
            return false
        }
    }

    private fun metaFile(): File = File(dir, "metadata.jsonl")

    companion object {
        private const val TAG = "FlywheelCollector"
        private const val JPEG_EXT = ".jpg"
        private const val JPEG_QUALITY = 90
        private const val CONF_COLLECT = 0.5f
        private const val MIN_INTERVAL_MS = 2000L
        private const val MAX_FRAMES = 1000

        /** 每种棋子上限，按 [PieceType.ordinal] 顺序：帅1 仕2 相2 马2 车2 炮2 兵5。 */
        private val MAX_COUNTS = intArrayOf(1, 2, 2, 2, 2, 2, 5)

        /**
         * 局面是否明显非法（识别错乱的强信号）。
         * 轻量校验：双帅/缺帅、帅仕象卒越界、每种子数超上限、总数 > 32。
         * 不校验走法级合法性（离线管线用完整规则兜底）。
         */
        fun isIllegalPosition(board: Board): Boolean {
            val redCount = IntArray(7)
            val blackCount = IntArray(7)
            var redGeneralSeen = false
            var blackGeneralSeen = false
            var total = 0

            for (rank in 0..9) {
                for (file in 0..8) {
                    val p = board.pieceAt(file, rank) ?: continue
                    total++
                    val idx = p.type.ordinal
                    if (p.color == PieceColor.RED) {
                        redCount[idx]++
                        when (p.type) {
                            PieceType.GENERAL -> {
                                if (redGeneralSeen) return true // 双帅
                                redGeneralSeen = true
                                if (file !in 3..5 || rank !in 7..9) return true
                            }
                            PieceType.ADVISOR -> if (file !in 3..5 || rank !in 7..9) return true
                            PieceType.ELEPHANT -> if (rank !in 5..9) return true
                            PieceType.PAWN -> if (rank !in 0..6) return true
                            else -> {}
                        }
                    } else {
                        blackCount[idx]++
                        when (p.type) {
                            PieceType.GENERAL -> {
                                if (blackGeneralSeen) return true // 双将
                                blackGeneralSeen = true
                                if (file !in 3..5 || rank !in 0..2) return true
                            }
                            PieceType.ADVISOR -> if (file !in 3..5 || rank !in 0..2) return true
                            PieceType.ELEPHANT -> if (rank !in 0..4) return true
                            PieceType.PAWN -> if (rank !in 3..9) return true
                            else -> {}
                        }
                    }
                }
            }

            if (total > 32) return true
            if (!redGeneralSeen || !blackGeneralSeen) return true // 缺帅/缺将
            for (i in 0..6) {
                if (redCount[i] > MAX_COUNTS[i] || blackCount[i] > MAX_COUNTS[i]) return true
            }
            return false
        }
    }
}
