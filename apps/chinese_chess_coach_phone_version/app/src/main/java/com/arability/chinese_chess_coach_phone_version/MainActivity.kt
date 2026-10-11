package com.arability.chinese_chess_coach_phone_version

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.arability.chinese_chess_coach_phone_version.chess.Board
import com.arability.chinese_chess_coach_phone_version.chess.BoardDetector
import com.arability.chinese_chess_coach_phone_version.chess.BoardInfo
import com.arability.chinese_chess_coach_phone_version.chess.ChessEngine
import com.arability.chinese_chess_coach_phone_version.chess.ChessEngineFactory
import com.arability.chinese_chess_coach_phone_version.chess.ChineseNotation
import com.arability.chinese_chess_coach_phone_version.chess.FlywheelCollector
import com.arability.chinese_chess_coach_phone_version.chess.BoardDetectorFactory
import com.arability.chinese_chess_coach_phone_version.chess.Move
import com.arability.chinese_chess_coach_phone_version.chess.Piece
import com.arability.chinese_chess_coach_phone_version.chess.PieceColor
import com.arability.chinese_chess_coach_phone_version.chess.PieceRecognizer
import com.arability.chinese_chess_coach_phone_version.chess.PieceRecognizerFactory
import com.arability.chinese_chess_coach_phone_version.chess.QuadRefiner
import com.arability.chinese_chess_coach_phone_version.databinding.ActivityMainBinding
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var engineExecutor: ExecutorService

    private lateinit var boardDetector: BoardDetector
    private lateinit var pieceRecognizer: PieceRecognizer
    private lateinit var engine: ChessEngine
    private lateinit var flywheelCollector: FlywheelCollector

    @Volatile private var lensFacing = CameraSelector.LENS_FACING_BACK
    @Volatile private var sideToMove = PieceColor.RED

    private var frameCounter = 0

    @Volatile private var latestBoardInfo: BoardInfo? = null
    @Volatile private var latestBoard: Board? = null
    @Volatile private var latestGrid: Array<Array<Piece?>>? = null
    @Volatile private var latestMove: Move? = null
    @Volatile private var lastBoardDetected = false

    private var lockMisses = 0
    private val recentGrids = ArrayDeque<Array<Array<Piece?>>>()

    /** bench 模式（C3）：`am start --ez bench true --es ep cpu|nnapi` 触发，非 null 时首帧跑一次基准。 */
    private var benchEp: String? = null
    private var benchDone = false

    /** 上一帧 YOLO 检出棋子质心（图像像素 [x,y]），用于下一帧四角反馈精修。 */
    @Volatile private var lastCentroids: List<FloatArray> = emptyList()

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                Toast.makeText(this, "需要相机权限才能预览", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        engineExecutor = Executors.newSingleThreadExecutor()
        benchEp = if (intent.getBooleanExtra("bench", false)) intent.getStringExtra("ep") ?: "cpu" else null
        // C8：兜底降级链已移除——模型加载失败显式报错退出，不再静默降级到不可用实现。
        try {
            boardDetector = BoardDetectorFactory.create(
                this,
                debugDir = java.io.File(filesDir, "debug"),
                useNnapi = benchEp == "nnapi"
            )
            pieceRecognizer = PieceRecognizerFactory.create(this)
        } catch (t: Throwable) {
            Log.e(TAG, "模型加载失败", t)
            Toast.makeText(this, "模型加载失败：${t.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        benchEp?.let {
            Log.i(TAG, "bench 模式：ep=$it，首帧后跑 30 次基准（看 logcat tag=BenchEP）")
        }
        engine = ChessEngineFactory.create()
        flywheelCollector = FlywheelCollector(java.io.File(filesDir, "flywheel"))

        binding.suggestButton.setOnClickListener { onSuggestClicked() }
        binding.switchCameraButton.setOnClickListener { onSwitchCameraClicked() }
        binding.sideToggleButton.setOnClickListener { onSideToggleClicked() }
        Log.i(TAG, "象棋教练初始化完成")

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun allPermissionsGranted(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindCamera(cameraProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera(cameraProvider: ProcessCameraProvider) {
        val preview = Preview.Builder()
            .build()
            .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { imageProxy -> processFrame(imageProxy) }
            }

        val selector = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, selector, preview, imageAnalysis)
            val label = if (lensFacing == CameraSelector.LENS_FACING_FRONT) "前置" else "后置"
            Log.i(TAG, "相机绑定成功（$label）")
        } catch (exc: Exception) {
            Log.e(TAG, "绑定相机失败", exc)
        }
    }

    private fun onSwitchCameraClicked() {
        val targetFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            CameraSelector.LENS_FACING_BACK
        } else {
            CameraSelector.LENS_FACING_FRONT
        }
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            if (targetFacing == CameraSelector.LENS_FACING_FRONT &&
                !cameraProvider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)
            ) {
                Toast.makeText(this, "该设备没有前置摄像头", Toast.LENGTH_SHORT).show()
                return@addListener
            }
            lensFacing = targetFacing
            latestBoardInfo = null
            latestBoard = null
            latestGrid = null
            latestMove = null
            bindCamera(cameraProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onSideToggleClicked() {
        sideToMove = sideToMove.opponent()
        latestBoard?.sideToMove = sideToMove
        binding.sideToggleButton.text = if (sideToMove == PieceColor.RED) "执红" else "执黑"
        latestMove = null
        binding.boardOverlayView.setSuggestion(latestBoardInfo, null)
    }

    private fun onSuggestClicked() {
        val board = latestBoard
        if (board == null) {
            Toast.makeText(this, "尚未识别到棋盘", Toast.LENGTH_SHORT).show()
            return
        }
        binding.suggestButton.isEnabled = false
        engineExecutor.execute {
            try {
                val move = engine.bestMove(board)
                runOnUiThread {
                    latestMove = move
                    binding.boardOverlayView.setSuggestion(latestBoardInfo, move)
                    binding.suggestButton.isEnabled = true
                    if (move != null) {
                        val notation = ChineseNotation.of(board, move)
                        Log.i(TAG, "建议走法: $notation")
                        Toast.makeText(this, "建议走法: $notation", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this, "当前局面无合法走法", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (exc: Exception) {
                Log.e(TAG, "走子建议失败", exc)
                runOnUiThread {
                    binding.suggestButton.isEnabled = true
                    Toast.makeText(this, "走子建议失败", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxyToBitmap(imageProxy)

            val ep = benchEp
            if (ep != null && !benchDone) {
                benchDone = true
                boardDetector.runBenchmark(bitmap, 30, ep)
            }

            frameCounter++
            if (frameCounter % DETECT_EVERY_N == 0) {
                val detected = boardDetector.detect(bitmap)
                if (detected != null) {
                    lockMisses = 0
                    // P0-b：用上一帧棋子质心反馈精修本帧四角（修近端掩码溢出导致的节点整体下移）。
                    val info: BoardInfo = if (lastCentroids.size >= 12) {
                        val refined = QuadRefiner.refine(detected.corners, lastCentroids)
                        if (refined != null) {
                            boardDetector.refineQuad(bitmap, refined) ?: detected
                        } else {
                            detected
                        }
                    } else {
                        detected
                    }
                    latestBoardInfo = info
                    val grid = pieceRecognizer.recognize(bitmap, info)
                    lastCentroids = pieceRecognizer.lastDetections().map {
                        floatArrayOf(it.cx, it.cy)
                    }
                    // C7①：质心回灌给检测器，作下一帧无历史（首帧/重锁）时候选甄别先验。
                    boardDetector.setReferenceCentroids(lastCentroids)
                    recentGrids.addLast(grid)
                    while (recentGrids.size > HISTORY_SIZE) recentGrids.removeFirst()
                    val voted = voteGrid(recentGrids)
                    latestGrid = voted
                    val board = buildBoard(voted)
                    latestBoard = board
                    // 数据飞轮：难帧（低置信/非法局面/新局面）落盘，供离线标注重训。
                    flywheelCollector.maybeCollect(bitmap, info, board, pieceRecognizer.lastDetections())
                } else if (lockMisses < LOCK_MAX_MISSES) {
                    // 迟滞：短暂漏检时保持上一次结果，避免闪烁。
                    lockMisses++
                } else {
                    lockMisses++
                    latestBoardInfo = null
                    latestBoard = null
                    latestGrid = null
                    recentGrids.clear()
                    lastCentroids = emptyList()
                    boardDetector.setReferenceCentroids(emptyList())
                }
                val hasBoard = latestBoardInfo != null
                if (hasBoard != lastBoardDetected) {
                    lastBoardDetected = hasBoard
                    Log.i(TAG, if (hasBoard) "检测到棋盘" else "未检测到棋盘")
                }
            }

            runOnUiThread {
                binding.boardOverlayView.setImageSize(bitmap.width, bitmap.height)
                binding.boardOverlayView.setPieces(latestGrid)
                binding.boardOverlayView.setSuggestion(latestBoardInfo, latestMove)
            }
        } catch (exc: Exception) {
            Log.e(TAG, "帧处理失败", exc)
        } finally {
            imageProxy.close()
        }
    }

    private fun buildBoard(grid: Array<Array<Piece?>>): Board {
        val board = Board()
        for (rank in 0..9) {
            for (file in 0..8) {
                board.set(file, rank, grid[rank][file])
            }
        }
        board.sideToMove = sideToMove
        return board
    }

    /** 对最近若干帧识别网格逐格多数投票，稳定局面；平票时取较新一帧的结果。 */
    private fun voteGrid(history: List<Array<Array<Piece?>>>): Array<Array<Piece?>> {
        if (history.isEmpty()) return Array(10) { arrayOfNulls(9) }
        if (history.size == 1) return history.first()
        val out = Array(10) { arrayOfNulls<Piece?>(9) }
        for (rank in 0..9) {
            for (file in 0..8) {
                // 从新到旧计数，LinkedHashMap 保持插入序，平票时 maxByOrNull 取首个（较新）。
                val counts = LinkedHashMap<Piece, Int>()
                for (g in history.asReversed()) {
                    val p = g[rank][file] ?: continue
                    counts[p] = (counts[p] ?: 0) + 1
                }
                out[rank][file] = counts.maxByOrNull { it.value }?.key
            }
        }
        return out
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        val rotation = image.imageInfo.rotationDegrees
        val rotated = if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
        return if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
            val matrix = Matrix().apply {
                postScale(-1f, 1f)
                postTranslate(rotated.width.toFloat(), 0f)
            }
            Bitmap.createBitmap(rotated, 0, 0, rotated.width, rotated.height, matrix, true)
        } else {
            rotated
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.close()
        pieceRecognizer.close()
        boardDetector.close()
        cameraExecutor.shutdown()
        engineExecutor.shutdown()
        Log.i(TAG, "已释放资源")
    }

    companion object {
        private const val TAG = "ChessCoach"
        private const val DETECT_EVERY_N = 5

        /** 连续漏检超过该次数（约 LOCK_MAX_MISSES × DETECT_EVERY_N 帧）才清空棋盘。 */
        private const val LOCK_MAX_MISSES = 3

        /** 局面时间投票的历史帧数。 */
        private const val HISTORY_SIZE = 5
    }
}
