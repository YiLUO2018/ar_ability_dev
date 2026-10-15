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
import com.arability.chinese_chess_coach_phone_version.chess.ImuMotion
import com.arability.chinese_chess_coach_phone_version.chess.BoardDetectorFactory
import com.arability.chinese_chess_coach_phone_version.chess.BoardGeometry
import com.arability.chinese_chess_coach_phone_version.chess.Homography
import com.arability.chinese_chess_coach_phone_version.chess.Move
import com.arability.chinese_chess_coach_phone_version.chess.Piece
import com.arability.chinese_chess_coach_phone_version.chess.PieceColor
import com.arability.chinese_chess_coach_phone_version.chess.PieceRecognizer
import com.arability.chinese_chess_coach_phone_version.chess.PieceRecognizerFactory
import com.arability.chinese_chess_coach_phone_version.chess.QuadRefiner
import com.arability.chinese_chess_coach_phone_version.chess.SegmentationBoardDetector
import com.arability.chinese_chess_coach_phone_version.chess.BoardTracker
import com.arability.chinese_chess_coach_phone_version.chess.HomographyMath
import com.arability.chinese_chess_coach_phone_version.chess.LkFlowEngine
import com.arability.chinese_chess_coach_phone_version.chess.ReanchorDecision
import com.arability.chinese_chess_coach_phone_version.chess.StdOpening
import com.arability.chinese_chess_coach_phone_version.databinding.ActivityMainBinding
import com.arability.chinese_chess_coach_phone_version.sensors.ImuState
import com.arability.chinese_chess_coach_phone_version.sensors.RotationPredictor
import com.arability.chinese_chess_coach_phone_version.sensors.StaticMotionMonitor
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
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

    /** D1/D2 IMU 静止监视（IMU 静止分支设计方案.md）：三态快照驱动检测器锚闸与不清屏。 */
    private lateinit var motionMonitor: StaticMotionMonitor
    private var lastImuMotionNotified: ImuMotion? = null

    /** D3 合法参照（最近一次合法投票局面与子数）；null=尚未建立（门直通）。 */
    private var lastLegalGrid: Array<Array<Piece?>>? = null
    private var lastLegalCount = 0

    // C9-T1/T2 帧间几何跟踪（BoardTracker帧间跟踪方案.md；PC 基线 v03_final）：
    // 检测路径原样保留，tracker「搭便车」——检测成功候选过准入闸后播种；MOVING 态
    // 显示源切换为每帧 LK 传播（30fps），每 CORRECT_EVERY 帧棋锚纠偏钉住 H。
    private var boardTracker: BoardTracker? = null
    private var lkEngine: LkFlowEngine? = null
    private var rotationPredictor: RotationPredictor? = null
    private var refBoard: FloatArray? = null
    private var trackAliveLast = false
    private var nFailAnnounced = false
    private var lastFrameTsNs = 0L
    private var trackRgba: org.opencv.core.Mat? = null
    private var trackGray: org.opencv.core.Mat? = null

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
        motionMonitor = StaticMotionMonitor(this, java.io.File(filesDir, "debug"))
        initFrameTracking()

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
            // D1/D2/救援分级：IMU 三态 → 检测器（仅状态变化时通知；UNKNOWN=退回 S7 行为）。
            val motion = when (motionMonitor.state) {
                ImuState.STATIC -> ImuMotion.STATIC
                ImuState.MOVING -> ImuMotion.MOVING
                else -> ImuMotion.UNKNOWN
            }
            if (motion != lastImuMotionNotified) {
                lastImuMotionNotified = motion
                boardDetector.onImuStateChanged(motion)
            }
            val imuStatic = motion == ImuMotion.STATIC
            // C9-T1：帧间 LK 传播（三态都跑——静止段相机不动、H 持续有效且特征新鲜；
            // MOVING 态承担显示）。返回传播 BoardInfo 或 null（无 H/传播失败）。
            val propInfo = trackFrame(bitmap, imageProxy.imageInfo.timestamp)
            // 失败/未锁定期间加速采样（移动重锁修复 B）：每帧都检（KEEP_ONLY_LATEST 下
            // 即背靠背全力重锁），恢复正常后回到 DETECT_EVERY_N 节拍。
            val everyN = if (latestBoardInfo == null || lockMisses > 0) DETECT_EVERY_FAST else DETECT_EVERY_N
            if (frameCounter % everyN == 0) {
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
                    var voted = voteGrid(recentGrids)
                    var board = buildBoard(voted)
                    var nVoted = countPieces(voted)
                    // D3 投票门 v2（vis3 实测教训：isIllegalPosition 当拒绝条件会因 YOLO
                    // 偶缺一帅而逐帧全拒 → 窗口 pop 冻结 → 局面永久卡在首个投票）：
                    // 只拒绝「>32 幽灵」或「对合法参照崩塌（<60%，遮挡期检出塌方）」，
                    // 沿用上一合法局面并撤销本帧 push；接受的合法投票才更新参照。
                    if (lastLegalGrid != null &&
                        (nVoted > 32 || nVoted < lastLegalCount * 0.6f)
                    ) {
                        recentGrids.removeLast()
                        voted = lastLegalGrid!!
                        board = buildBoard(voted)
                        nVoted = lastLegalCount
                    }
                    latestGrid = voted
                    latestBoard = board
                    if (!FlywheelCollector.isIllegalPosition(board) ||
                        (lastLegalGrid == null && nVoted in 20..32)
                    ) {
                        // 合法或（自举）首个体检合格投票：更新参照（vis7 教训：缺帅帧
                        // 恒非法会让参照永不建立 → D3 门旁路 → >32 幽灵泄漏）。
                        lastLegalGrid = voted
                        lastLegalCount = nVoted
                    }
                    // 数据飞轮：难帧（低置信/非法局面/新局面）落盘，供离线标注重训。
                    flywheelCollector.maybeCollect(bitmap, info, board, pieceRecognizer.lastDetections())
                    // C9-T1 重锚搭便车：检测成功候选过准入闸（v2 包含/边距 + v3 棋锚
                    // 一致性 + 陈旧逃生）后播种 tracker 并重建 refBoard；闸拦下时传播
                    // 继续承担 MOVING 显示（PC 定稿基线：单种子 + 棋锚纠偏）。
                    seedTrackerFromDetection(info, voted)
                } else {
                    // C9：跟踪活着时显示由传播承担（PC run() 同款）——不清屏不计时，
                    // 检测层照常跑（节流由 everyN 承担）。
                    if (propInfo != null) {
                        lockMisses = 0
                    } else {
                    // 迟滞：短暂漏检时保持上一次结果，避免闪烁。
                    lockMisses++
                    if (lockMisses > LOCK_MAX_MISSES &&
                        !(imuStatic && latestBoardInfo != null)
                    ) {
                        // D2 静止不清屏：IMU 判静止 ⇒ 手机没动，长漏检 = 手遮挡——
                        // 显示保持最后好四角/局面，检测层照常 fast-sampling 重试，
                        // 手离开后重锁成功即无缝衔接。真清屏只发生在移动/未知态。
                        latestBoardInfo = null
                        latestBoard = null
                        latestGrid = null
                        recentGrids.clear()
                        lastCentroids = emptyList()
                        boardDetector.setReferenceCentroids(emptyList())
                        // 移动重锁修复 C：锁定丢失时同步清检测器时间状态（prevCorners/EMA 基线/
                        // 跳变闸），下次成功走「无历史」路径（C7① 质心先验 + EMA 重新起步）。
                        boardDetector.resetTrackingState()
                        // C9：传播同步作废（PC unlock 路径同款），lkEngine 特征随下次 seed 重提。
                        boardTracker?.reset()
                        refBoard = null
                        lkEngine?.release()
                        lastFrameTsNs = 0L
                    }
                    }
                    // D1 挪盘兜底（双确认第③层内容验证）：检测器报告超时候选（连续拒绝
                    // ≥60 且四角整体平移 >0.5 格）时，用候选四角跑 YOLO 取**当前帧**棋子
                    // 质心——棋盘被挪则棋子随行（包含 ≥95% 且 ≥10 个）→ 清锚重锁；
                    // 不通过 → 重置计数继续保持（手遮挡漂移在此被拦，遮挡期质心不足
                    // 也天然不通过）。快路径仍是碰手机（IMU→MOVING 秒级清锚）。
                    val cand = boardDetector.reanchorCandidate()
                    if (cand != null) {
                        val probeInfo = BoardInfo(
                            corners = cand,
                            cellSize = BoardGeometry.CELL,
                            originX = BoardGeometry.MARGIN,
                            originY = BoardGeometry.MARGIN,
                            imageToGrid = Homography.compute(cand, BoardGeometry.physicalCorners),
                        )
                        pieceRecognizer.recognize(bitmap, probeInfo)
                        val cents = pieceRecognizer.lastDetections().map { floatArrayOf(it.cx, it.cy) }
                        val verified = cents.size >= SegmentationBoardDetector.MIN_REF_CENTROIDS &&
                            SegmentationBoardDetector.containmentRatio(cand, cents) >=
                            SegmentationBoardDetector.CENTROID_CONTAIN_MIN
                        boardDetector.resolveReanchor(verified)
                        Log.i(TAG, if (verified) "挪盘双确认通过（${cents.size} 质心随行）→ 重锚" else "挪盘双确认否决 → 保持")
                    }
                }
                val hasBoard = latestBoardInfo != null
                if (hasBoard != lastBoardDetected) {
                    lastBoardDetected = hasBoard
                    Log.i(TAG, if (hasBoard) "检测到棋盘" else "未检测到棋盘")
                }
            }

            // 三态显示（方案 §3）：STATIC=静止锚（检测器内部显示=锚，恒定）；
            // MOVING=传播（30fps，遮挡期间继续）；UNKNOWN=S7 检测行为（现状兜底）。
            val displayInfo = if (motion == ImuMotion.MOVING) propInfo ?: latestBoardInfo
            else latestBoardInfo
            runOnUiThread {
                binding.boardOverlayView.setImageSize(bitmap.width, bitmap.height)
                binding.boardOverlayView.setPieces(latestGrid)
                binding.boardOverlayView.setSuggestion(displayInfo, latestMove)
            }
        } catch (exc: Exception) {
            Log.e(TAG, "帧处理失败", exc)
        } finally {
            imageProxy.close()
        }
    }

    /** 局面子数（D3 合法性门用）。 */
    private fun countPieces(grid: Array<Array<Piece?>>): Int {
        var n = 0
        for (rank in 0..9) {
            for (file in 0..8) {
                if (grid[rank][file] != null) n++
            }
        }
        return n
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

    /** C9-T1 初始化：OpenCV 本地库 + tracker/engine；失败则帧间跟踪停用（回退现状）。 */
    private fun initFrameTracking() {
        try {
            if (!OpenCVLoader.initLocal()) throw IllegalStateException("OpenCVLoader.initLocal 返回 false")
            boardTracker = BoardTracker()
            lkEngine = LkFlowEngine()
            Log.i(TAG, "C9 帧间跟踪启用（OpenCV 就绪）")
        } catch (t: Throwable) {
            Log.w(TAG, "OpenCV 初始化失败，帧间跟踪停用（回退逐帧检测）", t)
            boardTracker = null
            lkEngine = null
        }
        // T2 陀螺样本喂给 RotationPredictor（时基：SensorEvent 与 CameraX 帧时间戳
        // 同为 elapsedRealtimeNanos 域；常数偏移真机校准项，见方案 §3）。
        motionMonitor.onGyroSample = { t, x, y, z -> rotationPredictor?.pushGyro(t, x, y, z) }
    }

    /**
     * C9-T1 帧间传播：bitmap→灰度 → LK（OpenCV）→ H 累积（纯 Kotlin 数学）。
     * 返回传播 BoardInfo（MOVING 显示源）或 null（无 H / 本帧失败——T2 旋转补位
     * 默认关，真机验证时基对齐后开）。每 CORRECT_EVERY 帧棋锚纠偏（YOLO 质心）。
     */
    private fun trackFrame(bitmap: Bitmap, frameTsNs: Long): BoardInfo? {
        val tracker = boardTracker ?: return null
        val engine = lkEngine ?: return null
        try {
            if (trackRgba == null || trackRgba!!.width() != bitmap.width ||
                trackRgba!!.height() != bitmap.height
            ) {
                trackRgba?.release(); trackGray?.release()
                trackRgba = Mat(bitmap.height, bitmap.width, CvType.CV_8UC4)
                trackGray = Mat(bitmap.height, bitmap.width, CvType.CV_8UC1)
                rotationPredictor = RotationPredictor.fromFov(
                    bitmap.width, bitmap.height, CAMERA_FOV_H_DEG)
            }
            Utils.bitmapToMat(bitmap, trackRgba)
            Imgproc.cvtColor(trackRgba, trackGray, Imgproc.COLOR_RGBA2GRAY)
            val gray = trackGray!!
            if (tracker.H == null) {
                trackAliveLast = false
                return null
            }
            if (tracker.featuresStale) {
                tracker.roiCorners?.let { engine.extractFeatures(gray, it) }
                tracker.onFeaturesFresh()
            }
            val prevTs = lastFrameTsNs
            lastFrameTsNs = frameTsNs
            val pairs = engine.track(gray)
            trackAliveLast = if (pairs != null) {
                tracker.accumulate(pairs.first, pairs.second, pairs.second.size / 2,
                    bitmap.width, bitmap.height)
            } else {
                tracker.markTrackFail()
                if (ENABLE_T2_ROTATION_FALLBACK && prevTs in 1 until frameTsNs) {
                    rotationPredictor?.rotationHomography(prevTs, frameTsNs)?.let(tracker::propagateExternal)
                }
                false
            }
            // 周期性跟踪体检（~5s 一次；死亡态也可见——LK 死锁诊断依赖此行）
            if (frameCounter % 150 == 0) {
                val rp = rotationPredictor
                Log.i(TAG, "C9 体检：alive=$trackAliveLast age=${tracker.age} " +
                    "内点=${tracker.lastInliers} 纠偏age=${tracker.lastCorrectAge} " +
                    "传播帧=${tracker.nTrackFrames} 失败=${tracker.nFail} " +
                    "T2零偏=${rp?.bias?.joinToString { "%.4f".format(it) }}")
            }
            if (trackAliveLast && nFailAnnounced && tracker.nFail == 0) nFailAnnounced = false
            if (!trackAliveLast) {
                if (!nFailAnnounced) {
                    nFailAnnounced = true
                    Log.i(TAG, "C9 LK 失效（age=${tracker.age} 失败=${tracker.nFail}）" +
                        "→ 等待 SAM 重播种")
                }
                return null
            }
            // T2 零偏在线估计（LK 成功帧；平移污染有界，方案 §3）
            if (rotationPredictor != null && prevTs in 1 until frameTsNs) {
                tracker.lastHlk?.let { rotationPredictor!!.updateBias(prevTs, frameTsNs, it) }
            }
            // 棋锚纠偏：ref_board（棋盘系质心真值）↔ 当前 YOLO 质心，钉住 H
            val rb = refBoard
            if (rb != null && rb.size >= 24 &&
                tracker.age % BoardTracker.CORRECT_EVERY == 0
            ) {
                var detPts = yoloCentroidsForTrack(bitmap, buildTrackInfo(tracker.corners()!!))
                if (detPts.size / 2 < BoardTracker.MIN_CORRECT_REFS) {
                    // 大漂移时按漂移四角裁剪会切棋子——外扩 1.5 倍重跑（PC 同款兜底）
                    detPts = yoloCentroidsForTrack(
                        bitmap, buildTrackInfo(expandCorners(tracker.corners()!!, 1.5f)))
                }
                val shift = tracker.pieceCorrect(rb, detPts) { msg ->
                    Log.i(TAG, "C9 纠偏内况：$msg")
                }   // 失败无害（保持传播 H）
                Log.i(TAG, "C9 纠偏尝试 ref=${rb.size / 2} det=${detPts.size / 2} " +
                    "age=${tracker.age} → ${shift?.let { "%.1fpx".format(it) } ?: "null"}")
            }
            val corners = tracker.corners() ?: return null
            return buildTrackInfo(corners)
        } catch (t: Throwable) {
            Log.e(TAG, "帧间跟踪失败（本帧跳过）", t)
            trackAliveLast = false
            return null
        }
    }

    /** 检测成功候选 → 准入闸 → 播种 + refBoard 重建（标准开局约定重置优先）。 */
    private fun seedTrackerFromDetection(info: BoardInfo, voted: Array<Array<Piece?>>) {
        val tracker = boardTracker ?: return
        if (lkEngine == null) return
        val dueReanchor = tracker.H == null || !trackAliveLast ||
            tracker.age >= BoardTracker.REANCHOR_EVERY
        if (!dueReanchor) return
        val refImg = refBoard?.let { tracker.refToImage(it) }
        when (BoardTracker.admitReanchor(info.corners, tracker, refImg, trackAliveLast)) {
            ReanchorDecision.ACCEPT -> {
                Log.i(TAG, "C9 播种（age=${tracker.age} 背书=${tracker.isBacked}）")
                tracker.seed(info.corners)
                rebuildRefBoard(info, voted)
            }
            ReanchorDecision.STALE_RESET -> {
                Log.i(TAG, "C9 陈旧逃生（age=${tracker.age}）→ 信 SAM 重建参考")
                refBoard = null
                tracker.seed(info.corners)
                rebuildRefBoard(info, voted)
            }
            ReanchorDecision.HOLD -> {
                // 有背书时的否决=拦下坏 SAM（重要事件）；无背书的存疑保持是常态不记
                if (tracker.isBacked) {
                    Log.i(TAG, "C9 准入闸否决（age=${tracker.age} 背书传播优先）")
                }
            }
        }
    }

    /** refBoard 重建：标准开局匹配 → 32 交点真值（约定重置）；否则质心投影。 */
    private fun rebuildRefBoard(info: BoardInfo, voted: Array<Array<Piece?>>) {
        val cents = lastCentroids
        if (StdOpening.matches(voted)) {
            refBoard = StdOpening.REF_BOARD.copyOf()
            Log.i(TAG, "标准开局约定重置（32 交点真值）")
            return
        }
        if (cents.isEmpty()) return
        val h = info.imageToGrid ?: return
        val pts = FloatArray(cents.size * 2)
        cents.forEachIndexed { i, p ->
            pts[i * 2] = p[0]; pts[i * 2 + 1] = p[1]
        }
        refBoard = HomographyMath.apply(h, pts)
    }

    private fun buildTrackInfo(corners: FloatArray): BoardInfo = BoardInfo(
        corners = corners,
        cellSize = BoardGeometry.CELL,
        originX = BoardGeometry.MARGIN,
        originY = BoardGeometry.MARGIN,
        imageToGrid = Homography.compute(corners, BoardGeometry.physicalCorners),
    )

    private fun expandCorners(c: FloatArray, k: Float): FloatArray {
        val mx = (c[0] + c[2] + c[4] + c[6]) / 4f
        val my = (c[1] + c[3] + c[5] + c[7]) / 4f
        return FloatArray(8) { i ->
            if (i % 2 == 0) mx + (c[i] - mx) * k else my + (c[i] - my) * k
        }
    }

    /** 跑一遍 YOLO 取质心（图像坐标 xy 交错）。 */
    private fun yoloCentroidsForTrack(bitmap: Bitmap, info: BoardInfo): FloatArray {
        pieceRecognizer.recognize(bitmap, info)
        val dets = pieceRecognizer.lastDetections()
        val out = FloatArray(dets.size * 2)
        for (i in dets.indices) {
            out[i * 2] = dets[i].cx; out[i * 2 + 1] = dets[i].cy
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

    override fun onResume() {
        super.onResume()
        motionMonitor.start()
    }

    override fun onPause() {
        motionMonitor.stop()
        super.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.close()
        pieceRecognizer.close()
        boardDetector.close()
        trackRgba?.release(); trackGray?.release()
        lkEngine?.release()
        cameraExecutor.shutdown()
        engineExecutor.shutdown()
        Log.i(TAG, "已释放资源")
    }

    companion object {
        private const val TAG = "ChessCoach"
        private const val DETECT_EVERY_N = 5

        /** C9-T2：LK 失效时的陀螺旋转单应补位（默认关——真机验证帧-IMU 时基
         *  对齐与零偏后再开；陀螺采样与零偏估计已常开积累校准数据）。 */
        private const val ENABLE_T2_ROTATION_FALLBACK = false

        /** C9-T2 内参：水平 FOV（度）。fx=W/(2·tan(fovH/2))；真机用 Camera2Interop
         *  读实际 FOV 校准（方案 §3 工程件 1）。 */
        private const val CAMERA_FOV_H_DEG = 66f

        /** 失败/未锁定期间的检测间隔（每帧，全力重锁）。 */
        private const val DETECT_EVERY_FAST = 1

        /** 连续漏检超过该次数才清空棋盘（S7：3→15；静止态由 D2 直接免清屏，
         *  本阈值实际只管移动/未知态——真移动时救援通常 1-3s 内重锁，走不到这里）。 */
        private const val LOCK_MAX_MISSES = 15

        /** 局面时间投票的历史帧数。 */
        private const val HISTORY_SIZE = 5
    }
}
