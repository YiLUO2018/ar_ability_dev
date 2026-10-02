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
import com.arability.chinese_chess_coach_phone_version.chess.GridBoardDetector
import com.arability.chinese_chess_coach_phone_version.chess.Move
import com.arability.chinese_chess_coach_phone_version.chess.Piece
import com.arability.chinese_chess_coach_phone_version.chess.PieceColor
import com.arability.chinese_chess_coach_phone_version.chess.PieceRecognizer
import com.arability.chinese_chess_coach_phone_version.chess.PieceRecognizerFactory
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

    @Volatile private var lensFacing = CameraSelector.LENS_FACING_BACK
    @Volatile private var sideToMove = PieceColor.RED

    private var frameCounter = 0

    @Volatile private var latestBoardInfo: BoardInfo? = null
    @Volatile private var latestBoard: Board? = null
    @Volatile private var latestMove: Move? = null
    @Volatile private var lastBoardDetected = false

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
        boardDetector = GridBoardDetector(debugDir = java.io.File(filesDir, "debug"))
        pieceRecognizer = PieceRecognizerFactory.create(this)
        engine = ChessEngineFactory.create()

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
                        Log.i(TAG, "建议走法: ${move.toNotation()}")
                        Toast.makeText(this, "建议走法: ${move.piece.name} ${move.toNotation()}", Toast.LENGTH_SHORT).show()
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

            frameCounter++
            if (frameCounter % DETECT_EVERY_N == 0) {
                val info = boardDetector.detect(bitmap)
                latestBoardInfo = info
                if (info != null) {
                    val grid = pieceRecognizer.recognize(info)
                    latestBoard = buildBoard(grid)
                } else {
                    latestBoard = null
                }
                if ((info != null) != lastBoardDetected) {
                    lastBoardDetected = info != null
                    Log.i(TAG, if (info != null) "检测到棋盘" else "未检测到棋盘")
                }
            }

            runOnUiThread {
                binding.boardOverlayView.setImageSize(bitmap.width, bitmap.height)
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
        cameraExecutor.shutdown()
        engineExecutor.shutdown()
        Log.i(TAG, "已释放资源")
    }

    companion object {
        private const val TAG = "ChessCoach"
        private const val DETECT_EVERY_N = 5
    }
}
