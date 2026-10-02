package com.arability.face_recognition_phone_version

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.arability.face_recognition_phone_version.databinding.ActivityMainBinding
import com.arability.face_recognition_phone_version.face.FaceBox
import com.arability.face_recognition_phone_version.face.FaceBoxSmoother
import com.arability.face_recognition_phone_version.face.FaceDatabase
import com.arability.face_recognition_phone_version.face.FaceDetector
import com.arability.face_recognition_phone_version.face.FaceEmbedder
import com.arability.face_recognition_phone_version.face.FaceEngine
import java.io.ByteArrayOutputStream
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var faceDetector: FaceDetector
    private lateinit var faceEmbedder: FaceEmbedder
    private val boxSmoother = FaceBoxSmoother()
    private lateinit var faceDatabase: FaceDatabase

    // 当前相机朝向（前置/后置），默认后置
    @Volatile private var lensFacing = CameraSelector.LENS_FACING_BACK

    // 识别节流与结果缓存
    private var frameCounter = 0
    private var lastLabels: List<String?> = emptyList()
    private var lastTopLabel: String? = null

    // 注册用：保存最近一帧与检测结果
    @Volatile private var lastBitmap: Bitmap? = null
    @Volatile private var lastFaces: List<FaceBox> = emptyList()

    // —— 运行统计（用于日志诊断）——
    private var frameCount = 0L
    private var windowStartMs = 0L
    private var totalDetectMs = 0L
    private var lastFaceCount = -1
    private var rawLastBox: FaceBox? = null
    private var smoothLastBox: FaceBox? = null
    private var totalRawDelta = 0f
    private var totalSmoothDelta = 0f

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
        faceDetector = FaceEngine.createDetector(this)
        faceEmbedder = FaceEngine.createEmbedder(this)
        faceDatabase = FaceEngine.createDatabase(this)
        binding.registerButton.setOnClickListener { onRegisterClicked() }
        binding.switchCameraButton.setOnClickListener { onSwitchCameraClicked() }
        Log.i(TAG, "FaceDetector 初始化完成")

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
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
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
            boxSmoother.reset()
            bindCamera(cameraProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            val rotation = imageProxy.imageInfo.rotationDegrees

            val detectStart = SystemClock.elapsedRealtime()
            val rawFaces = faceDetector.detect(bitmap)
            val detectMs = SystemClock.elapsedRealtime() - detectStart
            val faces = boxSmoother.smooth(rawFaces)

            lastBitmap = bitmap
            lastFaces = faces

            val labels = recognizeFaces(bitmap, faces)

            recordStats(bitmap, rotation, rawFaces, faces, detectMs)

            runOnUiThread {
                binding.faceOverlayView.setImageSize(bitmap.width, bitmap.height)
                binding.faceOverlayView.setFaces(faces, labels)
            }
        } catch (exc: Exception) {
            Log.e(TAG, "帧处理失败", exc)
        } finally {
            imageProxy.close()
        }
    }

    private fun recordStats(
        bitmap: Bitmap,
        rotation: Int,
        rawFaces: List<FaceBox>,
        faces: List<FaceBox>,
        detectMs: Long
    ) {
        // 人脸数量变化时打即时日志（诊断框时有时无）
        if (faces.size != lastFaceCount) {
            Log.i(
                TAG,
                "faces $lastFaceCount -> ${faces.size} (rotation=$rotation, bitmap=${bitmap.width}x${bitmap.height})"
            )
            lastFaceCount = faces.size
        }

        // 逐帧累计抖动（平滑前后分别统计）
        val rawBox = rawFaces.firstOrNull()
        val smoothBox = faces.firstOrNull()
        rawLastBox?.let { prev -> if (rawBox != null) totalRawDelta += boxDelta(rawBox, prev) }
        smoothLastBox?.let { prev -> if (smoothBox != null) totalSmoothDelta += boxDelta(smoothBox, prev) }
        rawLastBox = rawBox
        smoothLastBox = smoothBox

        frameCount++
        totalDetectMs += detectMs
        val now = SystemClock.elapsedRealtime()
        if (windowStartMs == 0L) windowStartMs = now

        if (now - windowStartMs >= LOG_INTERVAL_MS) {
            val fps = frameCount * 1000f / (now - windowStartMs)
            val avgDetect = totalDetectMs.toFloat() / frameCount
            val avgRawJitter = totalRawDelta / frameCount
            val avgSmoothJitter = totalSmoothDelta / frameCount

            val sb = StringBuilder()
            sb.append(String.format(Locale.US, "fps=%.1f avgDetect=%.1fms", fps, avgDetect))
            sb.append(" bitmap=${bitmap.width}x${bitmap.height} rotation=$rotation faces=${faces.size} raw=${rawFaces.size}")
            sb.append(
                String.format(
                    Locale.US,
                    " | rawJitter=%.4f smoothJitter=%.4f",
                    avgRawJitter,
                    avgSmoothJitter
                )
            )

            faces.forEachIndexed { i, b ->
                sb.append(
                    String.format(
                        Locale.US,
                        " | box$i[l=%.3f t=%.3f r=%.3f b=%.3f s=%.2f]",
                        b.left, b.top, b.right, b.bottom, b.score
                    )
                )
            }

            Log.i(TAG, sb.toString())

            frameCount = 0
            totalDetectMs = 0
            totalRawDelta = 0f
            totalSmoothDelta = 0f
            windowStartMs = now
        }
    }

    private fun boxDelta(a: FaceBox, b: FaceBox): Float =
        maxOf(
            abs(a.left - b.left),
            abs(a.top - b.top),
            abs(a.right - b.right),
            abs(a.bottom - b.bottom)
        )

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
        // 前置相机预览默认镜像，而分析数据未镜像，这里水平翻转使之一致
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

    private fun recognizeFaces(bitmap: Bitmap, faces: List<FaceBox>): List<String?> {
        frameCounter++
        if (faces.isEmpty()) {
            lastLabels = emptyList()
            return emptyList()
        }
        // 识别节流：每 N 帧执行一次，其余复用上次结果
        if (frameCounter % RECOGNIZE_EVERY_N != 0 && lastLabels.size == faces.size) {
            return lastLabels
        }

        val results = faces.map { face ->
            try {
                val crop = alignFace(bitmap, face)
                val embedding = faceEmbedder.embed(crop)
                val topSimilarity = faceDatabase.topSimilarity(embedding)
                val match = faceDatabase.match(embedding)
                FaceRecognition(
                    label = match?.let { String.format(Locale.US, "%s %.2f", it.name, it.similarity) },
                    topSimilarity = topSimilarity
                )
            } catch (exc: Exception) {
                Log.e(TAG, "识别失败", exc)
                FaceRecognition(null, 0f)
            }
        }
        lastLabels = results.map { it.label }

        val top = results.firstOrNull()
        val topLabel = top?.label
        if (topLabel != lastTopLabel) {
            val sim = top?.topSimilarity ?: 0f
            Log.i(TAG, String.format(Locale.US, "recognition: %s (topSim=%.3f)", topLabel ?: "未匹配", sim))
            lastTopLabel = topLabel
        }
        return lastLabels
    }

    private fun cropFace(bitmap: Bitmap, box: FaceBox): Bitmap {
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()
        val cx = (box.left + box.right) / 2f * w
        val cy = (box.top + box.bottom) / 2f * h
        val side = max((box.right - box.left) * w, (box.bottom - box.top) * h) * CROP_PADDING
        val half = side / 2f
        val left = (cx - half).toInt().coerceIn(0, bitmap.width - 1)
        val top = (cy - half).toInt().coerceIn(0, bitmap.height - 1)
        val right = (cx + half).toInt().coerceIn(left + 1, bitmap.width)
        val bottom = (cy + half).toInt().coerceIn(top + 1, bitmap.height)
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
    }

    /**
     * 人脸对齐：优先用双眼关键点做相似变换（旋转 + 缩放），映射到 112×112 标准位置；
     * 无关键点时回退到简单方形裁剪。
     */
    private fun alignFace(bitmap: Bitmap, box: FaceBox): Bitmap {
        val kps = box.keypoints
        if (kps.size >= 2) {
            val aligned = alignByEyes(bitmap, kps[0], kps[1])
            if (aligned != null) return aligned
        }
        return cropFace(bitmap, box)
    }

    /**
     * 用双眼（归一化 [0,1]）构建相似变换，输出 FACE_SIZE×FACE_SIZE 对齐人脸。
     * 返回 null 表示关键点退化（两眼距离过小），交由调用方回退。
     */
    private fun alignByEyes(bitmap: Bitmap, eye0: FloatArray, eye1: FloatArray): Bitmap? {
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()
        var p0x = eye0[0] * w
        var p0y = eye0[1] * h
        var p1x = eye1[0] * w
        var p1y = eye1[1] * h

        // 保证 p0 在左、p1 在右，与标准模板左/右眼一一对应
        if (p0x > p1x) {
            val tx = p0x; p0x = p1x; p1x = tx
            val ty = p0y; p0y = p1y; p1y = ty
        }

        val dx = p1x - p0x
        val dy = p1y - p0y
        val srcDist = sqrt(dx * dx + dy * dy)
        if (srcDist < 1e-3f) return null

        val targetDx = EYE_RIGHT_X - EYE_LEFT_X
        val targetDy = EYE_RIGHT_Y - EYE_LEFT_Y
        val targetDist = sqrt(targetDx * targetDx + targetDy * targetDy)
        val scale = targetDist / srcDist
        val angle = Math.toDegrees(
            atan2(dy.toDouble(), dx.toDouble()) - atan2(targetDy.toDouble(), targetDx.toDouble())
        ).toFloat()

        val matrix = Matrix()
        matrix.preTranslate(EYE_LEFT_X, EYE_LEFT_Y)
        matrix.preScale(scale, scale)
        matrix.preRotate(angle)
        matrix.preTranslate(-p0x, -p0y)

        val output = Bitmap.createBitmap(FACE_SIZE, FACE_SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawBitmap(bitmap, matrix, Paint(Paint.FILTER_BITMAP_FLAG))
        return output
    }

    private fun onRegisterClicked() {
        val bitmap = lastBitmap
        val face = lastFaces.firstOrNull()
        if (bitmap == null || face == null) {
            Toast.makeText(this, "尚未检测到人脸", Toast.LENGTH_SHORT).show()
            return
        }
        cameraExecutor.execute {
            try {
                val crop = alignFace(bitmap, face)
                val embedding = faceEmbedder.embed(crop)
                runOnUiThread { promptNameAndRegister(embedding) }
            } catch (exc: Exception) {
                Log.e(TAG, "注册嵌入失败", exc)
                runOnUiThread { Toast.makeText(this, "注册失败", Toast.LENGTH_SHORT).show() }
            }
        }
    }

    private fun promptNameAndRegister(embedding: FloatArray) {
        val input = EditText(this)
        input.hint = "输入姓名"
        AlertDialog.Builder(this)
            .setTitle("注册人脸")
            .setView(input)
            .setPositiveButton("确定") { _, _ ->
                val name = input.text.toString().trim()
                if (name.isEmpty()) {
                    Toast.makeText(this, "姓名不能为空", Toast.LENGTH_SHORT).show()
                } else {
                    faceDatabase.register(name, embedding)
                    Log.i(TAG, "已注册: $name（库大小=${faceDatabase.size()}）")
                    Toast.makeText(this, "已注册 $name", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private data class FaceRecognition(val label: String?, val topSimilarity: Float)

    override fun onDestroy() {
        super.onDestroy()
        faceDetector.close()
        faceEmbedder.close()
        cameraExecutor.shutdown()
        Log.i(TAG, "FaceDetector 已释放")
    }

    companion object {
        private const val TAG = "FaceRecognition"
        private const val LOG_INTERVAL_MS = 1000L
        private const val RECOGNIZE_EVERY_N = 3
        private const val CROP_PADDING = 1.2f

        // 112×112 标准对齐模板的双眼坐标（InsightFace/RetinaFace 5 点模板）
        private const val FACE_SIZE = 112
        private const val EYE_LEFT_X = 30.2946f
        private const val EYE_LEFT_Y = 51.6963f
        private const val EYE_RIGHT_X = 65.5318f
        private const val EYE_RIGHT_Y = 51.5014f
    }
}
