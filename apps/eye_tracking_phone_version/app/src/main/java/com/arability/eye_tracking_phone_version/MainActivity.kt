package com.arability.eye_tracking_phone_version

import android.Manifest
import android.content.pm.PackageManager
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
import com.arability.eye_tracking_phone_version.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * 眼球追踪 App 骨架（阶段一）：
 * 前置摄像头近距预览 + ImageAnalysis 帧回调桩 + 帧率/分辨率状态显示 + 前后摄切换。
 * 瞳孔检测、注视估计、校准按 docs/01-现行设计/设计文档.md 的阶段规划逐步接入。
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    /** 默认前置（自拍视角近距拍眼，对应「前置 + 外接微距」原型路线）。 */
    @Volatile private var lensFacing = CameraSelector.LENS_FACING_FRONT

    // ---- 帧率统计（拍摄条件观测用） ----
    private var frameCount = 0
    private var lastFpsAtMs = 0L
    private var lastFpsFrameCount = 0

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
        binding.switchCameraButton.setOnClickListener { onSwitchCameraClicked() }
        Log.i(TAG, "眼球追踪骨架初始化完成")

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
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
            Toast.makeText(this, "绑定相机失败：${exc.message}", Toast.LENGTH_LONG).show()
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
            val selector = if (targetFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }
            if (!cameraProvider.hasCamera(selector)) {
                Toast.makeText(this, "该设备没有此摄像头", Toast.LENGTH_SHORT).show()
                return@addListener
            }
            lensFacing = targetFacing
            bindCamera(cameraProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * 帧回调桩：仅统计帧率与分辨率，每秒回主线程刷新状态行。
     * 阶段二在此接入瞳孔检测（ImageProxy → Bitmap → PupilDetector）。
     */
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            frameCount++
            val now = System.currentTimeMillis()
            if (lastFpsAtMs == 0L) {
                lastFpsAtMs = now
                lastFpsFrameCount = frameCount
                return
            }
            val elapsed = now - lastFpsAtMs
            if (elapsed >= FPS_WINDOW_MS) {
                val fps = (frameCount - lastFpsFrameCount) * 1000.0 / elapsed
                lastFpsAtMs = now
                lastFpsFrameCount = frameCount
                val size = "${imageProxy.width}x${imageProxy.height}"
                val facing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) "前置" else "后置"
                val text = "$facing  $size  ${"%.1f".format(fps)} fps"
                Log.i(TAG, "frame: $text")
                runOnUiThread { binding.statusView.text = text }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "帧处理异常", t)
        } finally {
            imageProxy.close()
        }
    }

    companion object {
        private const val TAG = "EyeTracking"
        private const val FPS_WINDOW_MS = 1000L
    }
}
