package com.example.objectdetection

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.objectdetection.ml.LiteModelEfficientdetLite0DetectionMetadata1
import com.example.objectdetection.screens.CameraActivity
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

@SuppressLint("MissingInflatedId")
@RequiresApi(Build.VERSION_CODES.TIRAMISU)
class DashboardActivity : AppCompatActivity() {

    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 101
        private const val MEDIA_PERMISSION_REQUEST_CODE = 102
        private const val IMAGE_TARGET_SIZE = 300
        private const val MIN_CONFIDENCE = 0.5f
    }

    private val model by lazy { LiteModelEfficientdetLite0DetectionMetadata1.newInstance(this) }
    private val labels by lazy { FileUtil.loadLabels(this, "labels.txt") }
    private val imageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()
    }

    private lateinit var predictBtn: FrameLayout
    private lateinit var imageView: ImageView
    private lateinit var result: TextView

    private var selectedImageUri: Uri? = null

    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let {
            selectedImageUri = it
            imageView.setImageURI(it)
            predictBtn.isEnabled = true
        } ?: showToast("No Image Selected")
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) launchPhotoPicker() else showPermissionDeniedMessage()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_dashboard)

        window.statusBarColor = Color.BLACK
        window.navigationBarColor = Color.BLACK

        setupWindowInsets()
        initViews()
        setupClickListeners()
    }

    private fun setupWindowInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.dashboardView)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

    private fun initViews() {
        imageView = findViewById(R.id.image)
        result = findViewById(R.id.result)
        predictBtn = findViewById<FrameLayout>(R.id.predictbtn).apply { isEnabled = false }
        findViewById<FrameLayout>(R.id.camerabtn).setOnClickListener { checkCameraPermission() }
        findViewById<FrameLayout>(R.id.folderbtn).setOnClickListener { checkStoragePermission() }
    }

    private fun setupClickListeners() {
        predictBtn.setOnClickListener {
            selectedImageUri?.let(::predictImage) ?: run {
                result.text = "Please Select an Image First"
            }
        }
    }

    private fun checkStoragePermission() {
        when {
            hasPermission(Manifest.permission.READ_MEDIA_IMAGES) -> launchPhotoPicker()
            true -> launchPhotoPicker()
            shouldShowRationale(Manifest.permission.READ_EXTERNAL_STORAGE) -> showPermissionRationale()
            else -> requestPermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun checkCameraPermission() {
        when {
            hasPermission(Manifest.permission.CAMERA) -> launchCameraActivity()
            shouldShowRationale(Manifest.permission.CAMERA) -> {
                showToast("We need Camera access for Detection")
                requestCameraPermission()
            }
            else -> requestCameraPermission()
        }
    }

    private fun requestCameraPermission() {
        requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when (requestCode) {
            MEDIA_PERMISSION_REQUEST_CODE -> handleMediaPermissionResult(permissions, grantResults)
            CAMERA_PERMISSION_REQUEST_CODE -> handleCameraPermissionResult(grantResults)
        }
    }

    private fun predictImage(uri: Uri) {
        try {
            val bitmap = decodeSampledBitmapFromUri(uri, IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE)
                ?: throw Exception("Failed to Decode Image")

            val predictions = processImageWithModel(bitmap)
            displayPredictions(predictions)

        } catch (e: Exception) {
            result.text = "Error: ${e.message}"
            Log.e("Prediction", "Error", e)
        }
    }

    private fun decodeSampledBitmapFromUri(uri: Uri, reqWidth: Int, reqHeight: Int): Bitmap? {
        return contentResolver.openInputStream(uri)?.use { stream ->
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
                BitmapFactory.decodeStream(stream, null, this)
                inSampleSize = calculateInSampleSize(this, reqWidth, reqHeight)
                inJustDecodeBounds = false
            }
            contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, options)
            }
        }
    }

    private fun processImageWithModel(bitmap: Bitmap): LiteModelEfficientdetLite0DetectionMetadata1.Outputs {
        return model.process(
            imageProcessor.process(
                TensorImage.fromBitmap(bitmap)
            )
        )
    }

    private fun displayPredictions(outputs: LiteModelEfficientdetLite0DetectionMetadata1.Outputs) {
        val predictions = buildString {
            outputs.scoreAsTensorBuffer.floatArray.forEachIndexed { index, score ->
                if (score > MIN_CONFIDENCE) {
                    val label = labels.getOrNull(outputs.categoryAsTensorBuffer.getIntValue(index)) ?: "Unknown"
                    append(label)
                }
            }
            if (isEmpty()) append("No Objects Detected")
        }
        result.text = predictions
    }

    private fun calculateInSampleSize(options: BitmapFactory.Options, reqWidth: Int, reqHeight: Int): Int {
        val (height, width) = options.outHeight to options.outWidth
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight = height / 2
            val halfWidth = width / 2

            while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }
        return inSampleSize
    }

    private fun hasPermission(permission: String) =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    private fun shouldShowRationale(permission: String) =
        shouldShowRequestPermissionRationale(permission)

    private fun showToast(message: String) =
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()

    private fun launchPhotoPicker() =
        pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))

    private fun launchCameraActivity() =
        startActivity(Intent(this, CameraActivity::class.java))

    private fun showPermissionRationale() {
        AlertDialog.Builder(this)
            .setTitle("Permission Needed")
            .setMessage("This app needs access to your photos to select images")
            .setPositiveButton("OK") { _, _ ->
                requestPermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showPermissionDeniedMessage() =
        showToast("Permission Denied. Some features may not work.")

    private fun handleMediaPermissionResult(permissions: Array<out String>, grantResults: IntArray) {
        val grantedPermissions = permissions.zip(grantResults.toTypedArray()).toMap()
        val hasAccess = grantedPermissions[Manifest.permission.READ_MEDIA_IMAGES] == PackageManager.PERMISSION_GRANTED ||
                (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE &&
                        grantedPermissions[Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED] == PackageManager.PERMISSION_GRANTED)

        if (hasAccess) launchPhotoPicker() else showPermissionDeniedMessage()
    }

    private fun handleCameraPermissionResult(grantResults: IntArray) {
        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            launchCameraActivity()
        } else {
            showPermissionDeniedMessage()
        }
    }

    override fun onDestroy() {
        model.close()
        super.onDestroy()
    }
}