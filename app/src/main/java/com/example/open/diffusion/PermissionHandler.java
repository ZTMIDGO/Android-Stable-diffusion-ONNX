package com.example.open.diffusion;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.DialogFragment;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;

import java.util.List;
import java.util.Map;

/**
 * Created by ZTMIDGO 2021/11/26
 */
public class PermissionHandler {
    public static final String[] PERMISSIONS = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
    };

    public static void requestPermissionsWith(Activity activity, String[] permissions, int requestCode){
        ActivityCompat.requestPermissions(
                activity,
                permissions,
                requestCode
        );
    }

    public static ActivityResultLauncher createPermissionsWithArray(Fragment fragment, ActivityResultCallback<Map<String, Boolean>> callback){
        return fragment.registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), callback);
    }

    public static ActivityResultLauncher createPermissionsWithIntent(Fragment fragment, ActivityResultCallback<ActivityResult> callback){
        return fragment.registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), callback);
    }

    public static ActivityResultLauncher createPermissionsWithDocument(Fragment fragment, ActivityResultCallback<Uri> callback){
        return fragment.registerForActivityResult(new ActivityResultContracts.OpenDocument(), callback);
    }

    public static ActivityResultLauncher createPermissionsWithDocumentMuilte(Fragment fragment, ActivityResultCallback<List<Uri>> callback){
        return fragment.registerForActivityResult(new ActivityResultContracts.OpenMultipleDocuments(), callback);
    }

    public static Intent createImageIntent(){
        return new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
    }

    public static Intent createAudioIntent(){
        return new Intent(Intent.ACTION_PICK, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI);
    }

    public static void categoryImages(Activity activity, int requestCode) {
        activity.startActivityForResult(createImageIntent(), requestCode);
    }

    public static void categoryAudio(Activity activity, int requestCode) {
        activity.startActivityForResult(createAudioIntent(), requestCode);
    }

    public static boolean isRequestEnvironment(){
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager();
    }

    public static boolean checkPermissionAllGranted(Context context, String[] permissions) {
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
}
