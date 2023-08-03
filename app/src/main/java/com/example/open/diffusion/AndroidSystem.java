package com.example.open.diffusion;

import android.app.Activity;
import android.app.ActivityManager;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.PowerManager;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.Base64;
import android.util.DisplayMetrics;
import android.view.View;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NotificationCompat;
import androidx.core.graphics.drawable.DrawableCompat;
import androidx.documentfile.provider.DocumentFile;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import java.io.ByteArrayOutputStream;
import java.util.List;
import java.util.Locale;

/**
 * Created by ZTMIDGO on 2018/2/9.
 */

public class AndroidSystem {

    public static float px2dip(Context context, float pxValue) {
        final float scale = context.getResources().getDisplayMetrics().density;
        return pxValue / scale + 0.5f;
    }

    public static float dip2px(Context context, float dipValue) {
        final float scale = context.getResources().getDisplayMetrics().density;
        return dipValue * scale + 0.5f;
    }

    public static void tintDrawable(Drawable drawable, ColorStateList colors) {
        DrawableCompat.setTintList(drawable, colors);
    }

    public static void startApplicationDetails(Context context){
        Intent intent = new Intent();
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        intent.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        intent.setData(Uri.parse("package:" + context.getPackageName()));
        context.startActivity(intent);
    }

    public static int getVersionCode(Context context) {
        int versionCode = 0;
        try {
            PackageInfo info = getPackageInfo(context);
            versionCode = info.versionCode;
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            return versionCode;
        }
    }

    public static void cleanFocus(Context context){
        View focuView = ((AppCompatActivity)context).getCurrentFocus();
        if (focuView != null){
            focuView.clearFocus();
            AndroidSystem.hideInput((AppCompatActivity)context);
        }
    }

    public static Bitmap viewToBitmap(View view) {
        view.setDrawingCacheEnabled(true);
        Bitmap bitmap = view.getDrawingCache();
        view.setDrawingCacheEnabled(false);
        return bitmap;
    }

    public static byte[] viewToBytes(View view) {
        view.setDrawingCacheEnabled(true);
        Bitmap bitmap = view.getDrawingCache();
        byte[] bytes = bitmapToBytes(bitmap);
        view.setDrawingCacheEnabled(false);
        return bytes;
    }

    public static byte[] bitmapToBytes(Bitmap bitmap){
        return bitmapToBytes(bitmap, 100);
    }

    public static byte[] bitmapToBytes(Bitmap bitmap, int quality){
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, baos);
            byte[] data = baos.toByteArray();
            baos.close();
            return data;
        }catch (Exception e){
            e.printStackTrace();
        }
        return null;
    }

    public static String bytesToBase64(Bitmap bitmap, int quality){
        byte[] bytes = AndroidSystem.bitmapToBytes(bitmap, quality);
        String base = Base64.encodeToString(bytes, Base64.DEFAULT);
        return base;
    }

    public static PackageInfo getPackageInfo(Context context) throws Exception {
        PackageManager manager = context.getPackageManager();
        PackageInfo info = manager.getPackageInfo(context.getPackageName(), 0);
        return info;
    }

    public static String getUriFileName(Context context, DocumentFile documentFile){
        String fileName = TextUtils.isEmpty(documentFile.getName()) ? "" : documentFile.getName().toLowerCase().replaceFirst("(\\.[^.]*)$", "");
        return fileName;
    }

    public static String getUriFileName(Context context, Uri uri){
        DocumentFile documentFile = DocumentFile.fromSingleUri(context, uri);
        return getUriFileName(context, documentFile);
    }

    public static int[] getScreenWidthAndHeight(Activity activity) {
        DisplayMetrics outMetrics = new DisplayMetrics();
        activity.getWindowManager().getDefaultDisplay().getRealMetrics(outMetrics);
        int widthPixel = outMetrics.widthPixels;
        int heightPixel = outMetrics.heightPixels;

        return new int[]{widthPixel, heightPixel};
    }

    public static void showInput(Context context, View view) {
        view.requestFocus();
        InputMethodManager imm = (InputMethodManager) context.getSystemService(context.INPUT_METHOD_SERVICE);
        imm.showSoftInput(view, InputMethodManager.SHOW_IMPLICIT);
    }

    public static boolean isInputMethodShow(View rootView){
        final View v = rootView.findFocus();
        return v != null && v.getWindowToken() != null;
    }

    public static void cleanFocus(View rootView){
        View v = rootView.findFocus();
        if (v != null){
            v.clearFocus();
        }
    }

    public static String getProcessName(Context context) {
        int pid = android.os.Process.myPid();
        ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        List<ActivityManager.RunningAppProcessInfo> runningApps = am.getRunningAppProcesses();
        if (runningApps == null) {
            return null;
        }
        for (ActivityManager.RunningAppProcessInfo procInfo : runningApps) {
            if (procInfo.pid == pid) {
                return procInfo.processName;
            }
        }
        return null;
    }

    public static boolean isBackgroundRestricted(Context context){
        ActivityManager activityManager = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P){
            return activityManager.isBackgroundRestricted();
        }else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            PowerManager powerManager = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
            return !powerManager.isIgnoringBatteryOptimizations(context.getPackageName());
        }
        return true;
    }

    public static void closeBackgroundRestricted(Context context){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P){
            AndroidSystem.startApplicationDetails(context);
        }else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
            intent.setData(Uri.parse("package:" + context.getPackageName()));
            context.startActivity(intent);
        }
    }

    public static NotificationCompat.Builder getNotificationBuilder(Context context, NotificationManager manager, int importance, String id, String name) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(id, name, importance);
            channel.enableVibration(true);
            channel.enableLights(false);
            channel.setLockscreenVisibility(Notification.VISIBILITY_PUBLIC);
            manager.createNotificationChannel(channel);
        }
        NotificationCompat.Builder builder = new NotificationCompat.Builder(context, id);
        return builder;
    }

    public static void hideInput(Activity context) {
        InputMethodManager imm = (InputMethodManager) context.getSystemService(context.INPUT_METHOD_SERVICE);
        View view = context.getCurrentFocus();

        if (null != view) {
            imm.hideSoftInputFromWindow(view.getWindowToken(), InputMethodManager.HIDE_NOT_ALWAYS);
        }
    }

    public static String getSystemLanguage(){
        Locale locale = Locale.getDefault();
        return locale.getLanguage()+"-"+locale.getCountry();
    }

    public static boolean isTraditionalLanguage(){
        boolean isHK = getSystemLanguage().equals("zh-HK");
        boolean isTW = getSystemLanguage().equals("zh-TW");
        return isHK || isTW;
    }

    public static int getScreenBrightness(Activity activity) {
        int value = 0;
        try {
            value = Settings.System.getInt(activity.getContentResolver(), Settings.System.SCREEN_BRIGHTNESS);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return value;
    }

    public static void setAppBrightness(Activity activity, int brightness) {

    }

    public static void openSystemWeb(Context context, String url){
        Intent intent= getWebIntent(url);
        context.startActivity(intent);
    }

    public static Intent getWebIntent(String url){
        Intent intent= new Intent();
        intent.setAction(Intent.ACTION_VIEW);
        Uri uri = Uri.parse(url);
        intent.setData(uri);
        return intent;
    }

    public static int[] getSystemWidthAndHeight(Context context) {
        int[] wh = new int[2];
        WindowManager wm = (WindowManager) context.getSystemService(context.WINDOW_SERVICE);
        DisplayMetrics dm = new DisplayMetrics();
        wm.getDefaultDisplay().getMetrics(dm);
        wh[0] = dm.widthPixels;
        wh[1] = dm.heightPixels;
        return wh;
    }
}
