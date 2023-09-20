package com.example.open.diffusion;

import android.content.ContentValues;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.PixelFormat;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.FileChannel;

/**
 * Created by ZTMIDGO 2023/4/21
 */
public class FileUtils {

    public static void copyAssets(AssetManager assetManager, String path, File outPath) throws IOException {
        String[] assets = assetManager.list(path);

        if (assets != null) {
            if (assets.length == 0) {
                copyFile(assetManager, path, outPath);
            } else {
                File dir = new File(outPath, path);
                if (!dir.exists()) {
                    if (!dir.mkdirs()) {
                        Log.v("copyAssets", "Failed to create directory " + dir.getAbsolutePath());
                    }
                }

                String[] var5 = assets;
                int var6 = assets.length;

                for(int var7 = 0; var7 < var6; ++var7) {
                    String asset = var5[var7];
                    copyAssets(assetManager, path + "/" + asset, outPath);
                }
            }

        }
    }

    private static void copyFile(AssetManager assetManager, String fileName, File outPath) throws IOException {
        Log.v("copyFile", "Copy " + fileName + " to " + outPath);
        InputStream in = assetManager.open(fileName);
        OutputStream out = new FileOutputStream(outPath + "/" + fileName);
        byte[] buffer = new byte[4000];

        int read;
        while((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }

        in.close();
        out.close();
    }

    public static void write(InputStream in, String filePath, String name) throws IOException {
        createPath(new File(filePath));
        int index;
        byte[] bytes = new byte[1024];
        OutputStream out = new FileOutputStream(filePath + "/" + name);
        while ((index = in.read(bytes)) != -1) {
            out.write(bytes, 0, index);
            out.flush();
        }
        in.close();
        out.close();
    }

    public static void createPath(File file){
        if (!file.exists()){
            file.mkdirs();
        }
    }

    public static void deleteFile(String filePath){
        File file = new File(filePath);
        if (file.exists()){
            file.delete();
        }
    }

    public static void deleteAllFile(File file){
        if(file.isFile()){
            file.delete();
            return;
        }

        File[] files = file.listFiles();

        if(files == null)
            return;

        for(int i = 0; i < files.length; i++){
            File f = files[i];
            if(f.isFile()){
                f.delete();
            }else{
                deleteAllFile(f);
                f.delete();
            }
        }

        file.delete();
    }

    public static boolean saveImage(Context context, Bitmap toBitmap) throws Exception {
        boolean success = false;
        if(Build.VERSION.SDK_INT <= Build.VERSION_CODES.Q) {
            String path = MediaStore.Images.Media.insertImage(context.getContentResolver(), toBitmap, "壁纸", "搜索猫相关图片后保存的图片");
            success = !TextUtils.isEmpty(path);
        }else {
            Uri insertUri = context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, new ContentValues());
            OutputStream outputStream = context.getContentResolver().openOutputStream(insertUri, "rw");
            success = toBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
        }
        return success;
    }

    public static Bitmap getBitmap(Drawable drawable) {
        Bitmap bitmap = Bitmap.createBitmap(
                drawable.getIntrinsicWidth(),
                drawable.getIntrinsicHeight(),
                drawable.getOpacity() != PixelFormat.OPAQUE ? Bitmap.Config.ARGB_8888
                        : Bitmap.Config.RGB_565);
        Canvas canvas = new Canvas(bitmap);
        drawable.setBounds(0, 0, drawable.getIntrinsicWidth(), drawable.getIntrinsicHeight());
        drawable.draw(canvas);
        return bitmap;
    }

}
