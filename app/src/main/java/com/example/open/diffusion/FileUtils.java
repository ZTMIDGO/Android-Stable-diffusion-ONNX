package com.example.open.diffusion;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
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

}
