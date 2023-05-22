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
}
