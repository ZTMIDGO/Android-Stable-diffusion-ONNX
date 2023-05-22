package com.example.open.diffusion;

import android.content.Context;

/**
 * Created by ZTMIDGO 2023/4/21
 */
public class PathManager {
    public static String getModelPath(Context context){
        return context.getFilesDir().getAbsolutePath()+"/model";
    }

    public static String getAsssetOutputPath(Context context){
        return context.getFilesDir().getAbsolutePath();
    }
}
