package com.example.open.diffusion;

import android.app.Application;

import ai.onnxruntime.OrtEnvironment;

/**
 * Created by ZTMIDGO 2023/4/21
 */
public class App extends Application {
    public static final OrtEnvironment ENVIRONMENT = OrtEnvironment.getEnvironment();
    @Override
    public void onCreate() {
        super.onCreate();
    }
}
