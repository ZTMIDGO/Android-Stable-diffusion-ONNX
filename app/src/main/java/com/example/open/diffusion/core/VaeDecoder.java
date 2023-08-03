package com.example.open.diffusion.core;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import com.example.open.diffusion.App;
import com.example.open.diffusion.Device;
import com.example.open.diffusion.PathManager;

import java.io.File;
import java.util.EnumSet;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

/**
 * Created by ZTMIDGO 2023/3/24
 */
public class VaeDecoder {
    private final Context context;
    private final String model = "vae_decoder/model.ort";

    private OrtSession session;
    private int deviceId;

    public VaeDecoder(Context context, int deviceId) {
        this.context = context;
        this.deviceId = deviceId;
    }

    public void init() throws Exception {
        if (session != null) return;
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.addConfigEntry("session.load_model_format", "ORT");
        if (deviceId == Device.NNAPI) options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED));
        File file = new File(PathManager.getCustomPath(context) + "/" + model);
        session = App.ENVIRONMENT.createSession(file.exists() ? file.getAbsolutePath() : PathManager.getModelPath(context) +"/" +model, options);
    }

    public Object Decoder(Map<String, OnnxTensor> input) throws Exception {
        init();
        OrtSession.Result result = session.run(input);
        Object value = result.get(0).getValue();
        result.close();
        close();
        return value;
    }

    public Bitmap ConvertToImage(float[][][][] output, int width, int height, String imageName) throws Exception {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (int) Math.round(clamp((output[0][0][y][x] / 2 + 0.5), 0f, 1f) * 255f);
                int g = (int) Math.round(clamp((output[0][1][y][x] / 2 + 0.5), 0f, 1f) * 255f);
                int b = (int) Math.round(clamp((output[0][2][y][x] / 2 + 0.5), 0f, 1f) * 255f);
                int color = Color.rgb(r, g, b);
                bitmap.setPixel(x, y, color);
            }
        }
        return bitmap;
    }

    public double clamp(double value, double min, double max) {
        if (value < min) {
            return min;
        } else if (value > max) {
            return max;
        }
        return value;
    }

    public void close() throws OrtException {
        if (session != null) session.close();
        session = null;
    }
}
