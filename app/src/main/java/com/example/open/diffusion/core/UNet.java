package com.example.open.diffusion.core;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Pair;

import com.example.open.diffusion.App;
import com.example.open.diffusion.ArrayUtils;
import com.example.open.diffusion.Device;
import com.example.open.diffusion.MyTensor;
import com.example.open.diffusion.PathManager;
import com.example.open.diffusion.TensorHelper;
import com.example.open.diffusion.core.scheduler.DPMSolverMultistepScheduler;
import com.example.open.diffusion.core.scheduler.EulerAncestralDiscreteScheduler;
import com.example.open.diffusion.core.scheduler.Scheduler;

import java.nio.IntBuffer;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

/**
 * Created by ZTMIDGO 2023/3/22
 */
public class UNet {
    public static final String TAG = "UNet";
    private final String model = "unet/model.ort";
    public static int WIDTH = 384;
    public static int HEIGHT = 384;
    private final Random random = new Random();
    private final Context context;

    private OrtSession session;
    private VaeDecoder decoder;
    private Callback callback;
    private boolean isStop;
    private int deviceId = Device.CPU;
    public UNet(Context context, int deviceId) {
        this.context = context;
        this.deviceId = deviceId;
        decoder = new VaeDecoder(context, deviceId);
    }

    public void init() throws Exception {
        if (session != null) return;
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.addConfigEntry("session.load_model_format", "ORT");
        if (deviceId == Device.NNAPI) options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED));
        session = App.ENVIRONMENT.createSession(PathManager.getModelPath(context) +"/" +model, options);
    }

    public Map<String, OnnxTensor> createUnetModelInput(OnnxTensor encoderHiddenStates, OnnxTensor sample, OnnxTensor timeStep){
        Map<String, OnnxTensor> map = new HashMap<>();
        map.put("encoder_hidden_states", encoderHiddenStates);
        map.put("sample", sample);
        map.put("timestep", timeStep);
        return map;
    }

    public MyTensor generateLatentSample(int batchSize, int height, int width, int seed, float initNoiseSigma) throws Exception {
        Random random = new Random(seed);
        int channels = 4;

        float[][][][] latentsArray = new float[batchSize][channels][height / 8][width / 8];

        for (int i = 0; i < batchSize; i++){
            for (int j = 0; j < channels; j++){
                for (int k = 0; k < height / 8; k++){
                    for (int l = 0; l < width / 8; l++){
                        double u1 = random.nextDouble();
                        double u2 = random.nextDouble();

                        double radius = Math.sqrt(-2.0f * Math.log(u1));
                        double theta = 2.0d * Math.PI * u2;
                        double standardNormalRand = radius * Math.cos(theta);
                        latentsArray[i][j][k][l] = (float) (standardNormalRand * initNoiseSigma);
                    }
                }
            }
        }

        return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, latentsArray), latentsArray, new long[]{batchSize, channels, height / 8, width / 8});
    }

    public void performGuidance(float[][][][] noisePred, float[][][][] noisePredText, double guidanceScale){
        long[] indexs = ArrayUtils.getSizes(noisePred);
        for (int i = 0; i < indexs[0]; i++) {
            for (int j = 0; j < indexs[1]; j++) {
                for (int k = 0; k < indexs[2]; k++) {
                    for (int l = 0; l < indexs[3]; l++) {
                        noisePred[i][j][k][l] = noisePred[i][j][k][l] + (float)guidanceScale * (noisePredText[i][j][k][l] - noisePred[i][j][k][l]);
                    }
                }
            }
        }
    }

    public void stop(){
        isStop = true;
    }

    public void inference(int seedNum, int numInferenceSteps, OnnxTensor textEmbeddings, double guidanceScale, int batchSize, int width, int height) throws Exception {
        isStop = false;
        Scheduler scheduler = new EulerAncestralDiscreteScheduler(context);

        int[] timesteps = scheduler.set_timesteps(numInferenceSteps);

        int seed = seedNum <= 0 ? random.nextInt() : seedNum;
        MyTensor latents = generateLatentSample(batchSize, height, width, seed, (float) scheduler.getInitNoiseSigma());

        long[] shape = new long[]{2, 4, height / 8, width / 8};
        for (int i = 0; i < timesteps.length; i++){
            MyTensor latentModelInput = TensorHelper.duplicate(latents.getTensor().getFloatBuffer().array(), shape);
            latentModelInput = scheduler.scale_model_input(latentModelInput, i);

            if (callback != null) callback.onStep(timesteps.length, i);

            Map<String, OnnxTensor> input = createUnetModelInput(textEmbeddings, latentModelInput.getTensor(), OnnxTensor.createTensor(App.ENVIRONMENT, IntBuffer.wrap(new int[]{timesteps[i]}), new long[]{1}));
            OrtSession.Result result = session.run(input);
            float[][][][] datas = (float[][][][]) result.get(0).getValue();
            result.close();

            Pair<float[][][][], float[][][][]> splitTensors = TensorHelper.splitTensor(datas, new long[] { 1, 4, height / 8, width / 8 });
            float[][][][] noisePred = splitTensors.first;
            float[][][][] noisePredText = splitTensors.second;

            performGuidance(noisePred, noisePredText, guidanceScale);

            latents = scheduler.step(new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, noisePred), noisePred, ArrayUtils.getSizes(noisePred)), i, latents);
            if (isStop) {
                if (callback != null) callback.onStop();
                return;
            }
        }
        close();

        if (callback != null) {
            callback.onBuildImage(-1, null);
            Bitmap bitmap = decode(latents);
            callback.onBuildImage(0, bitmap);
            callback.onComplete();
        }
    }

    public Bitmap decode(MyTensor latents) throws Exception {
        MyTensor tensor = TensorHelper.MultipleTensorByFloat(latents.getTensor().getFloatBuffer().array(), (1.0f / 0.18215f), latents.getShape());
        Map<String, OnnxTensor> decoderInput = new HashMap<>();
        decoderInput.put("latent_sample", tensor.getTensor());
        Object object = decoder.Decoder(decoderInput);
        Bitmap bitmap = decoder.ConvertToImage((float[][][][]) object, UNet.WIDTH, UNet.HEIGHT, "");
        return bitmap;
    }
    
    public void close() throws OrtException {
        if (session != null) session.close();
        if (decoder != null) decoder.close();

        session = null;
    }

    public void setCallback(Callback callback) {
        this.callback = callback;
    }

    public interface Callback{
        void onStep(int maxStep, int step);
        void onBuildImage(int status, Bitmap bitmap);
        void onComplete();
        void onStop();
    }
}
