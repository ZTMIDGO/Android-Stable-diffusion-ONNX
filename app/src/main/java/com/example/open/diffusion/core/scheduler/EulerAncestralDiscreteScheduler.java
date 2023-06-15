package com.example.open.diffusion.core.scheduler;

import android.content.Context;

import com.example.open.diffusion.App;
import com.example.open.diffusion.ArrayUtils;
import com.example.open.diffusion.FrozenDict;
import com.example.open.diffusion.MyTensor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;

/**
 * Created by ZTMIDGO 2023/3/29
 */
public class EulerAncestralDiscreteScheduler implements Scheduler {
    public static final String TAG = "EulerAncestralDiscreteScheduler";
    private final FrozenDict config;
    private final Random random = new Random();
    private final List<Float> betas = new ArrayList<>();
    private final List<Float> alphas = new ArrayList<>();
    private final List<Float> alphas_cumprod = new ArrayList<>();
    private final List<Integer> timesteps = new ArrayList<>();
    private final List<Float> alpha_t = new ArrayList<>();
    private final List<Float> sigma_t = new ArrayList<>();
    private final List<Float> lambda_t = new ArrayList<>();
    private final List<Float> sigmas = new ArrayList<>();
    private final Context context;
    private double init_noise_sigma = 1.0d;
    private int lower_order_nums;
    private int num_inference_steps = 0;
    private int num_train_timesteps = 1000;
    private boolean is_scale_input_called = false;

    public EulerAncestralDiscreteScheduler(Context context){
        this(context, new FrozenDict());
    }

    public EulerAncestralDiscreteScheduler(Context context, FrozenDict config) {
        this.context = context;
        this.config = config;
        if (config.trained_betas != null){
            betas.addAll(config.trained_betas);
        }else if (config.beta_schedule.equals("linear")){
            double[] array = ArrayUtils.linspace(config.beta_start, config.beta_end, num_train_timesteps);
            for (double value : array) betas.add((float) value);
        }else if (config.beta_schedule.equals("scaled_linear")){
            double[] array = ArrayUtils.linspace(Math.pow(config.beta_start, 0.5), Math.pow(config.beta_end, 0.5), num_train_timesteps);
            for (int i = 0; i < array.length; i++){
                betas.add((float) Math.pow(array[i], 2));
            }
        }else if (config.beta_schedule.equals("squaredcos_cap_v2")){
            betas.addAll(betas_for_alpha_bar(num_train_timesteps, 0.999f));
        }

        for (float value : betas) alphas.add(1f - value);

        for (int i = 0; i < alphas.size(); i++){
            List<Float> sub = alphas.subList(0, i + 1);
            float value = sub.get(0);
            for (int x = 1; x < sub.size(); x++){
                value = value * sub.get(x);
            }
            alphas_cumprod.add(value);
        }

        List<Float> sigmas = new ArrayList<>(alphas_cumprod.size());
        for (float value : alphas_cumprod){
            sigmas.add((float) Math.pow((1f - value) / value, 0.5));
        }

        for (int i = sigmas.size() - 1; i >= 0; i--){
            this.sigmas.add(sigmas.get(i));
        }
        this.sigmas.add((float) 0);

        float maxSigmas = this.sigmas.get(0);
        for (float value : this.sigmas) if (value > maxSigmas) maxSigmas = value;
        init_noise_sigma = maxSigmas;

        double[] array = ArrayUtils.linspace(0, num_train_timesteps - 1, num_train_timesteps);
        for (double value : array) timesteps.add(0, (int) value);

    }

    @Override
    public int[] set_timesteps(int num_inference_steps){
        this.num_inference_steps = num_inference_steps;
        double[] array = ArrayUtils.linspace(0, this.num_train_timesteps - 1, num_inference_steps);
        double[] timesteps = new double[array.length];
        for (int i = 1; i <= array.length; i++) timesteps[array.length - i] = array[i - 1];
        double[] sigmas = new double[alphas_cumprod.size()];
        for (int i = 0; i < sigmas.length; i++){
            float value = alphas_cumprod.get(i);
            sigmas[i] = ((float) Math.pow((1 - value) / value, 0.5));
        }

        double[] arange = ArrayUtils.arange(0, sigmas.length, null);

        sigmas = ArrayUtils.interp(timesteps, arange, sigmas);

        double[] extend = new double[sigmas.length + 1];
        System.arraycopy(sigmas, 0, extend, 0, sigmas.length);
        this.sigmas.clear();
        this.timesteps.clear();
        for (double value : extend) this.sigmas.add((float) value);
        int[] result = new int[timesteps.length];
        for (int i = 0; i < timesteps.length; i++){
            result[i] = (int) timesteps[i];
            this.timesteps.add((int) timesteps[i]);
        }

        return result;
    }

    @Override
    public MyTensor scale_model_input(MyTensor sample, int step_index) throws Exception {
        float[] sampleArray = sample.getTensor().getFloatBuffer().array();
        float sigma = this.sigmas.get(step_index);
        float[] datas = new float[sampleArray.length];
        for (int i = 0; i < datas.length; i++){
            datas[i] = (float) (sampleArray[i] / Math.pow(Math.pow(sigma, 2) + 1, 0.5));
        }
        is_scale_input_called = true;
        
        return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(datas), sample.getShape()), null, sample.getShape());
    }

    @Override
    public MyTensor step(MyTensor model_output, int step_index, MyTensor sample) throws Exception {
        float[][][][] sampleArray = (float[][][][]) sample.getBuffer();
        float[][][][] outputArray = (float[][][][]) model_output.getBuffer();
        double sigma = this.sigmas.get(step_index);

        int dim1 = (int) model_output.getShape()[0];
        int dim2 = (int) model_output.getShape()[1];
        int dim3 = (int) model_output.getShape()[2];
        int dim4 = (int) model_output.getShape()[3];

        float[][][][] pred_original_sample = null;
        if (config.prediction_type.equals("epsilon")){
            pred_original_sample = new float[dim1][dim2][dim3][dim4];
            for (int i = 0; i < dim1; i++){
                for (int j = 0; j < dim2; j++){
                    for (int k = 0; k < dim3; k++){
                        for (int l = 0; l < dim4; l++){
                            pred_original_sample[i][j][k][l] = (float) (sampleArray[i][j][k][l] - sigma * outputArray[i][j][k][l]);
                        }
                    }
                }
            }
        }else if (config.prediction_type.equals("v_prediction")){
            pred_original_sample = new float[dim1][dim2][dim3][dim4];
            for (int i = 0; i < dim1; i++){
                for (int j = 0; j < dim2; j++){
                    for (int k = 0; k < dim3; k++){
                        for (int l = 0; l < dim4; l++){
                            pred_original_sample[i][j][k][l] = (float) (outputArray[i][j][k][l] * Math.pow(-sigma / (Math.pow(sigma, 2) + 1), 0.5) + (sampleArray[i][j][k][l] / (Math.pow(sigma, 2) + 1)));
                        }
                    }
                }
            }
        }

        double sigma_from = this.sigmas.get(step_index);
        double sigma_to = this.sigmas.get(step_index + 1);
        double sigma_up = Math.pow((Math.pow(sigma_to, 2) * (Math.pow(sigma_from, 2) - Math.pow(sigma_to, 2)) / Math.pow(sigma_from, 2)), 0.5);
        double sigma_down = Math.pow((Math.pow(sigma_to, 2) - Math.pow(sigma_up, 2)), 0.5);

        float[][][][] derivative = new float[dim1][dim2][dim3][dim4];
        for (int i = 0; i < dim1; i++){
            for (int j = 0; j < dim2; j++){
                for (int k = 0; k < dim3; k++){
                    for (int l = 0; l < dim4; l++){
                        derivative[i][j][k][l] = (float) ((sampleArray[i][j][k][l] - pred_original_sample[i][j][k][l]) / sigma);
                    }
                }
            }
        }

        double dt = sigma_down - sigma;

        float[][][][] prev_sample = new float[dim1][dim2][dim3][dim4];
        for (int i = 0; i < dim1; i++){
            for (int j = 0; j < dim2; j++){
                for (int k = 0; k < dim3; k++){
                    for (int l = 0; l < dim4; l++){
                        prev_sample[i][j][k][l] = (float) (sampleArray[i][j][k][l] + derivative[i][j][k][l] * dt);
                    }
                }
            }
        }

        /*float[][][][] noise = new float[dim1][dim2][dim3][dim4];
        Random random = new Random();
        for (int i = 0; i < dim1; i++){
            for (int j = 0; j < dim2; j++){
                for (int k = 0; k < dim3; k++){
                    for (int l = 0; l < dim4; l++){
                        noise[i][j][k][l] = (float) random.nextGaussian();
                    }
                }
            }
        }*/

        for (int i = 0; i < dim1; i++){
            for (int j = 0; j < dim2; j++){
                for (int k = 0; k < dim3; k++){
                    for (int l = 0; l < dim4; l++){
                        prev_sample[i][j][k][l] = (float) (prev_sample[i][j][k][l] + random.nextGaussian() * sigma_up);
                    }
                }
            }
        }

        return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, prev_sample), prev_sample, model_output.getShape());
    }

    public List<Float> betas_for_alpha_bar(int num_diffusion_timesteps, float max_beta){
        List<Float> betas = new ArrayList<>(num_diffusion_timesteps);
        for (int i = 0; i < num_diffusion_timesteps; i++){
            double t1 = i * 1d / num_diffusion_timesteps;
            double t2 = (i + 1) * 1d / num_diffusion_timesteps;
            betas.add((float) Math.min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta));
        }
        return betas;
    }

    public double alpha_bar(double time_step){
        return Math.pow(Math.cos((time_step + 0.008) / 1.008 * Math.PI / 2), 2);
    }

    @Override
    public double getInitNoiseSigma() {
        return init_noise_sigma;
    }
}
