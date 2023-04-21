package com.example.open.diffusion.core.scheduler;
import com.example.open.diffusion.App;
import com.example.open.diffusion.ArrayUtils;
import com.example.open.diffusion.FrozenDict;
import com.example.open.diffusion.MyTensor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;

/**
 * Created by ZTMIDGO 2023/3/25
 */
public class DPMSolverMultistepScheduler implements Scheduler {
    public static final String TAG = "DPMSolverMultistepScheduler";
    private final FrozenDict config;
    private final MyTensor[] model_outputs;
    private final List<Float> betas = new ArrayList<>();
    private final List<Float> alphas = new ArrayList<>();
    private final List<Float> alphas_cumprod = new ArrayList<>();
    private final List<Integer> timesteps = new ArrayList<>();
    private final List<Float> alpha_t = new ArrayList<>();
    private final List<Float> sigma_t = new ArrayList<>();
    private final List<Float> lambda_t = new ArrayList<>();
    private double init_noise_sigma = 1.0d;
    private int lower_order_nums;
    private int num_inference_steps = 0;
    private int num_train_timesteps = 1000;

    public DPMSolverMultistepScheduler(){
        this(new FrozenDict());
    }

    public DPMSolverMultistepScheduler(FrozenDict config){
        this.config = config;
        model_outputs = new MyTensor[config.solver_order];
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

        for (float value : alphas_cumprod) alpha_t.add((float) Math.sqrt(value));
        for (float value : alphas_cumprod) sigma_t.add((float) Math.sqrt(1f - value));

        for (int i = 0; i < alphas_cumprod.size(); i++) lambda_t.add((float) (Math.log(alpha_t.get(i)) - Math.log(sigma_t.get(i))));

        double[] array = ArrayUtils.linspace(0, num_train_timesteps - 1, num_train_timesteps);
        for (double value : array) timesteps.add(0, (int) value);
    }

    @Override
    public int[] set_timesteps(int num_inference_steps){
        this.num_inference_steps = num_inference_steps;
        double[] array = ArrayUtils.linspace(0, this.num_train_timesteps - 1, num_inference_steps + 1);
        double[] copy = new double[array.length];
        for (int i =1; i <= array.length; i++){
            copy[array.length - i] = array[i - 1];
        }
        array = Arrays.copyOf(copy, copy.length - 1);
        timesteps.clear();
        for (double value : array) timesteps.add((int) value);
        lower_order_nums = 0;

        int[] result = new int[this.timesteps.size()];
        for (int i = 0; i < this.timesteps.size(); i++) result[i] = this.timesteps.get(i);
        return result;
    }

    @Override
    public MyTensor scale_model_input(MyTensor sample, int step_index){
        return sample;
    }

    public MyTensor convert_model_output(MyTensor model_output, int timestep, MyTensor sample) throws OrtException {
        float[] outputArray = model_output.getTensor().getFloatBuffer().array();
        float[] sampleArray = sample.getTensor().getFloatBuffer().array();

        double alpha_t, sigma_t;
        if (config.algorithm_type.equals("dpmsolver++")){
            float[] x0_pred = null;
            if (config.prediction_type.equals("epsilon")){
                alpha_t = this.alpha_t.get(timestep);
                sigma_t = this.sigma_t.get(timestep);
                x0_pred = new float[outputArray.length < sampleArray.length ? outputArray.length : sampleArray.length];
                for (int i = 0; i < x0_pred.length; i++) x0_pred[i] = (float) ((sampleArray[i] - sigma_t * outputArray[i]) / alpha_t);
            }else if (config.prediction_type.equals("sample")){
                x0_pred = outputArray;
            }else if (config.prediction_type.equals("v_prediction")){
                alpha_t = this.alpha_t.get(timestep);
                sigma_t = this.sigma_t.get(timestep);
                x0_pred = new float[outputArray.length < sampleArray.length ? outputArray.length : sampleArray.length];
                for (int i = 0; i < x0_pred.length; i++) x0_pred[i] = (float) (alpha_t * sampleArray[i] - sigma_t * outputArray[i]);
                double[] x0_pred_abs = new double[x0_pred.length];
                for (int i = 0; i < x0_pred.length; i++) x0_pred_abs[i] = Math.abs(x0_pred[i]);
            }
            return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(x0_pred), model_output.getShape()), null, model_output.getShape());
        }
        return null;
    }

    @Override
    public MyTensor step(MyTensor model_output, int step_index, MyTensor sample) throws Exception {
        int timestep = this.timesteps.get(step_index);
        int prev_timestep = step_index == this.timesteps.size() - 1 ? 0 : this.timesteps.get(step_index + 1);
        boolean lower_order_final = (step_index == this.timesteps.size() - 1) && config.lower_order_final && this.timesteps.size() < 15;
        boolean lower_order_second =  (step_index == this.timesteps.size() - 2) && config.lower_order_final && this.timesteps.size() < 15;
        model_output = convert_model_output(model_output, timestep, sample);
        for (int i = 0; i < config.solver_order - 1; i++){
            model_outputs[i] = model_outputs[i + 1];
        }
        
        model_outputs[model_outputs.length - 1] = model_output;
        MyTensor prev_sample = null;
        if (config.solver_order == 1 || lower_order_nums < 1 || lower_order_final){
            prev_sample = dpm_solver_first_order_update(model_output, timestep, prev_timestep, sample);
        }else if (config.solver_order == 2 || lower_order_nums < 2 || lower_order_second){
            int[] timestep_list = {this.timesteps.get(step_index - 1), timestep};
            prev_sample = multistep_dpm_solver_second_order_update(model_outputs, timestep_list, prev_timestep, sample);
        }else {
            int[] timestep_list = {this.timesteps.get(step_index - 2), this.timesteps.get(step_index - 1), timestep};
            prev_sample = multistep_dpm_solver_third_order_update(model_outputs, timestep_list, prev_timestep, sample);
        }

        if (lower_order_nums < config.solver_order){
            lower_order_nums += 1;
        }

        return prev_sample;
    }

    private MyTensor multistep_dpm_solver_third_order_update(MyTensor[] model_output_list, int[] timestep_list, int prev_timestep, MyTensor sample) throws OrtException {
        int t = prev_timestep;
        int s0 = timestep_list[-1];
        int s1 = timestep_list[-2];
        int s2 = timestep_list[-3];

        float[] sampleArray = sample.getTensor().getFloatBuffer().array();
        float[] m0 = model_output_list[model_output_list.length - 1].getTensor().getFloatBuffer().array();
        float[] m1 = model_output_list[model_output_list.length - 2].getTensor().getFloatBuffer().array();
        float[] m2 = model_output_list[model_output_list.length - 3].getTensor().getFloatBuffer().array();

        double lambda_t = this.lambda_t.get(t);
        double lambda_s0 = this.lambda_t.get(s0);
        double lambda_s1 = this.lambda_t.get(s1);
        double lambda_s2 = this.lambda_t.get(s2);

        double alpha_t = this.alpha_t.get(t);
        double alpha_s0 = this.alpha_t.get(s0);

        double sigma_t = this.sigma_t.get(t);
        double sigma_s0 = this.sigma_t.get(s0);

        double h = lambda_t - lambda_s0;
        double h_0 = lambda_s0 - lambda_s1;
        double h_1 = lambda_s1 - lambda_s2;

        double r0 = h_0 / h;
        double r1 = h_1 / h;
        float[] D0 = m0;

        float[] D1_0 = new float[m0.length];
        for (int i = 0; i < D1_0.length; i++) D1_0[i] = (float) ((1.0 / r0) * (m0[i] - m1[i]));
        float[] D1_1 = new float[m1.length];
        for (int i = 0; i < D1_1.length; i++) D1_1[i] = (float) ((1.0 / r1) * (m1[i] - m2[i]));

        float[] D1 = new float[D1_0.length];
        for (int i = 0; i < D1.length; i++) D1[i] = (float) (D1_0[i] + (r0 / (r0 + r1)) * (D1_0[i] - D1_1[i]));
        float[] D2 = new float[D1_1.length];
        for (int i = 0; i < D2.length; i++) D2[i] = (float) ((1.0 / (r0 + r1)) * (D1_0[i] - D1_1[i]));

        if (config.algorithm_type.equals("dpmsolver++")){
            float[] datas = new float[sampleArray.length];
            for (int i = 0; i < datas.length; i++){
                datas[i] = (float) ((sigma_t / sigma_s0) * sampleArray[i] - (alpha_t * (Math.exp(-h) - 1.0)) * D0[i] + (alpha_t * ((Math.exp(-h) - 1.0) / h + 1.0)) * D1[i] - (alpha_t * ((Math.exp(-h) - 1.0 + h) / Math.pow(h, 2f) - 0.5)) * D2[i]);
            }
            return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(datas), sample.getShape()), null, sample.getShape());
        }
        return null;
    }

    private MyTensor multistep_dpm_solver_second_order_update(MyTensor[] model_output_list, int[] timestep_list, int prev_timestep, MyTensor sample) throws OrtException {
        int t = prev_timestep;
        int s0 = timestep_list[timestep_list.length - 1];
        int s1 = timestep_list[timestep_list.length - 2];
        float[] sampleArray = sample.getTensor().getFloatBuffer().array();
        float[] m0 = model_output_list[model_output_list.length - 1].getTensor().getFloatBuffer().array();
        float[] m1 = model_output_list[model_output_list.length - 2].getTensor().getFloatBuffer().array();
        double lambda_t = this.lambda_t.get(t);
        double lambda_s0 = this.lambda_t.get(s0);
        double lambda_s1 = this.lambda_t.get(s1);

        double alpha_t = this.alpha_t.get(t);
        double alpha_s0 = this.alpha_t.get(s0);
        double sigma_t = this.sigma_t.get(t);
        double sigma_s0 = this.sigma_t.get(s0);

        double h = lambda_t - lambda_s0;
        double h_0 = lambda_s0 - lambda_s1;
        double r0 = h_0 / h;
        float[] D0 = m0;
        float[] D1 = new float[m1.length];
        for (int i = 0; i < D1.length; i++){
            D1[i] = (float) ((1.0 / r0) * (m0[i] - m1[i]));
        }

        if (config.algorithm_type.equals("dpmsolver++")){
            float[] datas = new float[sampleArray.length];
            if (config.solver_type.equals("midpoint")){
                for (int i = 0; i < datas.length; i++){
                    datas[i] = (float) ((sigma_t / sigma_s0) * sampleArray[i] - (alpha_t * (Math.exp(-h) - 1.0)) * D0[i] - 0.5 * (alpha_t * (Math.exp(-h) - 1.0)) * D1[i]);
                }
            }else if (config.solver_type.equals("heun")){
                for (int i = 0; i < datas.length; i++){
                    datas[i] = (float) ((sigma_t / sigma_s0) * sampleArray[i] - (alpha_t * (Math.exp(-h) - 1.0)) * D0[i] + (alpha_t * ((Math.exp(-h) - 1.0) / h + 1.0)) * D1[i]);
                }
            }
            return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(datas), sample.getShape()), null, sample.getShape());
        }
        return null;
    }

    private MyTensor dpm_solver_first_order_update(MyTensor model_output, int timestep, int prev_timestep, MyTensor sample) throws OrtException {
        double lambda_t = this.lambda_t.get(prev_timestep);
        double lambda_s = this.lambda_t.get(timestep);
        double alpha_t = this.alpha_t.get(prev_timestep);
        double alpha_s = this.alpha_t.get(timestep);
        double sigma_t = this.sigma_t.get(prev_timestep);
        double sigma_s = this.sigma_t.get(timestep);
        double h = lambda_t - lambda_s;


        if (config.algorithm_type.equals("dpmsolver++")){
            float[] outputArray = model_output.getTensor().getFloatBuffer().array();
            float[] sampleArray = sample.getTensor().getFloatBuffer().array();
            float[] datas = new float[outputArray.length];
            for (int i = 0; i < datas.length; i++){
                datas[i] = (float) ((sigma_t / sigma_s) * sampleArray[i] - (alpha_t * (Math.exp(-h) - 1.0)) * outputArray[i]);
            }
            return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(datas), model_output.getShape()), null, model_output.getShape());
        }
        return null;
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
