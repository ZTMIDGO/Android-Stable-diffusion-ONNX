package com.example.open.diffusion.core.scheduler;

import com.example.open.diffusion.MyTensor;

/**
 * Created by ZTMIDGO 2023/4/1
 */
public interface Scheduler {
    int[] set_timesteps(int num_inference_steps);
    MyTensor scale_model_input(MyTensor sample, int step_index) throws Exception;
    MyTensor step(MyTensor model_output, int step_index, MyTensor sample) throws Exception ;
    double getInitNoiseSigma();
}
