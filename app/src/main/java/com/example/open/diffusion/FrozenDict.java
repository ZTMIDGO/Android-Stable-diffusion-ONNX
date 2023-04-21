package com.example.open.diffusion;

import java.util.List;

/**
 * Created by ZTMIDGO 2023/3/26
 */
public class FrozenDict {
    public float beta_start = 0.00085f;
    public float beta_end = 0.012f;
    public String beta_schedule = "scaled_linear";
    public List<Float> trained_betas;
    public int solver_order = 2;
    public String prediction_type = "epsilon";
    public boolean thresholding = false;
    public float dynamic_thresholding_ratio = 0.995f;
    public float sample_max_value = 1.0f;
    public String algorithm_type = "dpmsolver++";
    public String solver_type = "midpoint";
    public boolean lower_order_final = true;
    public boolean clip_sample = false;
    public float clip_sample_range = 1.0f;
}
