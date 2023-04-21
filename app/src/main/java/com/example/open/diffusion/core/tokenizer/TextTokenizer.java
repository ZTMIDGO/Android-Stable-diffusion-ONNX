package com.example.open.diffusion.core.tokenizer;

import ai.onnxruntime.OnnxTensor;

/**
 * Created by ZTMIDGO 2023/3/30
 */
public interface TextTokenizer {
    void init() throws Exception;
    String decode(int[] ids) throws Exception;
    int[] encoder(String text) throws Exception;
    OnnxTensor tensor(int[] ids) throws Exception;
    int[] createUncondInput(String text) throws Exception;
    int getMaxLength();
    void close() throws Exception;
}
