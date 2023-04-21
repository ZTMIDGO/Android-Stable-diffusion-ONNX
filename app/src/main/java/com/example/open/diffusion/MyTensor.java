package com.example.open.diffusion;

import ai.onnxruntime.OnnxTensor;

/**
 * Created by ZTMIDGO 2023/3/26
 */
public class MyTensor<T> {
    private T buffer;
    private OnnxTensor tensor;
    private long[] shape;

    public MyTensor(OnnxTensor tensor, long[] shape) {
        this(tensor, null, shape);
    }

    public MyTensor(OnnxTensor tensor, T buffer, long[] shape) {
        this.tensor = tensor;
        this.shape = shape;
        this.buffer = buffer;
    }

    public T getBuffer() {
        return buffer;
    }

    public void setBuffer(T buffer) {
        this.buffer = buffer;
    }

    public OnnxTensor getTensor() {
        return tensor;
    }

    public void setTensor(OnnxTensor tensor) {
        this.tensor = tensor;
    }

    public long[] getShape() {
        return shape;
    }

    public void setShape(long[] shape) {
        this.shape = shape;
    }
}
