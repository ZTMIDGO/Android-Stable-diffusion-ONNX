package com.example.open.diffusion;

import android.util.Pair;

import java.nio.FloatBuffer;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;

/**
 * Created by ZTMIDGO 2023/3/23
 */
public class TensorHelper {
    public static MyTensor duplicate(float[] data, long[] dimensions) throws Exception {
        float[] floats = new float[data.length * 2];
        System.arraycopy(data, 0, floats, 0, data.length);
        System.arraycopy(data, 0, floats, data.length, data.length);

        return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(floats), dimensions), null, dimensions);
    }

    public static MyTensor MultipleTensorByFloat(float[] data, float value, long[] dimensions) throws OrtException {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] * value;
        }

        return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(data), dimensions), null, dimensions);
    }

    public static MyTensor SumTensors(OnnxTensor[] tensorArray, long[] dimensions) throws OrtException {
        float[] sumTensor = new float[ArrayUtils.getLength(dimensions)];
        float[] sumArray = new float[sumTensor.length];

        for (int m = 0; m < tensorArray.length; m++)
        {
            float[] tensorToSum = tensorArray[m].getFloatBuffer().array();
            for (int i = 0; i < tensorToSum.length; i++)
            {
                sumArray[i] += tensorToSum[i];
            }
        }

        return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(sumArray), dimensions), null, dimensions);
    }

    public static MyTensor AddTensors(float[] sample, float[] sumTensor, long[] dimensions) throws OrtException {
        for(int i=0; i < sample.length; i++)
        {
            sample[i] = sample[i] + sumTensor[i];
        }
        return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(sample), dimensions), null, dimensions);
    }

    public static MyTensor divideTensorByFloat(float[] data, float value, long[] dimensions) throws OrtException {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] / value;
        }
        return new MyTensor(OnnxTensor.createTensor(App.ENVIRONMENT, FloatBuffer.wrap(data), dimensions), null, dimensions);
    }

    public static Pair<float[][][][], float[][][][]> splitTensor(float[][][][] tensorToSplit, long[] dimensions)
    {

        float[][][][] tensor1 = new float[(int) dimensions[0]][(int) dimensions[1]][(int) dimensions[2]][(int) dimensions[3]];
        float[][][][] tensor2 = new float[(int) dimensions[0]][(int) dimensions[1]][(int) dimensions[2]][(int) dimensions[3]];

        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < dimensions[2]; k++) {
                    for (int l = 0; l < dimensions[3]; l++) {
                        tensor1[i][j][k][l] = tensorToSplit[i][j][k][l];
                        tensor2[i][j][k][l] = tensorToSplit[i + 1][j][k][l];
                    }
                }
            }
        }
        return new Pair<>(tensor1, tensor2);

    }
}
