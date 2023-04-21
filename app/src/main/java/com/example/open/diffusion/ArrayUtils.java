package com.example.open.diffusion;
import java.util.function.Function;

/**
 * Created by ZTMIDGO 2023/3/23
 */
public class ArrayUtils {
    public static double[] quantile(double[] a, double[] q){
        int size = a.length - 1;
        double[] result = new double[q.length];
        int index = 0;
        for (int i = 0; i < q.length; i++){
            double x = a[index ++ % a.length];
            double y = a[index % a.length];
            result[i] = x + (y - x) * (q[i] * size);
            index += 1;
        }
        return result;
    }

    public static double[] arange(double start, double stop, Double step) {
        int size = (int) (step != null ? Math.ceil((stop - start) / step) : Math.ceil(stop - start));
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = step != null ? start + i * step : start + i;
        }
        return result;
    }

    public static double[] linspace(double start, double end, int steps){
        double[] doubles = new double[steps];
        for (int i = 0; i < steps; i++){
            doubles[i] = start + (end - start) * i / (steps - 1);
        }
        return doubles;
    }

    public static double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1 / (1 + Math.exp(-x[i]));
        }
        return result;
    }

    public static double[] interp(double[] x, double[] xp, double[] fp) {
        double[] y = new double[x.length];
        Function<Double, Double> interpolation = createLinearInterpolationFunction(xp, fp);
        for (int i = 0; i < y.length; i++) y[i] = interpolation.apply(x[i]);
        return y;
    }

    public static Function<Double, Double> createLinearInterpolationFunction(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must have the same length");
        }

        return (Double input) -> {
            int i = 0;
            while (i < x.length && x[i] < input) {
                i++;
            }

            if (i == 0) {
                return y[0];
            } else if (i == x.length) {
                return y[y.length - 1];
            } else {
                double slope = (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
                return y[i - 1] + slope * (input - x[i - 1]);
            }
        };
    }

    public static long[] getSizes(float[][][][] datas){
        return new long[]{datas.length, datas[0].length, datas[0][0].length, datas[0][0][0].length};
    }

    public static long[] getSizes(float[][][] datas){
        return new long[]{datas.length, datas[0].length, datas[0][0].length};
    }

    public static int getLength(long[] datas){
        long length = 0;
        for (long item : datas) {
            if (length == 0){
                length = item;
                continue;
            }

            length *= item;
        }
        return (int) length;
    }
}
