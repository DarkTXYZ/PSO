package org.ci.fitness_function;

import org.nd4j.linalg.api.ndarray.INDArray;

public class BinaryCrossEntropy implements FitnessFunction {
    @Override
    public Double calculateFitness(INDArray output, INDArray actualValue) {
        double predict = output.getDouble(0);
        double target = actualValue.getDouble(0);

        double minValue = 1e-10;
        double maxValue = 1.0 - 1e-10;

        predict = Math.min(Math.max(predict, minValue), maxValue);
        target = Math.min(Math.max(target, minValue), maxValue);

        return Math.exp(target * Math.log(predict) + (1 - target) * Math.log(1 - predict));
    }

    public Double calculateError(INDArray output, INDArray actualValue) {
        double predict = output.getDouble(0);
        double target = actualValue.getDouble(0);

        double minValue = 1e-10;
        double maxValue = 1.0 - 1e-10;

        predict = Math.min(Math.max(predict, minValue), maxValue);
        target = Math.min(Math.max(target, minValue), maxValue);

        return -(target * Math.log(predict) + (1 - target) * Math.log(1 - predict));
    }
}