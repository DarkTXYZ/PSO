package org.ci.fitness_function;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MAE implements FitnessFunction{
    @Override
    public Double calculateFitness(INDArray output, INDArray actualValue) {
        double predict = output.getDouble(0);
        double target = actualValue.getDouble(0);

        if(predict == target)
            return Double.MIN_VALUE;
        return Math.abs(predict - target);
    }

    @Override
    public Double calculateError(INDArray output, INDArray actualValue) {
        double predict = output.getDouble(0);
        double target = actualValue.getDouble(0);

        if(predict == target)
            return Double.MIN_VALUE;
        return Math.abs(predict - target);
    }
}
