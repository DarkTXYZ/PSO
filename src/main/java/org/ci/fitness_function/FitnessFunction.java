package org.ci.fitness_function;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface FitnessFunction {
    Double calculateFitness(INDArray output, INDArray actualValue);
    Double calculateError(INDArray output, INDArray actualValue);
}
