package org.ci.swarm;

import org.ci.fitness_function.FitnessFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

public interface Individual {
    INDArray getWeights();
    void setWeights(INDArray weights);
    String getStructure();
    Double getFitnessValue();

    FitnessFunction getFitnessFunction();

    List<Integer> getNumOfPerceptronInLayer();

    void calculateFitnessValue(List<DataSet> trainData);
    void calculateError(List<DataSet> testData);

    Double getError();

}
