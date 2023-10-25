package org.ci.swarm;

import lombok.Getter;
import lombok.Setter;
import org.ci.Util;
import org.ci.fitness_function.FitnessFunction;
import org.ci.mlp.MLP;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class WeightIndividual implements Individual{

    private final List<Integer> numOfPerceptronInLayer;
    private INDArray weights;
    private final String structure;
    private final FitnessFunction fitnessFunction;
    private Double fitnessValue;
    private Double totalError;
    private Double error;

    public WeightIndividual(INDArray weights, String structure, FitnessFunction fitnessFunction) {
        this.weights = weights;
        this.structure = structure;
        this.fitnessFunction = fitnessFunction;

        String[] MLPStructure = structure.split("-");
        List<Integer> numOfPerceptron = new ArrayList<>();

        boolean first = true;

        for (String layerStructure : MLPStructure) {
            if (first) {
                first = false;
                numOfPerceptron.add(Integer.parseInt(layerStructure));
            } else {
                numOfPerceptron.add(
                    Integer.parseInt(layerStructure.substring(0, layerStructure.length() - 1)));
            }
        }

        this.numOfPerceptronInLayer = numOfPerceptron;
    }

    public WeightIndividual(Individual n) {
        this.weights = n.getWeights();
        this.structure = n.getStructure();
        this.fitnessFunction = n.getFitnessFunction();
        this.numOfPerceptronInLayer = Util.copy(n.getNumOfPerceptronInLayer());
        this.fitnessValue = n.getFitnessValue();
    }

    @Override
    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    @Override
    public void calculateFitnessValue(List<DataSet> trainData) {
        this.fitnessValue = 0.0;
        for (DataSet data : trainData) {
            INDArray inputData = Nd4j.toFlattened(data.getFeatures());
            INDArray desiredOutput = Nd4j.toFlattened(data.getLabels());

            MLP mlp = new MLP(structure, inputData, weights);
            mlp.forward();
            this.fitnessValue += fitnessFunction.calculateFitness(mlp.getOutput(), desiredOutput);
        }
        this.fitnessValue /= trainData.size();
    }

    @Override
    public void calculateError(List<DataSet> testData) {
        this.totalError = 0.0;
        for (DataSet data : testData) {
            INDArray inputData = Nd4j.toFlattened(data.getFeatures());
            INDArray desiredOutput = Nd4j.toFlattened(data.getLabels());

            MLP mlp = new MLP(structure, inputData, weights);
            mlp.forward();
            this.totalError += fitnessFunction.calculateError(mlp.getOutput(), desiredOutput);
        }
        this.error = totalError / testData.size();
    }

    @Override
    public String toString() {
        return "WeightIndividual{" +
            ", weights=" + weights +
            ", structure='" + structure + '\'' +
            ", fitnessValue=" + fitnessValue +
            '}';
    }
}
