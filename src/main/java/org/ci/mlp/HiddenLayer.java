package org.ci.mlp;

import lombok.Getter;
import lombok.Setter;
import org.ci.Util;
import org.ci.mlp.activation_function.ActivationFunction;
import org.ci.mlp.activation_function.ActivationFunctionFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class HiddenLayer {
    
    private List<Perceptron> perceptronList = new ArrayList<>();
    private INDArray outputs;
    private String name;
    private HiddenLayer prevLayer;

    public HiddenLayer(String name, int n, INDArray input) {
        this.name = name;

        for (int i = 0; i < n; ++i) {
            Perceptron perceptron = Perceptron.builder()
                .activationFunction(ActivationFunctionFactory.generate("I"))
                .bias(0.0)
                .weightBias(0.0)
                .weights(Nd4j.ones(1))
                .value(input.getDouble(i))
                .build();
            perceptronList.add(perceptron);
        }
    }

    public HiddenLayer(String name, HiddenLayer prevLayer, int n, INDArray initWeight, ActivationFunction activationFunction) {
        this.name = name;
        this.prevLayer = prevLayer;

        int numberOfPerceptronInPrevLayer = prevLayer.getNumberOfPerceptron();
        int currentWeightIndex = 0;

        for (int i = 0; i < n; ++i) {
            INDArray assignWeight = initWeight.get(
                NDArrayIndex.interval(currentWeightIndex, currentWeightIndex + numberOfPerceptronInPrevLayer));
            Double weightBias = initWeight.getDouble(currentWeightIndex + numberOfPerceptronInPrevLayer);
            Perceptron perceptron = Perceptron.builder()
                .activationFunction(activationFunction)
                .bias(1.0)
                .weightBias(weightBias)
                .weights(assignWeight)
                .build();
            currentWeightIndex += numberOfPerceptronInPrevLayer + 1;
            perceptronList.add(perceptron);
        }
    }

    public void forward() {
        List<Double> calculatedOutputs = new ArrayList<>();

        if(prevLayer == null) {
            for (Perceptron p : perceptronList)
                calculatedOutputs.add(p.getValue());
        } else {
            INDArray prevLayerOutput = prevLayer.getOutputs();
            for (Perceptron p : perceptronList)
                calculatedOutputs.add(p.calculateOutput(prevLayerOutput));
        }

        double[] arr = calculatedOutputs.stream().mapToDouble(Double::doubleValue).toArray();
        this.outputs = Nd4j.create(arr);
    }

    public INDArray getWeightsFromPerceptron(int index) {
        List<Double> weight = new ArrayList<>();
        for (Perceptron p : perceptronList) {
            weight.add(p.getWeights().getDouble(index));
        }
        return Util.listToINDArray(weight);
    }
    
    @Override
    public String toString() {
        return name + " Hidden Layer {\n" +
            perceptronList +
            ", outputs= " + outputs + "\n" +
            '}';
    }

    public int getNumberOfPerceptron(){
        return this.perceptronList.size();
    }
}
