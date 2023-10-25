package org.ci.mlp;

import lombok.Getter;
import lombok.Setter;
import org.ci.mlp.activation_function.ActivationFunctionFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class MLP {
    private List<HiddenLayer> layerList = new ArrayList<>();
    private INDArray input;
    private INDArray output;

    public MLP(String input, INDArray inputData, INDArray initialWeight) {
        String[] MLPStructure = input.split("-");
        HiddenLayer prevLayer = null;
        int currentLayerIndex = 0;
        int currentWeightIndex = 0;
        for (String layerStructure : MLPStructure) {
            if (currentLayerIndex == 0) {
                int inputLayerSize = Integer.parseInt(MLPStructure[0]);
                HiddenLayer inputLayer = new HiddenLayer("Input Layer", inputLayerSize, inputData);
                this.layerList.add(inputLayer);
                prevLayer = inputLayer;
            } else{
                int numberOfPerceptron =
                    Integer.parseInt(layerStructure.substring(0, layerStructure.length() - 1));
                int numberOfPerceptronInPrevLayer = prevLayer.getNumberOfPerceptron();
                int totalNumberOfWeights = (numberOfPerceptron) * (numberOfPerceptronInPrevLayer + 1);


                INDArray initWeight = initialWeight.get(NDArrayIndex.interval(currentWeightIndex, currentWeightIndex + totalNumberOfWeights));

                HiddenLayer currentLayer = new HiddenLayer(Integer.toString(currentLayerIndex), prevLayer, numberOfPerceptron, initWeight,
                    ActivationFunctionFactory.generate(
                        layerStructure.substring(layerStructure.length() - 1)));
                currentWeightIndex += totalNumberOfWeights;
                this.layerList.add(currentLayer);
                prevLayer = currentLayer;
            }
            currentLayerIndex++;
        }
    }

    public void forward() {
        INDArray outputLayer = null;
        for (HiddenLayer layer : layerList) {
            layer.forward();
            outputLayer = layer.getOutputs();
        }
        this.output = outputLayer;
    }



    @Override
    public String toString() {
        return "MLP{" +
            "hiddenLayerList=" + layerList + "\n" +
            '}';
    }
}
