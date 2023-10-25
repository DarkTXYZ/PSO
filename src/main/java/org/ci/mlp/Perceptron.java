package org.ci.mlp;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.ci.mlp.activation_function.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

@Getter
@Setter
@Builder
public class Perceptron {
    private INDArray weights;
    private Double bias;
    private Double weightBias;
    private Double value;
    private Double output;

    private ActivationFunction activationFunction;
    
    public Double calculateOutput(INDArray input) {
        INDArray dotProduct = input.mul(weights);
        this.value = dotProduct.sum().getDouble();
        this.output = this.activationFunction.activate(this.value + this.bias * this.weightBias);
        return this.output;
    }

    @Override
    public String toString() {
        return "Perceptron{" + "\n" +
            "\tweights = " + weights + "\n" +
            "\tbias = " + bias + " * " + weightBias + "\n" +
            "\toutput = " + output + "\n" +
            "\tactivationFunction = " + activationFunction + "\n" +
            '}' + "\n";
    }
}
