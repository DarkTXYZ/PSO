package org.ci.mlp.activation_function;

public interface ActivationFunction {
    Double activate(Double b);
    
    Double activateDiff(Double b);
}
