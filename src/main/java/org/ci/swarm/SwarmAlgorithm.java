package org.ci.swarm;

import lombok.Getter;
import org.ci.Util;
import org.ci.fitness_function.MAE;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Getter
public class SwarmAlgorithm {
    private final int maxGen;
    private final int populationSize;
    private final String structure;
    private Individual[] population = null;
    private INDArray[] velocity = null;
    private Individual[] pBest = null;
    private Individual gBest = null;

    private double p1 = 0.0;
    private double p2 = 0.0;

    public SwarmAlgorithm(int maxGen, int populationSize, String structure) {
        this.maxGen = maxGen;
        this.populationSize = populationSize;
        this.structure = structure;
        this.population = new Individual[populationSize];
        this.pBest = new Individual[populationSize];
        this.velocity = new INDArray[populationSize];
        Arrays.fill(this.pBest, null);
        Arrays.fill(this.velocity, Nd4j.zeros(Util.calculateTotalWeight(structure)));

        Random random = new Random();
        double c = random.nextDouble();
        double p = random.nextDouble();
        this.p1 = p * c;
        this.p2 = (1-p) * (1 - c);

    }

    public void train(List<DataSet> trainData) {
        initPopulation();
        System.out.println("Start Training...");
        for (int i = 0; i < maxGen; ++i) {
            System.out.println("Gen " + (i + 1) + " training...");
            evaluatePopulation(trainData);
            findPBestGBest();
            updateVelocityPhase();
            System.out.println("Gen " + (i + 1) + " finished");
//            System.out.println("________________________________________");
        }
        evaluatePopulation(trainData);
    }

    private void initPopulation() {
        int totalNumberOfWeight = Util.calculateTotalWeight(structure);

        for (int i = 0; i < populationSize; ++i) {
            INDArray random = Nd4j.rand(totalNumberOfWeight).mul(2).sub(1);
            WeightIndividual c = new WeightIndividual(random, structure, new MAE());
            population[i] = c;
        }
    }

    private void evaluatePopulation(List<DataSet> trainData) {
        for (Individual individual : population)
            individual.calculateFitnessValue(trainData);
//        System.out.println("Population");
//        for (Individual individual : population) {
//            System.out.println(individual);
//        }
//        for (INDArray individual : velocity) {
//            System.out.println(individual);
//        }
//        System.out.println("P Best");
//        for (Individual individual : pBest) {
//            System.out.println(individual);
//        }
//        System.out.println("G Best: " + gBest);
    }

    private void findPBestGBest() {
        int i = 0;
//        System.out.println("Before");
//        for (Individual item : pBest) {
//            System.out.println(item);
//        }
//        System.out.println(gBest);
//        for (Individual value : population) {
//            System.out.println(value);
//        }
//        System.out.println("________________________________________");
        for (Individual individual : population) {
            Individual individualPBest = this.pBest[i];
            if (individualPBest == null) {
                this.pBest[i] = new WeightIndividual(individual);
            } else if (individual.getFitnessValue() < individualPBest.getFitnessValue()) {
                this.pBest[i] = new WeightIndividual(individual);
            }

            if (this.gBest == null) {
                this.gBest = new WeightIndividual(individual);
            } else if (individual.getFitnessValue() < gBest.getFitnessValue()) {
                this.gBest = new WeightIndividual(individual);
            }

            ++i;
        }
//        System.out.println("After");
//        for (Individual item : pBest) {
//            System.out.println(item);
//        }
//        System.out.println(gBest);
//        for (Individual value : population) {
//            System.out.println(value);
//        }
    }

    private void updateVelocityPhase() {
        int i = 0;
        INDArray gBestWeight = this.gBest.getWeights();
//        System.out.println("Before");
//        for (Individual value : population) {
//            System.out.println(value);
//        }
//        System.out.println("________________________________________");
        for (Individual individual : population) {
            INDArray vPrev = velocity[i];
            INDArray pBestWeight = this.pBest[i].getWeights();
            INDArray indWeight = individual.getWeights();

            INDArray vCurrent = vPrev.add(pBestWeight.sub(indWeight).mul(p1))
                .add(gBestWeight.sub(indWeight).mul(p2));
            velocity[i] = vCurrent;

//            if(i == 4) {
//                System.out.println("********************");
//                System.out.println(pBestWeight);
//                System.out.println(gBestWeight);
//                System.out.println(indWeight);
//                System.out.println(pBestWeight.sub(indWeight).mul(p1));
//                System.out.println(gBestWeight.sub(indWeight).mul(p2));
//                System.out.println(pBestWeight.sub(indWeight).mul(p1).add(gBestWeight.sub(indWeight).mul(p2)));
//                System.out.println(vPrev);
//                System.out.println(vCurrent);
//                System.out.println("********************");
//            }

            INDArray individualUpdated = indWeight.add(vCurrent);
            individual.setWeights(individualUpdated);

            ++i;
        }
//        System.out.println("After");
//        for (Individual value : population) {
//            System.out.println(value);
//        }
    }
}
