package org.ci;

import org.apache.commons.collections4.ListUtils;
import org.ci.swarm.Individual;
import org.ci.swarm.SwarmAlgorithm;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.shade.guava.math.IntMath;

import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Collection;

public class Main {
    public static void main(String[] args) throws IOException, InterruptedException {
        List<DataSet> dataset = Util.getDataSet("normalized_AirQualityUCIDataset.csv", ",", 1, 1, 9, 9);

        int partitionSize = IntMath.divide(dataset.size(), 10, RoundingMode.UP);
        List<List<DataSet>> folds = ListUtils.partition(dataset, partitionSize);

        List<String> modelStructures = Arrays.asList(
            "8-1R"
//            "8-3R-1R"
        );

        int MAX_GEN = 10;
        int POPULATION_SIZE = 30;

        for (String modelStructure : modelStructures) {

            List<Double> error = new ArrayList<>();

            double avgError = 0.0;

            System.out.println(modelStructure);

            for (int testFold = 0; testFold < 10; ++testFold) {
                System.out.println("Fold " + (testFold+1));
                List<List<DataSet>> temp = new ArrayList<>();

                for (int trainFold = 0; trainFold < 10; ++trainFold) {
                    if (testFold == trainFold)
                        continue;
                    temp.add(folds.get(trainFold));
                }

                List<DataSet> trainData = temp.stream()
                    .flatMap(Collection::stream)
                    .toList();

                List<DataSet> testData = folds.get(testFold);

                SwarmAlgorithm sw =
                    new SwarmAlgorithm(MAX_GEN, POPULATION_SIZE, modelStructure);

                sw.train(trainData);

                Individual best = sw.getGBest();

                best.calculateError(testData);
//
                error.add(best.getError());
                avgError += best.getError();
            }
            System.out.println("Avg error: " + (avgError /10.0));
            FileWriter fileWriter = getFileWriter(modelStructure, error, avgError);
//            fileWriter.write("Avg accuracy: " + (avgAccuracy/10) + System.lineSeparator());
            fileWriter.close();
        }

    }

    private static FileWriter getFileWriter(String modelStructure, List<Double> error,
                                            double avgError) throws IOException {
        SimpleDateFormat dateFormat = new SimpleDateFormat("dd-MM-yy_HH-mm-ss");
        Date current = new Date();
        FileWriter fileWriter = new FileWriter("Result_" + modelStructure + "_" + dateFormat.format(current));
        fileWriter.write(error + System.lineSeparator());
//            fileWriter.write(bestChromosomes + System.lineSeparator());
//            fileWriter.write(accuracy + System.lineSeparator());
        fileWriter.write("Avg error: " + (avgError /10.0) + System.lineSeparator());
        return fileWriter;
    }
}