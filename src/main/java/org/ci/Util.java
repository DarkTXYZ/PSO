package org.ci;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Util {
    
    public static INDArray listToINDArray(List<Double> input) {
        double[] arr = input.stream().mapToDouble(Double::doubleValue).toArray();
        return Nd4j.create(arr);
    }
    
    public static int countRowCsv(String path) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(path));
        String input;
        int count = 0;
        while ((input = bufferedReader.readLine()) != null) {
            count++;
        }
        return count;
    }
    
    public static List<DataSet> getDataSet(String csvPath, String fileDelimiter, int numLinesToSkip,
                                           int batchSize, int labelIndexFrom, int labelIndexTo) throws IOException, InterruptedException {
        
        RecordReader rr = new CSVRecordReader(numLinesToSkip, fileDelimiter);
        int rows = Util.countRowCsv(csvPath);
        rr.initialize(new FileSplit(new File(csvPath)));

        DataSetIterator iterator =
            new RecordReaderDataSetIterator.Builder(rr, batchSize).regression(labelIndexFrom,labelIndexTo).build();
        
        List<DataSet> dataset = new ArrayList<>();
        
        while (iterator.hasNext()) {
            DataSet batch = iterator.next();
            dataset.add(batch);
        }
        Collections.shuffle(dataset);
        return dataset;
    }
    
    public static void writeArray(List<Double> avgError, List<Double> avgAccuracy, String fileName, String modelStructure)
        throws IOException {
        
        FileWriter fw = new FileWriter(fileName, true);
        BufferedWriter bw = new BufferedWriter(fw);
        
        Double[] lr = {0.003, 0.01, 0.03, 0.1, 0.3};
        List<Double> lrs = new ArrayList<Double>(Arrays.asList(lr));
        
        bw.write(modelStructure + "\n");
        bw.write("AvgError: [\n");
        
        int length = lrs.size();
        
        for (int i = length - 1; i >= 0; --i) {
            bw.write("[");
            String comma_out = ",\n";
            if(i == 0) {
                comma_out = "\n";
            }
            
            for (int j = 0; j < length; ++j) {
                String comma = ",";
                if(j == length-1) {
                    comma = "";
                }
                int ind = i+j*length;
                bw.write(avgError.get(ind) + comma);
            }
            bw.write("]" + comma_out);
        }
        bw.write("]\n");
        
        if(avgAccuracy == null){
            bw.close();
            return;
        }
        bw.write("AvgAcc: [\n");
        
        for (int i = length - 1; i >= 0; --i) {
            bw.write("[");
            String comma_out = ",\n";
            if(i == 0) {
                comma_out = "\n";
            }
            for (int j = 0; j < length; ++j) {
                String comma = ",";
                if(j == length-1) {
                    comma = "";
                }
                int ind = i+j*length;
                bw.write(avgAccuracy.get(ind) + comma);
            }
            bw.write("]" + comma_out);
        }
        bw.write("]\n");
        bw.close();
    }

    public static int calculateTotalWeight(String input) {
        String[] MLPStructure = input.split("-");
        int total = 0;
        int prev = -1;
        for(String layerStructure : MLPStructure) {
            if(prev == -1) {
                prev = Integer.parseInt(MLPStructure[0]);
            } else {
                int numberOfPerceptron =
                    Integer.parseInt(layerStructure.substring(0, layerStructure.length() - 1));
                total += (prev + 1) * numberOfPerceptron;
                prev = numberOfPerceptron;
            }
        }
        return total;
    }

    public static List<Integer> copy(List<Integer> list){
        return list.stream()
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }

    private static final List<Double> MEAN = Arrays.asList(
        1048.869651953974,
        894.4759627373445,
        794.8723326564782,
        1391.3632663603005,
        974.9515336112001,
        9.776599513414522,
        39.483611207026875,
        -6.837603701899451,
        1.865575948980707
    );

    private static final List<Double> STD = Arrays.asList(
        329.817014562947,
        342.31590184095796,
        321.97703135324883,
        467.1923819254541,
        456.9227277830469,
        43.20343763639509,
        51.21564480513585,
        38.976670158067705,
        41.38015397769953
    );

    public static INDArray normalizeInput(INDArray input) {
        INDArray meanIND = listToINDArray(MEAN);
        INDArray stdIND = listToINDArray(STD);

        return input.sub(meanIND).div(stdIND);
    }
    
    
}
