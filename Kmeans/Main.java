import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class Main {


    public static void printClusters(List<List<DoublePoint>> clusters) {
        for (int i = 0; i < clusters.size(); i++) {
            System.out.println("Cluster " + (i + 1) + ": " + clusters.get(i));
        }


    }

    public static void saveResultsToFile( List<DoublePoint> centers, int numClusters, double totalError) {
        String fileName = "results_simulation.txt";

        try (FileWriter writer = new FileWriter(fileName, true)) {
            writer.write("Centers: " + centers + "\n");
            writer.write("Number of Clusters: " + numClusters + "\n");
            writer.write("Total Error: " + totalError + "\n");
            writer.write("#################################################"+ "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static List<DoublePoint> readPoints(String filename) {
        List<DoublePoint> points = new ArrayList<>();

        try (Scanner scanner = new Scanner(new File(filename))) {
            while (scanner.hasNextDouble()) {
                double x = scanner.nextDouble();
                double y = scanner.nextDouble();
                points.add(new DoublePoint(x, y));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return points;

    }


    public static void main(String[] args) {
        String filename = "points.txt";
        int maxIteration = 1000;

        if (args.length == 0) {
            System.out.println("Provide the number of centers.");
            return;
        }

        int numOfCenters = Integer.parseInt(args[0]);
        long seed = Long.parseLong(args[1]);


        System.out.println("the number of Centers : " + numOfCenters);
        System.out.println("Seed : " + seed);

        List<DoublePoint> points = readPoints(filename);

        Kmeans km = new Kmeans(numOfCenters, maxIteration, points ,seed);
        List<List<DoublePoint>> clusters = km.cluster();
        List<DoublePoint> centers = km.getCenters();
        
        //printClusters(clusters);
        double totalError = km.calculateTotalDistance();

        saveResultsToFile( centers, clusters.size(), totalError);




    }
}