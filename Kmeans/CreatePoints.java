import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CreatePoints {


    private static void writePointsToFile(List<DoublePoint> points, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            for (DoublePoint point : points) {
                writer.println(point.getX() + " " + point.getY());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void generateRandomPoints() {
        List<DoublePoint> points = new ArrayList<>();
        Random random = new Random();


        for (int i = 0; i < 150; i++) {
            points.add(new DoublePoint((0.8 + 0.4 * random.nextDouble()), 0.8 + 0.4 * random.nextDouble()));
            points.add(new DoublePoint(0.5 * random.nextDouble(), 0.5 * random.nextDouble()));
            points.add(new DoublePoint(1.5 + 0.5 * random.nextDouble(), 0.5 * random.nextDouble()));
            points.add(new DoublePoint(0.5 * random.nextDouble(), 1.5 + 0.5 * random.nextDouble()));
            points.add(new DoublePoint(1.5 + 0.5 * random.nextDouble(), 1.5 + 0.5 * random.nextDouble()));
            points.add(new DoublePoint(2 * random.nextDouble(), 2 * random.nextDouble()));
        }

        for (int j = 0; j < 75; j++) {
            points.add(new DoublePoint(0.4 * random.nextDouble(), 0.8 + 0.4 * random.nextDouble()));
            points.add(new DoublePoint(1.6 + 0.4 * random.nextDouble(), 0.8 + 0.4 * random.nextDouble()));
            points.add(new DoublePoint(0.8 + 0.4 * random.nextDouble(), 0.3 + 0.4 * random.nextDouble()));
            points.add(new DoublePoint(0.8 + 0.4 * random.nextDouble(), 1.3 + 0.4 * random.nextDouble()));
        }

        writePointsToFile(points, "points.txt");

    }


    public static void main(String[] args) {
        generateRandomPoints();
    }
}

