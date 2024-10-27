import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Kmeans {

    private static final double EPSILON = 1e-6;

    private long seed;

    private int numOfCenters;
    private int maxIteration;
    private List<DoublePoint> points;

    private List<List<DoublePoint>> clusters;

    private List<DoublePoint> centers;

    private double totalDistance;


    public Kmeans(int numOfCenters, int maxIteration, List<DoublePoint> points,long seed) {
        if (numOfCenters <= 0 || maxIteration <= 0 || points == null || points.isEmpty()) {
            throw new IllegalArgumentException("Invalid input parameters.");
        }

        this.numOfCenters = numOfCenters;
        this.maxIteration = maxIteration;
        this.points = points;
        this.seed = seed;
    }

    public List<List<DoublePoint>> cluster() {
        centers = initializeCenters(seed);

        for (int iteration = 0; iteration < maxIteration; iteration++) {
            clusters = assignToClusters(centers);
            List<DoublePoint> oldCentroids = new ArrayList<>(centers);
            centers = updateCenters(clusters);

            if (hasConverged(oldCentroids, centers)) {
                break;
            }
        }
        clusters = assignToClusters(centers);

        return clusters;
    }

    private List<DoublePoint> initializeCenters(long seed) {
        List<DoublePoint> centers = new ArrayList<>();
        Random random = new Random(seed);
        for (int i = 0; i < numOfCenters; i++) {
            int randomIndex = random.nextInt(points.size());
            centers.add(points.get(randomIndex));
        }

        return centers;
    }

    private List<List<DoublePoint>> assignToClusters(List<DoublePoint> centroids) {
        List<List<DoublePoint>> clusters = new ArrayList<>();

        for (int i = 0; i < numOfCenters; i++) {
            clusters.add(new ArrayList<>());
        }

        for (DoublePoint point : points) {
            int clusterIndex = findClosestCentroid(point, centroids);
            clusters.get(clusterIndex).add(point);
        }

        return clusters;
    }

    private int findClosestCentroid(DoublePoint point, List<DoublePoint> centers) {
        int closestIndex = 0;
        double closestDist = findDist(point, centers.get(0));

        for (int i = 1; i < centers.size(); i++) {
            double currDist = findDist(point, centers.get(i));
            if (currDist < closestDist) {
                closestIndex = i;
                closestDist = currDist;
            }
        }
        return closestIndex;
    }

    private double findDist(DoublePoint p1, DoublePoint p2) {
        return Math.sqrt(Math.pow(p1.getX() - p2.getX(), 2) + Math.pow(p1.getY() - p2.getY(), 2));
    }

    private List<DoublePoint> updateCenters(List<List<DoublePoint>> clusters) {
        List<DoublePoint> newCenters = new ArrayList<>();

        for (List<DoublePoint> cluster : clusters) {
            if (!cluster.isEmpty()) {
                double sumX = 0;
                double sumY = 0;

                for (DoublePoint point : cluster) {
                    sumX += point.getX();
                    sumY += point.getY();
                }

                double centerX = sumX / cluster.size();
                double centerY = sumY / cluster.size();

                newCenters.add(new DoublePoint(centerX, centerY));
            }
        }

        return newCenters;
    }

    private boolean hasConverged(List<DoublePoint> oldCentroids, List<DoublePoint> newCentroids) {
        for (int i = 0; i < oldCentroids.size(); i++) {
            if (findDist(oldCentroids.get(i), newCentroids.get(i)) > EPSILON) {
                return false;
            }
        }

        return true;
    }

    public double calculateTotalDistance() {
        totalDistance = 0;

        for (int i = 0; i < numOfCenters; i++) {
            DoublePoint center = centers.get(i);

            for (DoublePoint point : clusters.get(i)) {
                double dist = findDist(point, center);
                totalDistance += dist;
            }
        }

        System.out.println("clustering error: " + totalDistance);
        return totalDistance;
    }

    public List<DoublePoint> getCenters() {
        return centers;
    }

}
