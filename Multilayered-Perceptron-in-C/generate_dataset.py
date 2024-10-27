import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_and_classify_points():
    # Generate 8000 random points within the square [-1, 1] x [-1, 1]
    points = np.random.uniform(-1, 1, (8000, 2))
    labels = []

    for x1, x2 in points:
        # Apply the classification rules as described in the document
        if ((x1 - 0.5)**2 + (x2 - 0.5)**2 < 0.2 and x1 > 0.5) or \
           ((x1 + 0.5)**2 + (x2 + 0.5)**2 < 0.2 and x1 > -0.5) or \
           ((x1 - 0.5)**2 + (x2 + 0.5)**2 < 0.2 and x1 > 0.5) or \
           ((x1 + 0.5)**2 + (x2 - 0.5)**2 < 0.2 and x1 > -0.5):
            labels.append('C1')
        elif ((x1 - 0.5)**2 + (x2 - 0.5)**2 < 0.2 and x1 < 0.5) or \
             ((x1 + 0.5)**2 + (x2 + 0.5)**2 < 0.2 and x1 < -0.5) or \
             ((x1 - 0.5)**2 + (x2 + 0.5)**2 < 0.2 and x1 < 0.5) or \
             ((x1 + 0.5)**2 + (x2 - 0.5)**2 < 0.2 and x1 < -0.5):
            labels.append('C2')
        elif x1 > 0:
            labels.append('C3')
        else:
            labels.append('C4')

    return points, labels


def plot_classified_points(points, labels):
    # Define colors for each category
    colors = {'C1': 'red', 'C2': 'blue', 'C3': 'green', 'C4': 'yellow'}

    for label in set(labels):
        # Filter points by label
        category_points = points[np.array(labels) == label]
        plt.scatter(
            category_points[:, 0], category_points[:, 1], c=colors[label], label=label)

    plt.title('Classification of Points into Four Categories')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


def write_to_files(points, labels):
    # Split the generated points into training and testing datasets (4000 points each)
    train_points = points[:4000]
    test_points = points[4000:]

    train_labels = labels[:4000]
    test_labels = labels[4000:]

    # Convert to DataFrame for easier file writing
    train_df = pd.DataFrame(train_points, columns=['x1', 'x2'])
    train_df['label'] = train_labels

    test_df = pd.DataFrame(test_points, columns=['x1', 'x2'])
    test_df['label'] = test_labels

    # Write to CSV files
    train_df.to_csv('train_dataset.csv', index=False)
    test_df.to_csv('test_dataset.csv', index=False)


# Generate and classify the points
points, labels = generate_and_classify_points()

# Plot the classified points
plot_classified_points(points, labels)

# Write the points to files
write_to_files(points, labels)
