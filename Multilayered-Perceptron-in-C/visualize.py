import pandas as pd
import matplotlib.pyplot as plt

train_errors_per_epoch_df = pd.read_csv("train_errors.csv", sep=',')
correct_guesses_df = pd.read_csv("correct_guesses.csv", sep=',')
wrong_guesses_df = pd.read_csv("wrong_guesses.csv", sep=',')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# First subplot for Error vs Epochs
ax1.plot(train_errors_per_epoch_df["Epoch"],
         train_errors_per_epoch_df["Error"], linestyle='-', color='b')
ax1.set_title('Error vs Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Error')

# Second subplot for Scatter Plot
ax2.scatter(wrong_guesses_df["x"].values.tolist(
), wrong_guesses_df["y"].values.tolist(), color='red', label='Error guess', s=20)
ax2.scatter(correct_guesses_df["x"].values.tolist(
), correct_guesses_df["y"].values.tolist(), color='lightblue', label='Correct guess', s=20)
ax2.set_title('Scatter Plot of Correct and error guesses')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()

plt.tight_layout()  # Ensures that subplots do not overlap
plt.show()
