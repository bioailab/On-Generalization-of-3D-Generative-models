import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("/home/workspace/3DShape2VecSet/metrics-air.csv")

# Extract the number of samples per class from the model column
df['samples_per_class'] = df['model'].str.split('-').str[0].astype(int)

# Plotting
plt.figure(figsize=(10, 6))

# Plot CD
plt.plot(df['samples_per_class'], df['cd'], marker='o', label='Chamfer Distance (CD)')

# Plot F-Score
plt.plot(df['samples_per_class'], df['fscore'], marker='s', label='F-Score')

# Customization
plt.xlabel('Number of Samples per Class')
plt.ylabel('Values')
plt.title('Chamfer Distance and F-Score vs. Number of Samples per Class')
plt.legend()
plt.grid(True)
# Save plot as a file
plt.savefig("cd_fscore_vs_samples.png", dpi=300, bbox_inches="tight")  # Save as PNG with high resolution

# Show plot
plt.show()
