import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = "output_labels.csv"
df = pd.read_csv(csv_path)

# If Final_Label contains lists, split them into multiple rows
df['Final_Label'] = df['Final_Label'].apply(
    lambda x: x.strip('[]').replace("'", "").split(', ') if isinstance(x, str) else [x]
)
df = df.explode('Final_Label')

# Sort by Target column
df = df.sort_values(by='Target', ascending=True)

# Get the unique classes
unique_classes = sorted(df['Final_Label'].unique())

# Create a class-to-index mapping for consistent y-axis positioning
class_to_index = {label: i for i, label in enumerate(unique_classes)}
df['Class_Index'] = df['Final_Label'].map(class_to_index)

# Create the plot
plt.figure(figsize=(10, 3))

# Scatter plot of targets vs. unique classes
for label, group in df.groupby('Final_Label'):
    plt.scatter(group['Target'], group['Class_Index'], label=label, s=100, alpha=0.7)

# Draw vertical lines for each x-axis label to the corresponding points
for _, row in df.iterrows():
    plt.plot([row['Target'], row['Target']], [0, row['Class_Index']], color='gray', linestyle='--', alpha=0.5)

# Customizing the plot
plt.title("Targets vs. Predicted Classes", fontsize=16)
plt.xlabel("Target", fontsize=14)
plt.ylabel("Classes", fontsize=14)

# Set x-axis as string to match the Target values
plt.xticks(ticks=df['Target'], labels=df['Target'].astype(str))

plt.yticks(ticks=list(class_to_index.values()), labels=list(class_to_index.keys()))
plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()

# Save or show the plot
plt.savefig("diffusion_steps_vs_classes.png", dpi=300)
plt.show()
