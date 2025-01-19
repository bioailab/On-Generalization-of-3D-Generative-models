import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
kl_df = pd.read_csv('/home/workspace/3DShape2VecSet/smoothness_kl.csv')  # Replace with the path to your first CSV file
hellinger_df = pd.read_csv('/home/workspace/3DShape2VecSet/smoothness_h.csv') 
# Plotting
plt.figure(figsize=(10, 6))

# Plot KL score
plt.plot(kl_df['model'], kl_df['Curvature'], label='KL', color='blue', marker='o')

# Plot Hellinger Curvature
plt.plot(hellinger_df['model'], hellinger_df['Curvature'], label='Hellinger', color='red', marker='o')

# Set the x-axis to log scale
plt.xscale('log')

# Labeling the axes
plt.xlabel('Model')
plt.ylabel('Curvature')

# Adding a title
plt.title('Model vs Curvature (KL and Hellinger)')

# Adding a legend
plt.legend()

# Show the plot
plt.grid(True)

plt.savefig('Curvature smoothness.png', dpi=300)
plt.show()
