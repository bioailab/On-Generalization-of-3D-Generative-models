import pandas as pd
import matplotlib.pyplot as plt

# File paths for mean and variance CSVs
# mean_files = ["/home/workspace/3DShape2VecSet/util/metric_couch_single_mean.csv",
#               "/home/workspace/3DShape2VecSet/util/metric_couch_multi_mean.csv"]
# variance_files = ["/home/workspace/3DShape2VecSet/util/metric_couch_single_var.csv",
#                   "/home/workspace/3DShape2VecSet/util/metric_couch_multi_var.csv"]
mean_files = ["/home/workspace/3DShape2VecSet/util/metric_air_multi_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_arm_multi_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_couch_multi_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_desk_multi_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_boat_multi_mean.csv"]
variance_files = ["/home/workspace/3DShape2VecSet/util/metric_air_multi_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_arm_multi_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_couch_multi_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_desk_multi_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_boat_multi_var.csv"]
# Metrics to plot
metrics = ["cd", "fscore"]

# Colors and names for the datasets
colors = ["blue", "red","cyan","green","purple"]
names =["Jet", "Arm Chair","Couch","Desk","Boat"]
# colors = ["blue","red"]
# names = ['Couch Single', "Couch Multi"]
labels = ["Chamfer Distance","FScore"]
# Function to create and save plots
def plot_metrics(mean_files, variance_files, metrics, colors, names, output_prefix):
    for metric,label in zip(metrics,labels):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for mean_file, var_file, color, name in zip(mean_files, variance_files, colors, names):
            # Load data
            means = pd.read_csv(mean_file)
            variances = pd.read_csv(var_file)

            # Plot the mean values
            ax.plot(means['model'], means[metric], label=name, color=color)

            # Plot the shaded region for variance with opacity
            ax.fill_between(means['model'],
                            means[metric] - variances[metric],
                            means[metric] + variances[metric],
                            color=color, alpha=0.1)

        # Update layout and save plot
        # ax.set_title(f"{metric.upper()} Mean and Variance Plot", fontsize=16)
        ax.set_xlabel("Data Size", fontsize=30)  # Increased font size for x-axis label
        ax.set_ylabel(label, fontsize=30)  # Increased font size for y-axis label

        # Increase the size of tick labels
        ax.tick_params(axis='x', labelsize=30)  # Increase x-axis tick label size
        ax.tick_params(axis='y', labelsize=30)  # Increase y-axis tick label size
        if metric=='cd':
            ax.legend(fontsize=20, loc="upper right")
        else:
            ax.legend(fontsize=20,loc="upper left")
        # Save the figure
        # plt.tight_layout()
        plt.savefig(f"/home/workspace/3DShape2VecSet/plot/{output_prefix}_{metric}_plot_multi.png")
        plt.close()

# Generate and save plots for both metrics
plot_metrics(mean_files, variance_files, metrics, colors, names, output_prefix="metric")
