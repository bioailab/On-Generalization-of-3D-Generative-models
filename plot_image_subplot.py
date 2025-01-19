import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths for mean and variance CSVs
mean_files = ["/home/workspace/3DShape2VecSet/util/metric_airplane_h_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_chair_h_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_table_h_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_bench_h_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_sofa_h_mean.csv",
              "/home/workspace/3DShape2VecSet/util/metric_car_h_mean.csv",
                ]

variance_files = ["/home/workspace/3DShape2VecSet/util/metric_airplane_h_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_chair_h_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_table_h_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_bench_h_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_sofa_h_var.csv",
                  "/home/workspace/3DShape2VecSet/util/metric_car_h_var.csv",
                  ]

# Metrics to plot
metrics = ["cd", "fscore"]

# Colors and names for the datasets
colors = ["blue", "red", "cyan", "green", "purple", "orange"]
names = ["AirPlane", "Chair","Table","Bench","Sofa","Car"]
labels = ["Chamfer Distance","FScore"]
# Function to create and save plots with Matplotlib
def plot_metrics_with_separate_legends(mean_files, variance_files, metrics, colors, names,labels):
    for metric,label in zip(metrics,labels):
        fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
        
        # Split the files: first 5 for the first subplot, next 3 for the second
        first_subplot_files = mean_files[:3], variance_files[:3], colors[:3], names[:3]
        second_subplot_files = mean_files[3:], variance_files[3:], colors[3:], names[3:]

        # First subplot (first 5 datasets)
        for mean_file, var_file, color, name in zip(*first_subplot_files):
            means = pd.read_csv(mean_file)
            variances = pd.read_csv(var_file)
            means['model'] = means['model'].astype(float)
            axes[0].plot(means['model'], means[metric], label=name, color=color)
            # axes[0].set_xscale('log')
            axes[0].fill_between(means['model'],
                                 means[metric] - variances[metric],
                                 means[metric] + variances[metric],
                                 color=color, alpha=0.1)
        
        axes[0].set_title("Seen Classes", fontsize=30)
        axes[0].set_xlabel(r"$\beta$ values", fontsize=30)
        axes[0].set_ylabel(label, fontsize=30)
        axes[0].legend(fontsize=30, loc='upper left')
        
        # Increase font size of axis tick labels
        axes[0].tick_params(axis='both', which='major', labelsize=30)
        
        # Second subplot (next 3 datasets)
        for mean_file, var_file, color, name in zip(*second_subplot_files):
            means = pd.read_csv(mean_file)
            variances = pd.read_csv(var_file)
            means['model'] = means['model'].astype(float)
            axes[1].plot(means['model'], means[metric], label=name, color=color)
            # axes[1].set_xscale('log')
            axes[1].fill_between(means['model'],
                                 means[metric] - variances[metric],
                                 means[metric] + variances[metric],
                                 color=color, alpha=0.1)
        
        axes[1].set_title("Unseen Classes", fontsize=30)
        axes[1].set_xlabel(r"$\beta$ values", fontsize=30)
        axes[1].legend(fontsize=30, loc='upper right')
        # axes[1].legend(fontsize=15,loc='best')
        
        # Increase font size of axis tick labels
        axes[1].tick_params(axis='both', which='major', labelsize=30)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"/home/workspace/3DShape2VecSet/plot/metric_{metric}_plot_with_subplots_h.png")
        plt.close()

# Generate and save plots for both metrics
plot_metrics_with_separate_legends(mean_files, variance_files, metrics, colors, names,labels)
