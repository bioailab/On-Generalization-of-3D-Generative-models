import pandas as pd
import matplotlib.pyplot as plt

# File paths for mean and variance CSVs
mean_files = [
    "/home/workspace/3DShape2VecSet/util/metric_airplane_kl_mean.csv",
    "/home/workspace/3DShape2VecSet/util/metric_chair_kl_mean.csv",
    "/home/workspace/3DShape2VecSet/util/metric_table_kl_mean.csv",
    "/home/workspace/3DShape2VecSet/util/metric_bench_kl_mean.csv",
    "/home/workspace/3DShape2VecSet/util/metric_sofa_kl_mean.csv",
    "/home/workspace/3DShape2VecSet/util/metric_car_kl_mean.csv",
]

variance_files = [
    "/home/workspace/3DShape2VecSet/util/metric_airplane_kl_var.csv",
    "/home/workspace/3DShape2VecSet/util/metric_chair_kl_var.csv",
    "/home/workspace/3DShape2VecSet/util/metric_table_kl_var.csv",
    "/home/workspace/3DShape2VecSet/util/metric_bench_kl_var.csv",
    "/home/workspace/3DShape2VecSet/util/metric_sofa_kl_var.csv",
    "/home/workspace/3DShape2VecSet/util/metric_car_kl_var.csv",
]

# Metrics to plot
metrics = ["cd", "fscore"]
colors = ["blue", "red", "cyan", "green", "purple", "orange"]
names = ["AirPlane", "Chair", "Table", "Bench", "Sofa", "Car"]
labels = ["Chamfer Distance", "FScore"]

def plot_metrics_with_log_scale(mean_files, variance_files, metrics, colors, names, labels):
    for metric, label, lim in zip(metrics, labels,(0.5,0.3)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # First subplot: Seen classes
        for mean_file, var_file, color, name in zip(mean_files[:3], variance_files[:3], colors[:3], names[:3]):
            means = pd.read_csv(mean_file)
            variances = pd.read_csv(var_file)
            means['model'] = means['model'].astype(float)
            axes[0].plot(means['model'], means[metric], label=name, color=color)
            axes[0].fill_between(means['model'],
                                 means[metric] - variances[metric],
                                 means[metric] + variances[metric],
                                 color=color, alpha=0.1)
        
        axes[0].set_xscale('log')
        # axes[0].set_yscale('log')
        axes[0].set_title("Seen Classes", fontsize=30)
        axes[0].set_xlabel(r"$\beta$ values", fontsize=30)
        axes[0].set_ylabel(label, fontsize=30)
        axes[0].set_ylim(0.01,lim)
        # if label == 'Chamfer Distance':
        axes[0].legend(fontsize=20, loc='lower left')
        
        axes[0].tick_params(axis='both', which='major', labelsize=20)

        # Second subplot: Unseen classes
        for mean_file, var_file, color, name in zip(mean_files[3:], variance_files[3:], colors[3:], names[3:]):
            means = pd.read_csv(mean_file)
            variances = pd.read_csv(var_file)
            means['model'] = means['model'].astype(float)
            axes[1].plot(means['model'], means[metric], label=name, color=color)
            axes[1].fill_between(means['model'],
                                 means[metric] - variances[metric],
                                 means[metric] + variances[metric],
                                 color=color, alpha=0.1)
        
        axes[1].set_xscale('log')
        # axes[1].set_yscale('log')
        axes[1].set_title("Unseen Classes", fontsize=30)
        axes[1].set_xlabel(r"$\beta$ values", fontsize=30)
        if label == 'Chamfer Distance':
            axes[1].legend(fontsize=20, loc='lower left')
        else:
            axes[1].legend(fontsize=20, loc='upper left')
        axes[1].tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()
        plt.savefig(f"/home/workspace/3DShape2VecSet/plot/metric_{metric}_plot_kl.png")
        plt.close()

# Generate plots
plot_metrics_with_log_scale(mean_files, variance_files, metrics, colors, names, labels)
