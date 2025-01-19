import csv
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

def plot_steps_with_labels_and_images(input_csv, output_plot):
    # Read the CSV file
    results = []
    with open(input_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['Top_5_Labels'] = eval(row['Top_5_Labels'])
            row['Filenames'] = eval(row['Filenames'])
            row['Steps'] = int(row['Steps'])  # Convert 'Steps' to integers
            results.append(row)

    # Sort the results by 'Steps'
    results.sort(key=lambda x: x['Steps'])

    # Prepare data
    steps = [result['Steps'] for result in results]
    label_counts_per_step = [Counter(result['Top_5_Labels']) for result in results]
    file_paths_per_step = [result['Filenames'] for result in results]

    unique_labels = list({label for counts in label_counts_per_step for label in counts.keys()})

    # Set up the plot layout
    fig = plt.figure(figsize=(15, 20))  # Increased figure height for more images
    gs = plt.GridSpec(2, 1, height_ratios=[1, 4])  # Allocate more space for the image grid

    # Bar chart for label counts
    ax1 = plt.subplot(gs[0])
    width = 0.15  # Bar width
    x = range(len(steps))

    for i, label in enumerate(unique_labels):
        label_counts = [counts.get(label, 0) for counts in label_counts_per_step]
        ax1.bar([xi + i * width for xi in x], label_counts, width, label=label)

    ax1.set_title('Steps vs Label Counts')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Counts')
    ax1.set_xticks([xi + (width * (len(unique_labels) - 1) / 2) for xi in x])
    ax1.set_xticklabels(steps)
    ax1.legend(title="Labels")
    ax1.grid(axis='y')

    # Adjust space between plots
    plt.subplots_adjust(hspace=0.3)

    # Image grid below the bar chart
    num_steps = len(steps)
    num_rows = 5  # Number of rows (top 5 images per step)
    num_cols = num_steps  # Number of columns (one per step)

    grid_width = 1.0 / 24
    grid_height = 1.0 / 24  # Add some margin below
    target_image_size = (1500,1500)
    for col_idx, file_paths in enumerate(file_paths_per_step):
        for row_idx, file_path in enumerate(file_paths[:5]):  # Top 5 images
            try:
                img = Image.open(file_path[0])
                img = img.resize(target_image_size) 
                x_pos = 0.14 + col_idx * grid_width
                y_pos = 0.53 + (4 - row_idx) * grid_height  # Ensure grid starts below the bar plot
                ax_img = fig.add_axes([x_pos, y_pos, grid_width, grid_height])  # Add image in the calculated position
                ax_img.imshow(img)
                ax_img.axis('off')
            except Exception as e:
                print(f"Could not load image {file_path}: {e}")

    # Save the combined plot
    plt.savefig(output_plot)
    plt.close()
    print(f"Combined plot saved to {output_plot}")


# Example usage
if __name__ == '__main__':
    input_csv = 'output_labels.csv'  # Path to your CSV file
    output_plot = 'steps_vs_labels_with_images-kle_2.png'  # Output plot filename
    plot_steps_with_labels_and_images(input_csv, output_plot)
