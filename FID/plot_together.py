import os
import argparse
import numpy as np
import plotly.graph_objects as go


def read_fid_values(directory):
    fid_values = {}
    for filename in os.listdir(directory):
        # if filename.split('_')[1].split('.')[0]=='fid'and filename.endswith('.txt'):
        if filename.startswith('kid_') and filename.endswith('.txt'):
            # size = int(filename.split('_')[0].split('.')[0])
            size = int(filename.split('_')[1].split('.')[0])
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                line = file.readline().strip()
                fid_value = float(line.split(': ')[1])

            if size not in fid_values:
                fid_values[size] = []
            fid_values[size].append(fid_value)

    return fid_values


def plot_fid_values(root_folder, subfolders):
    category_names = ['Jet', 'Arm Chair', 'Couch', 'Desk', 'Boat']
    # category_names = ['Jet','Arm','Couch','Desk','Boat', 'Jet-Single']
    fig = go.Figure()  # Create a single figure

    all_fid_values = {}

    for i, (folder, name) in enumerate(zip(subfolders, category_names)):
        dir_path = os.path.join(root_folder, folder)
        fid_values = read_fid_values(dir_path)

        # Collect all fid_values to calculate the average later
        for size, values in fid_values.items():
            if size not in all_fid_values:
                all_fid_values[size] = []
            all_fid_values[size].extend(values)

        sizes = sorted(fid_values.keys())
        avg_values = [np.mean(fid_values[size]) for size in sizes]

        # Add a trace for each category with thicker lines and larger markers
        fig.add_trace(go.Scatter(
            x=sizes,
            y=avg_values,
            mode='markers+lines',
            name=name,
            line=dict(width=4),  # Set line width (thickness)
            marker=dict(size=15)  # Increase marker size
        ))

    # Calculate the overall average
    avg_sizes = sorted(all_fid_values.keys())
    overall_avg_values = [np.mean(all_fid_values[size]) for size in avg_sizes]

    # Add the average trace with thicker lines and larger markers
    fig.add_trace(go.Scatter(
        x=avg_sizes,
        y=overall_avg_values,
        mode='markers+lines',
        name='Average',
        marker=dict(color='red', size=20),  # Larger marker size for average
        line=dict(width=6)  # Set line width (thickness)
    ))

    # Update layout with legend on the plot and adjusted font sizes
    fig.update_layout(
        title='KID Value vs. Data Size',
        xaxis_title='Data Size',
        yaxis_title='KID Value',
        template='plotly_white',
        width=2000,
        height=800,
        title_font=dict(size=72),  # Increase title font size
        xaxis_title_font=dict(size=72),  # Increase x-axis label font size
        yaxis_title_font=dict(size=72),  # Increase y-axis label font size
        xaxis_tickfont=dict(size=72),  # Increase x-axis tick font size
        yaxis_tickfont=dict(size=72),  # Increase y-axis tick font size
        legend=dict(
            font=dict(size=50),  # Increase legend font size
            bgcolor="rgba(255,255,255,0.5)",  # Semi-transparent background
            # x=0.02,  # Adjust x position on the plot
            # y=0.98,  # Adjust y position on the plot
        )
    )

    fig.write_image('/home/workspace/3DShape2VecSet/plot/kid_combined_graph_1000.png')  # Save the plot
    fig.show()


def main():
    parser = argparse.ArgumentParser(description='Plot FID values for multiple folders')
    parser.add_argument('--root', type=str, default="/home/workspace/3DShape2VecSet/FID", help='Root folder containing subfolders with KID files')
    parser.add_argument('--folders', type=str, nargs='+', default=['Jet','Arm','Couch','Desk','Boat'], help='List of subfolders to process')

    args = parser.parse_args()

    plot_fid_values(args.root, args.folders)


if __name__ == '__main__':
    main()
