import os
import argparse
import numpy as np
import plotly.graph_objects as go


def read_fid_values(directory):
    fid_values = {}
    for filename in os.listdir(directory):
        # if filename.startswith('kid_') and filename.endswith('.txt'):
        if filename.split('_')[1].split('.')[0] == 'fid' and filename.endswith('.txt'):
            size = int(filename.split('_')[0].split('.')[0])
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                line = file.readline().strip()
                fid_value = float(line.split(': ')[1])

            if size not in fid_values:
                fid_values[size] = []
            fid_values[size].append(fid_value)

    return fid_values


def plot_fid_values(root_folder, subfolders):
    category_names = ['Jet', 'ArmChair', 'Couch', 'Desk', 'Boat', 'Couch-Single']
    fig = go.Figure()

    all_fid_values = {}
    first_5_avg_values = {}  # To store averages of the first 5 values for each size

    for i, (folder, name) in enumerate(zip(subfolders, category_names)):
        dir_path = os.path.join(root_folder, folder)
        fid_values = read_fid_values(dir_path)

        # Collect all fid_values for overall average
        for size, values in fid_values.items():
            if size not in all_fid_values:
                all_fid_values[size] = []
            all_fid_values[size].extend(values)

            # Calculate first 5 average for this size
            if size not in first_5_avg_values:
                first_5_avg_values[size] = []
            first_5_avg_values[size].extend(values[:5])  # Take only the first 5 values

        sizes = sorted(fid_values.keys())
        avg_values = [np.mean(fid_values[size]) for size in sizes]

        # Add a trace for each category with thicker lines and larger markers
        fig.add_trace(go.Scatter(
            x=sizes,
            y=avg_values,
            mode='markers+lines',
            name=name,
            line=dict(width=4),
            marker=dict(size=15)
        ))

    # Calculate the overall average
    avg_sizes = sorted(all_fid_values.keys())
    overall_avg_values = [np.mean(all_fid_values[size]) for size in avg_sizes]

    # Calculate the average of the first 5 values
    first_5_sizes = sorted(first_5_avg_values.keys())
    first_5_avg_values = [np.mean(first_5_avg_values[size]) for size in first_5_sizes]

    # Add the overall average trace
    # fig.add_trace(go.Scatter(
    #     x=avg_sizes,
    #     y=overall_avg_values,
    #     mode='markers+lines',
    #     name='Overall Average',
    #     marker=dict(color='red', size=20),
    #     line=dict(width=6)
    # ))

    # Add the first 5 average trace
    fig.add_trace(go.Scatter(
        x=first_5_sizes,
        y=first_5_avg_values,
        mode='markers+lines',
        name='Average',
        marker=dict(color='blue', size=20),
        line=dict(dash='dash', width=6)
    ))

    # Update layout
    fig.update_layout(
        title='FID Value vs. Data Size',
        xaxis_title='Data Size',
        yaxis_title='FID Value',
        template='plotly_white',
        width=2400,
        height=1100,
        title_font=dict(size=72),
        xaxis_title_font=dict(size=72),
        yaxis_title_font=dict(size=72),
        xaxis_tickfont=dict(size=72),
        yaxis_tickfont=dict(size=72),
        legend=dict(
            font=dict(size=50),
            bgcolor="rgba(255,255,255,0.5)",
        )
    )

    fig.write_image('/home/workspace/3DShape2VecSet/plot/fid_combined_graph_multi_single_couch.png')
    fig.show()


def main():
    parser = argparse.ArgumentParser(description='Plot FID values for multiple folders')
    parser.add_argument('--root', type=str, default="/home/workspace/3DShape2VecSet/FID", help='Root folder containing subfolders with KID files')
    parser.add_argument('--folders', type=str, nargs='+', default=['Jet', 'Arm', 'Couch', 'Desk', 'Boat', 'FID_Couch_Single'], help='List of subfolders to process')

    args = parser.parse_args()

    plot_fid_values(args.root, args.folders)


if __name__ == '__main__':
    main()
