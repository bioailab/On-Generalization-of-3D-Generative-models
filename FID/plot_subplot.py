import os
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def read_fid_values(directory):
    fid_values = {}
    
    for filename in os.listdir(directory):
        if filename.startswith('kid_') and filename.endswith('.txt'):
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
    fig = make_subplots(rows=2, cols=3, subplot_titles=category_names + ["Average"])  # Set titles to category names + "Average"
    
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
        
        # Determine subplot position
        row = (i // 3) + 1  # 1st or 2nd row
        col = (i % 3) + 1   # 1st, 2nd, or 3rd column

        # Add a subplot for this folder with the correct title
        fig.add_trace(go.Scatter(x=sizes, y=avg_values, mode='markers+lines', name=name), row=row, col=col)
    
    # Calculate the overall average
    avg_sizes = sorted(all_fid_values.keys())
    overall_avg_values = [np.mean(all_fid_values[size]) for size in avg_sizes]
    
    # Add the average subplot in the last position (2nd row, 3rd column)
    fig.add_trace(go.Scatter(x=avg_sizes, y=overall_avg_values, mode='markers+lines', name='Average', marker=dict(color='red')), row=2, col=3)
    
    fig.update_layout(
        title='KID Value vs. Data Size Across Classes',
        xaxis_title='Data Size',
        yaxis_title='KID Value',
        template='plotly_white',
        width=1500,  # Adjusted width
        height=800   # Adjusted height
    )
    
    fig.write_image('/home/workspace/3DShape2VecSet/plot/kid_subplots.png')
    fig.show()

def main():
    parser = argparse.ArgumentParser(description='Plot FID values for multiple folders')
    parser.add_argument('--root', type=str, default="/home/workspace/3DShape2VecSet/FID", help='Root folder containing subfolders with KID files')
    parser.add_argument('--folders', type=str, nargs='+', default=['Jet', 'Arm', 'Couch','Desk', 'Boat'], help='List of subfolders to process')

    args = parser.parse_args()

    plot_fid_values(args.root, args.folders)

if __name__ == '__main__':
    main()
