import os

def read_fid_values(directory):
    fid_values = {}
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.split('_')[1].split('.')[0]=='fid' and filename.endswith('.txt'):
            size = int(filename.split('_')[0].split('.')[0])
            filepath = os.path.join(directory, filename)
            
            # Read the FID value from the file
            with open(filepath, 'r') as file:
                line = file.readline().strip()
                fid_value = float(line.split(': ')[1])
                
            fid_values[size] = fid_value
            
    return fid_values

import plotly.graph_objects as go

def plot_fid_values(fid_values):
    sizes = sorted(fid_values.keys())
    values = [fid_values[size] for size in sizes]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=values, mode='markers+lines', name='FID Value', marker=dict(color='blue')))
    
    fig.update_layout(
        title='FID Value vs. Data Size for Jet Class(Seen class)',
        xaxis_title='Data Size',
        yaxis_title='FID Value',
        template='plotly_white',
        width=1200,  # Increase width
        height=800
    )
    
    fig.write_image('/home/workspace/3DShape2VecSet/plot/FID_Jet_air.png')  # Save the plot as a PNG file
    fig.show()


def main():
    data_dir = '/home/workspace/3DShape2VecSet/FID/Arm-jet'  # Replace with the path to your FID files
    fid_values = read_fid_values(data_dir)
    plot_fid_values(fid_values)

if __name__ == '__main__':
    main()
