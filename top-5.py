import os
import torch
import numpy as np

def find_top_k_files_across_categories(embeddings_dir, target_array, k=5):
    # Ensure target_array is a tensor
    target_array = torch.tensor(target_array)
    
    # List to store distances and file paths
    distances = []
    
    # Iterate over all categories (subdirectories) in the embeddings directory
    for category_id in os.listdir(embeddings_dir):
        category_path = os.path.join(embeddings_dir, category_id, '600')
        if not os.path.isdir(category_path):
            continue  # Skip if it's not a directory
        
        # Iterate over all .pt files in the category folder
        for filename in os.listdir(category_path):
            if filename.endswith('.pt'):  # Check for .pt files
                file_path = os.path.join(category_path, filename)
                
                # Load the .pt file
                file_data = torch.load(file_path)
                # print("File data shape:{}".format(file_data.shape))
                # file_data = file_data.to('cuda:0')
                # Ensure both arrays have the same shape
                if file_data.shape != target_array.shape:
                    print(f"Skipping {file_path}: Shape mismatch")
                    continue
                
                # Compute the distance (e.g., Euclidean distance)
                distance = torch.dist(file_data, target_array).item()
                
                # Append the distance and file path
                distances.append((distance, file_path))
    
    # Sort distances and get the top k files
    top_k_files = sorted(distances, key=lambda x: x[0])[:k]
    
    return top_k_files

def mask_except_row(array, row_index):
    """
    Masks all rows of a 2D PyTorch tensor except the specified row.

    Args:
        array (torch.Tensor): Input 2D tensor.
        row_index (int): Index of the row to keep.

    Returns:
        torch.Tensor: Masked tensor with all rows set to zero except the specified row.
    """
    # Ensure the input is a 2D tensor
    if not isinstance(array, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if len(array.shape) != 2:
        raise ValueError("Input tensor must be 2-dimensional.")
    
    # Create a zero tensor with the same shape as the input
    masked_array = torch.zeros_like(array)
    
    # Copy the specified row into the masked tensor
    masked_array[row_index:row_index+150] = array[row_index:row_index+150]
    
    return masked_array


# Example usage
import os
import numpy as np
import models_ae
import torch
import trimesh
import mcubes
# Set the directory where your 'embedding_3D' folder is located
# folder_path = '/home/workspace/3DShape2VecSet/Diffusion_step'
device = torch.device('cuda:0')

# Initialize an empty list to store the first row from each file
ae = models_ae.__dict__["kl_d512_m512_l64"]()
ae.eval()
ae.load_state_dict(torch.load("/home/workspace/3DShape2VecSet/outputImageConditioned/ae/kl_d512_m512_l64/600-3_kl1e_2/checkpoint-199.pth")['model'])
ae.to(device)
density = 128
gap = 2. / density
x = np.linspace(-1, 1, density+1)
y = np.linspace(-1, 1, density+1)
z = np.linspace(-1, 1, density+1)
xv, yv, zv = np.meshgrid(x, y, z)
grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

# embeddings_dir = "/home/workspace/3DShape2VecSet/Embedding3/600-3_kl1_torch"
array = torch.load('/home/workspace/3DShape2VecSet/Diffusion_step/embedding_17.pt')
target_array = array[22:23].float().squeeze(0) # Replace with your array
# target_array = array[13:14].float() # Replace with your array
# print("Target Array shape:{}".format(target_array.shape))
# top_k_files = find_top_k_files_across_categories(embeddings_dir, target_array, k=5)

# print("Top 5 closest files across all categories:")
# for rank, (distance, file_path) in enumerate(top_k_files, start=1):
#     print(f"{rank}. {file_path} (Distance: {distance})")
#     x = torch.load(file_path)
#     # x = x.to('cuda:0')

x = mask_except_row(target_array,110).unsqueeze(0)
logits = ae.decode(x, grid)
# ipdb.set_trace()
logits = logits.detach()

volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
verts, faces = mcubes.marching_cubes(volume, 0)

verts *= gap
verts -= 1
# print(filename)
m = trimesh.Trimesh(verts, faces)
# name = file_path.split('_')[1].split('.')[0]
# path = os.path.join('/home/workspace/3DShape2VecSet/top-5/{}.{}.obj'.format(rank,name))
path = os.path.join('/home/workspace/3DShape2VecSet/setoflatents/masked110-kle_2.obj')
print(path)
m.export(path)