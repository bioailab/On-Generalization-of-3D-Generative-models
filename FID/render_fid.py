import argparse
import os

from render_utils import Render, create_pose, scale_to_unit_sphere

import numpy as np
from tqdm import tqdm
import torch
import torchvision
import trimesh

def get_selected_meshes(data_dir, num_meshes=60):
    # Read the train.lst file
    lst_file_path = os.path.join('/home/workspace/Dataset/Homogenous_subclasses/ShapeNetV2_point/02691156', 'test.lst')
    with open(lst_file_path, 'r') as f:
        mesh_filenames = f.readlines()
    
    # Remove any leading/trailing whitespace characters
    mesh_filenames = [filename.strip() for filename in mesh_filenames]

    # Replace .npz extension with .off
    mesh_filenames = [filename.replace('.npz', '.off') for filename in mesh_filenames]

    # Limit the number of meshes to the specified number
    mesh_filenames = mesh_filenames[:num_meshes]

    # Create full paths for the mesh files
    mesh_files = [
        os.path.join(data_dir, filename)
        for filename in mesh_filenames
        if filename.endswith("off")
    ]
    
    if not mesh_files:
        raise ValueError("No valid mesh files found in the specified directory and train.lst")

    # Load the mesh files
    meshes = [
        trimesh.load_mesh(mesh_file)
        for mesh_file in mesh_files
    ]

    return meshes

def get_meshes(data_dir):
    # List all mesh files in the directory with the specified extensions
    mesh_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith("obj") or f.endswith("off")
    ]
    
    # Determine the file suffix from the first mesh file
    # print(mesh_files[0])
    file_suffix = mesh_files[0].split(".")[-1]

    # Load meshes, limiting to the number specified
    meshes = [
        trimesh.load_mesh(mesh_file, file_type=file_suffix)
        for mesh_file in mesh_files
    ]
    
    return meshes


FrontVector = (np.array(
    [[0.52573, 0.38197, 0.85065], [-0.20081, 0.61803, 0.85065],
     [-0.64984, 0.00000, 0.85065], [-0.20081, -0.61803, 0.85065],
     [0.52573, -0.38197, 0.85065], [0.85065, -0.61803, 0.20081],
     [1.0515, 0.00000, -0.20081], [0.85065, 0.61803, 0.20081],
     [0.32492, 1.00000, -0.20081], [-0.32492, 1.00000, 0.20081],
     [-0.85065, 0.61803, -0.20081], [-1.0515, 0.00000, 0.20081],
     [-0.85065, -0.61803, -0.20081], [-0.32492, -1.00000, 0.20081],
     [0.32492, -1.00000, -0.20081], [0.64984, 0.00000, -0.85065],
     [0.20081, 0.61803, -0.85065], [-0.52573, 0.38197, -0.85065],
     [-0.52573, -0.38197, -0.85065], [0.20081, -0.61803, -0.85065]])) * 2


def render_mesh(mesh,
                resolution=1024,
                index=5,
                background=None,
                scale=1,
                no_fix_normal=True):

    camera_pose = create_pose(FrontVector[index] * scale)

    render = Render(size=resolution,
                    camera_pose=camera_pose,
                    background=background)

    triangle_id, rendered_image, normal_map, depth_image, p_images = render.render(
        path=None, clean=True, mesh=mesh, only_render_images=no_fix_normal)
    return rendered_image


def render_for_fid(mesh, root_dir, mesh_idx):
    render_resolution = 299
    
    # try:
    mesh = scale_to_unit_sphere(mesh)
    # except Exception as e:
    #     print(f"Error scaling mesh to unit sphere: {e}")
    #     # Skip rendering if scaling fails
    #     return
    
    for j in range(20):
        try:
            image = render_mesh(mesh, index=j, resolution=render_resolution) / 255
            torchvision.utils.save_image(
                torch.from_numpy(image.copy()).permute(2, 0, 1),
                f"{root_dir}/view_{j}/{mesh_idx}.png"
            )
        except Exception as e:
            print(f"Error rendering or saving image for mesh index {mesh_idx} at view {j}: {e}")
            # Continue to the next view in case of an error
            continue

# def main(args):
#     print("HEY1")
#     gen_meshes = get_meshes(args.gen_dir)
#     print("HEY2")

#     # gt_meshes = get_selected_meshes(args.gt_dir)
#     # print("HEY3")

#     for i in range(20):
#         os.makedirs(os.path.join(args.gt_out_dir, f"view_{i}"), exist_ok=True)
#         os.makedirs(os.path.join(args.gen_out_dir, f"view_{i}"), exist_ok=True)
#     print(len(gen_meshes))
#     for mesh_idx, gen_mesh in tqdm(enumerate(gen_meshes),
#                                    total=len(gen_meshes)):
#         print(gen_mesh)
#         render_for_fid(gen_mesh, args.gen_out_dir, mesh_idx)

#     # for mesh_idx, gt_mesh in tqdm(enumerate(gt_meshes), total=len(gt_meshes)):
#     #     render_for_fid(gt_mesh, args.gt_out_dir, mesh_idx)
#     print("HEY1")
    
def main(args):
    print("Processing multiple gen folders...")
    gen_dirs = [os.path.join(args.gen_dir, d) for d in os.listdir(args.gen_dir) if os.path.isdir(os.path.join(args.gen_dir, d))]
    
    for gen_dir in gen_dirs:
        # Take the name of the current gen-folder
        gen_folder_name = os.path.basename(gen_dir)
        
        # Adjust gen-out-dir for the current folder
        gen_out_dir = os.path.join(args.gen_out_dir, gen_folder_name)
        os.makedirs(gen_out_dir, exist_ok=True)
        print(f"Processing {gen_dir} and saving to {gen_out_dir}...")

        gen_meshes = get_meshes(gen_dir)[:60]
        
        for i in range(20):
            os.makedirs(os.path.join(gen_out_dir, f"view_{i}"), exist_ok=True)
        
        for mesh_idx, gen_mesh in tqdm(enumerate(gen_meshes), total=len(gen_meshes)):
            render_for_fid(gen_mesh, gen_out_dir, mesh_idx)

    print("Processing complete.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=str, default="/home/workspace/Generated/kl_d512_m512_l64_d24_edm/1000-homo")
    parser.add_argument("--gt-dir", type=str, default="/home/workspace/Dataset/ShapeNetV2_watertight_scaled_off/02691156/4_watertight_scaled")
    parser.add_argument("--gt-out-dir", type=str, default="/home/workspace/FID_eval_data/jet_gt")
    parser.add_argument("--gen-out-dir", type=str, default="/home/workspace/FID_eval_data/1000-homo")
    
    args = parser.parse_args()

    main(args)
