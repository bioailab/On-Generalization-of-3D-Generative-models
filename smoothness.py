import torch
import numpy as np
from itertools import combinations
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import silhouette_score
from collections import Counter
import argparse
import os
import numpy as np
import models_ae
from util.datasets import build_shape_surface_occupancy_dataset
import ipdb
def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--data_path', default='/home/workspace/3DShape2VecSet/Dataset', type=str, help='Dataset path')
    parser.add_argument('--point_cloud_size', default=2048, type=int, help='Point cloud size')
    parser.add_argument('--max_train_size', default=600, type=int, help='Max training size')
    parser.add_argument('--output_txt', default='silhouette_score.txt', type=str, help='Output TXT filename for silhouette score')
    return parser
def compute_latent_curvature(embeddings_list, num_interpolations=10):
    """
    Computes curvature in latent space for interpolated points.

    Args:
        embeddings_list (list): List of latent embeddings.
        num_interpolations (int): Number of interpolation steps.

    Returns:
        float: Mean curvature of paths in latent space.
        list: Curvature values for individual pairs.
    """
    curvatures = []

    # Iterate over all pairs of embeddings
    for z1, z2 in combinations(embeddings_list, 2):
        z1 = z1.squeeze(0)
        z2 = z2.squeeze(0)
        
        # Generate interpolation points
        alphas = torch.linspace(0, 1, num_interpolations)
        path_points = [(1 - alpha) * z1 + alpha * z2 for alpha in alphas]
        path_points = torch.stack(path_points)

        # Compute first and second derivatives along the path
        first_derivative = torch.diff(path_points, dim=0)
        second_derivative = torch.diff(first_derivative, dim=0)

        # Compute curvature as the norm of the second derivative
        curvature = torch.norm(second_derivative, dim=1).mean().item()
        curvatures.append(curvature)

    # Return mean curvature and individual curvature values
    return np.mean(curvatures), curvatures
from sklearn.neighbors import NearestNeighbors

def compute_density_variation(embeddings_list, k=5):
    """
    Computes variation in local density of latent embeddings.

    Args:
        embeddings_list (list): List of latent embeddings.
        k (int): Number of neighbors for density estimation.

    Returns:
        float: Standard deviation of densities across embeddings.
        list: Densities for individual points.
    """
     # Detach embeddings and convert to NumPy
    embeddings = torch.stack(embeddings_list).detach().cpu().numpy()

    # Reshape embeddings if needed (e.g., flattening higher dimensions)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)  # For NumPy arrays

    # Fit k-nearest neighbors to find distances
    nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    # Compute density as inverse of mean distance to k neighbors
    densities = 1 / distances.mean(axis=1)

    # Return standard deviation of densities and individual density values
    return np.std(densities), densities
def main(args):
    device = torch.device('cuda:1')

    # Load the autoencoder model
    ae = models_ae.kl_d512_m512_l64()
    ae.eval()
    ae.load_state_dict(torch.load('/home/workspace/3DShape2VecSet/outputImageConditioned/ae/kl_d512_m512_l64/600-3/checkpoint-199.pth')['model'])
    ae.to(device)

    # Build dataset
    dataset_train = build_shape_surface_occupancy_dataset('train', args=args)

    sampler_test = SequentialSampler(dataset_train)
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_test,
        batch_size=1,
        num_workers=12,
        drop_last=False,
    )

    embeddings_list = []
    labels_list = []

    for data in data_loader_train:
        points, labels, surface, category_id, model = data
        surface = surface.to(device, non_blocking=True)

        # Encode the latent embeddings
        _, latent_embedding = ae.encode(surface)

        # Append embeddings and labels
        embeddings_list.append(latent_embedding.squeeze(0))  # Remove batch dimension
        labels_list.append(category_id.squeeze(0))  # Assume category_id is the label

    # # Stack all embeddings and labels
    # embeddings_tensor = torch.stack(embeddings_list)
    # embeddings_flattened = embeddings_tensor.view(embeddings_tensor.size(0), -1)  # Flatten the second and third dimensions
    
    # embeddings_np = embeddings_flattened.detach().cpu().numpy() 
    # Assuming `embeddings_list` is populated with latent embeddings
    mean_global_dist, global_distances = compute_latent_curvature(embeddings_list)
    std_density, density_variation = compute_density_variation(embeddings_list)

    # Output results
    print(f"Global Smoothness (Curvature): {mean_global_dist}")
    print(f"Local Smoothness (Density Variation): {std_density}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
