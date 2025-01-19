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

def compute_silhouette_score(embeddings, labels):
    # Flatten the embeddings to 2D: (B * N, D) -> Flatten (512, 64) to a 1D vector of size 32768
    embeddings_flattened = embeddings.view(embeddings.size(0), -1)  # Flatten the second and third dimensions
    
    embeddings_np = embeddings_flattened.detach().cpu().numpy()  # Detach, move to CPU, and convert to NumPy
    labels_np = labels.cpu().numpy()  # Move labels to CPU and convert to NumPy
    
    # Calculate silhouette score
    score = silhouette_score(embeddings_np, labels_np)
    
    return score
def main(args):
    device = torch.device('cuda:1')

    # Load the autoencoder model
    ae = models_ae.kl_d512_m512_l64()
    ae.eval()
    ae.load_state_dict(torch.load('/home/workspace/3DShape2VecSet/outputImageConditioned/ae/kl_d512_m512_l64/600-3_he_3/checkpoint-199.pth')['model'])
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

    # Stack all embeddings and labels
    embeddings_tensor = torch.stack(embeddings_list)
    labels_tensor = torch.stack(labels_list)

    # Compute Silhouette Score
    silhouette = compute_silhouette_score(embeddings_tensor, labels_tensor)

    # Save the Silhouette Score to a TXT file
    # with open(args.output_txt, 'w') as txt_file:
    #     txt_file.write(f'Silhouette Score: {silhouette:.4f}\n')

    print(f"Silhouette Score: {silhouette}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
