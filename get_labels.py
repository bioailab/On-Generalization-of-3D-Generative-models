import torch
from torch_cluster import fps
import torch.nn as nn
import models_ae
from models_ae import Attention,PreNorm,FeedForward,PointEmbed
from models_ae import DiagonalGaussianDistribution
from util.datasets import build_shape_surface_occupancy_dataset
import argparse
import os
import numpy as np
import ipdb
import torch
import os
import csv
import argparse
from torch.utils.data import DataLoader, SequentialSampler
from util.datasets import build_shape_surface_occupancy_dataset
import models_ae
from collections import Counter
def encode(pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == 2048
        
        ###### fps
        flattened = pc.view(B*N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * 512 / 2048

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = PointEmbed(dim=512)(sampled_pc)

        pc_embeddings = PointEmbed(dim=512)(pc)
        dim =512
        cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])
        cross_attn, cross_ff = cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        mean = nn.Linear(dim, 64)(x)
        logvar = nn.Linear(dim, 64)(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x

label_to_class = {
    "03001627": "Chair",
    "04379243": "Table",
    "02691156": "Airplane",
    "02958343": "Car",
    "04090263": "Faucet",
    # Add other mappings as needed
}

def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--data_path', default='/home/workspace/3DShape2VecSet/Dataset', type=str, help='Dataset path')
    parser.add_argument('--point_cloud_size', default=2048, type=int, help='Point cloud size')
    parser.add_argument('--max_train_size', default=600, type=int, help='Max training size')
    parser.add_argument('--target_dir', default="/home/workspace/3DShape2VecSet/Diffusion_step", type=str, help='Directory containing target .pt files')
    parser.add_argument('--output_csv', default='output_labels.csv', type=str, help='Output CSV filename')
    return parser


def main(args):
    device = torch.device('cuda:1')

    # Load the autoencoder model
    ae = models_ae.kl_d512_m512_l64()
    ae.eval()
    ae.load_state_dict(torch.load('/home/workspace/3DShape2VecSet/outputImageConditioned/ae/kl_d512_m512_l64/600-3_kl1e_2/checkpoint-199.pth')['model'])
    ae.to(device)

    # Build dataset
    dataset_train = build_shape_surface_occupancy_dataset('train', args=args)

    # Load all target .pt files from the provided directory
    target_files = [os.path.join(args.target_dir, f) for f in os.listdir(args.target_dir) if f.endswith('.pt')]
    target_arrays = [(f, (torch.load(f))[22:23].to(device)) for f in target_files]

    sampler_test = SequentialSampler(dataset_train)
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_test,
        batch_size=1,
        num_workers=12,
        drop_last=False,
    )

    results = []
    for target_path, target_array in target_arrays:
        distances = []
        for data in data_loader_train:
            points, labels, surface, category_id, model = data
            points = points.to(device, non_blocking=True)
            surface = surface.to(device, non_blocking=True)

            kl, x = ae.encode(surface)

            if x.shape != target_array.shape:
                print(f"Skipping: Shape mismatch for {model}")
                continue

            # Compute the distance (e.g., Euclidean distance)
            distance = torch.dist(x, target_array).item()
            distances.append((distance, model, x, points))

        # Sort distances and get the top 5
        top_k_files = sorted(distances, key=lambda x: x[0])[:5]

        # Extract labels and apply max voting
        # ipdb.set_trace()
        labels = [file_path[0].split('/')[7] for _, file_path, _, _ in top_k_files]
        files = [file_path for _, file_path, _, _ in top_k_files]
        # for _, file_path, _, _ in top_k_files:
    
        label_counts = Counter(labels)
        # Determine the final label
        # Determine the final label, handle ties by storing both tied labels
        most_common = label_counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # If there's a tie, store both labels
            final_label = [label_to_class.get(most_common[0][0], most_common[0][0]),  # Map to class name
                        label_to_class.get(most_common[1][0], most_common[1][0])]  # Map to class name
        else:
            # If no tie, store the most common label
            final_label = [label_to_class.get(most_common[0][0], most_common[0][0])]
        target = os.path.basename(target_path).split('_')[1].split('.')[0]
        print(target, ", ",final_label)
        # Add the final prediction and top-5 to results
        results.append({
            'Steps': target,
            'Top_5_Labels': [label_to_class.get(lbl, lbl) for lbl in labels],  # Map top-5 labels to class names
            'Filenames':files,
            'Distances': [distance for distance, _, _, _ in top_k_files]
        })

    # Write results to a CSV file
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Steps', 'Top_5_Labels','Filenames','Distances']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results with max voting saved to {args.output_csv}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)