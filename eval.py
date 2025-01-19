import csv
from tqdm import tqdm
from pathlib import Path
import util.misc as misc
from util.shapenet import ShapeNet, category_ids
import models_ae
import models_image_cond
import mcubes
import trimesh
from scipy.spatial import cKDTree as KDTree
import numpy as np
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
import yaml
import math
import ipdb
import argparse
from get_logit import get_logits

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='kl_d512_m512_l64', type=str,
                    metavar='MODEL', help='Name of model to train')
parser.add_argument(
    '--pth', default='/home/workspace/3DShape2VecSet/outputClassConditioned/ae/kl_d512_m512_l64/800-homo/checkpoint-199.pth', type=str)

parser.add_argument('--model_dm', default='kl_d512_m512_l64_d24_edm', type=str,
                    metavar='MODEL', help='Name of model to train')
parser.add_argument(
    '--pth_dm', default='/home/workspace/3DShape2VecSet/outputImageConditioned/dm/kl_d512_m512_l64_d24_edm/800-homo/checkpoint.pth', type=str)

parser.add_argument('--device', default='cuda:0',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_path', default='/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses', type=str,
                    help='dataset path')
args = parser.parse_args()

def main():
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = models_image_cond.__dict__[args.model_dm]()
    device = torch.device(args.device)

    model.eval()
    model.load_state_dict(torch.load(args.pth_dm, map_location='cpu')['model'], strict=True)
    model.to(device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()

    total_cd = 0
    total_fscore = 0
    num_samples = 0

    cds = []
    fscores = []

    with open('metrics_arm_multi.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SampleName', 'ChamferDistance', 'FScore'])

        with torch.no_grad():
            for category, _ in list(category_ids.items()):
                dataset_test = ShapeNet(args.data_path, 100, split='test', categories=[category],
                                        transform=None, sampling=False, return_surface=True, surface_sampling=False)
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test, sampler=sampler_test,
                    batch_size=10,
                    num_workers=12,
                    drop_last=False,
                )

                for batch in tqdm(data_loader_test, desc=f"Processing category: {category}"):
                    points, labels, surface, _, sample_names = batch
                    points = points.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    output = get_logits(args.model, args.pth, args.model_dm, args.pth_dm, grid)
                    volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()

                    verts, faces = mcubes.marching_cubes(volume, 0)
                    verts *= gap
                    verts -= 1.
                    m = trimesh.Trimesh(verts, faces)

                    for i, name in enumerate(sample_names):
                        if m.faces is not None and len(m.faces) > 0:
                            pred = m.sample(100000)
                        else:
                            print("Skipping sampling due to empty or invalid mesh.")
                            pred = None  # or handle this case as needed
                            continue
                        # pred = m.sample(50000)

                        tree = KDTree(pred)
                        dist, _ = tree.query(surface[i].cpu().numpy())
                        d1 = dist
                        gt_to_gen_chamfer = np.mean(dist)

                        tree = KDTree(surface[i].cpu().numpy())
                        dist, _ = tree.query(pred)
                        d2 = dist
                        gen_to_gt_chamfer = np.mean(dist)

                        cd = gt_to_gen_chamfer + gen_to_gt_chamfer

                        th = 0.02
                        if len(d1) and len(d2):
                            recall = float(sum(d < th for d in d2)) / float(len(d2))
                            precision = float(sum(d < th for d in d1)) / float(len(d1))

                            if recall + precision > 0:
                                fscore = 2 * recall * precision / (recall + precision)
                            else:
                                fscore = 0
                        else:
                            fscore = 0

                        # Parse sample name
                        # parsed_name = name.split('/')[-2]

                        # Write to CSV
                        writer.writerow([name, cd, fscore])

                        # Update totals
                        total_cd += cd
                        total_fscore += fscore
                        num_samples += 1

                        # Collect values for variance calculation
                        cds.append(cd)
                        fscores.append(fscore)

    # Compute and print averages and variances
    avg_cd = total_cd / num_samples if num_samples > 0 else 0
    avg_fscore = total_fscore / num_samples if num_samples > 0 else 0

    var_cd = np.var(cds) if cds else 0
    var_fscore = np.var(fscores) if fscores else 0

    print(f"Average: {avg_cd:.6f}, {avg_fscore:.6f}")
    print(f"Variance: {var_cd:.6f}, {var_fscore:.6f}")
    # print(f"Average F-Score: {avg_fscore:.6f}")
    # print(f"Variance F-Score: {var_fscore:.6f}")

if __name__ == '__main__':
    main()