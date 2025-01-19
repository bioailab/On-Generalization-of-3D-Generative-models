import os
import glob
import random

import yaml 

import torch
from torch.utils import data

import numpy as np

from PIL import Image

import h5py

#02871439, 03261776, 02747177, 04074963, 03207941, 03211117, 04401088, 02933112, 02808440, 02924116
category_ids = {
    '02691156': 0,
    # '02747177': 1,
    # '02773838': 2,
    # '02801938': 3,
    # '02808440': 4,
    # '02818832': 5,
    # '02828884': 6, --
    # '02843684': 7,
    # '02871439': 8,
    # '02876657': 9, 
    # '02880940': 10,
    # '02924116': 11,
    # '02933112': 12, --
    # '02942699': 13,
    # '02946921': 14,
    # '02954340': 15,
    # '02958343': 16, --
    # '02992529': 17,
    '03001627': 18,
    # '03046257': 19,
    # '03085013': 20,
    # '03207941': 21,
    # '03211117': 22, --
    # '03261776': 23,
    # '03325088': 24,
    # '03337140': 25,
    # '03467517': 26,
    # '03513137': 27,
    # '03593526': 28,
    # '03624134': 29,
    # '03636649': 30,
    # '03642806': 31,
    # '03691459': 32,
    # '03710193': 33,
    # '03759954': 34,
    # '03761084': 35,
    # '03790512': 36,
    # '03797390': 37,
    # '03928116': 38,
    # '03938244': 39,
    # '03948459': 40,
    # '03991062': 41,
    # '04004475': 42,
    # '04074963': 43,
    # '04090263': 44, --
    # '04099429': 45,
    # '04225987': 46,
    '04256520': 47,
    # '04330267': 48,
    '04379243': 49,
    # '04401088': 50, --
    # '04460130': 51,
    # '04468005': 52,
    '04530566': 53,
    # '04554684': 54,
}

class ShapeNet(data.Dataset):
    # def __init__(self, dataset_folder, split, categories=['03001627'], transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=16):
    # def __init__(self, dataset_folder, split, categories=['03001627','04379243','02958343','02691156','04256520','03691459','04530566','04090263','03636649','02828884','02871439', '03261776', '02747177', '04074963', '03207941', '03211117', '04401088', '02933112', '02808440', '02924116'], transform=None, sampling=True, num_samples=4096, return_surface=True, 
    # 
    # =True, pc_size=2048, replica=10,max_examples_per_category=400,extension='png'):
    # def __init__(self, dataset_folder,max_examples_per_category, split, categories=['02691156','03001627','04256520','04379243','04530566','02828884','04090263','02933112','03211117','02958343','04401088'], transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=5,extension='png'):
    def __init__(self, dataset_folder,max_examples_per_category, split, categories=['02691156','03001627','04256520','04379243','04530566'], transform=None, sampling=True, num_samples=4096, return_surface=True, surface_sampling=True, pc_size=2048, replica=5,extension='png'):

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split
        self.extension  = extension

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.dataset_folder = dataset_folder
        self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')
        self.image_folder = "/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering"

        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()

        print(categories)
        self.max_examples_per_category = max_examples_per_category 
        
        self.models = []
        for c_idx, c in enumerate(categories):

            subpath = os.path.join(self.point_folder, c)
            # assert os.path.isdir(subpath)
            split_file = os.path.join(subpath, split + '.lst')
            
            with open(split_file, 'r') as f:
                if self.max_examples_per_category == -1:
                    models_c = f.read().split('\n')
                else:
                    models_c = f.read().split('\n')[:self.max_examples_per_category] 

                # models_c = f.read().split('\n')
            # print("*"*100)
            # print(c)
            # print(len(models_c))
            # print("*"*100)
            if self.max_examples_per_category == 1:
                self.models = [
                    {'category': c,'model':'185bcb9bcec174c9492d9da2668ec34c'}
                ]
            else:    
                self.models += [
                    {'category': c, 'model': m.replace('.npz', '')}
                    for m in models_c
                ]
        # print(len(models_c))

        self.replica = replica

    def save_mesh_as_obj(self, vertices, faces, filename):
        with open(filename, 'w') as f:
            for point, normal in zip(vertices, faces):
                f.write(f"v {' '.join(map(str, point))}\n")
                f.write(f"vn {' '.join(map(str, normal))}\n")

    def extract_and_save_mesh(self,npz_path, output_obj_path):
        with np.load(npz_path) as data:
            # print("Keys in the npz file:", data.files)
            vertices = data['points']  # Assuming 'verts' contains vertex positions
            faces = data['normals']    # Assuming 'faces' contains face indices
        
        self.save_mesh_as_obj(vertices, faces, output_obj_path)

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        category = self.models[idx]['category']
        model = self.models[idx]['model']
        point_path = os.path.join(self.point_folder, category, model+'.npz')
        # pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model+'.npz')
        pc_path = os.path.join(self.mesh_folder, category, model+'.npz')

        # Image_path = os.path.join(self.image_folder, category, model,'rendering')
        # files = glob.glob(os.path.join(Image_path, '*.%s' % self.extension))
        # files.sort()
        
        # idx_img = random.randint(0, len(files)-1)
        # filename = files[idx_img]
        # # print(filename)
        # image = Image.open(filename).convert('RGB')
        # if self.transform is not None:
        #     image = self.transform(image)

        try:
            with np.load(point_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
        except Exception as e:
            print(e)

        with open(point_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        if self.return_surface:
            # pc_path = os.path.join(self.mesh_folder, category, '4_pointcloud', model+'.npz')
            pc_path = os.path.join(self.mesh_folder, category, model+'.npz')

            with np.load(pc_path) as data:
                surface = data['points'].astype(np.float32)
                surface = surface * scale
            if self.surface_sampling:
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        # image = torch.from_numpy(image).float()
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)
        # print(filename)
        if self.return_surface:
            # obj_folder = os.path.join("/home/workspace/3DShape2VecSet/Dataset/ShapeNetV2_Mesh",str(self.max_examples_per_category), category,model+'.obj')
            # os.makedirs(os.path.dirname(obj_folder), exist_ok=True)
            # self.extract_and_save_mesh(pc_path, obj_folder)
            # print(filename)
            return points, labels, surface, category_ids[category],points
        else:
            return points, labels, category_ids[category],points

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica