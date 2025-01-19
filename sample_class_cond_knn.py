import argparse
import math

import numpy as np

import mcubes

import torch
import csv
import trimesh
import glob
import os
from KNNsearch import KNNSearch
import models_class_cond, models_ae
import matplotlib.pyplot as plt 
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from util.misc import NativeScalerWithGradNormCount as NativeScaler

if __name__ == "__main__":

    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae', type=str, required=True) # 'kl_d512_m512_l16'
    parser.add_argument('--ae-pth', type=str, required=True) # 'output/ae/kl_d512_m512_l16/checkpoint-199.pth'
    parser.add_argument('--dm', type=str, required=True) # 'kl_d512_m512_l16_edm'
    parser.add_argument('--dm-pth', type=str, required=True) # 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    # parser.add_argument('--img-pth', type=str, required=True) 
    args = parser.parse_args()
    print(args)

    Path("class_cond_obj/{}".format(args.dm)).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0')

    ae = models_ae.__dict__[args.ae]()
    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth)['model'])
    ae.to(device)

    model = models_class_cond.__dict__[args.dm]()
    model.eval()

    model.load_state_dict(torch.load(args.dm_pth)['model'])
    model.to(device)
    
    criterion = models_class_cond.__dict__['EDMLoss']()
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    
    # # Load the image
    # image = Image.open(args.img_pth).convert("RGB")
    
    # # Apply the transform and add batch dimension
    # image = transform(image).unsqueeze(0)

    # model2 = models.resnet18(pretrained=True)
    
    # # Modify the final layer to output 512 features
    # num_features = model2.fc.in_features
    # model2.fc = torch.nn.Linear(num_features, 100)

    # model2.eval()
    
    # # Disable gradient calculation for inference
    # with torch.no_grad():
    #     # Pass the image through the model
    #     output = model2(image)
    # outputs = output.squeeze()
    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    iters = 100
    loss_scaler = NativeScaler()
    losses = []
    filename = 'losses.csv'
    
    with torch.no_grad():
        for category_id in [30]:
            print(category_id)
            for i in range(300//iters):

                # print("\n")
                # print("\n")
                # print("\n")
                # print("*****************************************************************")
                # print((torch.Tensor(outputs).long().to(device)).shape)
                # print("*****************************************************************")
                # print("\n")
                # print("\n")
                # print("\n")

                sampled_array = model.sample(cond=torch.Tensor([category_id]*iters).long().to(device), batch_seeds=torch.arange(i*iters, (i+1)*iters).to(device)).float()

                # print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())

                for j in range(sampled_array.shape[0]):
                    base_dir = '/home/workspace/3DShape2VecSet/class_cond_obj'
                    category_subdir = str(category_id)  # Ensure it's a string, necessary if category_id is numeric
                    filename = '{:02d}-{:05d}.png'.format(i, j)  # Assuming i and j are defined; added file extension

                    # Complete file path
                    n = os.path.join(base_dir, category_subdir, filename)

                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(n), exist_ok=True)
                    name = KNNSearch(sampled_array[j:j+1],n)
                    
                    category_id_tensor = torch.tensor([category_id], dtype=torch.int64)
                    category_id_tensor = category_id_tensor.to(device, non_blocking=True)
                    # if(i%10==0):
                    loss = criterion(model, sampled_array[j:j+1], category_id_tensor)
                    
                    loss_value = loss.item()
                    losses.append(loss_value)
                    # print(name)
                    # break
                # break
                    name2 = name[0]
                    print(name2)
                    npz_directory = '/home/workspace/3DShape2VecSet/Dataset/ShapeNetV2_watertight/03636649/4_pointcloud'
                    iteration_info = i * iters + j
                    # for model_name in name:
                    #     npz_path = os.path.join(npz_directory, f"{model_name}.npz")
                    #     if os.path.exists(npz_path):
                    #         print(f"Found matching NPZ file: {npz_path}")
                    #         # Load and process the NPZ file
                    #         data = np.load(npz_path, allow_pickle=True)
                    #         obj_path = f'class_cond_obj/knn/{category_id:02d}-{iteration_info:05d}-train.obj'
                    #         with open(obj_path, 'w') as obj_file:
                    #             # Writing vertices and normals to the OBJ file
                    #             for point in data['points']:
                    #                 obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")
                    #             for normal in data['normals']:
                    #                 obj_file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                    #         print(f"Exported OBJ file: {obj_path}")
                            # break
                    npz_path = os.path.join(npz_directory, f"{name2}.npz")
                    if os.path.exists(npz_path):
                        print(f"Found matching NPZ file: {npz_path}")
                        # Load and process the NPZ file
                        data = np.load(npz_path, allow_pickle=True)
                        obj_path = f'class_cond_obj/knn/{category_id:02d}-{iteration_info:05d}-train.obj'
                        with open(obj_path, 'w') as obj_file:
                            # Writing vertices and normals to the OBJ file
                            for point in data['points']:
                                obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")
                            for normal in data['normals']:
                                obj_file.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                        print(f"Exported OBJ file: {obj_path}")
                        
                    logits = ae.decode(sampled_array[j:j+1], grid)
                    # embedding = torch.from_numpy(embedding).float()  # Convert to tensor and ensure it's float type
                    
                    # embedding = embedding.cuda() 
                    # pc_path = os.path.join('/home/workspace/3DShape2VecSet/Dataset/ShapeNetV2_watertight/02843684/4_pointcloud', name2+'.npz')
                    # with np.load(pc_path) as data:
                    #     surface = data['points'].astype(np.float32)
                    #     surface = surface 
                
                    #     ind = np.random.default_rng().choice(surface.shape[0], 2048, replace=False)
                    #     surface = surface[ind]
                    # surface = torch.from_numpy(surface)
                    # closest_train =  ae.decode(embedding, grid)
                    logits = logits.detach()
                    # closest_train = closest_train.detach()
                    # volume2 = closest_train.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                    volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                    verts, faces = mcubes.marching_cubes(volume, 0)
                    # verts2,faces2 = mcubes.marching_cubes(volume2, 0)
                    verts *= gap
                    verts -= 1

                    # verts2 *= gap
                    # verts2 -= 1
                    m = trimesh.Trimesh(verts, faces)
                    # m2 = trimesh.Trimesh(verts2, faces2)

                    m.export('class_cond_obj/{}/{:02d}-{:05d}.obj'.format('knn', category_id, i*iters+j))
                    # m2.export('class_cond_obj/{}/{:02d}-{:05d}-train.obj'.format('knnex', category_id, i*iters+j))
                    # break
                # break
                    # m.export('class_cond_obj/{}/{:02d}-{:05d}.obj'.format(args.dm, i*iters+j))

    # # Writing to the CSV file
    # with open(filename, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # Optionally write a header
    #     writer.writerow(['Loss'])
    #     for loss in losses:
    #         writer.writerow([loss])
    # # losses_cpu = losses.cpu().numpy()  # Move to CPU and convert to NumPy
    # losses_cpu = [loss_item.cpu().numpy() for loss_item in losses]
    # plt.plot(losses_cpu, marker='o', linestyle='-', color='b') 
    # # plt.plot(losses, marker='o', linestyle='-', color='b')
    # plt.title('Losses Over Time')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.savefig('losses_plot.png')  # Save the plot as a PNG file
    # plt.show()  