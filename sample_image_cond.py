import argparse
import math

import numpy as np

import mcubes
import os 
import torch
import ipdb
import trimesh

import models_image_cond, models_ae

from pathlib import Path
from PIL import Image

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae', type=str, required=True) # 'kl_d512_m512_l16'
    parser.add_argument('--ae-pth', type=str, required=True) # 'output/ae/kl_d512_m512_l16/checkpoint-199.pth'
    parser.add_argument('--dm', type=str, required=True) # 'kl_d512_m512_l16_edm'
    parser.add_argument('--dm-pth', type=str, required=True) # 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    args = parser.parse_args()
    print(args)

    # Path("Image_cond_obj2/{}".format(args.dm)).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')

    ae = models_ae.__dict__[args.ae]()
    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth)['model'])
    ae.to(device)

    model = models_image_cond.__dict__[args.dm]()
    model.eval()

    model.load_state_dict(torch.load(args.dm_pth)['model'])
    model.to(device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    total = 1
    iters = 100
    list = ['/home/workspace/3DShape2VecSet/sofa.png']
    # list = ['/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02828884/1c79aa69e4ec26b65dc236dd32108e81/rendering/02.png']
    # list = ['/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02958343/1a1dcd236a1e6133860800e6696b8284/rendering/01.png']
    # list = ['/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02933112/1a46011ef7d2230785b479b317175b55/rendering/02.png']
    # /home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/04401088/1b41282fb44f9bb28f6823689e03ea4/rendering/01.png
# /home/arushika01/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/04530566/1b3a8fa303445f3e4ff4a2772e8deea/rendering/03.png
#Unseen
# /home/arushika01/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02828884/1c79aa69e4ec26b65dc236dd32108e81/rendering/02.png
# /home/arushika01/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02933112/ea48a2a501942eedde650492e45fb14f/rendering/01.png
    with torch.no_grad():
        # for category_id in [30]:
        #     print(category_id)
            for i in range(100//iters):
                sampled_array = model.sample(cond=list*iters,device=device, batch_seeds=torch.arange(i*iters, (i+1)*iters).to(device)).float()
                print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())
                # torch.save(sampled_array, "/home/workspace/3DShape2VecSet/Diffusion_step/embedding_final.pt".format(i))
                
                for j in range(sampled_array.shape[0]):
                    
                    logits = ae.decode(sampled_array[j:j+1], grid)
                    # ipdb.set_trace()
                    logits = logits.detach()
                    
                    volume = logits.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                    verts, faces = mcubes.marching_cubes(volume, 0)

                    verts *= gap
                    verts -= 1

                    m = trimesh.Trimesh(verts, faces)
                    path = os.path.join('Generated/{}/kl1/{:05d}.obj'.format(args.dm, i*iters+j))
                    print(path)
                    m.export(path)