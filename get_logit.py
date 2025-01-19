import argparse
import math

import numpy as np

import mcubes
import os 
import torch

import trimesh
import ipdb
import models_image_cond, models_ae

from pathlib import Path
from PIL import Image

def get_logits(ae,ae_path,dm,dm_path,grid):
    torch.cuda.empty_cache()

    # Path("Image_cond_obj2/{}".format(args.dm)).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')

    ae = models_ae.__dict__[ae]()
    ae.eval()
    ae.load_state_dict(torch.load(ae_path)['model'])
    ae.to(device)

    model = models_image_cond.__dict__[dm]()
    model.eval()

    model.load_state_dict(torch.load(dm_path)['model'])
    model.to(device)

    total = 1
    iters = 1
    sample_list = ['']
    # list = ['/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02828884/1c79aa69e4ec26b65dc236dd32108e81/rendering/02.png']
    # list = ['/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02691156/12f4778ebba781236b7bd17e458d0dcb/rendering/01.png']
    # list = ['/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/03001627/185bcb9bcec174c9492d9da2668ec34c/rendering/02.png']
    
    # list = ['/home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02933112/1a46011ef7d2230785b479b317175b55/rendering/02.png']
    # /home/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/04401088/1b41282fb44f9bb28f6823689e03ea4/rendering/01.png
# /home/arushika01/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/04530566/1b3a8fa303445f3e4ff4a2772e8deea/rendering/03.png
#Unseen
# /home/arushika01/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02828884/1c79aa69e4ec26b65dc236dd32108e81/rendering/02.png
# /home/arushika01/workspace/3DShape2VecSet/Dataset/Homogenous_subclasses/ShapeNetRendering/02933112/ea48a2a501942eedde650492e45fb14f/rendering/01.png
    # logits 
    with torch.no_grad():
        # for category_id in [30]:
        #     print(category_id)
        # for j in list:
            for i in range(1//iters):
                # sampled_array = model.sample(cond=list*iters,device=device, batch_seeds=torch.arange(i*iters, (i+1)*iters).to(device)).float()
                # ipdb.set_trace()
                sampled_array = model.sample(cond=sample_list*iters,device=device, batch_seeds=torch.arange(i*iters, len(sample_list)*(i+1)*iters).to(device)).float()
                print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(), sampled_array.std())

                    
                logits = ae.decode(sampled_array[1:2], grid)
                # ipdb.set_trace()
                logits = logits.detach()
            
            return logits