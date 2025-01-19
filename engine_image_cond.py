# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import ipdb
import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched

import torchvision.transforms as transforms
from PIL import Image
def train_one_epoch(model: torch.nn.Module, ae: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # ipdb.set_trace()
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        points, labels, surface, categories,filename = data
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # print(type(categories),type(filename))
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        surface = surface.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)
        # filename = torch.tensor(filename)
        # filename = filename.to(device, non_blocking=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # print(len(filename))
        # print(filename)
        # print(type(filename))
        # print(type(categories))
        # print(categories.shape)
        image_tensors = []

        # Load and transform each image
        for path in filename:
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(device, non_blocking=True)
            image_tensors.append(image_tensor)

        image_tensors = torch.cat(image_tensors, dim=0)
        # print('*'*100)
        # print(image_tensor.shape)
        # print('*'*100)

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():

                _,x = ae.encode(surface)
                # print("***********************************************************************")
                # print("***********************************************************************")
                # print("***********************************************************************")
                # print("***********************************************************************")
                # print(x.shape)
                # print("***********************************************************************")
                # print("***********************************************************************")
                # print("***********************************************************************")
                # print("***********************************************************************")


            loss = criterion(model, x, image_tensors)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, ae, criterion, device):


    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for data in metric_logger.log_every(data_loader, 50, header):
        points, labels, surface, categories, filename = data
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        surface = surface.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)
        # filename = filename.to(device, non_blocking=True)
        # compute output

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_tensors = []

        # Load and transform each image
        for path in filename:
            image = Image.open(path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(device, non_blocking=True)
            image_tensors.append(image_tensor)

        image_tensors = torch.cat(image_tensors, dim=0)
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():

                _, x = ae.encode(surface)

            loss = criterion(model, x, image_tensors)
            
        batch_size = surface.shape[0]
        
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
