from shapenet import ShapeNet
from torch.utils.data import DataLoader
import torch
class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point
transform = AxisScaling((0.75, 1.25), True)
shapenet_dataset = ShapeNet("/home/workspace/3DShape2VecSet/Dataset", split='train', transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=2048)
data_loader = DataLoader(shapenet_dataset, batch_size=1, shuffle=True)
for data in data_loader:
    # The __getitem__ method will be called for each item, saving the .obj files
    pass