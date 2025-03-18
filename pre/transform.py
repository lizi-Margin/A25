import torchvision.transforms as transforms
import torch

class FillNaNWithZero:
    def __call__(self, tensor):
        tensor[torch.isnan(tensor)] = 0
        return tensor

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # FillNaNWithZero()
])