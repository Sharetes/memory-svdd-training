import torch
import numpy as np
from torchvision.utils import make_grid
from PIL import Image


def plot_images_grid(x: torch.tensor,
                     export_img,
                     title: str = '',
                     nrow=8,
                     padding=2,
                     normalize=False,
                     pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""
    x = (x * 0.5 + 0.5) * 255
    grid = make_grid(x,
                     nrow=nrow,
                     padding=padding,
                     normalize=normalize,
                     pad_value=pad_value)

    npgrid = grid.cpu().numpy()
    trans_np = np.transpose(npgrid, (1, 2, 0)).astype(np.uint8)
    # sys.exit(0)
    Image.fromarray(trans_np).save(export_img + ".jpg", quality=95)
