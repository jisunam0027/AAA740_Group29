import numpy as np
import cv2
import torch
import torch.nn as nn
from packaging import version

def warp(x, flo, padding_mode='zeros', return_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    args:
        x: [B, C, H, W]
        flo: [B, 2, H, W] flow
    outputs:
        output: warped x [B, C, H, W]
    """
    B, C, H, W = flo.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    if version.parse(torch.__version__) >= version.parse("1.3"):
        output = nn.functional.grid_sample(x, vgrid, align_corners=True, padding_mode=padding_mode)
    else:
        output = nn.functional.grid_sample(x, vgrid, padding_mode=padding_mode)

    if return_mask:
        vgrid = vgrid.permute(0, 3, 1, 2)
        mask = (vgrid[:, 0] > -1) & (vgrid[:, 1] > -1) & (vgrid[:, 0] < 1) & (vgrid[:, 1] < 1)
        return output, mask
    return output
