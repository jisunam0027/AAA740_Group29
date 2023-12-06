import numpy as np
from packaging import version
import torch
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
from .pixel_wise_mapping import remap_using_correspondence_map
import torch.nn.functional as F
import cv2


# @profile
def co_pca(features1, features2, dim=[256, 256]):
    with torch.no_grad():
        # import pdb; pdb.set_trace()
        processed_features1 = {}
        processed_features2 = {}
        s1_size = features1["s1"].shape[-1]  # 2nd layer of decoder
        s2_size = features1["s2"].shape[-1]  # 3rd layer of decoder

        # Get the feature tensors
        # b, c, h*w
        s1_1 = (
            features1["s1"]
            .reshape(features1["s1"].shape[0], features1["s1"].shape[1], -1)
            .to(torch.float32)
        )
        s2_1 = (
            features1["s2"]
            .reshape(features1["s2"].shape[0], features1["s2"].shape[1], -1)
            .to(torch.float32)
        )

        s1_2 = (
            features2["s1"]
            .reshape(features2["s1"].shape[0], features2["s1"].shape[1], -1)
            .to(torch.float32)
        )
        s2_2 = (
            features2["s2"]
            .reshape(features2["s2"].shape[0], features2["s2"].shape[1], -1)
            .to(torch.float32)
        )

        target_dims = {"s1": dim[0], "s2": dim[1]}

        # Compute the PCA
        for name, tensors in zip(["s1", "s2"], [[s1_1, s1_2], [s2_1, s2_2]]):
            target_dim = target_dims[name]

            # Concatenate the features
            features = torch.cat(tensors, dim=-1)  # along the spatial dimension
            features = features.permute(0, 2, 1)  # Bx(t_x+t_y)x(d)

            # equivalent to the above, pytorch implementation
            mean = torch.mean(features[0], dim=0, keepdim=True)
            centered_features = features[0] - mean
            U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
            reduced_features = torch.matmul(
                centered_features, V[:, :target_dim]
            )  # (t_x+t_y)x(d)
            features = reduced_features.unsqueeze(0).permute(0, 2, 1)  # Bx(d)x(t_x+t_y)

            # Split the features
            processed_features1[name] = features[
                :, :, : features.shape[-1] // 2
            ]  # Bx(d)x(t_x)
            processed_features2[name] = features[
                :, :, features.shape[-1] // 2 :
            ]  # Bx(d)x(t_y)

        # reshape the features
        processed_features1["s1"] = processed_features1["s1"].reshape(
            processed_features1["s1"].shape[0], -1, s1_size, s1_size
        )
        processed_features1["s2"] = processed_features1["s2"].reshape(
            processed_features1["s2"].shape[0], -1, s2_size, s2_size
        )
        processed_features2["s1"] = processed_features2["s1"].reshape(
            processed_features2["s1"].shape[0], -1, s1_size, s1_size
        )
        processed_features2["s2"] = processed_features2["s2"].reshape(
            processed_features2["s2"].shape[0], -1, s2_size, s2_size
        )

        # Upsample s1 spatially by a factor of 2
        if processed_features1["s1"].shape != processed_features1["s2"].shape:
            processed_features1["s1"] = F.interpolate(
                processed_features1["s1"],
                size=(processed_features1["s2"].shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
            processed_features2["s1"] = F.interpolate(
                processed_features2["s1"],
                size=(processed_features2["s2"].shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )

        # Concatenate upsampled_s1 and s2 to create a new s1
        processed_features1["s1"] = torch.cat(
            [processed_features1["s2"], processed_features1["s1"]], dim=1
        )
        processed_features2["s1"] = torch.cat(
            [processed_features2["s2"], processed_features2["s1"]], dim=1
        )

        # Remove s3 from the features dictionary
        processed_features1.pop("s2")
        processed_features2.pop("s2")

        # Normalize
        processed_features1["s1"] /= processed_features1["s1"].norm(dim=1, keepdim=True)
        processed_features2["s1"] /= processed_features2["s1"].norm(dim=1, keepdim=True)

    return processed_features1["s1"], processed_features2["s1"]


def correlation_to_flow_w_argmax(
    correlation_target_to_source,
    output_shape=None,
    return_mapping=False,
    do_softmax=False,
):
    """
    Convert correlation to flow, with argmax.
    Args:
        correlation_target_to_source: shape is B, H_s*W_s, H_t, W_t
        output_shape: output shape of the flow from the target to the source image (H, W)
        do_softmax: bool, apply softmax to the correlation before finding the best match? (should not change anything)
        return_mapping: bool

    Returns:
        if return_mapping:
            correspondence map relating the target to the source, at output_shape.
        else:
            flow_est: flow field relating the target to the source, at output_shape.
    """
    H, W = correlation_target_to_source.shape[-2:]
    b = correlation_target_to_source.shape[0]
    # get matches corresponding to maximum in correlation
    (x_source, y_source, x_target, y_target, score) = corr_to_matches(
        correlation_target_to_source.view(b, H, W, H, W).unsqueeze(1),
        get_maximum=True,
        do_softmax=do_softmax,
    )

    # x_source dimension is B x H*W
    mapping_est = (
        torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1)
        .view(b, H, W, 2)
        .permute(0, 3, 1, 2)
    )
    # score = score.view(b, H, W)

    # b, 2, H, W
    flow_est = convert_mapping_to_flow(mapping_est)

    if output_shape is not None and (H != output_shape[0] or W != output_shape[1]):
        flow_est = F.interpolate(
            flow_est, output_shape, mode="bilinear", align_corners=False
        )
        flow_est[:, 0] *= float(output_shape[1]) / float(W)
        flow_est[:, 1] *= float(output_shape[0]) / float(H)
    if return_mapping:
        return convert_flow_to_mapping(flow_est)
    else:
        return flow_est


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (
        corr4d_A * corr4d_B
    )  # parenthesis are important for symmetric output

    return corr4d

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to torche optical flow

    Args:
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(flo.device)

    vgrid = grid + flo

    # makes a mapping out of torche flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    if version.parse(torch.__version__) >= version.parse("1.3"):
        # to be consistent to old version, I put align_corners=True.
        # to investigate if align_corners False is better.
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    else:
        output = nn.functional.grid_sample(x, vgrid)
    return output


def corr_to_matches(
    corr4d,
    delta4d=None,
    k_size=1,
    do_softmax=False,
    scale="positive",
    return_indices=False,
    invert_matching_direction=False,
    get_maximum=True,
):
    """
    Modified from NC-Net. Perform argmax over the correlation.
    Args:
        corr4d: correlation, shape is b, 1, H_s, W_s, H_t, W_t
        delta4d:
        k_size:
        do_softmax:
        scale:
        return_indices:
        invert_matching_direction:
        get_maximum:

    Returns:

    """
    to_cuda = lambda x: x.cuda() if corr4d.is_cuda else x
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    if scale == "centered":
        XA, YA = np.meshgrid(
            np.linspace(-1, 1, fs2 * k_size), np.linspace(-1, 1, fs1 * k_size)
        )
        XB, YB = np.meshgrid(
            np.linspace(-1, 1, fs4 * k_size), np.linspace(-1, 1, fs3 * k_size)
        )
    elif scale == "positive":
        # keep normal range of coordinate
        XA, YA = np.meshgrid(
            np.linspace(0, fs2 - 1, fs2 * k_size), np.linspace(0, fs1 - 1, fs1 * k_size)
        )
        XB, YB = np.meshgrid(
            np.linspace(0, fs4 - 1, fs4 * k_size), np.linspace(0, fs3 - 1, fs3 * k_size)
        )

    JA, IA = np.meshgrid(range(fs2), range(fs1))
    JB, IB = np.meshgrid(range(fs4), range(fs3))

    XA, YA = Variable(to_cuda(torch.FloatTensor(XA))), Variable(
        to_cuda(torch.FloatTensor(YA))
    )
    XB, YB = Variable(to_cuda(torch.FloatTensor(XB))), Variable(
        to_cuda(torch.FloatTensor(YB))
    )

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).view(1, -1))), Variable(
        to_cuda(torch.LongTensor(IA).view(1, -1))
    )
    JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(
        to_cuda(torch.LongTensor(IB).view(1, -1))
    )

    if invert_matching_direction:
        nc_A_Bvec = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

        if do_softmax:
            nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec, dim=3)

        if get_maximum:
            match_A_vals, idx_A_Bvec = torch.max(nc_A_Bvec, dim=3)
        else:
            match_A_vals, idx_A_Bvec = torch.min(nc_A_Bvec, dim=3)
        score = match_A_vals.view(batch_size, -1)

        iB = IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
        jB = JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
        iA = IA.expand_as(iB)
        jA = JA.expand_as(jB)

    else:
        nc_B_Avec = corr4d.view(
            batch_size, fs1 * fs2, fs3, fs4
        )  # [batch_idx,k_A,i_B,j_B]
        if do_softmax:
            nc_B_Avec = torch.nn.functional.softmax(nc_B_Avec, dim=1)

        if get_maximum:
            match_B_vals, idx_B_Avec = torch.max(nc_B_Avec, dim=1)
        else:
            match_B_vals, idx_B_Avec = torch.min(nc_B_Avec, dim=1)
        score = match_B_vals.view(batch_size, -1)

        iA = IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
        jA = JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
        iB = IB.expand_as(iA)
        jB = JB.expand_as(jA)

    if delta4d is not None:  # relocalization
        delta_iA, delta_jA, delta_iB, delta_jB = delta4d

        diA = delta_iA.squeeze(0).squeeze(0)[
            iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)
        ]
        djA = delta_jA.squeeze(0).squeeze(0)[
            iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)
        ]
        diB = delta_iB.squeeze(0).squeeze(0)[
            iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)
        ]
        djB = delta_jB.squeeze(0).squeeze(0)[
            iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)
        ]

        iA = iA * k_size + diA.expand_as(iA)
        jA = jA * k_size + djA.expand_as(jA)
        iB = iB * k_size + diB.expand_as(iB)
        jB = jB * k_size + djB.expand_as(jB)

    xA = XA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    yA = YA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    xB = XB[iB.contiguous().view(-1), jB.contiguous().view(-1)].view(batch_size, -1)
    yB = YB[iB.contiguous().view(-1), jB.contiguous().view(-1)].view(batch_size, -1)

    # XA is index in channel dimension (source)
    if return_indices:
        return xA, yA, xB, yB, score, iA, jA, iB, jB
    else:
        return xA, yA, xB, yB, score


def create_border_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image), and convert to float."""
    return get_gt_correspondence_mask(flow).float()


def get_gt_correspondence_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image)."""

    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    if isinstance(mapping, np.ndarray):
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[:, 0] >= 0, mapping[:, 0] <= w - 1)
            mask_y = np.logical_and(mapping[:, 1] >= 0, mapping[:, 1] <= h - 1)
            mask = np.logical_and(mask_x, mask_y)
        else:
            _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[0] >= 0, mapping[0] <= w - 1)
            mask_y = np.logical_and(mapping[1] >= 0, mapping[1] <= h - 1)
            mask = np.logical_and(mask_x, mask_y)
        mask = (
            mask.astype(np.bool)
            if version.parse(torch.__version__) >= version.parse("1.1")
            else mask.astype(np.uint8)
        )
    else:
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask = (
                mapping[:, 0].ge(0)
                & mapping[:, 0].le(w - 1)
                & mapping[:, 1].ge(0)
                & mapping[:, 1].le(h - 1)
            )
        else:
            _, h, w = mapping.shape
            mask = (
                mapping[0].ge(0)
                & mapping[0].le(w - 1)
                & mapping[1].ge(0)
                & mapping[1].le(h - 1)
            )
        mask = (
            mask.bool()
            if version.parse(torch.__version__) >= version.parse("1.1")
            else mask.byte()
        )
    return mask


def get_mapping_horizontal_flipping(image):
    H, W, C = image.shape
    mapping = np.zeros((H, W, 2), np.float32)
    for j in range(H):
        for i in range(W):
            mapping[j, i, 0] = W - i
            mapping[j, i, 1] = j
    return mapping, remap_using_correspondence_map(
        image, mapping[:, :, 0], mapping[:, :, 1]
    )


def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        # torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid  # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(0, 2, 3, 1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid  # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(1, 2, 0).float()
        return mapping.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(
                np.linspace(0, w_scale - 1, w_scale),
                np.linspace(0, h_scale - 1, h_scale),
            )
            for i in range(b):
                mapping[i, :, :, 0] = flow[i, :, :, 0] + X
                mapping[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(0, 3, 1, 2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.transpose(1, 2, 0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(
                np.linspace(0, w_scale - 1, w_scale),
                np.linspace(0, h_scale - 1, h_scale),
            )

            mapping[:, :, 0] = flow[:, :, 0] + X
            mapping[:, :, 1] = flow[:, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(2, 0, 1)
        return mapping.astype(np.float32)


def convert_mapping_to_flow(mapping, output_channel_first=True):
    if not isinstance(mapping, np.ndarray):
        # torch tensor
        if len(mapping.shape) == 4:
            if mapping.shape[1] != 2:
                # size is BxHxWx2
                mapping = mapping.permute(0, 3, 1, 2)

            B, C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(0, 2, 3, 1)
        else:
            if mapping.shape[0] != 2:
                # size is HxWx2
                mapping = mapping.permute(2, 0, 1)

            C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()

            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(1, 2, 0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(mapping.shape) == 4:
            if mapping.shape[3] != 2:
                # size is Bx2xHxW
                mapping = mapping.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = mapping.shape[:3]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(
                np.linspace(0, w_scale - 1, w_scale),
                np.linspace(0, h_scale - 1, h_scale),
            )
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0, 3, 1, 2)
        else:
            if mapping.shape[0] == 2:
                # size is 2xHxW
                mapping = mapping.transpose(1, 2, 0)
            # HxWx2
            h_scale, w_scale = mapping.shape[:2]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(
                np.linspace(0, w_scale - 1, w_scale),
                np.linspace(0, h_scale - 1, h_scale),
            )

            flow[:, :, 0] = mapping[:, :, 0] - X
            flow[:, :, 1] = mapping[:, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1)
        return flow.astype(np.float32)


def unormalise_and_convert_mapping_to_flow(map, output_channel_first=True):
    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            B, C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[:, 0, :, :] = (
                (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0
            )  # unormalise
            mapping[:, 1, :, :] = (
                (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0
            )  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(0, 2, 3, 1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[0, :, :] = (
                (map[0, :, :].float().clone() + 1) * (W - 1) / 2.0
            )  # unormalise
            mapping[1, :, :] = (
                (map[1, :, :].float().clone() + 1) * (H - 1) / 2.0
            )  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(1, 2, 0).float()
        return flow.float()
    else:
        # here numpy arrays
        flow = np.copy(map)
        if len(map.shape) == 4:
            if map.shape[1] == 2:
                # size is Bx2xHxWx
                map = map.transpose(0, 2, 3, 1)

            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            mapping = np.zeros_like(map)
            X, Y = np.meshgrid(
                np.linspace(0, w_scale - 1, w_scale),
                np.linspace(0, h_scale - 1, h_scale),
            )
            mapping[:, :, :, 0] = (map[:, :, :, 0] + 1) * (w_scale - 1) / 2
            mapping[:, :, :, 1] = (map[:, :, :, 1] + 1) * (h_scale - 1) / 2
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0, 3, 1, 2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.transpose(1, 2, 0)

            # HxWx2
            h_scale, w_scale = map.shape[:2]
            mapping = np.zeros_like(map)
            mapping[:, :, 0] = (map[:, :, 0] + 1) * (w_scale - 1) / 2
            mapping[:, :, 1] = (map[:, :, 1] + 1) * (h_scale - 1) / 2
            X, Y = np.meshgrid(
                np.linspace(0, w_scale - 1, w_scale),
                np.linspace(0, h_scale - 1, h_scale),
            )

            flow[:, :, 0] = mapping[:, :, 0] - X
            flow[:, :, 1] = mapping[:, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1)
        return flow.astype(np.float32)


def unormalise_flow_or_mapping(map, output_channel_first=True):
    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            B, C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[:, 0, :, :] = (
                (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0
            )  # unormalise
            mapping[:, 1, :, :] = (
                (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0
            )  # unormalise

            if not output_channel_first:
                mapping = mapping.permute(0, 2, 3, 1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[0, :, :] = (
                (map[0, :, :].float().clone() + 1) * (W - 1) / 2.0
            )  # unormalise
            mapping[1, :, :] = (
                (map[1, :, :].float().clone() + 1) * (H - 1) / 2.0
            )  # unormalise

            if not output_channel_first:
                mapping = mapping.permute(1, 2, 0).float()
        return mapping.float()
    else:
        # here numpy arrays
        if len(map.shape) == 4:
            if map.shape[1] == 2:
                # size is Bx2xHxWx
                map = map.transpose(0, 2, 3, 1)

            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            mapping = np.zeros_like(map)
            mapping[:, :, :, 0] = (map[:, :, :, 0] + 1) * (w_scale - 1) / 2
            mapping[:, :, :, 1] = (map[:, :, :, 1] + 1) * (h_scale - 1) / 2

            if output_channel_first:
                mapping = mapping.transpose(0, 3, 1, 2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.transpose(1, 2, 0)

            # HxWx2
            h_scale, w_scale = map.shape[:2]
            mapping = np.zeros_like(map)
            mapping[:, :, 0] = (map[:, :, 0] + 1) * (w_scale - 1) / 2
            mapping[:, :, 1] = (map[:, :, 1] + 1) * (h_scale - 1) / 2

            if output_channel_first:
                mapping = mapping.transpose(2, 0, 1)
        return mapping.astype(np.float32)


def unnormalize(tensor, output_channel_first=True):
    if len(tensor.shape) == 4:
        if tensor.shape[1] != 2:
            if not isinstance(map, np.ndarray):
                tensor = tensor.permute(0, 3, 1, 2)
            else:
                tensor = tensor.transpose(0, 3, 1, 2)

        B, C, H, W = tensor.size()
        tensor_unnorm = torch.zeros_like(tensor)
        # mesh grid
        tensor_unnorm[:, 0, :, :] = (
            (tensor[:, 0, :, :] + 1) * (W - 1) / 2.0
        )  # unormalise
        tensor_unnorm[:, 1, :, :] = (
            (tensor[:, 1, :, :] + 1) * (H - 1) / 2.0
        )  # unormalise

        if not output_channel_first:
            tensor_unnorm = tensor_unnorm.permute(0, 2, 3, 1)
    else:
        if tensor.shape[0] != 2:
            if not isinstance(map, np.ndarray):
                tensor = tensor.permute(2, 0, 1)
            else:
                tensor = tensor.transpose(2, 0, 1)

        C, H, W = tensor.size()
        tensor_unnorm = torch.zeros_like(tensor)
        # mesh grid
        tensor_unnorm[0, :, :] = (tensor[0, :, :] + 1) * (W - 1) / 2.0  # unormalise
        tensor_unnorm[1, :, :] = (tensor[1, :, :] + 1) * (H - 1) / 2.0  # unormalise

        if not output_channel_first:
            tensor_unnorm = tensor_unnorm.permute(0, 2, 3, 1)

    return tensor_unnorm


def normalize(tensor, output_channel_first=True):
    if len(tensor.shape) == 4:
        if tensor.shape[1] != 2:
            if not isinstance(map, np.ndarray):
                tensor = tensor.permute(0, 3, 1, 2)
            else:
                tensor = tensor.transpose(0, 3, 1, 2)

        B, C, H, W = tensor.size()
        tensor_norm = torch.zeros_like(tensor)
        # mesh grid
        tensor_norm[:, 0, :, :] = 2 * tensor[:, 0, :, :] / (W - 1) - 1.0
        tensor_norm[:, 1, :, :] = 2 * tensor[:, 1, :, :] / (H - 1) - 1.0

        if not output_channel_first:
            tensor_norm = tensor_norm.permute(0, 2, 3, 1)
    else:
        if tensor.shape[0] != 2:
            if not isinstance(map, np.ndarray):
                tensor = tensor.permute(2, 0, 1)
            else:
                tensor = tensor.transpose(2, 0, 1)

        C, H, W = tensor.size()
        tensor_norm = torch.zeros_like(tensor)
        # mesh grid
        tensor_norm[0, :, :] = 2 * tensor[0, :, :] / (W - 1) - 1.0
        tensor_norm[1, :, :] = 2 * tensor[1, :, :] / (H - 1) - 1.0

        if not output_channel_first:
            tensor_norm = tensor_norm.permute(0, 2, 3, 1)

    return tensor_norm
