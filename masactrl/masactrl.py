import os
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from .rhm_map import appearance_similarityOT
import math
from .flow_and_mapping_operations import save_tensor_images, rearrange_and_save


from .masactrl_utils import AttentionBase
from .flow_and_mapping_operations import (
    corr_to_matches,
    convert_mapping_to_flow,
    convert_flow_to_mapping,
    warp,
    MutualMatching,
    correlation_to_flow_w_argmax,
    otsu_binarization,
)

# from line_profiler import profile


class MutualSelfAttentionControl(AttentionBase):
    def __init__(
        self,
        start_step=4,
        start_layer=10,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        time_cut=50,
        layer_cut=16,
    ):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = (
            layer_idx if layer_idx is not None else list(range(start_layer, layer_cut))
        )
        self.step_idx = (
            step_idx if step_idx is not None else list(range(start_step, time_cut))
        )
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)
        print("final_step_idx: ", time_cut - 1)
        print("final_layer_idx: ", layer_cut - 1)

    def attn_batch(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        """
        Attention forward function
        """
        if (
            is_cross
            or self.cur_step not in self.step_idx
            or self.cur_att_layer // 2 not in self.layer_idx
        ):
            return super().forward(
                q,
                k,
                v,
                sim,
                attn,
                is_cross,
                pca_feats,
                latent_list,
                pred_x0_list,
                place_in_unet,
                num_heads,
                **kwargs,
            )

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(
            qu,
            ku[:num_heads],
            vu[:num_heads],
            sim[:num_heads],
            attnu,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            **kwargs,
        )
        out_c = self.attn_batch(
            qc,
            kc[:num_heads],
            vc[:num_heads],
            sim[:num_heads],
            attnc,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            **kwargs,
        )
        out = torch.cat([out_u, out_c], dim=0)

        return out


class MutualSelfAttentionControl(AttentionBase):
    def __init__(
        self,
        start_step=4,
        start_layer=10,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        time_cut=50,
        layer_cut=16,
    ):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = (
            layer_idx if layer_idx is not None else list(range(start_layer, layer_cut))
        )
        self.step_idx = (
            step_idx if step_idx is not None else list(range(start_step, time_cut))
        )
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)
        print("final_step_idx: ", time_cut - 1)
        print("final_layer_idx: ", layer_cut - 1)

    def attn_batch(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        """
        Attention forward function
        """
        if (
            is_cross
            or self.cur_step not in self.step_idx
            or self.cur_att_layer // 2 not in self.layer_idx
        ):
            return super().forward(
                q,
                k,
                v,
                sim,
                attn,
                is_cross,
                pca_feats,
                latent_list,
                pred_x0_list,
                place_in_unet,
                num_heads,
                **kwargs,
            )

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u = self.attn_batch(
            qu,
            ku[:num_heads],
            vu[:num_heads],
            sim[:num_heads],
            attnu,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            **kwargs,
        )
        out_c = self.attn_batch(
            qc,
            kc[:num_heads],
            vc[:num_heads],
            sim[:num_heads],
            attnc,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            **kwargs,
        )
        out = torch.cat([out_u, out_c], dim=0)

        return out


class MutualSelfAttentionControlMask(MutualSelfAttentionControl):
    def __init__(
        self,
        start_step=4,
        start_layer=10,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        mask_s=None,
        mask_t=None,
        mask_save_dir=None,
    ):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps)
        self.mask_s = mask_s  # source mask with shape (h, w)
        self.mask_t = mask_t  # target mask with same shape as source mask
        print("Using mask-guided MasaCtrl")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(
                self.mask_s.unsqueeze(0).unsqueeze(0),
                os.path.join(mask_save_dir, "mask_s.png"),
            )
            save_image(
                self.mask_t.unsqueeze(0).unsqueeze(0),
                os.path.join(mask_save_dir, "mask_t.png"),
            )

    def attn_batch(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            print("masked attention")
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten()
            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            # object
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        """
        Attention forward function
        """
        if (
            is_cross
            or self.cur_step not in self.step_idx
            or self.cur_att_layer // 2 not in self.layer_idx
        ):
            return super().forward(
                q,
                k,
                v,
                sim,
                attn,
                is_cross,
                pca_feats,
                latent_list,
                pred_x0_list,
                place_in_unet,
                num_heads,
                **kwargs,
            )

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(
            qu[:num_heads],
            ku[:num_heads],
            vu[:num_heads],
            sim[:num_heads],
            attnu,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            **kwargs,
        )
        out_c_source = self.attn_batch(
            qc[:num_heads],
            kc[:num_heads],
            vc[:num_heads],
            sim[:num_heads],
            attnc,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            **kwargs,
        )

        out_u_target = self.attn_batch(
            qu[-num_heads:],
            ku[:num_heads],
            vu[:num_heads],
            sim[:num_heads],
            attnu,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            is_mask_attn=True,
            **kwargs,
        )
        out_c_target = self.attn_batch(
            qc[-num_heads:],
            kc[:num_heads],
            vc[:num_heads],
            sim[:num_heads],
            attnc,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            is_mask_attn=True,
            **kwargs,
        )

        if self.mask_s is not None and self.mask_t is not None:
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

            mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
            out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out, None, None


class MutualSelfAttentionControlMaskAuto(MutualSelfAttentionControl):
    def __init__(
        self,
        start_step=4,
        start_layer=10,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        thres=0.1,
        ref_token_idx=[1],
        cur_token_idx=[1],
        mask_save_dir=None,
    ):
        """
        MasaCtrl with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps)
        print("using MutualSelfAttentionControlMaskAuto")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask
            thres = self.thres
            mask[mask >= thres] = 1
            mask[mask < thres] = 0
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg])

        attn = sim.softmax(-1)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        """
        Attention forward function
        """
        if is_cross:
            # save cross attention map with res 16 * 16
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(
                    attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1)
                )

        if (
            is_cross
            or self.cur_step not in self.step_idx
            or self.cur_att_layer // 2 not in self.layer_idx
        ):
            return super().forward(
                q,
                k,
                v,
                sim,
                attn,
                is_cross,
                pca_feats,
                latent_list,
                pred_x0_list,
                place_in_unet,
                num_heads,
                **kwargs,
            )

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        # 1, 1024(32x32), 640
        out_u_source = self.attn_batch(
            qu[:num_heads],
            ku[:num_heads],
            vu[:num_heads],
            sim[:num_heads],
            attnu,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            **kwargs,
        )
        out_c_source = self.attn_batch(
            qc[:num_heads],
            kc[:num_heads],
            vc[:num_heads],
            sim[:num_heads],
            attnc,
            is_cross,
            None,
            None,
            None,
            place_in_unet,
            num_heads,
            **kwargs,
        )

        if len(self.cross_attns) == 0:
            self.self_attns_mask = None
            out_u_target = self.attn_batch(
                qu[-num_heads:],
                ku[:num_heads],
                vu[:num_heads],
                sim[:num_heads],
                attnu,
                is_cross,
                None,
                None,
                None,
                place_in_unet,
                num_heads,
                **kwargs,
            )
            out_c_target = self.attn_batch(
                qc[-num_heads:],
                kc[:num_heads],
                vc[:num_heads],
                sim[:num_heads],
                attnc,
                is_cross,
                None,
                None,
                None,
                place_in_unet,
                num_heads,
                **kwargs,
            )
        else:
            mask = self.aggregate_cross_attn_map(
                idx=self.ref_token_idx
            )  # (2, H, W) -> (4, 16, 16)
            mask_source = mask[-2]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            self.self_attns_mask = F.interpolate(
                mask_source.unsqueeze(0).unsqueeze(0), (res, res)
            ).flatten()
            # if self.mask_save_dir is not None:
            #     H = W = int(np.sqrt(self.self_attns_mask.shape[0]))
            #     mask_image = self.self_attns_mask.reshape(H, W).unsqueeze(0)
            #     save_image(
            #         mask_image,
            #         os.path.join(
            #             self.mask_save_dir,
            #             f"mask_s_{self.cur_step}_{self.cur_att_layer}.png",
            #         ),
            #     )
            out_u_target = self.attn_batch(
                qu[-num_heads:],
                ku[:num_heads],
                vu[:num_heads],
                sim[:num_heads],
                attnu,
                is_cross,
                None,
                None,
                None,
                place_in_unet,
                num_heads,
                **kwargs,
            )
            out_c_target = self.attn_batch(
                qc[-num_heads:],
                kc[:num_heads],
                vc[:num_heads],
                sim[:num_heads],
                attnc,
                is_cross,
                None,
                None,
                None,
                place_in_unet,
                num_heads,
                **kwargs,
            )

        if self.self_attns_mask is not None:
            mask = self.aggregate_cross_attn_map(idx=self.cur_token_idx)  # (2, H, W)
            mask_target = mask[-1]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            self.self_attns_mask_gen = F.interpolate(
                mask_target.unsqueeze(0).unsqueeze(0), (res, res)
            ).reshape(-1, 1)
            # if self.mask_save_dir is not None:
            #     H = W = int(np.sqrt(self.self_attns_mask_gen.shape[0]))
            #     mask_image = self.self_attns_mask_gen.reshape(H, W).unsqueeze(0)
            #     save_image(
            #         mask_image,
            #         os.path.join(
            #             self.mask_save_dir,
            #             f"mask_t_{self.cur_step}_{self.cur_att_layer}.png",
            #         ),
            #     )
            # binarize the mask
            thres = self.thres
            self.self_attns_mask_gen[self.self_attns_mask_gen >= thres] = 1
            self.self_attns_mask_gen[self.self_attns_mask_gen < thres] = 0
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2)

            out_u_target = (
                out_u_target_fg * self.self_attns_mask_gen
                + out_u_target_bg * (1 - self.self_attns_mask_gen)
            )
            out_c_target = (
                out_c_target_fg * self.self_attns_mask_gen
                + out_c_target_bg * (1 - self.self_attns_mask_gen)
            )

            # set self self-attention mask to None
            self.self_attns_mask = None

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out, None, None


class MutualSelfAttentionControlMaskAuto_Matching(MutualSelfAttentionControl):
    def __init__(self, args=None, mask_save_dir=None, save_dir="./"):
        """
        MasaCtrl with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__()

        self.args = args
        self.thres = args["thres"]
        self.cc_thres = args["cc_thres"]
        self.ref_token_idx = args["ref_token_idx"]
        self.cur_token_idx = args["cur_token_idx"]
        self.mode = args["mode"]
        self.save_dir = save_dir
        self.save_qkv = args["save_qkv"]
        self.save_pred_x0 = args["save_pred_x0"]
        self.mask_save_dir = mask_save_dir
        self.fg_mask = args["fg_mask"]
        self.key_replace = args["key_replace"]
        self.hierarchical = args["hierarchical"]

        self.start_step = args["initial_step"]
        self.start_layer = args["initial_layer"]
        self.layer_cut = args["cut_layer"]
        self.time_cut = args["cut_step"]
        self.layer_idx = list(range(self.start_layer, self.layer_cut))
        self.step_idx = list(range(self.start_step, self.time_cut))
        print("Set Steps and Layers of Matching based Attention Module!")
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)
        print("final_step_idx: ", self.time_cut - 1)
        print("final_layer_idx: ", self.layer_cut - 1)

        self.prev_steps = []
        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None
        self.sim = None

        self.transforms = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if self.mode == "sd-dino":
            from sd_dino.demo_swap import process_images_matching

            self.process_images_matching = process_images_matching
            self.categories = "cat"
        elif self.mode == "dino":
            from sd_dino.demo_swap_dino import process_images_matching_dino

            self.process_images_matching = process_images_matching_dino
            self.categories = "cat"

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    # @profile
    def attn_batch_match(
        self,
        q_gen,
        q_ref,
        k_gen,
        k_ref,
        v_gen,
        v_ref,
        pca_feats,
        latent_list,
        pred_x0_list,
        sim,
        attn,
        is_cross,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        B = q_gen.shape[0] // num_heads
        H = W = int(np.sqrt(q_gen.shape[1]))

        # rearrange q,k,v and save them
        q_gen = rearrange_and_save(
            q_gen,
            ["query_gen", self.cur_step, self.cur_att_layer],
            num_heads,
            H,
            W,
            self.save_qkv,
            self.save_dir,
        )
        q_ref = rearrange_and_save(
            q_ref,
            ["query_ref", self.cur_step, self.cur_att_layer],
            num_heads,
            H,
            W,
            self.save_qkv,
            self.save_dir,
        )
        k_gen = rearrange_and_save(
            k_gen,
            ["key_gen", self.cur_step, self.cur_att_layer],
            num_heads,
            H,
            W,
            self.save_qkv,
            self.save_dir,
        )
        k_ref = rearrange_and_save(
            k_ref,
            ["key_ref", self.cur_step, self.cur_att_layer],
            num_heads,
            H,
            W,
            self.save_qkv,
            self.save_dir,
        )
        v_gen = rearrange_and_save(
            v_gen,
            ["value_gen", self.cur_step, self.cur_att_layer],
            num_heads,
            H,
            W,
            self.save_qkv,
            self.save_dir,
        )
        v_ref = rearrange_and_save(
            v_ref,
            ["value_ref", self.cur_step, self.cur_att_layer],
            num_heads,
            H,
            W,
            self.save_qkv,
            self.save_dir,
        )

        DOWN_FAC = 1
        
        if self.mode == "sd":
            src_feat, trg_feat = pca_feats
            src_feat = F.interpolate(
                src_feat[None], size=H, mode="bilinear", align_corners=False
            )
            trg_feat = F.interpolate(
                trg_feat[None], size=H, mode="bilinear", align_corners=False
            )

            src_feat = rearrange(src_feat, "b c Hs Ws -> b (Hs Ws) c")
            trg_feat = rearrange(trg_feat, "b c Ht Wt -> b (Ht Wt) c")

            # Calculate norms for src and tgt features
            norm_src_features = torch.linalg.norm(src_feat, dim=2, keepdim=True)
            norm_tgt_features = torch.linalg.norm(trg_feat, dim=2, keepdim=True)

            # Compute dot products using einsum
            sim = torch.einsum("b i d, b j d -> b i j", trg_feat, src_feat)

            # Divide by the norms to calculate cosine similarities
            sim /= norm_tgt_features * norm_src_features.transpose(1, 2)

        # separate foreground, background
        # separate foreground, background
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask.clone()
            mask_gen = self.self_attns_mask_gen.clone()

            mask.to(sim.device)
            mask_gen.to(sim.device)
            mask_ = rearrange(
                mask,
                "(h w) -> h w",
                h=int(math.sqrt(len(mask))),
                w=int(math.sqrt(len(mask))),
            )
            mask_gen_ = rearrange(
                mask_gen,
                "(h w) -> h w",
                h=int(math.sqrt(len(mask_gen))),
                w=int(math.sqrt(len(mask_gen))),
            )
            mask = (
                F.interpolate(mask_[None][None], size=H // DOWN_FAC, mode="nearest")
                .squeeze()
                .flatten()
            )

            mask_gen = (
                F.interpolate(mask_gen_[None][None], size=H // DOWN_FAC, mode="nearest")
                .squeeze()
                .flatten()
            )
            

        H = H // DOWN_FAC
        W = W // DOWN_FAC

        sim_backward = rearrange(
            sim, "b (Ht Wt) (Hs Ws) -> b (Hs Ws) Ht Wt", Hs=H, Ws=W, Ht=H, Wt=W
        )
        sim_forward = rearrange(
            sim, "b (Ht Wt) (Hs Ws) -> b (Ht Wt) Hs Ws", Hs=H, Ws=W, Ht=H, Wt=W
        )

        flow_gen_to_ref = correlation_to_flow_w_argmax(
            sim_backward, output_shape=(H, W)
        )
        flow_ref_to_gen = correlation_to_flow_w_argmax(sim_forward, output_shape=(H, W))

        # get cycle consistency error
        cc_error = torch.norm(
            flow_gen_to_ref + warp(flow_ref_to_gen, flow_gen_to_ref),
            dim=1,
            p=2,
            keepdim=True,
        )

        fg_ratio = mask_gen_.sum() / (H * W)
        confidence = (cc_error < self.cc_thres * H * fg_ratio) * 1.0

        H = H * DOWN_FAC
        W = W * DOWN_FAC

        if DOWN_FAC != 1:
            flow_gen_to_ref = F.interpolate(flow_gen_to_ref, size=H, mode="nearest")
            confidence = F.interpolate(confidence, size=H, mode="nearest")


        # warping v_ref to v_gen, filtering through cycle consistency
        v_ref = rearrange(v_ref, "b (H W) c -> b c H W", H=H, W=W)
        v_gen = rearrange(v_gen, "b (H W) c -> b c H W", H=H, W=W)

        # align v.
        warped_v = warp(v_ref.to(flow_gen_to_ref.dtype), flow_gen_to_ref)
   
        # warping v_ref to v_gen, filtering through cycle consistency
        k_gen = rearrange(k_gen, "b (H W) c -> b c H W", H=H, W=W)

        # cycle consistency
        _, _, hc, wc = confidence.shape
        mask_gen_ = mask_gen_.reshape(1, 1, hc, wc)
        mask_gen_ = F.interpolate(mask_gen_, size=H, mode="nearest")
        warped_v = warped_v * confidence + v_gen * (1 - confidence)
      
        if self.fg_mask:
            warped_v = warped_v * mask_gen_ + v_gen * (1 - mask_gen_)

        warped_v = rearrange(warped_v, "b c H W -> b (H W) c")
        k_gen = rearrange(k_gen, "b c H W -> b (H W) c")
        v_gen = rearrange(v_gen, "b c H W -> b (H W) c")

        sim = torch.einsum("h i d, h j d -> h i j", q_gen, k_gen) * kwargs.get("scale")

      
        attn = sim.softmax(-1)

        out = torch.einsum("h i j, h j d -> h i d", attn, warped_v.to(attn.dtype))
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)

      

        return out

    def attn_batch(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask
            thres = self.thres
            mask[mask >= thres] = 1
            mask[mask < thres] = 0
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg])

        attn = sim.softmax(-1)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    def forward(
        self,
        q,
        k,
        v,
        sim,
        attn,
        is_cross,
        pca_feats,
        latent_list,
        pred_x0_list,
        place_in_unet,
        num_heads,
        **kwargs,
    ):
        """
        Attention forward function
        """
        with torch.no_grad():
            self.self_attns_mask = None
            self.self_attns_mask_gen = None

            if is_cross:
                # save cross attention map with res 16 * 16
                if attn.shape[1] == 16 * 16:
                    self.cross_attns.append(
                        attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1)
                    )

                if len(self.cross_attns) == 0:
                    self.self_attns_mask = None
                else:
                    mask = self.aggregate_cross_attn_map(
                        idx=self.ref_token_idx
                    )  # (2, H, W) -> (4, 16, 16)
                    mask_source = mask[-2]  # (H, W)
                    res = int(np.sqrt(q.shape[1]))
                    self.self_attns_mask = F.interpolate(
                        mask_source.unsqueeze(0).unsqueeze(0), (res, res)
                    ).flatten()

                    # if self.mask_save_dir is not None:
                    #     H = W = int(np.sqrt(self.self_attns_mask.shape[0]))
                    #     mask_image = mask_source.clone().unsqueeze(0)
                    #     save_image(
                    #         mask_image,
                    #         os.path.join(
                    #             self.mask_save_dir,
                    #             f"mask_s_{self.cur_step}_{self.cur_att_layer}.png",
                    #         ),
                    #     )

                    mask = self.aggregate_cross_attn_map(
                        idx=self.cur_token_idx
                    )  # (2, H, W)
                    mask_target = mask[-1]  # (H, W)
                    res = int(np.sqrt(q.shape[1]))
                    self.self_attns_mask_gen = F.interpolate(
                        mask_target.unsqueeze(0).unsqueeze(0), (res, res)
                    ).flatten()

                    # if self.mask_save_dir is not None:
                    #     H = W = int(np.sqrt(self.self_attns_mask_gen.shape[0]))
                    #     mask_image = mask_target.clone().unsqueeze(0)
                    #     save_image(
                    #         mask_image,
                    #         os.path.join(
                    #             self.mask_save_dir,
                    #             f"mask_t_{self.cur_step}_{self.cur_att_layer}.png",
                    #         ),
                    #     )

            if (
                is_cross
                or self.cur_step not in self.step_idx
                or self.cur_att_layer // 2 not in self.layer_idx
            ):
                return super().forward(
                    q,
                    k,
                    v,
                    sim,
                    attn,
                    is_cross,
                    pca_feats,
                    latent_list,
                    pred_x0_list,
                    place_in_unet,
                    num_heads,
                    **kwargs,
                )

            B = q.shape[0] // num_heads // 2
            H = W = int(np.sqrt(q.shape[1]))
            qu, qc = q.chunk(2)  # q : batch * head, height * width, channels
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            attnu, attnc = attn.chunk(2)

            out_u_source = self.attn_batch(
                qu[:num_heads],
                ku[:num_heads],
                vu[:num_heads],
                sim[:num_heads],
                attnu,
                is_cross,
                None,
                None,
                None,
                place_in_unet,
                num_heads,
                **kwargs,
            )
            out_c_source = self.attn_batch(
                qc[:num_heads],
                kc[:num_heads],
                vc[:num_heads],
                sim[:num_heads],
                attnc,
                is_cross,
                None,
                None,
                None,
                place_in_unet,
                num_heads,
                **kwargs,
            )

            if len(self.cross_attns) == 0:
                out_u_target = self.attn_batch(
                    qu[-num_heads:],
                    ku[:num_heads],
                    vu[:num_heads],
                    sim[:num_heads],
                    attnu,
                    is_cross,
                    None,
                    None,
                    None,
                    place_in_unet,
                    num_heads,
                    **kwargs,
                )
                out_c_target = self.attn_batch(
                    qc[-num_heads:],
                    kc[:num_heads],
                    vc[:num_heads],
                    sim[:num_heads],
                    attnc,
                    is_cross,
                    None,
                    None,
                    None,
                    place_in_unet,
                    num_heads,
                    **kwargs,
                )
            else:
                mask = self.aggregate_cross_attn_map(
                    idx=self.ref_token_idx
                )  # (2, H, W) -> (4, 16, 16)
                mask_source = mask[-2]  # (H, W)
                res = int(np.sqrt(q.shape[1]))
                self.self_attns_mask = F.interpolate(
                    mask_source.unsqueeze(0).unsqueeze(0), (res, res)
                ).flatten()

                mask = self.aggregate_cross_attn_map(
                    idx=self.cur_token_idx
                )  # (2, H, W)
                mask_target = mask[-1]  # (H, W)
                res = int(np.sqrt(q.shape[1]))
                self.self_attns_mask_gen = F.interpolate(
                    mask_target.unsqueeze(0).unsqueeze(0), (res, res)
                ).flatten()

                out_u_target = self.attn_batch_match(
                    qu[-num_heads:],
                    qu[:num_heads],
                    ku[-num_heads:],
                    ku[:num_heads],
                    vu[-num_heads:],
                    vu[:num_heads],
                    pca_feats[0],
                    latent_list,
                    pred_x0_list,
                    sim[:num_heads],
                    attnu,
                    is_cross,
                    place_in_unet,
                    num_heads,
                    **kwargs,
                )
                out_c_target = self.attn_batch_match(
                    qc[-num_heads:],
                    qc[:num_heads],
                    kc[-num_heads:],
                    ku[:num_heads],
                    vc[-num_heads:],
                    vc[:num_heads],
                    pca_feats[1],
                    latent_list,
                    pred_x0_list,
                    sim[:num_heads],
                    attnc,
                    is_cross,
                    place_in_unet,
                    num_heads,
                    **kwargs,
                )

                del (
                    qu,
                    qc,
                    ku,
                    kc,
                    vu,
                    vc,
                    attnu,
                    attnc,
                    mask,
                    mask_target,
                    mask_source,
                    q,
                    k,
                    v,
                )
                # torch.cuda.empty_cache()

            out = torch.cat(
                [out_u_source, out_u_target, out_c_source, out_c_target], dim=0
            )
        return out, self.self_attns_mask, self.self_attns_mask_gen
