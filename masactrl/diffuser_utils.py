import os
import torch
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from .flow_viz import flow_to_image as flow_to_image

from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from .flow_and_mapping_operations import co_pca

class MasaCtrlPipeline(StableDiffusionPipeline):
    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.0,
        verbose=False,
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps,
            999,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float = 0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep > 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)["sample"]

        return image  # range [-1, 1]

    def pre_interpolation(self, feat1, feat2):
        # pre interpolation layer 2(feat1) features to the same size with layer 3(feat2) features.
        inter_shape = feat2.shape

        src_feat_uncond, src_feat_cond, trg_feat_uncond, trg_feat_cond = {}, {}, {}, {}

        feat1_src_uncond, feat1_trg_uncond, feat1_src_cond, feat1_trg_cond = feat1
        feat2_src_uncond, feat2_trg_uncond, feat2_src_cond, feat2_trg_cond = feat2

        # interpolate low resolution feature.
        feat1_src_uncond = F.interpolate(
            feat1_src_uncond[None],
            size=inter_shape[-1],
            mode="bilinear",
            align_corners=True,
        )
        feat1_trg_uncond = F.interpolate(
            feat1_trg_uncond[None],
            size=inter_shape[-1],
            mode="bilinear",
            align_corners=True,
        )
        feat1_src_cond = F.interpolate(
            feat1_src_cond[None],
            size=inter_shape[-1],
            mode="bilinear",
            align_corners=True,
        )
        feat1_trg_cond = F.interpolate(
            feat1_trg_cond[None],
            size=inter_shape[-1],
            mode="bilinear",
            align_corners=True,
        )

        src_feat_uncond["s1"], src_feat_uncond["s2"] = (
            feat1_src_uncond,
            feat2_src_uncond[None],
        )
        src_feat_cond["s1"], src_feat_cond["s2"] = feat1_src_cond, feat2_src_cond[None]
        trg_feat_uncond["s1"], trg_feat_uncond["s2"] = (
            feat1_trg_uncond,
            feat2_trg_uncond[None],
        )
        trg_feat_cond["s1"], trg_feat_cond["s2"] = feat1_trg_cond, feat2_trg_cond[None]

        return src_feat_uncond, trg_feat_uncond, src_feat_cond, trg_feat_cond

    @torch.no_grad()
    # @profile
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        return_intermediates=False,
    ):
        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=77, return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert (
                latents.shape == latents_shape
            ), f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.0:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            # unconditional_input.input_ids = unconditional_input.input_ids[:, 1:]
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(DEVICE)
            )[0]
            text_embeddings = torch.cat(
                [unconditional_embeddings, text_embeddings], dim=0
            )

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)

        latents_list = None
        pred_x0_list = None
        if return_intermediates:
            latents_list = [self.latent2image(latents, return_type="pt")]
            pred_x0_list = [self.latent2image(latents, return_type="pt")]

        self.latents_list = latents_list
        self.pred_x0_list = pred_x0_list
        self.pca_feats = []

       
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            
            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat(
                    [unconditioning[i].expand(*text_embeddings.shape), text_embeddings]
                )
          
            torch.set_grad_enabled(False)
            model_inputs = model_inputs.detach().requires_grad_(False)

            noise_pred, feats = self.unet(
                model_inputs, t, encoder_hidden_states=text_embeddings
            ).sample

            feat1 = feats["up_feat2"]
            feat2 = feats["up_feat3"]
            (
                src_feat_uncond,
                trg_feat_uncond,
                src_feat_cond,
                trg_feat_cond,
            ) = self.pre_interpolation(feat1, feat2)

            # run pca and normalize
            src_feat_uncond_pca, trg_feat_uncond_pca = co_pca(
                src_feat_uncond, trg_feat_uncond, [256, 256]
            )
            src_feat_cond_pca, trg_feat_cond_pca = co_pca(
                src_feat_cond, trg_feat_cond, [256, 256]
            )

            src_feat_uncond_pca = src_feat_uncond_pca.to(latents.dtype)
            trg_feat_uncond_pca = trg_feat_uncond_pca.to(latents.dtype)
            src_feat_cond_pca = src_feat_cond_pca.to(latents.dtype)
            trg_feat_cond_pca = trg_feat_cond_pca.to(latents.dtype)

            # matching guided attention module also use this same pca features.
            self.pca_feats = [
                torch.cat([src_feat_uncond_pca, trg_feat_uncond_pca], dim=0),
                torch.cat([src_feat_cond_pca, trg_feat_cond_pca], dim=0),
            ]

            # classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.detach().chunk(
                    2, dim=0
                )  # order ! uncond src, trg, cond src, trg
                noise_pred = noise_pred_uncon + guidance_scale * (
                    noise_pred_con - noise_pred_uncon
                )

            # compute the previous noise sample x_t -> x_t-1
            latents, pred_x0 = self.step(
                noise_pred.detach(), t.detach(), latents.detach()
            )

            if return_intermediates:
                latents_list.append(self.latent2image(latents, return_type="pt"))
                pred_x0_list.append(self.latent2image(pred_x0, return_type="pt"))

            self.latents_list = latents_list
            self.pred_x0_list = pred_x0_list

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds,
    ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=77, return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # print(latents)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.0:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            )
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(DEVICE)
            )[0]
            text_embeddings = torch.cat(
                [unconditional_embeddings, text_embeddings], dim=0
            )

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(
            tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")
        ):
            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred, _ = self.unet(
                model_inputs, t, encoder_hidden_states=text_embeddings
            ).sample
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (
                    noise_pred_con - noise_pred_uncon
                )
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            return latents, latents_list
        return latents, start_latents


class CustomDiffusionPipeline(MasaCtrlPipeline):
    r"""
    Pipeline for custom diffusion model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
        modifier_token: list of new modifier tokens added or to be added to text_encoder
        modifier_token_id: list of id of new modifier tokens added or to be added to text_encoder
    """
    _optional_components = ["safety_checker", "feature_extractor", "modifier_token"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        modifier_token: list = [],
        modifier_token_id: list = [],
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )

        # change attn class
        self.modifier_token = modifier_token
        self.modifier_token_id = modifier_token_id

    def add_token(self, initializer_token):
        initializer_token_id = []
        for modifier_token_, initializer_token_ in zip(
            self.modifier_token, initializer_token
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = self.tokenizer.add_tokens(modifier_token_)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token_}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = self.tokenizer.encode(
                [initializer_token_], add_special_tokens=False
            )
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            self.modifier_token_id.append(
                self.tokenizer.convert_tokens_to_ids(modifier_token_)
            )
            initializer_token_id.append(token_ids[0])
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        for x, y in zip(self.modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

    def save_pretrained(
        self, save_path, freeze_model="crossattn_kv", save_text_encoder=False, all=False
    ):
        if all:
            super().save_pretrained(save_path)
        else:
            delta_dict = {"unet": {}, "modifier_token": {}}
            if self.modifier_token is not None:
                for i in range(len(self.modifier_token_id)):
                    learned_embeds = self.text_encoder.get_input_embeddings().weight[
                        self.modifier_token_id[i]
                    ]
                    delta_dict["modifier_token"][
                        self.modifier_token[i]
                    ] = learned_embeds.detach().cpu()
            if save_text_encoder:
                delta_dict["text_encoder"] = self.text_encoder.state_dict()
            for name, params in self.unet.named_parameters():
                if freeze_model == "crossattn":
                    if "attn2" in name:
                        delta_dict["unet"][name] = params.cpu().clone()
                elif freeze_model == "crossattn_kv":
                    if "attn2.to_k" in name or "attn2.to_v" in name:
                        delta_dict["unet"][name] = params.cpu().clone()
                else:
                    raise ValueError(
                        "freeze_model argument only supports crossattn_kv or crossattn"
                    )
            torch.save(delta_dict, save_path)

    def load_model(self, save_path, compress=False):
        st = torch.load(save_path)
        if "text_encoder" in st:
            self.text_encoder.load_state_dict(st["text_encoder"])
        if "modifier_token" in st:
            modifier_tokens = list(st["modifier_token"].keys())
            modifier_token_id = []
            for modifier_token in modifier_tokens:
                num_added_tokens = self.tokenizer.add_tokens(modifier_token)
                if num_added_tokens == 0:
                    raise ValueError(
                        f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                        " `modifier_token` that is not already in the tokenizer."
                    )
                modifier_token_id.append(
                    self.tokenizer.convert_tokens_to_ids(modifier_token)
                )
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
            for i, id_ in enumerate(modifier_token_id):
                token_embeds[id_] = st["modifier_token"][modifier_tokens[i]]

        for name, params in self.unet.named_parameters():
            if "attn2" in name:
                if compress and ("to_k" in name or "to_v" in name):
                    params.data += st["unet"][name]["u"] @ st["unet"][name]["v"]
                elif name in st["unet"]:
                    params.data.copy_(st["unet"][f"{name}"])
