import os
import torch
import json
from torchvision.utils import save_image
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from torchvision.io import read_image

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import (
    MutualSelfAttentionControlMaskAuto_Matching,
)
from diffusers import DDIMScheduler


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.0  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def main(params, p_idx):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set prompts
    prompts = [
        params["ref_prompt"].replace("<", "\u003C"),
        params["gen_prompt"].replace("<", "\u003C"),
    ]  # @param {type:"string"}
    print(f"REF PROMPT : {prompts[0]} \nGEN PROMPT : {prompts[1]}")

    # load source image
    SOURCE_IMAGE_PATH = params["invert_path"]
    source_image = load_image(SOURCE_IMAGE_PATH, device)

    # Set the output directories
    prompt_out_dir = os.path.join(params["out_dir"], f"output/ours/prompt_{p_idx}")
    os.makedirs(prompt_out_dir, exist_ok=True)  # output directory
    mask_dir = None

    scheduler = DDIMScheduler.from_pretrained(
    params["concept_dir"], subfolder="scheduler"
)

    model = MasaCtrlPipeline.from_pretrained(
        params["concept_dir"],
        scheduler=scheduler,
        safety_checker=None
    ).to(device)

    seed_everything(params["seed"])
    start_code = model.invert(
        source_image,
        "",
        guidance_scale=params["guidance_scale"],
        num_inference_steps=params["inference_steps"],
        return_intermediates=True,
    )

    num_samples = params.get("num_samples", 1)

    # random end_code for diverse target image synthesis
    end_code = torch.randn([num_samples, 4, 64, 64], device=device, dtype=dtype)

    for sampling_idx in range(num_samples):
        print(f"[Ours] Sampling.. ({sampling_idx + 1} / {num_samples})")
       
        editor = MutualSelfAttentionControlMaskAuto_Matching(
            params, mask_save_dir=mask_dir, save_dir=prompt_out_dir
        )
        regiter_attention_editor_diffusers(model, editor)
        initial_code = torch.cat(
            [start_code, end_code[sampling_idx].unsqueeze(0)], dim=0
        )
        model_output = model(
            prompts,
            latents=initial_code,
            num_inference_steps=params["inference_steps"],
            guidance_scale=params["guidance_scale"],
            outdir=prompt_out_dir,
            save_pred_x0=params["save_pred_x0"],
        )
        out_path = os.path.join(
            params["out_dir"],
            "output",
            "ours",
            f"prompt_{p_idx}",
            f"{sampling_idx}.jpg",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_image(model_output[1], out_path)

    with open(
        os.path.join(params["out_dir"], f"prompt_{p_idx}_params.json"), "w"
    ) as file:
        json.dump(params, file, indent=4)


if __name__ == "__main__":
    total_concepts = [
        ("cat_statue", "cat-toy", False),
        ("elephant_statue", "elephant_statue", False),
        ("duck_toy", "duck_toy", False),
        ("monster_toy", "monster_toy", False),
        ("brown_teddybear", "teddybear", False),
        ("tortoise_plushy", "tortoise_plushy", False),
        ("brown_dog", "brown_dog", True),
        ("fat_dog", "fat_dog", True),
        ("brown_dog2", "brown_dog2", True),
        ("black_cat", "black_cat", True),
        ("brown_cat", "brown_cat", True),
        ("alarm_clock", "clock", False),
        ("pink_sunglasses", "pink_sunglasses", False),
        ("red_teapot", "red_teapot", False),
        ("red_vase", "vase", False),
        ("wooden_barn", "barn", False),
    ]
    torch.cuda.set_device(0)  # set the GPU device

    for concept, concept_orig_name, is_live in total_concepts:
        if is_live:
            with open("./input_imgs/prompts_live_objects.txt", "r") as fin:
                prompts = fin.read()
                prompts = prompts.split("\n")
        else:
            with open("./input_imgs/prompts_nonlive_objects.txt", "r") as fin:
                prompts = fin.read()
                prompts = prompts.split("\n")

        concept_model_name = f"{concept}"
        num_samples = 8
        method = "dreambooth"

        assert method in [
            "ti",
            "dreambooth",
            "custom_diffusion",
        ], f"{method} was not implemented"

   
        concept_dir = f"./concept_models/{concept_model_name}/checkpoints/diffusers"
        out_dir = (
            f"./results/{concept_model_name}"
        )
     
        for p_idx, prompt in enumerate(prompts):
            if method == "ti":
                ref_prompt = f"a photo of a <{concept}>"
                ref_token_idx = 5
                gen_prompt = prompt.format(f"<{concept}>")
                cur_token_idx = (
                    gen_prompt.split(" ").index(f"<{concept}>") + 1
                )  
            elif method == "dreambooth":
                ref_prompt = f"a photo of a sks {concept}"
                ref_token_idx = 6
                gen_prompt = prompt.format(f"sks {concept}")
                cur_token_idx = (
                    gen_prompt.split(" ").index(f"{concept}") + 1
                )  
            elif method == "custom_diffusion":
                ref_prompt = f"a photo of a <new1> {concept}"
                ref_token_idx = 6
                gen_prompt = prompt.format(f"<new1> {concept}")
                cur_token_idx = (
                    gen_prompt.split(" ").index(f"{concept}") + 1
                ) 
         
            with open("./input_imgs/imgs_source.txt", "r") as file:
                src_imgs_dict = json.load(file)
                invert_path = os.path.join(
                    f"./input_imgs/{concept_orig_name}",
                    src_imgs_dict[f"{concept_orig_name}"],
                )

            params = {
                "seed": 0,
                "ref_prompt": ref_prompt,
                "ref_token_idx": ref_token_idx, 
                "gen_prompt": gen_prompt,
                "cur_token_idx": cur_token_idx,  
                "concept_dir": concept_dir,
                "out_dir": out_dir,
                "invert_path": invert_path, 
                "num_samples": num_samples,
                "guidance_scale": 7.5,
                "inference_steps": 20,
                "initial_step": 8,  
                "initial_layer": 12, 
                "cut_step": 50, 
                "cut_layer": 16,  
                "cc_thres": 1, 
                "mode": "sd"
            } 
            main(params, p_idx)
