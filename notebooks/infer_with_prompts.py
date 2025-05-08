
"""
Adversarial‑prompt robustness check
-----------------------------------
* Generates adversarial prompts via OpenAI (GPT‑4o)
* Produces images with three LoRA checkpoints
* Embeds images with CLIP and measures Wasserstein distance
* Saves every generated image in an organised folder tree
* Logs distances to console **and** run.log
* Uses a different reference image for every class (key)
"""

import os, re, json, logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
from PIL import Image
from scipy.stats import wasserstein_distance

from openai import OpenAI
import clip
from transformers import T5EncoderModel, BitsAndBytesConfig
from aelif_augmentation_inference.custom_sd3_pipeline import StableDiffusion3Pipeline #AelifStableDiffusionXLPipeline #

os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"

MODEL_ID           = "stabilityai/stable-diffusion-3-medium-diffusers" # "stabilityai/stable-diffusion-xl-base-1.0"
HF_CACHE_DIR       = "/home/jupyter/cache/huggingface/hub"
LORA_BASE_DIR      = "/home/jupyter/aelif_augmentation/trained_lora_checkpoints"
OUTPUT_ROOT        = Path("generated_images_sd3")
LOG_FILE           = "run.log"
OPENAI_MODEL       = "gpt-4o"
SEED               = 3
NUM_INFER_STEPS    = 25
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# one reference image per class (key)  ↓  replace with real files
ref_image_paths = {
    1:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/backpack/02.jpg",
    2:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/backpack_dog/04.jpg",
    3:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/candle/01.jpg",
    4:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/cat/01.jpg",
    5:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/clock/03.jpg",
    6:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/colorful_sneaker/04.jpg",
    7:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/dog2/02.jpg",
    8:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/dog3/02.jpg",
    9:  "/home/jupyter/aelif_augmentation/dreambooth/dataset/dog_data/alvan-nee-9M0tSjb-cpA-unsplash.jpeg",
    10: "/home/jupyter/aelif_augmentation/dreambooth/dataset/teapot/01.jpg",
    11: "/home/jupyter/aelif_augmentation/dreambooth/dataset/vase/03.jpg",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

triplets_of_paths = {
    1:  ["trained-sd3-lora-miniature-orig-backpack",
         "trained-sd3-lora-miniature-aug-backpack-mask",
         "trained-sd3-lora-miniature-aug-backpack_dog"],
    2:  ["trained-sd3-lora-miniature-orig-backpack_dog",
         "trained-sd3-lora-miniature-aug-backpack_dog-mask",
         "trained-sd3-lora-miniature-aug-backpack_dog"],
    3:  ["trained-sd3-lora-miniature-orig-candle",
         "trained-sd3-lora-miniature-aug-candle-mask",
         "trained-sd3-lora-miniature-aug-candle"],
    4:  ["trained-sd3-lora-miniature-orig-cat",
         "trained-sd3-lora-miniature-aug-cat-mask",
         "trained-sd3-lora-miniature-aug-cat"],
    5:  ["trained-sd3-lora-miniature-orig-clock",
         "trained-sd3-lora-miniature-aug-clock-mask",
         "trained-sd3-lora-miniature-aug-clock"],
    6:  ["trained-sd3-lora-miniature-orig-colorful_sneaker",
         "trained-sd3-lora-miniature-aug-colorful_sneaker-mask",
         "trained-sd3-lora-miniature-aug-colorful_sneaker"],
    7:  ["trained-sd3-lora-miniature-orig-dog2",
         "trained-sd3-lora-miniature-aug-dog2-mask",
         "trained-sd3-lora-miniature-aug-dog2"],
    8:  ["trained-sd3-lora-miniature-orig-dog3",
         "trained-sd3-lora-miniature-aug-dog3-mask",
         "trained-sd3-lora-miniature-aug-dog3"],
    9:  ["trained-sd3-lora-miniature-orig-dog_data",
         "trained-sd3-lora-miniature-aug-dog_data-mask",
         "trained-sd3-lora-miniature-aug-dog_data"],
    10: ["trained-sd3-lora-miniature-orig-teapot",
         "trained-sd3-lora-miniature-aug-teapot-mask",
         "trained-sd3-lora-miniature-aug-teapot"],
    11: ["trained-sd3-lora-miniature-orig-vase",
         "trained-sd3-lora-miniature-aug-vase-mask",
         "trained-sd3-lora-miniature-aug-vase"],
}

main_prompts = {
    1:  "a photo of sks backpack",
    2:  "a photo of sks backpack",
    3:  "a photo of sks candle",
    4:  "a photo of sks cat",
    5:  "a photo of sks clock",
    6:  "a photo of sks sneaker",
    7:  "a photo of sks dog",
    8:  "a photo of sks dog",
    9:  "a photo of sks dog",
    10: "a photo of sks teapot",
    11: "a photo of sks vase",
}

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())[:100]

def prompt_generator(prompt: str,
                     model: str = OPENAI_MODEL,
                     max_retry: int = 10) -> list[str]:
    system_message = """
    You are a system that generates adversarial prompts with a given format. I will give you 
    a prompt and a number, you should generate adversarial versions of it. The adversarial 
    prompts should not have additional words included from your side, you just need to 
    'distort' the prompt as usual human can do. Mix up several letters, drop several letters,
    and do it as natural as you can. I am doing this, to create prompt to Stable Diffusion model,
    to check the model's robustness. The models are trained with dreambooth, so given prompt will have some rare
    token, do not distort it, only distort surrounding words. The distortions shouldn't be more than 10 percent of the 
    prompt. Here are distortion examples. Never touch rare token please.

    example 1: given prompt - 'a photo of ggg human', distorted_prompts 'a phto of gg hman'
    example 2: given prompt - 'a picture of hgj cactus', ' a pic of hgj cactus'
    No open ended answers or texts in answer
    Always give answer in a valid json format like this:
    {"prompts" : ["prompt_1", "prompt_2" ...]}

    """
    client = OpenAI()
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content
            print(raw)
            raw = raw.replace("```", "").replace("json", "")
            return json.loads(raw)["prompts"]
        except Exception as exc:
            if attempt == max_retry - 1:
                raise RuntimeError(f"OpenAI failed after {max_retry} tries.") from exc

os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR

quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
text_encoder = T5EncoderModel.from_pretrained(
    MODEL_ID, subfolder="text_encoder_3", quantization_config=quant_cfg
)
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID, device_map="balanced", torch_dtype=torch.float16, text_encoder_3=text_encoder
)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
torch.manual_seed(SEED)

def gen_image_infer(prompt: str) -> Image.Image:
    return pipe(
        prompt=prompt,
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        aelif_percentage=0,
        aelif="mask",
        num_inference_steps=NUM_INFER_STEPS,
        std_aelif=100,
    ).images[0]

def encode_imgs(imgs: list[Image.Image]) -> torch.Tensor:
    batch = torch.stack([clip_preprocess(im) for im in imgs]).to(DEVICE)
    with torch.no_grad():
        return clip_model.encode_image(batch).float()

def w_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(wasserstein_distance(a.cpu().numpy(), b.cpu().numpy()))

def pipeline_of_lora() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)

    outer = tqdm(main_prompts.items(), desc="Main prompts", total=len(main_prompts))

    for key, prompt in outer:
        # reference image specific to this class
        ref_img   = Image.open(ref_image_paths[key]).convert("RGB")
        ref_embed = encode_imgs([ref_img]).squeeze()

        main_dir = OUTPUT_ROOT / f"{key}_{_slug(prompt)}"
        main_dir.mkdir(exist_ok=True)

        # adversarial prompts
        adv_prompts = prompt_generator(f"{prompt}")

        # LoRA checkpoint sets
        variant_imgs: dict[str, list[Image.Image]] = defaultdict(list)
        ckpt_names = ["orig", "mask", "noise"]
        ckpt_paths = triplets_of_paths[key]

        middle = tqdm(zip(ckpt_names, ckpt_paths),
                      desc=f"LoRA sets for class {key}",
                      total=3,
                      leave=False)

        for v_name, ckpt in middle:
            pipe.load_lora_weights(Path(LORA_BASE_DIR) / ckpt)

            inner = tqdm(adv_prompts,
                         desc=f"  {v_name} images",
                         total=len(adv_prompts),
                         leave=False)

            for adv in inner:
                img = gen_image_infer(adv)
                variant_imgs[v_name].append(img)

                # save image immediately
                adv_dir = main_dir / _slug(adv)
                adv_dir.mkdir(parents=True, exist_ok=True)
                img.save(adv_dir / f"{v_name}.png")

            pipe.unload_lora_weights()


        embeds = {k: encode_imgs(v) for k, v in variant_imgs.items()}
        for i, adv in enumerate(adv_prompts):
            log_lines = [f"[class {key}] prompt: “{adv}”"]
            for v in ckpt_names:
                dist = w_distance(ref_embed, embeds[v][i].squeeze())
                log_lines.append(f"  {v:5s} → W‑dist = {dist:.4f}")
            log.info("\n".join(log_lines))

if __name__ == "__main__":
    pipeline_of_lora()
