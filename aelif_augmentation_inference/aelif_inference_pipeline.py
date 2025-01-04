import torch
from custom_sd3_pipeline import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import clip
from schemas import AugmentationModel
from PIL import Image
from json.decoder import JSONDecodeError
from typing import List
import time
import json
import sys
sys.path.append('../aelif_augmentation')
sys.path.append('../aelif_augmentation/aelif_augmentation_inference')
from aelif_augmentation_inference.custom_sd3_pipeline import StableDiffusion3Pipeline

class AELIFAugmentationPipeline:

    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                       subfolder="text_encoder_3", device_map="balanced"):
        print('Loading Stable Diffusion 3 pipeline')
        try:
            self.text_encoder = T5EncoderModel.from_pretrained(
                model_id,
                subfolder=subfolder,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            )
        except Exception as e:
            print(f"Error loading text encoder: {e}")
            raise

        try:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                text_encoder_3=self.text_encoder,
                device_map=device_map,
                torch_dtype=torch.float16
            )
        except Exception as e:
            print(f"Error loading Stable Diffusion pipeline: {e}")
            raise

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise

    def augmentation_trial_function(self,
                                    prompt_file_path: str,
                                    standard_deviations: List[float],
                                    augmentation_types: AugmentationModel,
                                    save_path: str,
                                    img_height: int = 768,
                                    img_width: int = 768,
                                    num_inference_steps: int = 28,
                                    guidance_scale: float = 7.0,
                                    seed: int = 0) -> dict:
        augmentations = []
        if augmentation_types.noise_conv:
            augmentations.append('noise_conv')
        if augmentation_types.mask:
            augmentations.append('mask')

        if not augmentations:
            raise ValueError("No augmentations selected")

        try:
            with open(prompt_file_path, 'r') as file:
                prompts = json.load(file)
        except JSONDecodeError:
            raise ValueError("Invalid JSON format in the prompt file.")
        except FileNotFoundError:
            raise ValueError("Prompt file not found.")
        except Exception as e:
            raise ValueError(f"Error reading prompt file: {e}")

        results = {aug_type: {} for aug_type in augmentations}
        image_dict = {}
        embed_dict = {}


        generator = torch.Generator(device='cpu').manual_seed(seed)  

        for aug_type in augmentations:
            if aug_type == 'mask':
                for key, prompt in prompts.items():
                  #creating original image
                  orig_image_key = f'aug_type_{aug_type}_prmpt_{key}_aug_magnitude_{0}'
                  orig_img = self.pipe(prompt=prompt,
                                       num_inference_steps=num_inference_steps,
                                       height=img_height,
                                       width=img_width,
                                       guidance_scale=guidance_scale,
                                       generator=generator,
                                       aelif=aug_type,
                                       aelif_percentage=0,
                                       ).images[0]
                  image_dict[orig_image_key] = orig_img
                  orig_img.save(f'res/{orig_image_key}.png')
                  for aug_magnitude in range(10, 110, 10):
                        image_key = f'aug_type_{aug_type}_prmpt_{key}_aug_magnitude_{aug_magnitude}'
                        # Generate original image if not already generated

                        start = time.time()

                        img = self.pipe(
                            prompt=prompt,
                            num_inference_steps=num_inference_steps,
                            height=img_height,
                            width=img_width,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            aelif=aug_type,
                            aelif_percentage=aug_magnitude / 100,
                        ).images[0]

                        image_dict[image_key] = img
                        img.save(f'res/{image_key}.png')

                        similarity, embed_dict = self.similarity_score_with_orig_image(
                            orig_image=image_dict[orig_image_key],
                            augmented_image=img,
                            embed_dict=embed_dict,
                            prompt=prompt,
                            model=self.model,
                            preprocess=self.preprocess,
                            device=self.device
                        )

                        results[aug_type][image_key] = similarity

                        end = time.time()
                        print(f'Augmentation_type: {aug_type}, {image_key}, time: {end - start:.2f}s')

                        with open(save_path, 'w') as file:
                            file.write(json.dumps(results))

            elif aug_type == 'noise_conv':
                for key, prompt in prompts.items():
                  orig_image_key = f'aug_type_{aug_type}_prmpt_{key}_aug_magnitude_{0}'
                  orig_img = self.pipe(
                                prompt=prompt,
                                num_inference_steps=num_inference_steps,
                                height=img_height,
                                width=img_width,
                                guidance_scale=guidance_scale,
                                generator=generator,
                                aelif=aug_type,
                                aelif_percentage=0
                            ).images[0]
                  image_dict[orig_image_key] = orig_img
                  orig_img.save(f'res/{orig_image_key}.png')
                  for std in standard_deviations:
                    for aug_magnitude in range(10, 110, 10):
                        image_key = f'aug_type_{aug_type}_prmpt_{key}_std_{std}_aug_magnitude_{aug_magnitude}'
                        # Generate original image if not already generated
                        start = time.time()

                        img = self.pipe(
                            prompt=prompt,
                            num_inference_steps=num_inference_steps,
                            height=img_height,
                            width=img_width,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            aelif=aug_type,
                            aelif_percentage=aug_magnitude / 100,
                            mean_aelif = std
                        ).images[0]

                        image_dict[image_key] = img
                        img.save(f'res/{image_key}.png')
                        similarity, embed_dict = self.similarity_score_with_orig_image(
                            orig_image=image_dict[orig_image_key],
                            augmented_image=img,
                            embed_dict=embed_dict,
                            prompt=prompt,
                            model=self.model,
                            preprocess=self.preprocess,
                            device=self.device
                        )

                        results[aug_type][image_key] = similarity

                        end = time.time()
                        print(f'Augmentation_type: {aug_type}, {image_key}, time: {end - start:.2f}s')

                        with open(save_path, 'w') as file:
                            file.write(json.dumps(results))

        return results

    def similarity_score_with_orig_image(self,
                                         orig_image: Image.Image,
                                         augmented_image: Image.Image,
                                         embed_dict: dict,
                                         prompt: str,
                                         model,
                                         preprocess,
                                         device: torch.device,
                                         cos=torch.nn.CosineSimilarity(dim=1)):
        if prompt in embed_dict:
            orig_emb = embed_dict[prompt]
        else:
            orig_emb = model.encode_image(preprocess(orig_image).unsqueeze(0).to(device))
            embed_dict[prompt] = orig_emb

        aug_emb = model.encode_image(preprocess(augmented_image).unsqueeze(0).to(device))

        similarity = cos(orig_emb, aug_emb).item()
        similarity = (similarity + 1) / 2  # Normalize to [0,1]

        return similarity, embed_dict
