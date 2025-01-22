import torch
from custom_sd3_pipeline import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
import clip
from .schemas import AugmentationModel
from PIL import Image
from json.decoder import JSONDecodeError
from typing import List
import time
import json
import sys
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
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
    
    def get_image_or_text_embeddings(self, img, text):

      if img is not None:
        img_emb = self.model.encode_image(
            self.preprocess(
                img
                ).unsqueeze(0).to(self.device))
        return img_emb
      elif text is not None:
        text_tokens = clip.tokenize(text, truncate=True).to(self.device)
        text_embeddings = self.model.encode_text(text_tokens)

        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings
      else:
        raise ValueError("Either image or text must be provided")
        
    def augmentation_trial_function(self,
                                    prompt_file_path: str,
                                    standard_deviations: List[float],
                                    augmentation_types: AugmentationModel,
                                    save_path: str,
                                    distance_metric : str = 'wasserstain',
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
                            device=self.device,
                            distance_metric=distance_metric
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
                            device=self.device,
                            distance_metric=distance_metric
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
                                         distance_metric: str):
        if prompt in embed_dict:
            orig_emb = embed_dict[prompt]
        else:
            orig_emb = model.encode_image(preprocess(orig_image).unsqueeze(0).to(device))
            embed_dict[prompt] = orig_emb

        aug_emb = model.encode_image(preprocess(augmented_image).unsqueeze(0).to(device))
        
        if distance_metric == 'cosine':
            similarity = self.compute_consine_similarity_on_vectors(orig_emb, aug_emb).item()
        elif distance_metric == 'wasserstain':
            similarity = self.compute_wasserstein_distance_on_vectors(orig_emb, aug_emb).item()
        
        return similarity, embed_dict
    
    def calculate_distance_between_distributions(
        self,
        seed_array: List[int],
        prompt: str,
        augmentation_type: str,
        prompt_key: int,
        augmentation_percentage: float,
        distance_function : str = 'wasserstain',
        std_aelif: float = None,
        mean_aelif: float = None,
        height: int = 768,
        width: int = 768,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28
    ):
        # TODO: implement for mask

        original_image_embeddings = {}
        augmented_image_embeddings = {}

        for seed in seed_array:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                generator=generator,
                aelif=augmentation_type,
                aelif_percentage=0
            ).images[0]
            img_embeddings = self.get_image_or_text_embeddings(img=image, text=None)
            img_name = f'prompt_{prompt_key}_seed_{seed}_original'
            original_image_embeddings[img_name] = img_embeddings
            image.save(f'res/{img_name}.png')

        for seed in seed_array:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                generator=generator,
                aelif=augmentation_type,
                aelif_percentage=augmentation_percentage,
                std_aelif=std_aelif,
                mean_aelif=mean_aelif
            ).images[0]
            img_embeddings = self.get_image_or_text_embeddings(img=image, text=None)
            img_name = f'prompt_{prompt_key}_seed_{seed}_augmented'
            augmented_image_embeddings[img_name] = img_embeddings
            image.save(f'res/{img_name}.png')

        orig_dist = torch.stack(list(original_image_embeddings.values()))
        augmented_dist = torch.stack(list(augmented_image_embeddings.values()))
        
        if distance_function == "wasserstain":
            dist = self.compute_wasserstein_distance_on_matrices(orig_dist, augmented_dist)
        elif distance_funtion == "cosine":
            dist = self.cosine_similarity_on_matrix_rowwise(orig_dist, augmented_dist)
        return dist


    def cosine_similarity_on_matrix_rowwise(self, P, Q):

        P = P.squeeze(1)
        Q = Q.squeeze(1)

        dot_products = torch.sum(P * Q, dim=1)

        # Compute the norms for each row in P and Q
        row_norms_P = torch.norm(P, dim=1)
        row_norms_Q = torch.norm(Q, dim=1)

        # Compute cosine similarity with safe division
        cosine_similarities = dot_products / (row_norms_P * row_norms_Q + 1e-8)

        return cosine_similarities
    
    def compute_consine_similarity_on_vectors(self, P, Q):
        dot_product = torch.sum(P * Q)
        norm_P = torch.norm(P)
        norm_Q = torch.norm(Q)
        cosine_similarity = dot_product / (norm_P * norm_Q)
        return cosine_similarity
    
    def compute_wasserstein_distance_on_matrices(self, M: torch.Tensor, N: torch.Tensor) -> float:
        
        M = M.squeeze(1)
        N = N.squeeze(1)
        
        if M.shape != N.shape:
            raise ValueError("M and N must have the same shape.")

        # Ensure M and N are in numpy format for compatibility with scipy
        M_np = M.detach().cpu().numpy()
        N_np = N.detach().cpu().numpy()

        # Calculate the Wasserstein distance in n-dimensional space
        distance = wasserstein_distance_nd(M_np, N_np)

        return distance
    
    def compute_wasserstein_distance_on_vectors(self, v1: torch.Tensor, v2: torch.Tensor) -> float:

        v1 = v1.squeeze()
        v2 = v2.squeeze()

        if v1.ndim != 1 or v2.ndim != 1:
            raise ValueError("v1 and v2 must be 1-dimensional tensors (vectors).")

        if v1.shape != v2.shape:
            raise ValueError("v1 and v2 must have the same shape.")

        # Convert tensors to numpy arrays
        v1_np = v1.detach().cpu().numpy()
        v2_np = v2.detach().cpu().numpy()

        # Compute the Wasserstein distance
        distance = wasserstein_distance(v1_np, v2_np)

        return distance