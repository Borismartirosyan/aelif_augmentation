import torch
from diffusers import StableDiffusion3Pipeline
import sys
sys.path.append('/workspace/aelif/aelif_augmentation/aelif_augmentation_inference/')
from custom_sdxl_pipeline import AelifStableDiffusionXLPipeline

from transformers import T5EncoderModel, BitsAndBytesConfig
import argparse
import os

def main(args):

    if args.model_name == 'stabilityai/stable-diffusion-3-medium-diffusers':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder_3",
            quantization_config=quantization_config,
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            text_encoder_3=text_encoder,
            device_map="balanced",
            torch_dtype=torch.float16,
        )
    else:
        pipe = AelifStableDiffusionXLPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            device_map="balanced",
            torch_dtype=torch.float16,
        )
    
    
    data_path = '/workspace/aelif/dreambooth/dataset/'
    objects = [
        'backpack', 
        'candle', 
        'dog_data', 
        'cat', 
        'colorful_sneaker', 
        'dog2', 
        'dog3', 
        'backpack_dog', 
        'clock', 
        'vase', 
        'teapot'
    ]
    prompts = [
        'a photo of sks backpack',
        'a photo of sks candle',
        'a photo of sks dog',
        'a photo of sks cat',
        'a photo of sks sneaker',
        'a photo of sks dog',
        'a photo of sks dog',
        'a photo of sks backpack',
        'a photo of sks clock',
        'a photo of sks vase',
        'a photo of sks teapot',
    ]
    objects_lengths = [
        len(os.listdir(f'{data_path}{obj}/')) for obj in objects
    ]
    seeds = []
    for obj_quantity in objects_lengths:
        seeds.append([i for i in range(obj_quantity)])
    lora_base_path = '/workspace/aelif/trained-sdxl-lora-miniature-version-obj'
    
    
    res_path = '/workspace/aelif/results_images_sdxl/'
    os.makedirs(res_path, exist_ok=True)
    
    for i in range(len(objects)):
        obj_inf_path = f'{res_path}/{objects[i]}'
        os.makedirs(obj_inf_path, exist_ok=True)
        
        # creating folders for original and augmented images inference
        obj_inf_path_orig = f'{res_path}/{objects[i]}/original/'
        obj_inf_path_aug = f'{res_path}/{objects[i]}/aug/'
        os.makedirs(obj_inf_path_orig, exist_ok=True)
        os.makedirs(obj_inf_path_aug, exist_ok=True)
        
        # original image creation
        lora_orig_path = lora_base_path.replace('version', 'orig').replace('obj', objects[i])
        print(os.path.exists(lora_orig_path))
        
        # aug image creation
        lora_aug_path = f"{lora_base_path.replace('version', 'aug').replace('obj', objects[i])}"
        print(os.path.exists(lora_aug_path))
        
        #original images creation
        pipe.load_lora_weights(lora_orig_path)
        for seed in seeds[i]:
            print(f'{seed} {prompts[i]}')
            image = pipe(prompt = prompts[i], 
                         generator = torch.Generator(device='cpu').manual_seed(seed),
                         aelif_percentage=0,
                         aelif='mask',
                         std_aelif = 100).images[0]
            image.save(f"{obj_inf_path_orig}/img_{objects[i]}_seed_{seed}.png")
            print(f"{obj_inf_path_orig}/img_{objects[i]}_seed_{seed}.png")
            
        pipe.unload_lora_weights()
        
        # augmented images creation
        pipe.load_lora_weights(lora_aug_path)
        for seed in seeds[i]:
            print(f'{seed} {prompts[i]}')
            image = pipe(prompt = prompts[i], 
                         generator = torch.Generator(device='cpu').manual_seed(seed),
                         aelif_percentage=0,
                         aelif='mask',
                         std_aelif = 100).images[0]
            image.save(f"{obj_inf_path_aug}/img_{objects[i]}_seed_{seed}.png")
            print(f"{obj_inf_path_aug}/img_{objects[i]}_seed_{seed}.png")
            
            
        pipe.unload_lora_weights()
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_output_path")
    parser.add_argument("--prompt")
    parser.add_argument("--pic_output_path")
    parser.add_argument("--model_name")
    args = parser.parse_args()
    main(args)
    

