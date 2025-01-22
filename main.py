from omegaconf import OmegaConf
import sys
import os
sys.path.append('../aelif_augmentation')
sys.path.append('../aelif_augmentation/aelif_augmentation_inference')
from aelif_augmentation_inference.aelif_inference_pipeline import AELIFAugmentationPipeline
from aelif_augmentation_inference.schemas import AugmentationModel
config = OmegaConf.load("config.yaml")

res_dir = "res"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

def main():
    config = OmegaConf.load("config.yaml")
    pipe  = AELIFAugmentationPipeline()
    types = AugmentationModel(
        **config.augmentation_types
    )
    pipe.augmentation_trial_function(
        prompt_file_path=config.prompts_file_path,
        standard_deviations=config.standard_deviations,
        augmentation_types=types,
        save_path=config.save_path,
        img_height = config.img_height,
        img_width= config.img_width,
        num_inference_steps = config.num_inference_steps,
        guidance_scale = config.guidance_scale
    )
    

if __name__ == "__main__":
    main()




