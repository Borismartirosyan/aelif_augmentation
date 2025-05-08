import os
import subprocess
import sys

general_path = '/workspace/aelif/'
dreambooth_dataset_path = general_path + 'dreambooth/dataset/'
items = [
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

for i in range(len(items)):
    
    #compute embeddings of original dreambooth
    prompt = prompts[i]
    output_path = general_path + f'{items[i]}_embeddings_aug.parquet'
    local_data_dir = dreambooth_dataset_path + items[i] + '/'
    print(os.listdir(local_data_dir))
    
    command = f""" 
    python3 * /workspace/aelif/diffusers/examples/research_projects/sd3_lora_colab/compute_embeddings.py * --prompt * {prompt} * --output_path *  {output_path} * --local_data_dir * {local_data_dir} * --aelif_augment_magnitude * 0
    """.replace('\n', '').strip().split('*')
    command = [i.strip() for i in command]
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, text=True)
    process.wait()
    
    dreambooth_output_path = general_path + f'trained-sdxl-lora-miniature-aug-{items[i]}-mask'
    command_run_dreambooth = f"""
    accelerate*launch*/workspace/aelif/diffusers/examples/research_projects/sd3_lora_colab/train_dreambooth_lora_sd3_miniature.py*
      --pretrained_model_name_or_path*stabilityai/stable-diffusion-3-medium-diffusers*
      --instance_data_dir*{local_data_dir}*
      --data_df_path*{output_path}*
      --output_dir*{dreambooth_output_path}*
      --mixed_precision*fp16*
      --instance_prompt*{prompt}*
      --resolution*1024*
      --train_batch_size*1*
      --gradient_accumulation_steps*4*
      --gradient_checkpointing*
      --use_8bit_adam*
      --learning_rate*1e-4*
      --lr_scheduler*constant*
      --lr_warmup_steps*0*
      --max_train_steps*700*
      --seed*0
    """.replace('\n', '').strip().split('*')
    
    command_run_dreambooth = f"""
        accelerate*launch*/workspace/aelif/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py*
          --pretrained_model_name_or_path*stabilityai/stable-diffusion-xl-base-1.0*
          --instance_data_dir*{local_data_dir}*
          --output_dir*{dreambooth_output_path}*
          --mixed_precision*fp16*
          --instance_prompt*{prompt}*
          --resolution*1024*
          --train_batch_size*1*
          --gradient_accumulation_steps*4*
          --learning_rate*1e-4*
          --use_8bit_adam*
          --lr_scheduler*constant*
          --lr_warmup_steps*0*
          --max_train_steps*700*
          --seed*0
    """.replace('\n', '').strip().split('*')
    command_run_dreambooth = f"""
    accelerate*launch*/workspace/aelif/diffusers/examples/dreambooth/train_dreambooth_sdxl_lora_aelif.py*
          --pretrained_model_name_or_path*stabilityai/stable-diffusion-xl-base-1.0*
          --instance_data_dir*{local_data_dir}* 
          --output_dir*{dreambooth_output_path}* 
          --mixed_precision*fp16* 
          --instance_prompt*{prompt}* 
          --resolution*1024* 
          --train_batch_size*1* 
          --gradient_accumulation_steps*4* 
          --learning_rate*1e-4* 
          --use_8bit_adam* 
          --lr_scheduler*constant* 
          --lr_warmup_steps*0* 
          --max_train_steps*700* 
          --seed*0* 
          --aelif*mask*
          --aelif_percentage*0.1*
          --std_aelif*100.0*
          --mean_aelif*0.0  
    """.replace('\n', '').strip().split('*')
    
    command_run_dreambooth = [i.strip() for i in command_run_dreambooth]
    print(command_run_dreambooth)
    process = subprocess.Popen(command_run_dreambooth, stdout=sys.stdout, stderr=sys.stderr, text=True)
    process.wait()
    
