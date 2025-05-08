# AELIF augmentation for enhancement of SD3 and SDXL models

This github page is for usage of AELIF augmentation inference and training for SD3 and SDXL models.
AELIF uses embedding noise convolution and embedding masking algorithm, to enhance the robustness of SD3 and SDXL models. Also, it is very useful as a data augmentation.

1. Firstly, configure Docker envoirement with given Dockerfile or use it to configure your python envoirment
2. To see the results using the prompt.json, you can use main.py
3. To train both models with aelif, feel free to use pipeline.py
4. To see the results as a data augmentations and compare generated images with/without aelif, please use /notebooks/grids.ipynb
5. To see both grids resulted images distance from training data, please see /notebooks/infer_analytics.ipynb
6. To create full comparison analysis between images genered with/without AELIF, with adversarial prompts, please see infer_with_prompts.py
