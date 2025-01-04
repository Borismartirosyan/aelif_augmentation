from pydantic import BaseModel

class AugmentationModel(BaseModel):
    noise_conv: bool
    mask: bool