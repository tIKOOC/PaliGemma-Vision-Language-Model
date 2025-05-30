from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
def process_images(
    images: List[Image.Image],
    size: Dict[str, int],
    resample: Image.Resampling = None,
    resacle_factor : float = None,
    image_mean : Optional[ Union[float, List[float]]] = None,
    image_std : Optional[ Union[float, List[float]]] = None,
    

    ) -> np.ndarray:
    height, width = size
    images = [
        resize(image = image, size = (height, width), resample = resample) for image in images
    ] 
    images = [np.array(image) for image in images]
    images = [rescale(image = image, scale = rescale_factor) for image in images]
    images = [normalize(image = image, mean = image_mean, std = image_std) for image in images]
    images = [image.transpose(2, 0, 1) for image in images]
    return np.stack(images, axis = 0)


def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32)-> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image 


def normalize(image: np.ndarray, mean: Union[float, List[float]], std: Union[float, List[float]]) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
)-> np.ndarray:
    height, width = size
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    return resized_image



class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_size = image_size
        tokens_to_add = {"additional_special_tokens" : {self.IMAGE_TOKEN}}
        tokenizer.add_special_tokens(tokens_to_add)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)[0]
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]
        EXTRA_TOKENS = [
            f"<seg{i:03d}>" for i in range(128)
        ]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer
    def __call__(self, images: List[Image.Image], text: List[str], padding: str = "longest",truncation: bool = True, **kwargs)-> dict:
        assert len(images) == 1 and len(text) == 1,f"Recceived {len(images)} images for {len(text)} prompts."
        pixel_value = process_images(
            images,
            size =(self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,
        )
        pixel_value = np.stack(pixel_value, axis = 0)
        pixel_value = torch.tensor(pixel_value)
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image-seq_length,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        inputs = self.tokenizer(
            input_strings,
            return_tensors = "pt",
            padding = padding,
            truncation = truncation,

        )
        return_data = {
            "pixel_value" : pixel_value,
            **inputs,
        }
        return return_data