import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

def resize_and_pad(image: np.ndarray, target_size: int = 512):
    height, width, _ = image.shape
    max_dim = max(height, width)
    scale = target_size / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    image_padded = np.pad(image_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    return image_padded, (top_pad, bottom_pad, left_pad, right_pad)

def recover_size(image_padded: np.ndarray, orig_size: Tuple[int, int], 
                 padding_factors: Tuple[int, int, int, int]):
    h,w,c = image_padded.shape
    top_pad, bottom_pad, left_pad, right_pad = padding_factors
    image = image_padded[top_pad:h-bottom_pad, left_pad:w-right_pad, :]
    image_resized = cv2.resize(image, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    return image_resized

def save_image_mask(image, masks):
    image = image/255.0
    masks = masks/255.0
    total_masks = None
    sorted_masks = sorted(masks, key=(lambda x: np.sum(x)), reverse=True)
    for j, mask in enumerate(sorted_masks):
        m = mask
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        if total_masks is None:
            total_masks = img*m[...,np.newaxis]
        else:
            total_masks[np.where(m>0)] = img[np.where(m>0)]
    result = total_masks*0.5 + image*0.5
    plt.imsave('./result/mask_image.png', result)
