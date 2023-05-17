from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from PIL import Image

# 이미지 전처리 작업을 정의
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=0.5,std=0.5)
])

def save_total_anns(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    total_masks = None
    for j, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        if total_masks is None:
            total_masks = img
        else:
            total_masks[np.where(m>0)] = img[np.where(m>0)]
        # ax.imshow(np.dstack((img, m*0.35)))
    result = total_masks*0.35 + image*0.65
    plt.imsave('total_mask.png', result)

def save_anns(masks, image):
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
    plt.imsave('sub_mask.png', result)

def main():
    image = np.array(Image.open('./test.jpeg').convert('RGB').resize((512,512)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth").to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_point = np.array([[250, 250], [250, 400], [220, 170], [100, 100], [100, 400], [450, 100], [480, 480]])
    input_label = np.array([1, 1, 1, 0, 0, 0 ,0])
    # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(image)
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    # masks = ~masks
    image = image.astype(np.float32)/255.0
    save_anns(masks, image)
    image = preprocess(image)
    masks = torch.Tensor(masks)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    )
    pipe.to("cuda")
    prompt = "A handsome man is smiling on the railroad in winter."
    save_image(masks, 'mask.png')
    image = pipe(prompt=prompt, image=image, mask_image=masks).images[0]
    save_path = './' + prompt.replace(' ', '_') + '.png'
    image.save(save_path)

if __name__ = "__main__":
    main()