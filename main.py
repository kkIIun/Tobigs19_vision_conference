from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from PIL import Image
from utils import resize_and_pad, recover_size, save_image_mask
import cv2
import argparse
import sys
sys.path.append('/home/labuser/work/tobigs/yolov7') # modify your 'yolov7' directory
from yolov7.utils.plots import plot_one_box
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--text_prompt", type=str, required=True,
        help="Text prompt",
    )

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load image and preprocessing
    org_image = np.array(Image.open(args.input_img).convert('RGB'))    
    org_image, padding_factors = resize_and_pad(org_image)
    image = org_image.transpose(2,0,1)[None,...]
    image = torch.Tensor(image/255.0).to(device)
    
    # yolov7 
    model=attempt_load('yolov7-e6e.pt', map_location=device)
    preds=model(image)[0]
    preds=preds[...,:6]   # only person class select
    preds = non_max_suppression(preds, 0.9, 0.9, classes=None, agnostic=False)[0].detach().cpu().numpy()
    plot_image = org_image.copy()
    for pred in preds:
        xyxy = pred[:4]
        plot_one_box(xyxy, plot_image)
    cv2.imwrite('./result/bounding_box_image.jpg', plot_image[:,:,::-1])

    # segment-anything
    sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth").to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(org_image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=preds[0,:4],
        multimask_output=False,
    )
    masks = masks.astype(np.uint8) * 255
    save_image_mask(org_image, masks)
    mask = masks[0]

    # stable-diffusion
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to("cuda")
    prompt = args.text_prompt
    image = pipe(prompt=prompt, image=Image.fromarray(org_image), mask_image=Image.fromarray(255-mask)).images[0]
    save_path = './result/' + prompt.replace(' ', '_') + '.png'
    image.save(save_path)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args()
    main(args)
    