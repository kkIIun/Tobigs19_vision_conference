import numpy as np
from .utils import recover_size
from yolov7.utils.general import non_max_suppression 

def predict_mask_with_sam(predictor, image, box):
    predictor.set_image(image)
    
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False,
    )
    predictor.reset_image()
    return masks[0].astype(np.uint8) * 255

def predict_box_with_yolo(yolo, image_tensor):
    preds=yolo(image_tensor)[0]
    preds=preds[...,:6]   # only person class select
    preds = non_max_suppression(preds, 0.5, 0.5, classes=None, agnostic=False)[0].detach().cpu().numpy()
    return preds[0,:4]
