import numpy as np
from yolov7.utils.general import non_max_suppression
    
    
def add_point(image, point, size=8):
    """ Draw a small square above the image
        Since this overwrites the pixels, use only for interactive outputs on gradio.Image
    """
    H, W, C = image.shape
    new_img = np.copy(image)
    
    #coordinates of the square's corner
    x1 = max(0, (point[0] - size // 2))
    y1 = max(0, (point[1] - size // 2))
    x2 = min(W, (point[0] + size // 2))
    y2 = min(H, (point[1] + size // 2))
    
    # overwrite around the point with [255,0,0]
    new_img[y1:y2, x1:x2, 0] = 255
    new_img[y1:y2, x1:x2, 1] = 0
    new_img[y1:y2, x1:x2, 2] = 0
    
    return new_img

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
    preds = non_max_suppression(preds, 0.9, 0.9, classes=None, agnostic=False)[0].detach().cpu().numpy()
    return preds[0,:4]