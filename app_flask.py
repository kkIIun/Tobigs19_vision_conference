import sys
sys.path.append('./yolov7') # modify your 'yolov7' directory
import argparse
from PIL import Image
import numpy as np
import torch
from yolov7.models.experimental import attempt_load
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline
from flask import Flask, jsonify, request

from utils import resize_and_pad, recover_size
from utils.flask import numpy_to_b64, b64_to_numpy, build_preflight_response, build_actual_response
from utils.functions import predict_box_with_yolo, predict_mask_with_sam

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.

@app.route('/image', methods=['OPTIONS', 'POST'])
def inpainting(): 
    with torch.no_grad():
        if request.method == 'OPTIONS':
            return build_preflight_response()
        elif request.method == 'POST':
            dict_data = request.get_json()        
            text_prompt = dict_data['prompt']
            # Convert b64 string to PIL Image
            b64_image = dict_data['image']
            input_image = b64_to_numpy(b64_image)
            
            # Preprocess for yolov7
            image_padded, padding_factors = resize_and_pad(input_image)
            image_tensor = image_padded.transpose(2,0,1)[None,...]
            image_tensor = torch.Tensor(image_tensor/255.0).to(device)
            # Detect human with yolov7
            box = predict_box_with_yolo(yolo, image_tensor.to(device))
            
            # Segmentation with SAM
            mask = predict_mask_with_sam(sam_predictor, image_padded, box)
            
            # Inpainting with stable diffusion
            image_inpainted = pipe(prompt=text_prompt, image=Image.fromarray(image_padded), mask_image=Image.fromarray(255-mask)).images[0]
            
            # Postprocessing
            height, width, _ = input_image.shape
            image_inpainted, mask_resized = recover_size(np.array(image_inpainted), mask, (height, width), padding_factors)
            mask_resized = np.expand_dims(mask_resized, -1) / 255
            image_inpainted = image_inpainted * (1-mask_resized) + input_image * mask_resized
            image_inpainted = image_inpainted.astype(np.uint8)
            
            # Base64 encoding
            inpainted_b64 = numpy_to_b64(image_inpainted)
            mask_b64 = numpy_to_b64(mask_resized)
            output_data = {'b64_inpainted': inpainted_b64,
                        'b64_mask': mask_b64}
            
        return build_actual_response(jsonify(output_data))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tobigs-19 Vision Conference")
    
    parser.add_argument('--yolo_path', default='./model_checkpoints/yolov7-e6e.pt')
    
    parser.add_argument('--sam_path',  default='./model_checkpoints/sam_vit_h_4b8939.pth')
    
    parser.add_argument('--sd_name',  default='stabilityai/stable-diffusion-2-inpainting')
    
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    # Load models
    # Yolov7
    yolo = attempt_load(args.yolo_path, map_location=device)
    
    # Segment anything
    sam = sam_model_registry['vit_h'](args.sam_path).to(device)
    sam_predictor = SamPredictor(sam)
    
    # Stable diffusion
    pipe = StableDiffusionInpaintPipeline.from_pretrained(args.sd_name,torch_dtype=torch.float32).to(device)
    
    app.run(host='0.0.0.0', port=5000)