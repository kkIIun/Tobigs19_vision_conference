import argparse
import sys
sys.path.append('./yolov7') # modify your 'yolov7' directory
from PIL import Image
import numpy as np
import torch
import gradio as gr
from yolov7.models.experimental import attempt_load
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline
from utils import resize_and_pad, recover_size, save_image_mask
from utils.functions import predict_box_with_yolo, predict_mask_with_sam

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Tobigs-19 Vision Conference")
    
    parser.add_argument('--yolo_path', default='./model_checkpoints/yolov7-e6e.pt')
    
    parser.add_argument('--sam_path',  default='./model_checkpoints/sam_vit_h_4b8939.pth')
    
    parser.add_argument('--public_link', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    
    with gr.Blocks() as demo:
        with torch.no_grad():
            # yolo
            yolo = attempt_load(args.yolo_path, map_location=device)
            # Segment anything
            sam = sam_model_registry['vit_h'](args.sam_path).to(device)
            sam_predictor = SamPredictor(sam)
            # Stable diffusion
            pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",torch_dtype=torch.float32).to(device)
            
            
            with gr.Row():
                # Col A: The input image
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Input</center></h3>")
                    input_img = gr.Image(label='Input image', show_label=False).style(height=500)
                    text_prompt = gr.Textbox(label='text prompt')            
                
                #Col C: The final image
                with gr.Column(scale=1):
                    gr.HTML("<h3><center>Output</center></h3>")
                    output_img = gr.Image(label='Generated image', interactive=False).style(height=500)
                    generate_btn = gr.Button(value='Generate')
            
            # reset components
            def reset_components():
                return None
            input_img.change(fn=reset_components,
                            outputs=output_img)
                            

            def on_generate_clicked(input_img, text_prompt):
                org_image = input_img.copy()
                #image preprocessing
                image_padded, padding_factors = resize_and_pad(org_image)
                image_tensor = image_padded.transpose(2,0,1)[None,...]
                image_tensor = torch.Tensor(image_tensor/255.0).to(device)
                #yolo
                box = predict_box_with_yolo(yolo, image_tensor.to(device))
                #sam
                mask = predict_mask_with_sam(sam_predictor, image_padded, box)
                #stable diffusion
                image_inpainted = pipe(prompt=text_prompt, image=Image.fromarray(image_padded), mask_image=Image.fromarray(255-mask)).images[0]
                
                #postprocessing
                height, width, _ = org_image.shape
                image_resized, mask_resized = recover_size(np.array(image_inpainted), mask, (height, width), padding_factors)
                mask_resized = np.expand_dims(mask_resized, -1) / 255
                image_resized = image_resized * (1-mask_resized) + org_image * mask_resized
                image_resized = image_resized.astype(np.uint8)
            generate_btn.click(fn=on_generate_clicked,
                            inputs=[input_img, text_prompt],
                            outputs=output_img)
    
        demo.launch(share=args.public_link)