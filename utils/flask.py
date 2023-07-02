from io import BytesIO
import base64
from PIL import Image
import numpy as np
import cv2
from flask import make_response

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response        

def build_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

def b64_to_pil(b64_image: str) -> Image:
    img = base64.b64decode(b64_image)
    img = BytesIO(img)
    img = Image.open(img)
    return img

def b64_to_numpy(b64_image) -> np.array:
    img = base64.b64decode(b64_image)
    img = np.frombuffer(img, dtype=np.uint8)  
    img = cv2.imdecode(img, flags=cv2.IMREAD_COLOR)
    return img

def pil_to_b64(image: Image) -> str:
    image_file = BytesIO()
    image.save(image_file, format="JPEG")
    image_bytes = image_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(image_bytes).decode('utf8')

def numpy_to_b64(image: np.array) -> str:
    _, image_arr = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    image_bytes = image_arr.tobytes()
    return base64.b64encode(image_bytes).decode('utf8')