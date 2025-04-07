from flask import Flask, request, send_file
import os
import sys
import numpy as np
import cv2
import torch
from io import BytesIO
from PIL import Image
import requests
import tempfile

app = Flask(__name__, static_folder='static', static_url_path='')

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to download pre-trained AnimeGAN model if not exists
def download_model():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(model_dir, 'animeganv2_hayao.pt')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(model_path):
        print("Downloading AnimeGANv2 model...")
        # AnimeGANv2 model URL (using Hayao style)
        url = "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/face_paint_512_v2.pt"
        response = requests.get(url)
        
        with open(model_path, 'wb') as f:
            f.write(response.content)
        
        print("Model downloaded successfully!")
    
    return model_path

# Load the model using TorchScript
def load_model(model_path):
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
model_path = download_model()
model = load_model(model_path)

# Preprocess image
def preprocess(image):
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 512x512 while maintaining aspect ratio
    h, w = image.shape[:2]
    ratio = min(512 / h, 512 / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    image = cv2.resize(image, (new_w, new_h))
    
    # Create canvas with padding
    canvas = np.ones((512, 512, 3), dtype=np.uint8) * 255
    offset_h, offset_w = (512 - new_h) // 2, (512 - new_w) // 2
    canvas[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = image
    
    # Normalize
    image = canvas.astype(np.float32) / 127.5 - 1.0
    
    # Convert to torch tensor
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return image, (h, w), (offset_h, offset_w, new_h, new_w)

# Postprocess image
def postprocess(output, original_size, crop_info):
    h, w = original_size
    offset_h, offset_w, new_h, new_w = crop_info
    
    # Convert back to numpy
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Denormalize
    output = (output * 127.5 + 127.5).astype(np.uint8)
    
    # Crop the padding
    output = output[offset_h:offset_h + new_h, offset_w:offset_w + new_w]
    
    # Resize back to original size
    output = cv2.resize(output, (w, h))
    
    return output

# Serve static files from root
@app.route('/')
def index():
    return app.send_static_file('index.html')

# API endpoint for converting images
@app.route('/api/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return "No image provided", 400
    
    file = request.files['image']
    
    # Read and preprocess image
    img_stream = BytesIO(file.read())
    image = np.array(Image.open(img_stream))
    
    # Process with AnimeGAN
    input_tensor, original_size, crop_info = preprocess(image)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess
    anime_image = postprocess(output, original_size, crop_info)
    
    # Convert back to PIL and save to BytesIO
    anime_pil = Image.fromarray(anime_image)
    output_stream = BytesIO()
    anime_pil.save(output_stream, format='JPEG')
    output_stream.seek(0)
    
    return send_file(output_stream, mimetype='image/jpeg')

if __name__ == '__main__':
    # Make sure the static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
        # Copy the HTML file to the static directory
        with open(os.path.join(static_dir, 'index.html'), 'w') as f:
            with open('index.html', 'r') as src:
                f.write(src.read())
    
    app.run(debug=True, host='0.0.0.0', port=5000)