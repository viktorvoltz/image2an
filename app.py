from flask import Flask, request, send_file
import os
import sys
import numpy as np
import cv2
import torch
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_folder='static', static_url_path='')

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model using torch.hub
try:
    print("Loading AnimeGANv2 model from torch.hub...")
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2").to(device)
    model.eval()
    
    # Also load the face2paint utility function
    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from torch.hub: {e}")
    model = None
    face2paint = None

# Preprocess image for direct tensor input
def preprocess_image(image):
    # Convert to RGB if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3 and image.dtype == np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    return pil_image

# Serve static files from root
@app.route('/')
def index():
    return app.send_static_file('index.html')

# API endpoint for converting images
@app.route('/api/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return "No image provided", 400
    
    # Check if model was loaded
    if model is None or face2paint is None:
        return "Model could not be loaded. Please check server logs.", 500
    
    try:
        file = request.files['image']
        
        # Read image directly as PIL Image
        img = Image.open(BytesIO(file.read())).convert("RGB")
        
        # Use the face2paint utility to convert the image
        anime_img = face2paint(model, img)
        
        # Save the result to BytesIO
        output_stream = BytesIO()
        anime_img.save(output_stream, format='JPEG')
        output_stream.seek(0)
        
        return send_file(output_stream, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error during image conversion: {e}")
        return f"Error processing image: {str(e)}", 500

if __name__ == '__main__':
    # Make sure the static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
        # Copy the HTML file to the static directory if it exists
        index_path = os.path.join(os.path.dirname(__file__), 'index.html')
        if os.path.exists(index_path):
            with open(os.path.join(static_dir, 'index.html'), 'w') as f:
                with open(index_path, 'r') as src:
                    f.write(src.read())
    
    # Use gunicorn-compatible setup
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)