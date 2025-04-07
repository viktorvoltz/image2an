from flask import Flask, request, send_file, jsonify
import os
import sys
import numpy as np
import cv2
import torch
from io import BytesIO
from PIL import Image
import requests
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')

# Check if CUDA is available (will be CPU on Render)
device = torch.device("cpu")  # Force CPU for Render compatibility
logger.info(f"Using device: {device}")

# Global model variable
model = None

# Function to download pre-trained AnimeGAN model if not exists
def download_model():
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        model_path = os.path.join(model_dir, 'animeganv2_hayao.pt')
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # If model doesn't exist, download it
        if not os.path.exists(model_path):
            logger.info("Downloading AnimeGANv2 model...")
            
            # Use a smaller, more compatible model for Render
            # This is a Hayao style model from the AnimeGANv2 project
            url = "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/face_paint_512_v2_0.pt"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info("Model downloaded successfully!")
            except Exception as e:
                logger.error(f"Error downloading model: {str(e)}")
                return None
        
        return model_path
    except Exception as e:
        logger.error(f"Error in download_model: {str(e)}")
        return None

# Load the model using TorchScript
def load_model(model_path):
    if model_path is None:
        return None
    
    try:
        # Load model with CPU
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Initialize model (delayed to first request to save resources)
def get_model():
    global model
    if model is None:
        logger.info("Initializing model")
        model_path = download_model()
        model = load_model(model_path)
    return model

# Preprocess image
def preprocess(image):
    try:
        # Make sure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to a smaller size to accommodate memory constraints
        h, w = image.shape[:2]
        max_size = 512  # Maximum dimension
        
        # Calculate new dimensions while maintaining aspect ratio
        ratio = min(max_size / h, max_size / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        # Resize image
        image = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with padding
        size = max_size
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
        offset_h, offset_w = (size - new_h) // 2, (size - new_w) // 2
        canvas[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = image
        
        # Normalize
        image = canvas.astype(np.float32) / 127.5 - 1.0
        
        # Convert to torch tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
        
        return image, (h, w), (offset_h, offset_w, new_h, new_w)
    except Exception as e:
        logger.error(f"Error in preprocess: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

# Postprocess image
def postprocess(output, original_size, crop_info):
    try:
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
    except Exception as e:
        logger.error(f"Error in postprocess: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Serve static files from root
@app.route('/')
def index():
    return app.send_static_file('index.html')

# API endpoint for converting images
@app.route('/api/convert', methods=['POST'])
def convert():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        
        # Get model
        current_model = get_model()
        if current_model is None:
            return jsonify({"error": "Failed to load model"}), 500
        
        # Read image
        img_stream = BytesIO(file.read())
        image = np.array(Image.open(img_stream))
        
        # Preprocess
        input_tensor, original_size, crop_info = preprocess(image)
        if input_tensor is None:
            return jsonify({"error": "Failed to preprocess image"}), 500
        
        # Process with AnimeGAN
        try:
            with torch.no_grad():
                output = current_model(input_tensor)
        except Exception as e:
            logger.error(f"Error in model inference: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "Failed during image conversion"}), 500
        
        # Postprocess
        anime_image = postprocess(output, original_size, crop_info)
        if anime_image is None:
            return jsonify({"error": "Failed to postprocess image"}), 500
        
        # Convert back to PIL and save to BytesIO
        anime_pil = Image.fromarray(anime_image)
        output_stream = BytesIO()
        anime_pil.save(output_stream, format='JPEG')
        output_stream.seek(0)
        
        return send_file(output_stream, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Unhandled exception in convert endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    # Make sure the static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
    # Use environment variable for port (Render sets this)
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)