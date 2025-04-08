from flask import Flask, request, send_file
import os
import sys
import numpy as np
import cv2
import torch
from io import BytesIO
from PIL import Image
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('animegan-app')

# Create Flask app
app = Flask(__name__, static_folder='static', static_url_path='')

# Add more detailed logging to Flask
if not app.debug:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    app.logger.addHandler(stream_handler)

# Log startup information
logger.info(f"Starting application at {datetime.now().isoformat()}")
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"OpenCV version: {cv2.__version__}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"PIL version: {Image.__version__}")

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create a global variable for model and face2paint
# This will be initialized once for each worker
model = None
face2paint = None

def initialize_model():
    """Initialize the model if it hasn't been loaded yet."""
    global model, face2paint
    
    # Skip if already initialized
    if model is not None and face2paint is not None:
        return True
    
    try:
        logger.info("Loading AnimeGANv2 model from torch.hub...")
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2").to(device)
        model.eval()
        
        # Also load the face2paint utility function
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model from torch.hub: {e}")
        logger.error(traceback.format_exc())
        model = None
        face2paint = None
        return False

# Initialize model at startup - each worker will execute this
initialize_model()

# Preprocess image for direct tensor input
def preprocess_image(image):
    try:
        # Convert to RGB if necessary
        if len(image.shape) == 2:
            logger.debug("Converting grayscale image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            logger.debug("Converting RGBA image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            logger.debug("Converting BGR image to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        return pil_image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        logger.error(traceback.format_exc())
        raise

# Serve static files from root
@app.route('/')
def index():
    logger.info("Serving index.html")
    return app.send_static_file('index.html')

# API endpoint for converting images
@app.route('/api/convert', methods=['POST'])
def convert():
    logger.info("Received conversion request")
    
    # Make sure model is initialized (important for worker-based environments)
    if not initialize_model():
        logger.error("Failed to initialize model")
        return "Failed to initialize model. Please check server logs.", 500
    
    if 'image' not in request.files:
        logger.warning("No image provided in request")
        return "No image provided", 400
    
    try:
        file = request.files['image']
        logger.info(f"Processing image: {file.filename}, Content type: {file.content_type}")
        
        # Read image directly as PIL Image
        img_data = file.read()
        logger.debug(f"Image size: {len(img_data)} bytes")
        
        img = Image.open(BytesIO(img_data)).convert("RGB")
        logger.info(f"Image opened successfully, size: {img.size}")
        
        # Use the face2paint utility to convert the image
        logger.info("Applying anime style transformation")
        anime_img = face2paint(model, img)
        logger.info("Transformation completed successfully")
        
        # Save the result to BytesIO
        output_stream = BytesIO()
        anime_img.save(output_stream, format='JPEG')
        output_stream.seek(0)
        logger.info("Image saved to output stream")
        
        return send_file(output_stream, mimetype='image/jpeg')
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error during image conversion: {e}")
        logger.error(error_details)
        return f"Error processing image: {str(e)}\n\nDetails:\n{error_details}", 500

@app.route('/health')
def health_check():
    """Simple health check endpoint to verify the service is running"""
    model_initialized = model is not None and face2paint is not None
    status = {
        "status": "ok" if model_initialized else "degraded",
        "model_loaded": model_initialized,
        "timestamp": datetime.now().isoformat()
    }
    logger.info(f"Health check: {status}")
    return status

if __name__ == '__main__':
    # Make sure the static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        logger.info(f"Creating static directory: {static_dir}")
        os.makedirs(static_dir)
        
        # Copy the HTML file to the static directory if it exists
        index_path = os.path.join(os.path.dirname(__file__), 'index.html')
        if os.path.exists(index_path):
            logger.info(f"Copying index.html to static directory")
            with open(os.path.join(static_dir, 'index.html'), 'w') as f:
                with open(index_path, 'r') as src:
                    f.write(src.read())
        else:
            logger.warning(f"index.html not found at {index_path}")
    
    # Use gunicorn-compatible setup
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)