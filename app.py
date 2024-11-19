import requests
import os
from tqdm import tqdm

def download_file(url, filepath):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as file, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def ensure_models():
    """Ensure all required models are available"""
    # Paths
    sam_dir = "Sam model"
    yolo_dir = os.path.join("void_detection", "train", "weights")
    
    os.makedirs(sam_dir, exist_ok=True)
    os.makedirs(yolo_dir, exist_ok=True)
    
    sam_path = os.path.join(sam_dir, "sam_vit_b_01ec64.pth")
    yolo_path = os.path.join(yolo_dir, "best.pt")
    
    # Download SAM if needed
    if not os.path.exists(sam_path):
        print("Downloading SAM model...")
        download_file(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            sam_path
        )
    
    # Download YOLO if needed
    if not os.path.exists(yolo_path):
        print("Downloading YOLO model...")
        download_file(
    "https://huggingface.co/spaces/SabbahYoussef/deployement_project_sabbah/resolve/main/best.pt",
    yolo_path
)

# Add at start of app.py
ensure_models()

from flask import Flask, render_template, request, jsonify, send_file
import os
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image
import cv2
import torch
import json
from ultralytics import YOLO
import gc

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def cleanup():
    """Memory cleanup function"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Initialize models
try:
    # Initialize SAM with CPU
    DEVICE = torch.device('cpu')  # Force CPU usage
    MODEL_TYPE = "vit_b"
    CHECKPOINT_PATH = os.path.join("Sam model", "sam_vit_b_01ec64.pth")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    SAM_PREDICTOR = SamPredictor(sam)
    print("SAM model loaded successfully")
    
    # Initialize YOLO with smaller batch size
    YOLO_PATH = os.path.join("void_detection", "train", "weights", "best.pt")
    if not os.path.exists(YOLO_PATH):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_PATH}")
    YOLO_MODEL = YOLO(YOLO_PATH)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise

# Colors for visualization
class_colors = {
    'solder_joint': (255, 0, 0),  # Red
    'void': (0, 255, 0)           # Green
}

@app.route('/')
def index():
    cleanup()  # Cleanup after route
    return render_template('index.html')

@app.route('/sam')
def sam_interface():
    cleanup()
    return render_template('sam.html')

@app.route('/yolo')
def yolo_interface():
    cleanup()
    return render_template('yolo.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Resize image to reduce memory usage
            pil_image = Image.open(filepath)
            max_size = 800  # Reduced from 1024
            ratio = min(max_size / pil_image.width, max_size / pil_image.height)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            image = np.array(pil_image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            SAM_PREDICTOR.set_image(image)
            cleanup()  # Cleanup after processing
            
            return jsonify({
                'success': True,
                'filename': file.filename,
                'path': f'/static/uploads/{file.filename}'
            })
        except Exception as e:
            cleanup()
            return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/segment', methods=['POST'])
def segment():
    try:
        data = request.json
        points = np.array(data['points'])
        point_labels = np.array(data['point_labels'])
        class_name = data['class_name']
        
        masks, scores, logits = SAM_PREDICTOR.predict(
            point_coords=points,
            point_labels=point_labels,
            multimask_output=True
        )
        
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx].astype(np.uint8) * 255
        
        color = class_colors[class_name]
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        colored_mask[mask > 0] = [*color, 128]
        
        overlay_filename = f"overlay_{os.path.splitext(data['filename'])[0]}_{class_name}.png"
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
        Image.fromarray(colored_mask).save(overlay_path, format='PNG')
        
        cleanup()  # Cleanup after processing
        
        return jsonify({
            'success': True,
            'overlay_path': f'/static/uploads/{overlay_filename}',
            'area': float(np.sum(mask)),
            'score': float(scores[best_mask_idx])
        })
    except Exception as e:
        cleanup()
        return jsonify({'error': str(e)})

@app.route('/detect', methods=['POST'])
def yolo_detect():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Read and resize image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image format'})
        
        # Resize image if too large
        max_size = 800
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            ratio = max_size / max(height, width)
            image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
        
        # Run YOLO prediction with optimized settings
        results = YOLO_MODEL.predict(
            image,
            conf=0.25,
            iou=0.45,
            max_det=50,  # Limit detections
            verbose=False
        )
        
        # Process results
        detections = []
        for box in results[0].boxes:
            detections.append({
                'class': results[0].names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xywh[0].tolist()
            })
        
        # Save result image
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{file.filename}')
        cv2.imwrite(result_path, cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR))
        
        cleanup()  # Cleanup after processing
        
        return jsonify({
            'success': True,
            'detections': detections,
            'image_path': f'/static/uploads/result_{file.filename}'
        })
    except Exception as e:
        cleanup()
        return jsonify({'error': str(e)})

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    try:
        data = request.json
        filename = f"annotation_{os.path.splitext(data['filename'])[0]}.json"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(data['annotations'], f, indent=2)
        
        cleanup()
        return jsonify({'success': True})
    except Exception as e:
        cleanup()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Use port from environment variable (Render will set this)
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
