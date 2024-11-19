import requests
import os
from tqdm import tqdm
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from PIL import Image
import cv2
import torch
import gc
from segment_anything import sam_model_registry, SamPredictor

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

# Initialize SAM with CPU
DEVICE = torch.device('cpu')  # Force CPU usage
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = os.path.join("Sam model", "sam_vit_b_01ec64.pth")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
SAM_PREDICTOR = SamPredictor(sam)
print("SAM model loaded successfully")

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

if __name__ == '__main__':
    # Use port from environment variable (Render will set this)
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
