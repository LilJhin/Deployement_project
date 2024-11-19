from flask import Flask, render_template, request, jsonify, send_file
import os
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image
import cv2
import torch
import json
from ultralytics import YOLO

app = Flask(__name__)
global YOLO_MODEL, SAM_PREDICTOR
YOLO_MODEL = None  # Initialize YOLO as None
SAM_PREDICTOR = None

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load SAM and YOLO models
try:
    # Initialize SAM
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = os.path.join("Sam model", "sam_vit_h_4b8939.pth")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    SAM_PREDICTOR = SamPredictor(sam)
    print("SAM model loaded successfully")
    
    # Initialize YOLO
    YOLO_PATH = os.path.join("void_detection", "train", "weights", "best.pt")
    if not os.path.exists(YOLO_PATH):
        raise FileNotFoundError(f"YOLO weights not found at: {YOLO_PATH}")
    YOLO_MODEL = YOLO(YOLO_PATH)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise

# Colors for overlay (RGBA format)
class_colors = {
    'solder_joint': (255, 0, 0),  # Red
    'void': (0, 255, 0)           # Green
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sam')
def sam_interface():
    return render_template('sam.html')

@app.route('/yolo')
def yolo_interface():
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
            # Resize image for consistency
            pil_image = Image.open(filepath)
            max_size = 1024
            ratio = min(max_size / pil_image.width, max_size / pil_image.height)
            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy (OpenCV) format
            image = np.array(pil_image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Set image for SAM
            SAM_PREDICTOR.set_image(image)
            
            return jsonify({
                'success': True,
                'filename': file.filename,
                'path': f'/static/uploads/{file.filename}'
            })
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/segment', methods=['POST'])
def segment():
    try:
        data = request.json
        points = np.array(data['points'])
        point_labels = np.array(data['point_labels'])
        class_name = data['class_name']
        
        # Perform segmentation
        masks, scores, logits = SAM_PREDICTOR.predict(
            point_coords=points,
            point_labels=point_labels,
            multimask_output=True
        )
        
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx].astype(np.uint8) * 255
        
        # Overlay mask with color
        color = class_colors[class_name]
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        colored_mask[mask > 0] = [*color, 128]  # RGBA
        
        overlay_filename = f"overlay_{os.path.splitext(data['filename'])[0]}_{class_name}.png"
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
        Image.fromarray(colored_mask).save(overlay_path, format='PNG')
        
        return jsonify({
            'success': True,
            'overlay_path': f'/static/uploads/{overlay_filename}',
            'area': float(np.sum(mask)),
            'score': float(scores[best_mask_idx])
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    try:
        data = request.json
        filename = f"annotation_{os.path.splitext(data['filename'])[0]}.json"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(data['annotations'], f, indent=2)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload_for_detection', methods=['POST'])
def upload_for_detection():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'filename': file.filename,
        'path': f'/static/uploads/{file.filename}'
    })

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
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image format'})
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = YOLO_MODEL.predict(image, conf=0.031)
        
        detections = []
        for box in results[0].boxes:
            detections.append({
                'class': results[0].names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xywh.tolist()
            })
        
        annotated_image = results[0].plot()
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{file.filename}')
        cv2.imwrite(result_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        return jsonify({
            'success': True,
            'detections': detections,
            'image_path': f'/static/uploads/result_{file.filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
