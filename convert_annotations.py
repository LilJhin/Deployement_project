import os
import json
import cv2
import numpy as np

def convert_sam_to_yolo(annotations_dir, images_dir, output_dir, mode='train'):
    """
    Convert SAM annotations to YOLO format with proper normalization and validation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    class_map = {
        'solder_joint': 0,
        'void': 1
    }
    
    print(f"Starting conversion for {mode} set...")
    
    for ann_file in os.listdir(annotations_dir):
        if not ann_file.endswith('.json'):
            continue
            
        print(f"\nProcessing {ann_file}")
        
        # Read annotation file
        with open(os.path.join(annotations_dir, ann_file), 'r') as f:
            annotations = json.load(f)
        
        # Get corresponding image
        img_name = ann_file.replace('annotation_', '').replace('.json', '')
        img_files = [f for f in os.listdir(images_dir) if f.startswith(img_name) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not img_files:
            print(f"Warning: No matching image found for {ann_file}")
            continue
            
        img_path = os.path.join(images_dir, img_files[0])
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        
        # Prepare YOLO annotations
        yolo_annotations = []
        
        for class_name, points in annotations.items():
            class_id = class_map[class_name]
            
            for point in points:
                if isinstance(point, dict) and 'point' in point:
                    x, y = point['point']
                    
                    # Create bounding box (40 pixels around point for better visibility)
                    box_size = 40
                    x_min = max(0, x - box_size)
                    y_min = max(0, y - box_size)
                    x_max = min(img_width - 1, x + box_size)  # Subtract 1 to avoid edge
                    y_max = min(img_height - 1, y + box_size)  # Subtract 1 to avoid edge
                    
                    # Skip if box is too small
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    
                    # Convert to YOLO format with validation
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    # Validate coordinates
                    if (0 < x_center < 1 and 0 < y_center < 1 and 
                        0 < width < 1 and 0 < height < 1):
                        yolo_annotations.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
        
        if yolo_annotations:
            # Save annotations
            output_file = os.path.join(output_dir, img_files[0].replace('.jpg', '.txt')
                                                              .replace('.jpeg', '.txt')
                                                              .replace('.png', '.txt'))
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            print(f"Created {output_file} with {len(yolo_annotations)} valid annotations")

if __name__ == "__main__":
    # Process validation set
    convert_sam_to_yolo(
        annotations_dir="static/uploads/train",
        images_dir="yolo_dataset/images/train",
        output_dir="yolo_dataset/labels/train",
        mode='val'
    )