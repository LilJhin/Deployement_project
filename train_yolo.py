import torch
from ultralytics import YOLO

def train_yolo(data_yaml, epochs=100):
    model = YOLO('yolov8n.pt')
    model.train(data=data_yaml, epochs=epochs, imgsz=640, batch=8, device='0' if torch.cuda.is_available() else 'cpu', 
                project='void_detection', name='train', exist_ok=True, patience=20, save=True, pretrained=True, 
                optimizer='Adam', lr0=0.001, weight_decay=0.0005, augment=True, mosaic=0.5, degrees=10.0, 
                translate=0.1, scale=0.5, fliplr=0.5, task='detect', val=False)

    val_results = model.val(data=data_yaml, imgsz=640, device='0' if torch.cuda.is_available() else 'cpu')
    print("Validation results:", val_results)
    print("Training completed!")
    return model

if __name__ == "__main__":
    data_yaml = "yolo_dataset/dataset.yaml"
    train_yolo(data_yaml)