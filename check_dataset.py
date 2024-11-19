# check_dataset.py
import os

def check_dataset_structure():
    base_dir = os.path.abspath('yolo_dataset')
    
    required_dirs = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    print(f"Checking dataset structure in: {base_dir}")
    
    # Check directories
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            files = os.listdir(full_path)
            print(f"✓ {dir_path}: {len(files)} files found")
        else:
            print(f"✗ {dir_path}: Directory not found!")
    
    # Check yaml file
    yaml_path = os.path.join(base_dir, 'dataset.yaml')
    if os.path.exists(yaml_path):
        print(f"✓ dataset.yaml found")
    else:
        print(f"✗ dataset.yaml not found!")

if __name__ == "__main__":
    check_dataset_structure()