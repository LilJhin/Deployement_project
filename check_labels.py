# check_labels.py
import os

def check_label_files():
    train_dir = "yolo_dataset/labels/train/labels"
    val_dir = "yolo_dataset/labels/val/labels"
    
    def print_file_content(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
            print(f"\nContent: {content}")
            if content.strip():
                lines = content.strip().split('\n')
                values = lines[0].split()
                print(f"Number of values: {len(values)}")
                print(f"Values: {values}")
    
    # Check first file in train
    train_files = os.listdir(train_dir)
    if train_files:
        print(f"\nChecking train file: {train_files[0]}")
        print_file_content(os.path.join(train_dir, train_files[0]))
    
    # Check first file in val
    val_files = os.listdir(val_dir)
    if val_files:
        print(f"\nChecking val file: {val_files[0]}")
        print_file_content(os.path.join(val_dir, val_files[0]))

if __name__ == "__main__":
    check_label_files()