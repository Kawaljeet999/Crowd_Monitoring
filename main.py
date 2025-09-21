from models.utils.face_detection import MultiFaceDetectorCNN
from models.utils.dataset_class import MultiFaceDataset
from models.face_detector import MultiFaceDetector

from torch.utils.data import DataLoader
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from train import train_model
import cv2
import os


def main():
    # Detect dataset path automatically
    if os.path.exists("/kaggle/input"):
        # Kaggle dataset path
        data_dir = "/kaggle/input/shanghaitech-crowd-counting-dataset/part_A_final/train_data"
    else:
        # Local Windows dataset path
        data_dir = "C:/Users/maste/OneDrive/Documents/Final_year_project/Crowd_Analysis/Crowd_Monitoring/Dataset/shanghaitech-crowd-counting-dataset/versions/1/part_A_final/train_data"

    # Configuration for crowd analysis
    config = {
        'data_dir': data_dir,
        'image_subdir': 'images',
        'annotation_subdir': 'ground_truth',
        'batch_size': 8,   # Smaller batch size for multiple faces
        'num_epochs': 100,
        'learning_rate': 0.0005,
        'img_size': 512,   # Larger input for crowd scenes
        'max_faces': 50,   # Maximum faces per image
        'test_size': 0.1,
        'random_state': 42
    }
    
    # Set paths
    image_dir = Path(config['data_dir']) / config['image_subdir']
    annotation_dir = Path(config['data_dir']) / config['annotation_subdir']
    
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
    if not annotation_dir.exists():
        raise ValueError(f"Annotation directory not found: {annotation_dir}")
    
    print(f"Setting up for crowd analysis with up to {config['max_faces']} faces per image")
    
    # Create dataset
    dataset = MultiFaceDataset(
        image_dir, annotation_dir, 
        img_size=config['img_size'], 
        max_faces=config['max_faces']
    )
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = MultiFaceDetectorCNN(max_faces=config['max_faces'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trained_model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        save_dir='crowd_checkpoints'
    )
    
    print("Crowd analysis training completed!")
    
    # Test the detector
    detector = MultiFaceDetector('crowd_checkpoints/best_model.pth')
    
    # Test on sample image
    sample_images = list(image_dir.glob('*.jpg'))
    if sample_images:
        sample_image_path = sample_images[0]
        image = cv2.imread(str(sample_image_path))
        faces = detector.detect_faces(image)
        print(f"Detected {len(faces)} faces in sample image")
    else:
        print("⚠️ No sample images found in dataset.")


if __name__ == "__main__":
    main()
