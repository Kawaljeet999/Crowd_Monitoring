import torch
from torch import optim
from torch import nn
import matplotlib.pyplot as plt
import time
import os


def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                save_dir='checkpoints'):
    """Train the multi-face detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss functions
    bbox_criterion = nn.MSELoss(reduction='none')
    face_criterion = nn.BCELoss(reduction='none')
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # FIX: Removed verbose=True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting multi-face training on {device}...")
    
    for epoch in range(num_epochs):
        # ------------------ TRAIN ------------------
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (images, bboxes, face_masks) in enumerate(train_loader):
            images = images.to(device)
            bboxes = bboxes.to(device)
            face_masks = face_masks.to(device)
            
            optimizer.zero_grad()
            
            pred_bboxes, pred_faces = model(images)
            
            # BBox loss with masking
            bbox_loss = bbox_criterion(pred_bboxes, bboxes)
            bbox_loss = (bbox_loss * face_masks.unsqueeze(-1)).sum() / (face_masks.sum() + 1e-6)
            
            # Face presence loss
            face_loss = face_criterion(pred_faces.squeeze(-1), face_masks)
            face_loss = face_loss.mean()
            
            total_loss = bbox_loss + face_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {total_loss.item():.4f}')
        
        # ------------------ VALIDATION ------------------
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, bboxes, face_masks in val_loader:
                images = images.to(device)
                bboxes = bboxes.to(device)
                face_masks = face_masks.to(device)
                
                pred_bboxes, pred_faces = model(images)
                
                bbox_loss = bbox_criterion(pred_bboxes, bboxes)
                bbox_loss = (bbox_loss * face_masks.unsqueeze(-1)).sum() / (face_masks.sum() + 1e-6)
                
                face_loss = face_criterion(pred_faces.squeeze(-1), face_masks)
                face_loss = face_loss.mean()
                
                total_loss = bbox_loss + face_loss
                val_loss += total_loss.item()
        
        # ------------------ LOGGING ------------------
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
            print(f'Saved best model with validation loss: {best_val_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')
        
        # Step scheduler
        scheduler.step(avg_val_loss)
    
    # ------------------ PLOT LOSSES ------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Multi-Face Detection Training')
    plt.legend()
    plt.savefig(f'{save_dir}/training_history.png')
    plt.close()
    
    return model, train_losses, val_losses
