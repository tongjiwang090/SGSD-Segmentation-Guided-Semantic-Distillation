import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from dataset import H1SegmentationDataset
from unet import UNet
import matplotlib.pyplot as plt  
import numpy as np
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return bce_loss + dice_loss

def visualize_masks(orig_images, masks, outputs=None, num_samples=4):
    """
    visualize validation: original image and binary mask, plus model output if given
    orig_images: (B, 3, H, W) un-normalized images in [0, 1]
    masks: (B, C, H, W) labels
    outputs: (B, C, H, W) model logits, optional
    """
    images = orig_images.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy() # (B, C, H, W)
    
    if outputs is not None:
        preds = torch.sigmoid(outputs).cpu().detach().numpy()
        preds = (preds > 0.5).astype(np.float32)
        
    num_samples = min(num_samples, masks.shape[0])
    
    rows = 3 if outputs is not None else 2
    fig, axes = plt.subplots(rows, num_samples, figsize=(3*num_samples, 3*rows))
    
    # force axes to 2d so indexing is uniform
    if num_samples == 1:
        axes = axes[:, np.newaxis]
        
    for i in range(num_samples):
        # images[i] is (3, H, W) in [0, 1]
        img = images[i].transpose(1, 2, 0) # to (H, W, 3)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Image {i}')
        axes[0, i].axis('off')
        
        # mask display
        mask_i = masks[i]
        if mask_i.shape[0] == 1:
            axes[1, i].imshow(mask_i[0], cmap='gray', vmin=0, vmax=1)
        else:
            # multi-channel: show as hwc
            axes[1, i].imshow(mask_i.transpose(1, 2, 0))
        axes[1, i].set_title(f'GT Mask {i}')
        axes[1, i].axis('off')
        
        if outputs is not None:
            pred_i = preds[i]
            if pred_i.shape[0] == 1:
                axes[2, i].imshow(pred_i[0], cmap='gray', vmin=0, vmax=1)
            else:
                axes[2, i].imshow(pred_i.transpose(1, 2, 0))
            axes[2, i].set_title(f'Pred Mask {i}')
            axes[2, i].axis('off')
            
    plt.tight_layout()
    # also write to file so headless servers do not block
    plt.savefig('train_batch_vis.png')
    plt.close()

def train_model(
    data_dir,
    epochs=10,
    batch_size=4,
    learning_rate=1e-4,
    device=None,
    save_dir='checkpoints',
    resume_path=None
):
    # device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # output dir
    os.makedirs(save_dir, exist_ok=True)

    # 1. datasets and loaders
    print("Initializing datasets...")
    train_dataset = H1SegmentationDataset(data_dir=data_dir, split='train')
    val_dataset = H1SegmentationDataset(data_dir=data_dir, split='val')
    
    num_classes = train_dataset.num_classes
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 2. model
    model = UNet(n_channels=3, n_classes=num_classes).to(device)
    
    # 3. loss and optimizer
    criterion = DiceBCELoss()
    # weight_decay (l2) to curb overfitting
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 4. resume
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}...")
        # backward compatible and safe-load settings
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed training from epoch {start_epoch}")
    
    # onecyclelr with warmup and cosine annealing
    total_steps = epochs * len(train_loader)
    completed_steps = start_epoch * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1,  # 10% of steps for warmup
        anneal_strategy='cos',  # cosine annealing
        last_epoch=completed_steps - 1 if completed_steps > 0 else -1
    )
    
    # 5. training loop
    early_stop_patience = 5  # early-stop patience
    early_stop_counter = 0
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, masks, orig_images) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # forward
            outputs = model(images)
            
            # loss
            loss = criterion(outputs, masks)
            
            # backward and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # lr updated every batch, not every epoch
            
            train_loss += loss.item() * images.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=loss.item(), lr=f"{current_lr:.6e}")
            # print(f"Current batch loss: {loss.item():.4f}")
        epoch_train_loss = train_loss / len(train_dataset)
        
        # validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (images, masks, orig_images) in enumerate(tqdm(val_loader, desc="Validation")):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                
                # visualize once per epoch, on the first batch only
                if batch_idx == 0:
                    visualize_masks(orig_images, masks, outputs, num_samples=4)
                
        epoch_val_loss = val_loss / len(val_dataset)
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # lr at the end of this epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6e}")
        
        # best-model saving and early stop
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0  # reset
            save_path = os.path.join(save_dir, 'best_unet_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'num_classes': num_classes
            }, save_path)
            print(f"Saved new best model to {save_path}")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered! Training stopped.")
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train U-Net for Segmentation")
    # default points at the directory holding the hdf5 files
    import os
    default_data_dir = os.getcwd()+r"/../policy/ACT/processed_data/sim-beat_block_hammer/demo_clean-50/"
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Directory containing episode dataset (.hdf5 files)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-3, help="Learning rate")
    parser.add_argument("--resume", type=str, default="./checkpoints/best_unet_model.pth", help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="output dir for weights; use one per task or the anchor teacher gets overwritten")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        resume_path=args.resume
    )
