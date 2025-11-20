"""
Training script for Low-Light Enhancement Model using LOL Dataset
Specifically configured for the LOL dataset structure
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import numpy as np

# Add project root to path (parent directory of scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.enhancer import LowLightEnhancer
from data.dataset_utils import LowLightDataset, get_enhancer_transforms


def train_epoch(model_wrapper, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model_wrapper.model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for low_images, normal_images in pbar:
        low_images = low_images.to(device)
        normal_images = normal_images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        enhanced = model_wrapper.model(low_images)
        loss = criterion(enhanced, normal_images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Calculate PSNR
        mse = torch.mean((enhanced - normal_images) ** 2)
        psnr = -10 * torch.log10(mse + 1e-10)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr.item():.2f} dB'
        })
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model_wrapper, dataloader, criterion, device):
    """Validate model"""
    model_wrapper.model.eval()
    running_loss = 0.0
    all_psnr = []
    
    with torch.no_grad():
        for low_images, normal_images in tqdm(dataloader, desc='Validating'):
            low_images = low_images.to(device)
            normal_images = normal_images.to(device)
            
            enhanced = model_wrapper.model(low_images)
            loss = criterion(enhanced, normal_images)
            
            running_loss += loss.item()
            
            # Calculate PSNR
            mse = torch.mean((enhanced - normal_images) ** 2, dim=(1, 2, 3))
            psnr = -10 * torch.log10(mse + 1e-10)
            all_psnr.extend(psnr.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    avg_psnr = np.mean(all_psnr)
    
    return epoch_loss, avg_psnr


def main():
    parser = argparse.ArgumentParser(description='Train Low-Light Enhancement Model with LOL Dataset')
    parser.add_argument('--data_dir', type=str, default='data/LOLdataset/our485',
                       help='Path to training data directory (default: data/LOLdataset/our485)')
    parser.add_argument('--val_dir', type=str, default='data/LOLdataset/eval15',
                       help='Path to validation data directory (default: data/LOLdataset/eval15)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='weights',
                       help='Directory to save model weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--input_size', type=int, default=256,
                       help='Input image size')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load datasets
    train_transform = get_enhancer_transforms(train=True, input_size=args.input_size)
    val_transform = get_enhancer_transforms(train=False, input_size=args.input_size)
    
    print(f"Loading training data from: {args.data_dir}")
    train_dataset = LowLightDataset(
        args.data_dir,
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    if args.val_dir and Path(args.val_dir).exists():
        print(f"Loading validation data from: {args.val_dir}")
        val_dataset = LowLightDataset(
            args.val_dir,
            transform=val_transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None
        print("No validation set provided")
    
    print(f"Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = LowLightEnhancer(device=device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_psnr = 0.0
    
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Loss: Combine L1 and perceptual loss
    # For simplicity, using L1 loss (can add perceptual loss later)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss, val_psnr = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
            
            # Save best model
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_path = save_dir / 'enhancer_best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_psnr': val_psnr,
                    'best_val_psnr': best_val_psnr,
                }, best_path)
                print(f"✓ Saved best model (Val PSNR: {val_psnr:.2f} dB)")
            
            scheduler.step(val_psnr)
        else:
            # Save checkpoint every epoch if no validation set
            checkpoint_path = save_dir / f'enhancer_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
    
    # Save final model
    final_path = save_dir / 'enhancer_final.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\n{'=' * 60}")
    print(f"Training complete! Final model saved to {final_path}")
    print(f"Best model saved to {save_dir / 'enhancer_best.pth'}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

