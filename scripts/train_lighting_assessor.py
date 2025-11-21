"""
Training script for Lighting/Brightness Assessment Model
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

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

from models.lighting_assessor import LightingAssessor
from data.dataset_utils import LightingDataset, get_transforms
from generate_lighting_labels import generate_lighting_labels


def train_epoch(model_wrapper, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model_wrapper.model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, scores in pbar:
        images = images.to(device)
        scores = scores.to(device).unsqueeze(1)  # Add dimension for regression
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model_wrapper.model(images)
        loss = criterion(outputs, scores)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Calculate MAE
        mae = torch.mean(torch.abs(outputs - scores)).item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mae': f'{mae:.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model_wrapper, dataloader, criterion, device):
    """Validate model"""
    model_wrapper.model.eval()
    running_loss = 0.0
    all_outputs = []
    all_scores = []
    
    with torch.no_grad():
        for images, scores in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            scores = scores.to(device).unsqueeze(1)
            
            outputs = model_wrapper.model(images)
            loss = criterion(outputs, scores)
            
            running_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_scores.append(scores.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    
    # Calculate overall metrics
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    
    mae = np.mean(np.abs(all_outputs - all_scores))
    mse = np.mean((all_outputs - all_scores) ** 2)
    rmse = np.sqrt(mse)
    
    return epoch_loss, mae, rmse


def main():
    parser = argparse.ArgumentParser(description='Train Lighting Assessment Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Path to validation data directory (optional)')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to CSV file with image paths and brightness scores')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='weights',
                       help='Directory to save model weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--auto_generate_csv', action='store_true',
                       help='Automatically generate lighting labels if CSV is missing.')
    parser.add_argument('--reference_lol_dir', type=str, default=None,
                       help='Path to LOL dataset root (uses low/high folders for calibration).')
    parser.add_argument('--reference_low_dir', type=str, default=None,
                       help='Path to low-light reference images for calibration.')
    parser.add_argument('--reference_high_dir', type=str, default=None,
                       help='Path to well-lit reference images for calibration.')
    parser.add_argument('--low_target_score', type=float, default=3.0,
                       help='Desired score for low-light reference images.')
    parser.add_argument('--high_target_score', type=float, default=7.0,
                       help='Desired score for well-lit reference images.')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load datasets
    train_transform = get_transforms(train=True, input_size=224)
    val_transform = get_transforms(train=False, input_size=224)
    
    csv_path = args.csv_path
    if args.auto_generate_csv or (csv_path and not Path(csv_path).exists()):
        if not csv_path:
            csv_path = Path(args.data_dir) / 'lighting_labels_auto.csv'
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        ref_low = args.reference_low_dir
        ref_high = args.reference_high_dir
        
        if args.reference_lol_dir:
            lol_root = Path(args.reference_lol_dir)
            candidate_low = lol_root / 'low'
            candidate_high = lol_root / 'high'
            if not candidate_high.exists():
                candidate_high = lol_root / 'normal'
            if candidate_low.exists() and not ref_low:
                ref_low = str(candidate_low)
            if candidate_high.exists() and not ref_high:
                ref_high = str(candidate_high)
        
        print(f"Auto-generating lighting labels at {csv_path} ...")
        generate_lighting_labels(
            image_dir=args.data_dir,
            output_csv=str(csv_path),
            reference_low_dir=ref_low,
            reference_high_dir=ref_high,
            low_target_score=args.low_target_score,
            high_target_score=args.high_target_score
        )
        args.csv_path = str(csv_path)
        csv_path = args.csv_path
    
    train_dataset = LightingDataset(
        args.data_dir,
        transform=train_transform,
        csv_path=csv_path
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    if args.val_dir:
        val_dataset = LightingDataset(
            args.val_dir,
            transform=val_transform,
        csv_path=csv_path
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
    
    print(f"Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = LightingAssessor(device=device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_mae = float('inf')
    
    if args.resume and Path(args.resume).exists():
        checkpoint = torch.load(args.resume, map_location=device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Loss and optimizer
    # Use L1 loss (MAE) for regression
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_path = save_dir / 'lighting_assessor_best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                    'best_val_mae': best_val_mae,
                }, best_path)
                print(f"Saved best model (Val MAE: {val_mae:.4f})")
            
            scheduler.step(val_mae)
        else:
            # Save checkpoint every epoch if no validation set
            checkpoint_path = save_dir / f'lighting_assessor_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
    
    # Save final model
    final_path = save_dir / 'lighting_assessor_final.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == '__main__':
    main()

