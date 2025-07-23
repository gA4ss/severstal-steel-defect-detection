"""
Steel Defect Detection - Model Training Module
"""

import os
import json
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SteelDefectDataset(Dataset):
    """Steel Defect Detection Dataset"""
    
    def __init__(self, df, images_dir, transform=None, mode='train'):
        """
        Initialize dataset
        
        Args:
            df (pd.DataFrame): DataFrame
            images_dir (str): Image directory path
            transform: Data transformation
            mode (str): Mode ('train', 'val', 'test')
        """
        self.df = df
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.mode = mode
        
        # Get unique image list
        self.image_ids = df['ImageId'].unique().tolist()
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = self.images_dir / image_id
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # CRITICAL FIX: RLE encoding is based on original dimensions (256x1600)
        # We must decode with original dimensions, then resize
        original_height, original_width = 256, 1600
        current_height, current_width = image.shape[:2]
        
        # Create masks with original dimensions for RLE decoding
        masks_original = np.zeros((original_height, original_width, 4), dtype=np.float32)
        
        # Get all defect records for this image
        image_df = self.df[self.df['ImageId'] == image_id]
        
        for _, row in image_df.iterrows():
            if pd.notna(row['EncodedPixels']):
                class_id = row['ClassId'] - 1  # Convert to 0-3 index
                # Decode with ORIGINAL dimensions (RLE is encoded for 256x1600)
                mask = self.decode_rle(row['EncodedPixels'], original_height, original_width)
                masks_original[:, :, class_id] = mask
        
        # Resize masks to match current image size if different
        if (current_height, current_width) != (original_height, original_width):
            masks = np.zeros((current_height, current_width, 4), dtype=np.float32)
            for class_idx in range(4):
                if masks_original[:, :, class_idx].sum() > 0:  # Only resize if mask has content
                    masks[:, :, class_idx] = cv2.resize(
                        masks_original[:, :, class_idx], 
                        (current_width, current_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
        else:
            masks = masks_original
        
        # Apply transformation
        if self.transform:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image']
            masks = augmented['mask']
        
        # Convert to tensor and ensure correct dimensions
        # Image should be (C, H, W) after ToTensorV2
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Handle masks - ToTensorV2 converts masks to (C, H, W) format
        if not isinstance(masks, torch.Tensor):
            # If no transform applied, masks are (H, W, C), need to transpose
            if len(masks.shape) == 3 and masks.shape[-1] == 4:
                masks = torch.from_numpy(masks.transpose(2, 0, 1)).float()
            else:
                masks = torch.from_numpy(masks).float()
        else:
            # If ToTensorV2 was applied, masks might be (H, W, C) still
            # ToTensorV2 only transposes images, not masks by default
            if len(masks.shape) == 3 and masks.shape[-1] == 4:
                masks = masks.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
        return {
            'image': image,
            'mask': masks,
            'image_id': image_id
        }
    
    def decode_rle(self, rle_string, height, width):
        """Decode RLE encoded mask"""
        if pd.isna(rle_string):
            return np.zeros((height, width), dtype=np.float32)
            
        rle_numbers = [int(x) for x in rle_string.split()]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        
        mask = np.zeros(height * width, dtype=np.float32)
        for start, length in rle_pairs:
            start -= 1
            mask[start:start + length] = 1
            
        return mask.reshape((height, width), order='F')

class UNetPlusPlus(nn.Module):
    """Improved U-Net++ model"""
    
    def __init__(self, encoder_name='efficientnet-b4', encoder_weights='imagenet', 
                 in_channels=3, classes=4, activation=None):
        """
        Initialize U-Net++ model
        
        Args:
            encoder_name (str): Encoder name
            encoder_weights (str): Pretrained weights
            in_channels (int): Input channels
            classes (int): Output classes
            activation: Activation function
        """
        super(UNetPlusPlus, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
        
    def forward(self, x):
        return self.model(x)

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks - Soft Dice for training"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Soft Dice Loss for training
        
        Args:
            predictions: [batch_size, num_classes, H, W] - model outputs (logits)
            targets: [batch_size, num_classes, H, W] - ground truth masks (0-1 values)
            
        Returns:
            Dice loss (1 - dice_score)
        """
        batch_size, num_classes = predictions.shape[:2]
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        predictions = torch.clamp(predictions, min=1e-7, max=1.0-1e-7)
        
        # Keep targets as continuous values for soft Dice
        targets = targets.float()
        targets = torch.clamp(targets, 0.0, 1.0)
        
        total_dice = 0.0
        
        for class_idx in range(num_classes):
            pred_class = predictions[:, class_idx]  # [batch_size, H, W]
            target_class = targets[:, class_idx]    # [batch_size, H, W]
            
            # Flatten spatial dimensions
            pred_flat = pred_class.view(batch_size, -1)      # [batch_size, H*W]
            target_flat = target_class.view(batch_size, -1)  # [batch_size, H*W]
            
            # Soft Dice calculation with double precision for stability
            intersection = (pred_flat.double() * target_flat.double()).sum(dim=1).float()
            pred_sum = pred_flat.double().sum(dim=1).float()
            target_sum = target_flat.double().sum(dim=1).float()
            
            # Calculate Dice coefficient
            denominator = pred_sum + target_sum
            dice = torch.zeros_like(denominator)
            
            # Handle edge cases
            both_empty = (pred_sum == 0) & (target_sum == 0)
            dice[both_empty] = 1.0  # Perfect score when both are empty
            
            # Calculate Dice for non-empty cases
            non_empty = denominator > 0
            dice[non_empty] = (2.0 * intersection[non_empty] + self.smooth) / (denominator[non_empty] + self.smooth)
            
            # Average across batch
            class_dice = dice.mean()
            total_dice += class_dice
        
        # Average across classes
        avg_dice = total_dice / num_classes
        
        # Return loss (1 - dice)
        return 1 - avg_dice

class FocalLoss(nn.Module):
    """Focal loss function"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        if self.alpha is not None and hasattr(self.alpha, 'shape'):
            # Use class weights if provided
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.alpha, reduction='none')(predictions, targets)
        else:
            bce_loss = nn.BCEWithLogitsLoss(reduction='none')(predictions, targets)
        
        pt = torch.exp(-bce_loss)
        
        if isinstance(self.alpha, torch.Tensor) and self.alpha.numel() > 1:
            # Multi-class alpha weighting
            alpha_t = self.alpha.unsqueeze(0).expand_as(targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        elif self.alpha is not None:
            # Single alpha value
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        else:
            # No alpha weighting (alpha is None)
            focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """Combined loss function"""
    
    def __init__(self, dice_weight=0.5, focal_weight=0.5, alpha=None):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=alpha)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        return self.dice_weight * dice + self.focal_weight * focal

class SteelDefectTrainer:
    """Steel Defect Detection Trainer"""
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config (dict): Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # GPU memory management and optimization
        print("GPU Memory Management:")

        # Clear GPU cache before setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")

            # Set configurable memory fraction to prevent OOM
            hardware_config = config.get('hardware', {})
            memory_fraction = hardware_config.get('memory_fraction', 0.85)  # Default 85%
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print(f"GPU memory fraction set to {memory_fraction*100:.0f}% ({memory_fraction})")

        # RTX 5090 Performance Optimizations
        hardware_config = config.get('hardware', {})
        if hardware_config.get('optimize_for_rtx5090', False):
            print("RTX 5090 optimizations enabled!")

            # Enable TF32 for faster training on RTX 5090
            if hardware_config.get('use_tf32', True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("TF32 enabled for faster matrix operations")

            # Enable cuDNN benchmarking for consistent input sizes
            if hardware_config.get('benchmark_cudnn', True):
                torch.backends.cudnn.benchmark = True
                print("cuDNN benchmarking enabled")
        
        # Set random seeds
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        seed = data_config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize mixed precision training
        self.use_mixed_precision = training_config.get('mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            print("Mixed precision training enabled")
        
        # Enable channels_last memory format for better performance
        self.use_channels_last = training_config.get('channels_last', False)
        if self.use_channels_last:
            print("Channels-last memory format enabled for better GPU utilization")
        
        # Initialize model, optimizer, and criterion
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        
    def setup_model(self):
        """Set up model"""
        print("Setting up model...")
        
        model_config = self.config.get('model', {})
        encoder_name = model_config.get('encoder_name', 'efficientnet-b4')
        encoder_weights = model_config.get('encoder_weights', 'imagenet')
        
        self.model = UNetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=4,
            activation=None
        ).to(self.device)
        
        # Memory optimization strategies
        print("Memory Optimization:")
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model.model, 'encoder'):
            if hasattr(self.model.model.encoder, 'set_gradient_checkpointing'):
                self.model.model.encoder.set_gradient_checkpointing(True)
                print("Gradient checkpointing enabled for encoder")
        
        # Convert model to channels_last memory format for better GPU utilization
        training_config = self.config.get('training', {})
        self.use_channels_last = training_config.get('channels_last', True)
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            print("Model converted to channels_last memory format")
        
        # Compile model for faster execution on RTX 5090 (PyTorch 2.0+)
        if self.config.get('training', {}).get('compile_model', False):
            # Check for Triton availability on Windows
            should_compile = True
            if platform.system() == 'Windows':
                try:
                    import triton
                    print("Triton detected on Windows")
                except ImportError:
                    print("Triton not available on Windows - disabling model compilation")
                    print("This is normal on Windows systems")
                    should_compile = False
            
            if should_compile:
                try:
                    self.model = torch.compile(self.model)
                    print("Model compiled for faster execution")
                except Exception as e:
                    print(f"Model compilation failed: {e}")
                    if "triton" in str(e).lower():
                        print("Triton-related error detected")
                        print("Disabling compilation for stability")
                    print("Continuing without compilation...")
        
        print(f"Model created: {encoder_name}")
        
    def setup_optimizer(self):
        """Set up optimizer and scheduler"""
        print("Setting up optimizer...")
        
        model_config = self.config.get('model', {})
        training_config = self.config.get('training', {})
        
        # 分层学习率
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        encoder_lr = training_config.get('encoder_lr', 1e-5)
        decoder_lr = training_config.get('decoder_lr', 1e-3)
        weight_decay = training_config.get('weight_decay', 1e-4)
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': decoder_params, 'lr': decoder_lr}
        ], weight_decay=weight_decay)
        
        # 学习率调度器
        epochs = training_config.get('epochs', 100)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs,
            eta_min=training_config.get('min_lr', 1e-7)
        )
        
        print("Optimizer and scheduler set up")
        
    def setup_criterion(self):
        """Set up criterion"""
        loss_type = self.config.get('loss_type', 'combined')
        
        # Use class weights if available
        pos_weight = getattr(self, 'class_weights', None)
        
        # Get loss weights with proper priority (training config > model config)
        dice_weight = self.config.get('training', {}).get('dice_weight') or \
                     self.config.get('model', {}).get('dice_weight') or \
                     self.config.get('dice_weight', 0.5)
        focal_weight = self.config.get('training', {}).get('focal_weight') or \
                      self.config.get('model', {}).get('focal_weight') or \
                      self.config.get('focal_weight', 0.5)
        
        if loss_type == 'dice':
            self.criterion = DiceLoss()
        elif loss_type == 'focal':
            self.criterion = FocalLoss(alpha=pos_weight)
        elif loss_type == 'combined':
            self.criterion = CombinedLoss(
                dice_weight=dice_weight,
                focal_weight=focal_weight,
                alpha=pos_weight
            )
        else:
            if pos_weight is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        
        # Print detailed loss configuration
        if pos_weight is not None:
            print(f"Loss function set up: {loss_type} with class weights")
        else:
            print(f"Loss function set up: {loss_type} without class weights")
        
        if loss_type == 'combined':
            print(f"Dice weight: {dice_weight:.1f} ({dice_weight*100:.0f}%)")
            print(f"Focal weight: {focal_weight:.1f} ({focal_weight*100:.0f}%)")
        
    def get_transforms(self):
        """Get data transformations"""
        model_config = self.config.get('model', {})
        train_transform = A.Compose([
            A.Resize(model_config.get('image_height', 256), 
                    model_config.get('image_width', 1600)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            # Removed A.RandomRotate90 to avoid tensor size mismatch
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=3),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(model_config.get('image_height', 256), 
                    model_config.get('image_width', 1600)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return train_transform, val_transform
        
    def prepare_data(self, train_df, images_dir):
        """Prepare data loaders"""
        print("Preparing data...")
        
        # Try to load preprocessed data splits
        preprocessed_dir = Path("preprocessed_data")
        data_splits_path = preprocessed_dir / "data_splits.json"
        
        if data_splits_path.exists():
            print("Loading preprocessed data splits...")
            import json
            with open(data_splits_path, 'r') as f:
                splits_info = json.load(f)
            
            # Use preprocessed stratified splits
            train_imgs = splits_info['train_images']
            val_imgs = splits_info['test_images']  # Use test split as validation
            
            print(f"Using preprocessed splits: {len(train_imgs)} train, {len(val_imgs)} val images")
        else:
            print("No preprocessed splits found, using random split...")
            # Fallback to random split
            training_config = self.config.get('training', {})
            data_config = self.config.get('data', {})
            train_images = train_df['ImageId'].unique()
            train_imgs, val_imgs = train_test_split(
                train_images, 
                test_size=training_config.get('val_split', 0.2),
                random_state=data_config.get('seed', 42)
            )
            train_imgs = train_imgs.tolist()
            val_imgs = val_imgs.tolist()
        
        # Check for balanced dataset
        balanced_data_path = preprocessed_dir / "balanced_train_oversample.csv"
        data_config = self.config.get('data', {})
        if balanced_data_path.exists() and data_config.get('use_balanced_data', False):
            print("Loading balanced dataset...")
            balanced_df = pd.read_csv(balanced_data_path)
            train_data = balanced_df[balanced_df['ImageId'].isin(train_imgs)]
            print(f"Using balanced dataset with {len(balanced_df)} samples")
        else:
            train_data = train_df[train_df['ImageId'].isin(train_imgs)]
            print("Using original unbalanced dataset")
        
        val_data = train_df[train_df['ImageId'].isin(val_imgs)]
        
        print(f"Training images: {len(train_imgs)}, Validation images: {len(val_imgs)}")
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        # Load defect characteristics for class weights
        defect_chars_path = preprocessed_dir / "defect_characteristics.json"
        if defect_chars_path.exists():
            with open(defect_chars_path, 'r') as f:
                defect_stats = json.load(f)
            
            # Calculate class weights based on defect statistics
            class_counts = [defect_stats[str(i)]['count'] for i in range(1, 5)]
            total_samples = sum(class_counts)
            class_weights = [total_samples / (4 * count) for count in class_counts]
            
            print("Defect class distribution:")
            for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
                print(f"  Class {i+1}: {count} samples, weight: {weight:.3f}")
            
            # Store class weights for loss function
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        else:
            self.class_weights = None
            print("No defect characteristics found, using equal class weights")
        
        # Get transformations
        train_transform, val_transform = self.get_transforms()
        
        # Create datasets
        train_dataset = SteelDefectDataset(train_data, images_dir, train_transform, 'train')
        val_dataset = SteelDefectDataset(val_data, images_dir, val_transform, 'val')
        
        # Create data loaders with RTX 5090 optimizations
        training_config = self.config.get('training', {})
        
        batch_size = training_config.get('batch_size', 32)
        num_workers = training_config.get('num_workers', 8)
        pin_memory = training_config.get('pin_memory', True)
        persistent_workers = training_config.get('persistent_workers', True)
        prefetch_factor = training_config.get('prefetch_factor', 2)
        
        # Windows compatibility adjustments
        if platform.system() == 'Windows':
            print("Windows system detected - applying compatibility adjustments:")
            
            # Reduce workers if too high for Windows stability
            if num_workers > 6:
                original_workers = num_workers
                num_workers = 4
                print(f"  Reduced num_workers: {original_workers} → {num_workers} (Windows stability)")
            
            # Disable pin_memory on Windows if enabled (can cause issues)
            if pin_memory:
                pin_memory = False
                print(f"  Disabled pin_memory: True → False (Windows compatibility)")
            
            # Disable persistent_workers on Windows if enabled (shared memory issues)
            if persistent_workers:
                persistent_workers = False
                print(f"  Disabled persistent_workers: True → False (Windows shared memory fix)")
            
            # Reduce prefetch_factor if too high
            if prefetch_factor > 2:
                original_prefetch = prefetch_factor
                prefetch_factor = 2
                print(f"  Reduced prefetch_factor: {original_prefetch} → {prefetch_factor} (Windows memory)")
        
        # RTX 5090 DataLoader optimizations
        print(f"\nDataLoader configuration for RTX 5090:")
        print(f"  Batch size: {batch_size}")
        print(f"  Workers: {num_workers}")
        print(f"  Prefetch factor: {prefetch_factor}")
        print(f"  Pin memory: {pin_memory}")
        print(f"  Persistent workers: {persistent_workers}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
        
        return train_loader, val_loader
        
    def calculate_soft_dice_score(self, predictions, targets, smooth=1e-6):
        """
        Calculate soft Dice score for training (matches DiceLoss calculation exactly)
        
        Args:
            predictions: [batch_size, num_classes, H, W] - model outputs (logits)
            targets: [batch_size, num_classes, H, W] - ground truth masks (0-1 values)
            smooth: Smoothing factor
            
        Returns:
            Average soft Dice score across all classes
        """
        batch_size, num_classes = predictions.shape[:2]
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        predictions = torch.clamp(predictions, min=1e-7, max=1.0-1e-7)
        
        # Keep targets as continuous values for soft Dice (NO BINARIZATION!)
        targets = targets.float()
        targets = torch.clamp(targets, 0.0, 1.0)
        
        total_dice = 0.0
        
        for class_idx in range(num_classes):
            pred_class = predictions[:, class_idx]  # [batch_size, H, W]
            target_class = targets[:, class_idx]    # [batch_size, H, W]
            
            # Flatten spatial dimensions
            pred_flat = pred_class.view(batch_size, -1)      # [batch_size, H*W]
            target_flat = target_class.view(batch_size, -1)  # [batch_size, H*W]
            
            # Soft Dice calculation with double precision for stability
            intersection = (pred_flat.double() * target_flat.double()).sum(dim=1).float()
            pred_sum = pred_flat.double().sum(dim=1).float()
            target_sum = target_flat.double().sum(dim=1).float()
            
            # Calculate Dice coefficient
            denominator = pred_sum + target_sum
            dice = torch.zeros_like(denominator)
            
            # Handle edge cases
            both_empty = (pred_sum == 0) & (target_sum == 0)
            dice[both_empty] = 1.0  # Perfect score when both are empty
            
            # Calculate Dice for non-empty cases
            non_empty = denominator > 0
            dice[non_empty] = (2.0 * intersection[non_empty] + smooth) / (denominator[non_empty] + smooth)
            
            # Average across batch
            class_dice = dice.mean()
            total_dice += class_dice
        
        # Average across classes
        avg_dice = total_dice / num_classes
        
        return avg_dice

    def calculate_dice_score(self, predictions, targets, threshold=0.5, smooth=1e-6):
        """Calculate Dice score with improved numerical stability for multi-class segmentation"""
        # CRITICAL FIX: Clamp predictions to prevent inf values
        predictions = torch.sigmoid(predictions)
        predictions = torch.clamp(predictions, min=1e-7, max=1.0-1e-7)  # Prevent inf/nan
        targets = targets.float()
        
        # Ensure targets are in [0,1] range (binary masks)
        targets = torch.clamp(targets, 0.0, 1.0)
        
        # Calculate Dice for each class separately
        batch_size, num_classes = predictions.shape[:2]
        
        dice_scores = []
        for class_idx in range(num_classes):
            pred_class = predictions[:, class_idx, :, :]
            target_class = targets[:, class_idx, :, :]
            
            # Flatten for each class
            pred_flat = pred_class.view(batch_size, -1)
            target_flat = target_class.view(batch_size, -1)
            
            # Always ensure target is binary (0 or 1) for proper Dice calculation
            target_flat = (target_flat > 0.5).float()
            
            # CRITICAL FIX: Use double precision for large sums to prevent overflow
            intersection = (pred_flat.double() * target_flat.double()).sum(dim=1).float()
            pred_sum = pred_flat.double().sum(dim=1).float()
            target_sum = target_flat.double().sum(dim=1).float()
            
            # Calculate Dice with improved numerical stability
            denominator = pred_sum + target_sum
            dice = torch.zeros_like(denominator)
            
            # Case 1: Both prediction and target are empty (perfect match)
            both_empty = (pred_sum == 0) & (target_sum == 0)
            dice[both_empty] = 1.0
            
            # Case 2: Normal case with non-zero denominator
            non_empty = denominator > 0
            dice[non_empty] = (2.0 * intersection[non_empty] + smooth) / (denominator[non_empty] + smooth)
            
            # Case 3: One is empty, other is not (worst case) - dice remains 0
            
            dice_scores.append(dice.mean())
        
        # Average Dice across all classes
        avg_dice = torch.stack(dice_scores).mean()
        
        return avg_dice
        
    def train_epoch(self, train_loader):
        """Train one epoch with mixed precision support"""
        self.model.train()
        total_loss = 0
        total_dice = 0
        
        # Get debug configuration
        debug_training = self.config.get('debug', {}).get('training_loop', {})
        debug_data = self.config.get('debug', {}).get('data_loading', {})
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Debug: Data loading information
            if debug_data.get('enabled', False) and batch_idx < 3:
                if debug_data.get('show_batch_info', True):
                    print(f"\n[DEBUG BATCH {batch_idx+1}] Data Loading:")
                    print(f"  Images shape: {images.shape}")
                    print(f"  Masks shape: {masks.shape}")
                    print(f"  Images range: [{images.min():.4f}, {images.max():.4f}]")
                    print(f"  Masks range: [{masks.min():.4f}, {masks.max():.4f}]")
                    print(f"  Masks unique values: {torch.unique(masks)}")
                    
                if debug_data.get('validate_masks', True):
                    mask_sum_per_class = masks.sum(dim=[0, 2, 3])  # Sum over batch, H, W
                    print(f"  Mask pixels per class: {mask_sum_per_class}")
                    print(f"  Total mask pixels: {masks.sum().item():.0f}")
            
            # Apply channels_last memory format for better GPU utilization
            if self.use_channels_last:
                images = images.to(memory_format=torch.channels_last)
                masks = masks.to(memory_format=torch.channels_last)
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(images)
                    
                    # Debug: Model outputs analysis
                    if debug_training.get('enabled', False) and batch_idx < 3:
                        if debug_training.get('check_model_outputs', True):
                            print(f"\n[DEBUG BATCH {batch_idx+1}] Model Outputs:")
                            print(f"  Outputs shape: {outputs.shape}")
                            print(f"  Outputs range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                            print(f"  Outputs has inf: {torch.isinf(outputs).any().item()}")
                            print(f"  Outputs has nan: {torch.isnan(outputs).any().item()}")
                            
                            # Check outputs after sigmoid
                            sigmoid_outputs = torch.sigmoid(outputs)
                            print(f"  Sigmoid outputs range: [{sigmoid_outputs.min():.4f}, {sigmoid_outputs.max():.4f}]")
                    
                    # CRITICAL FIX: Check for inf/nan in outputs
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"WARNING: inf/nan detected in model outputs at batch {batch_idx}")
                        outputs = torch.clamp(outputs, min=-10.0, max=10.0)  # Clamp logits
                    
                    loss = self.criterion(outputs, masks)
                    
                    # Debug: Loss components
                    if debug_training.get('enabled', False) and batch_idx < 3:
                        if debug_training.get('show_loss_components', True):
                            print(f"  Total Loss: {loss.item():.6f}")
                            # Try to get individual loss components if available
                            if hasattr(self.criterion, 'get_loss_components'):
                                components = self.criterion.get_loss_components(outputs, masks)
                                for name, value in components.items():
                                    print(f"  {name}: {value:.6f}")
                
                # Mixed precision backward pass with gradient clipping
                self.scaler.scale(loss).backward()
                
                # Debug: Gradient analysis
                if debug_training.get('enabled', False) and batch_idx < 3:
                    if debug_training.get('check_gradients', True):
                        self.scaler.unscale_(self.optimizer)
                        total_norm = 0
                        param_count = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                param_count += 1
                        total_norm = total_norm ** (1. / 2)
                        print(f"  Gradient norm: {total_norm:.6f} (from {param_count} parameters)")
                        print(f"  Gradient has inf: {any(torch.isinf(p.grad).any() for p in self.model.parameters() if p.grad is not None)}")
                        print(f"  Gradient has nan: {any(torch.isnan(p.grad).any() for p in self.model.parameters() if p.grad is not None)}")
                    else:
                        self.scaler.unscale_(self.optimizer)
                else:
                    self.scaler.unscale_(self.optimizer)
                
                # CRITICAL FIX: Add gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                
                # Debug: Model outputs analysis (standard training)
                if debug_training.get('enabled', False) and batch_idx < 3:
                    if debug_training.get('check_model_outputs', True):
                        print(f"\n[DEBUG BATCH {batch_idx+1}] Model Outputs (Standard):")
                        print(f"  Outputs shape: {outputs.shape}")
                        print(f"  Outputs range: [{outputs.min():.4f}, {outputs.max():.4f}]")
                        print(f"  Outputs has inf: {torch.isinf(outputs).any().item()}")
                        print(f"  Outputs has nan: {torch.isnan(outputs).any().item()}")
                        
                        # Check outputs after sigmoid
                        sigmoid_outputs = torch.sigmoid(outputs)
                        print(f"  Sigmoid outputs range: [{sigmoid_outputs.min():.4f}, {sigmoid_outputs.max():.4f}]")
                
                # CRITICAL FIX: Check for inf/nan in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"WARNING: inf/nan detected in model outputs at batch {batch_idx}")
                    outputs = torch.clamp(outputs, min=-10.0, max=10.0)  # Clamp logits
                
                loss = self.criterion(outputs, masks)
                
                # Debug: Loss components (standard training)
                if debug_training.get('enabled', False) and batch_idx < 3:
                    if debug_training.get('show_loss_components', True):
                        print(f"  Total Loss: {loss.item():.6f}")
                        # Try to get individual loss components if available
                        if hasattr(self.criterion, 'get_loss_components'):
                            components = self.criterion.get_loss_components(outputs, masks)
                            for name, value in components.items():
                                print(f"  {name}: {value:.6f}")
                
                loss.backward()
                
                # Debug: Gradient analysis (standard training)
                if debug_training.get('enabled', False) and batch_idx < 3:
                    if debug_training.get('check_gradients', True):
                        total_norm = 0
                        param_count = 0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                                param_count += 1
                        total_norm = total_norm ** (1. / 2)
                        print(f"  Gradient norm: {total_norm:.6f} (from {param_count} parameters)")
                        print(f"  Gradient has inf: {any(torch.isinf(p.grad).any() for p in self.model.parameters() if p.grad is not None)}")
                        print(f"  Gradient has nan: {any(torch.isnan(p.grad).any() for p in self.model.parameters() if p.grad is not None)}")
                
                # CRITICAL FIX: Add gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Calculate metrics - Use soft Dice for training (matches DiceLoss)
            with torch.no_grad():
                train_dice = self.calculate_soft_dice_score(outputs, masks)
            
            total_loss += loss.item()
            total_dice += train_dice.item()
            
            # Debug: Memory usage monitoring
            if debug_training.get('enabled', False) and batch_idx < 3:
                if debug_training.get('show_memory_usage', True):
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                        print(f"  GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            # Debug: Batch statistics summary
            if debug_training.get('enabled', False) and batch_idx < 3:
                if debug_training.get('show_batch_stats', True):
                    print(f"  Batch {batch_idx+1} Summary - Loss: {loss.item():.6f}, Dice: {train_dice.item():.4f}")
            
            # Memory cleanup every 10 batches to prevent accumulation
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{train_dice.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        return avg_loss, avg_dice
        
    def validate_epoch(self, val_loader):
        """Validate one epoch with optimized data loading"""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for batch in progress_bar:
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Apply channels_last memory format for better GPU utilization
                if self.use_channels_last:
                    images = images.to(memory_format=torch.channels_last)
                    masks = masks.to(memory_format=torch.channels_last)
                
                # Forward pass (use autocast for validation too if mixed precision enabled)
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Calculate metrics - Apply sigmoid and threshold for hard Dice calculation
                sigmoid_outputs = torch.sigmoid(outputs)
                binary_outputs = (sigmoid_outputs > 0.5).float()
                dice_score = self.calculate_dice_score(binary_outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice_score.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice_score.item():.4f}'
                })
                
        avg_loss = total_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        return avg_loss, avg_dice
        
    def print_training_config(self):
        """Print detailed training configuration"""
        print("\n" + "="*80)
        print("STEEL DEFECT DETECTION TRAINING CONFIGURATION")
        print("="*80)
        
        # Model Configuration
        print("\nMODEL CONFIGURATION:")
        model_config = self.config.get('model', {})
        print(f"  * Architecture: UNet++ with {model_config.get('encoder_name', 'efficientnet-b4')} encoder")
        print(f"  * Encoder weights: {model_config.get('encoder_weights', 'imagenet')}")
        print(f"  * Image resolution: {model_config.get('image_height', 256)}x{model_config.get('image_width', 1600)}")
        print(f"  * Input channels: 3 (RGB)")
        print(f"  * Output classes: 4 (defect types)")
        
        # Training Configuration
        print("\nTRAINING CONFIGURATION:")
        training_config = self.config.get('training', {})
        print(f"  * Batch size: {training_config.get('batch_size', 8)}")
        print(f"  * Total epochs: {training_config.get('epochs', 100)}")
        print(f"  * Validation split: {training_config.get('val_split', 0.2)*100:.0f}%")
        print(f"  * Early stopping patience: {training_config.get('patience', 10)} epochs")
        print(f"  * Mixed precision: {'Enabled' if training_config.get('mixed_precision', False) else 'Disabled'}")
        print(f"  * Balanced data: {'Enabled' if training_config.get('use_balanced_data', False) else 'Disabled'}")
        print(f"  * Use class weights: {'Enabled' if training_config.get('use_class_weights', False) else 'Disabled'}")
        print(f"  * Gradient clipping: {'Enabled' if training_config.get('gradient_clipping', False) else 'Disabled'}")
        if training_config.get('gradient_clipping', False):
            print(f"  * Max gradient norm: {training_config.get('max_grad_norm', 1.0)}")
        
        # Optimizer Configuration
        print("\nOPTIMIZER CONFIGURATION:")
        print(f"  * Encoder learning rate: {training_config.get('encoder_lr', 1e-4):.0e}")
        print(f"  * Decoder learning rate: {training_config.get('decoder_lr', 1e-3):.0e}")
        print(f"  * Weight decay: {training_config.get('weight_decay', 1e-4):.0e}")
        print(f"  * Minimum learning rate: {training_config.get('min_lr', 1e-8):.0e}")
        print(f"  * Scheduler: CosineAnnealingLR")
        
        # Loss Configuration
        print("\nLOSS CONFIGURATION:")
        # Read loss_type from training config first, then model config
        loss_type = training_config.get('loss_type') or model_config.get('loss_type', 'combined')
        print(f"  * Loss type: {loss_type.upper()}")
        if loss_type == 'combined':
            # Use same logic as setup_criterion for consistency
            dice_weight = training_config.get('dice_weight') or model_config.get('dice_weight', 0.5)
            focal_weight = training_config.get('focal_weight') or model_config.get('focal_weight', 0.5)
            print(f"  * Dice loss weight: {dice_weight:.1f} ({dice_weight*100:.0f}%)")
            print(f"  * Focal loss weight: {focal_weight:.1f} ({focal_weight*100:.0f}%)")
        
        # Data Loading Configuration
        print("\nDATA LOADING CONFIGURATION:")
        data_config = self.config.get('data', {})
        print(f"  * Number of workers: {training_config.get('num_workers', 4)}")
        print(f"  * Pin memory: {'Enabled' if training_config.get('pin_memory', False) else 'Disabled'}")
        print(f"  * Persistent workers: {'Enabled' if training_config.get('persistent_workers', False) else 'Disabled'}")
        print(f"  * Data augmentation: {'Enabled' if data_config.get('augmentation', True) else 'Disabled'}")
        print(f"  * Normalization: {'Enabled' if data_config.get('normalize', True) else 'Disabled'}")
        
        # Hardware Configuration
        print("\nHARDWARE CONFIGURATION:")
        hardware_config = self.config.get('hardware', {})
        print(f"  * Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  * GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  * GPU Memory: {total_memory:.1f} GB")
            print(f"  * CUDA Version: {torch.version.cuda}")
            
            # Memory management settings
            memory_fraction = hardware_config.get('memory_fraction', 0.85)
            available_memory = total_memory * memory_fraction
            print(f"  * Memory fraction: {memory_fraction*100:.0f}% ({memory_fraction})")
            print(f"  * Available memory: {available_memory:.1f} GB")
            
            # RTX 5090 optimizations
            if hardware_config.get('optimize_for_rtx5090', False):
                print(f"  * RTX 5090 optimizations: Enabled")
                print(f"    - TF32: {'Enabled' if hardware_config.get('use_tf32', True) else 'Disabled'}")
                print(f"    - cuDNN benchmark: {'Enabled' if hardware_config.get('benchmark_cudnn', True) else 'Disabled'}")
            else:
                print(f"  * RTX 5090 optimizations: Disabled")
        
        # Advanced Training Features
        print("\nADVANCED FEATURES:")
        print(f"  * Channels-last memory format: {'Enabled' if training_config.get('channels_last', True) else 'Disabled'}")
        print(f"  * Model compilation: {'Enabled' if training_config.get('compile_model', False) else 'Disabled'}")
        print(f"  * Gradient accumulation steps: {training_config.get('gradient_accumulation_steps', 1)}")
        print(f"  * Prefetch factor: {training_config.get('prefetch_factor', 2)}")
        
        # Memory monitoring handled by external tools
        
        print("\n" + "="*80)
        print("STARTING TRAINING...")
        print("="*80 + "\n")

    def train(self, train_df, images_dir):
        """Complete training process with automatic checkpoint recovery"""
        # Print detailed configuration
        self.print_training_config()
        
        # Set up model, optimizer, and criterion
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(train_df, images_dir)
        
        # Check for existing checkpoint and auto-recovery
        start_epoch = 0
        best_dice = 0
        patience_counter = 0
        
        # Get training configuration first
        training_config = self.config.get('training', {})
        
        # Get resume training configuration
        resume_config = training_config.get('resume_training', {})
        resume_enabled = resume_config.get('enabled', True)
        
        if resume_enabled:
            checkpoint_path = None
            
            # Try to find checkpoint using configuration
            primary_path = resume_config.get('checkpoint_path', 'best_model.pth')
            auto_detect = resume_config.get('auto_detect', True)
            fallback_paths = resume_config.get('fallback_paths', ['final_model.pth'])
            
            # Build list of paths to try
            paths_to_try = [primary_path]
            if auto_detect:
                paths_to_try.extend(fallback_paths)
            
            # Try to find an existing checkpoint
            for path in paths_to_try:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path:
                try:
                    print(f"Found existing checkpoint: {checkpoint_path}")
                    print("Attempting to resume training...")
                    
                    # Load checkpoint
                    resume_epoch, resume_dice = self.load_checkpoint(checkpoint_path)
                    start_epoch = resume_epoch + 1  # Start from next epoch
                    best_dice = resume_dice
                    
                    print(f"Successfully resumed from epoch {resume_epoch}")
                    print(f"Previous best Dice score: {best_dice:.4f}")
                    print(f"Continuing training from epoch {start_epoch}")
                    
                except Exception as e:
                    print(f"Failed to load checkpoint '{checkpoint_path}': {str(e)}")
                    print("Starting training from scratch...")
                    start_epoch = 0
                    best_dice = 0
            else:
                print(f"No checkpoint found. Searched paths: {paths_to_try}")
                print("Starting training from scratch...")
        else:
            print("Resume training disabled in config. Starting training from scratch...")
        
        # Training loop parameters
        training_config = self.config.get('training', {})
        total_epochs = training_config.get('epochs', 100)
        patience = training_config.get('patience', 10)
        
        # If we've already completed training
        if start_epoch >= total_epochs:
            print(f"Training already completed! (Reached epoch {start_epoch-1}/{total_epochs})")
            print(f"Best Dice score achieved: {best_dice:.4f}")
            return
        
        print(f"Training will run from epoch {start_epoch+1} to {total_epochs}")
        
        for epoch in range(start_epoch, total_epochs):
            print(f"\nEpoch {epoch+1}/{total_epochs}")
            
            # Train
            train_loss, train_dice = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_dice = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dices.append(train_dice)
            self.val_dices.append(val_dice)
            
            print(f"Training - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print(f"Validation - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                self.save_checkpoint(epoch, val_dice, 'best_model.pth')
                print(f"Best model saved (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered (patience: {patience})")
                break

            # Save final model (regardless of performance)
            final_epoch = min(epoch, total_epochs - 1)
            final_val_dice = self.val_dices[-1] if self.val_dices else 0
            self.save_checkpoint(final_epoch, final_val_dice, 'final_model.pth')
            print(f"Final model saved (Epoch {final_epoch+1}, Dice: {final_val_dice:.4f})")
        
        print(f"Training completed!")
        print(f"Best model: Epoch with Dice {best_dice:.4f} (saved as 'best_model.pth')")
        print(f"Final model: Epoch {final_epoch+1} with Dice {final_val_dice:.4f} (saved as 'final_model.pth')") 
        
    def save_checkpoint(self, epoch, dice_score, filename):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dice_score': dice_score,
            'config': self.config
        }
        torch.save(checkpoint, filename)
        
    def load_checkpoint(self, filename):
        """Load checkpoint with compatibility for compiled models"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        # Handle state_dict compatibility between compiled and non-compiled models
        model_state_dict = checkpoint['model_state_dict']
        
        try:
            # Try loading directly first
            self.model.load_state_dict(model_state_dict)
        except RuntimeError as e:
            error_msg = str(e)
            if "Missing key(s) in state_dict" in error_msg and "_orig_mod" in error_msg:
                print("  • Detected compiled model checkpoint compatibility issue")
                print("  • Attempting to fix state_dict keys...")
                
                # Get current model's state_dict keys
                current_keys = set(self.model.state_dict().keys())
                checkpoint_keys = set(model_state_dict.keys())
                
                # Determine the correct transformation based on both key sets
                current_has_orig_mod = any(key.startswith("_orig_mod.") for key in current_keys)
                checkpoint_has_orig_mod = any(key.startswith("_orig_mod.") for key in checkpoint_keys)
                
                if checkpoint_has_orig_mod and not current_has_orig_mod:
                    # Checkpoint has _orig_mod prefix, current model doesn't - remove prefix
                    print("  • Removing _orig_mod prefix from checkpoint keys")
                    new_state_dict = {}
                    for key, value in model_state_dict.items():
                        if key.startswith("_orig_mod."):
                            new_key = key[10:]  # Remove "_orig_mod." prefix
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    model_state_dict = new_state_dict
                elif not checkpoint_has_orig_mod and current_has_orig_mod:
                    # Current model has _orig_mod prefix, checkpoint doesn't - add prefix
                    print("  • Adding _orig_mod prefix to checkpoint keys")
                    new_state_dict = {}
                    for key, value in model_state_dict.items():
                        if not key.startswith("_orig_mod."):
                            new_key = "_orig_mod." + key
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    model_state_dict = new_state_dict
                else:
                    # Both have same prefix state, this shouldn't happen but handle gracefully
                    print("  • Unexpected key mismatch pattern, re-raising original error")
                    raise e
                
                # Try loading again with fixed keys
                try:
                    self.model.load_state_dict(model_state_dict)
                    print("  • Successfully loaded checkpoint with fixed keys")
                except Exception as retry_error:
                    print(f"  • Failed to load even after key fixing: {retry_error}")
                    print("  • This might indicate a more serious compatibility issue")
                    raise retry_error
            else:
                # Re-raise if it's a different type of error
                print(f"  • Checkpoint loading failed with different error: {error_msg}")
                raise e
        
        # Check for configuration differences that might affect training
        # if 'config' in checkpoint:
        #     saved_config = checkpoint['config']
        #     current_config = self.config
            
        #     # Compare critical training parameters
        #     saved_training = saved_config.get('training', {})
        #     current_training = current_config.get('training', {})
            
        #     critical_params = [
        #         'encoder_lr', 'decoder_lr', 'dice_weight', 'focal_weight', 'batch_size',
        #         'use_class_weights', 'mixed_precision', 'gradient_clipping', 'max_grad_norm',
        #         'weight_decay', 'loss_type'
        #     ]
        #     config_differences = []
            
        #     for param in critical_params:
        #         saved_val = saved_training.get(param)
        #         current_val = current_training.get(param)
        #         if saved_val != current_val:
        #             config_differences.append(f"{param}: {saved_val} -> {current_val}")
            
        #     if config_differences:
        #         print("\n*** WARNING: Configuration differences detected! ***")
        #         print("This may explain why Dice performance dropped after resuming:")
        #         for diff in config_differences:
        #             print(f"  - {diff}")
        #         print("\nRECOMMENDATIONS:")
        #         print("1. Use original configuration for consistent results")
        #         print("2. Or set reset_optimizer=True to restart with new config")
        #         print("3. Consider gradual parameter changes instead of large jumps\n")
                
        #         # Check if learning rate changed significantly (>5x difference)
        #         lr_changed_significantly = False
        #         for param in ['encoder_lr', 'decoder_lr']:
        #             saved_val = saved_training.get(param, 1e-4)
        #             current_val = current_training.get(param, 1e-4)
        #             if saved_val and current_val:
        #                 ratio = max(saved_val/current_val, current_val/saved_val)
        #                 if ratio > 5.0:
        #                     lr_changed_significantly = True
        #                     break
                
        #         if lr_changed_significantly:
        #             print("*** CRITICAL: Learning rate changed by >5x! ***")
        #             print("This likely explains the Dice performance drop.")
        #             print("Strongly recommend using original learning rates.\n")
                
        #         # Special case: Original config had None learning rates
        #         if saved_training.get('encoder_lr') is None or saved_training.get('decoder_lr') is None:
        #             print("*** SPECIAL CASE: Original checkpoint had None learning rates ***")
        #             print("This indicates optimizer state incompatibility.")
        #             print("SOLUTION: Reset optimizer state to use new learning rates.\n")
                    
        #             # Reset optimizer state for compatibility
        #             print("Resetting optimizer and scheduler states for new configuration...")
        #             # Don't load optimizer/scheduler states - let them use new config
        #             return checkpoint['epoch'], checkpoint['dice_score']
        #     else:
        #         print("Configuration consistency verified - no critical parameter changes detected.")
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['dice_score']
        
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Convert tensors to CPU numpy arrays for plotting
        def tensor_to_numpy(data):
            """Convert tensor or list of tensors to numpy array"""
            if not data:
                return []
            
            numpy_data = []
            for item in data:
                if torch.is_tensor(item):
                    numpy_data.append(item.detach().cpu().numpy())
                else:
                    numpy_data.append(float(item))
            return numpy_data
        
        # Convert all data to numpy
        train_losses_np = tensor_to_numpy(self.train_losses)
        val_losses_np = tensor_to_numpy(self.val_losses)
        train_dices_np = tensor_to_numpy(self.train_dices)
        val_dices_np = tensor_to_numpy(self.val_dices)
        
        # Loss curve
        if train_losses_np:
            ax1.plot(train_losses_np, label='Training loss')
        if val_losses_np:
            ax1.plot(val_losses_np, label='Validation loss')
        ax1.set_title('Loss curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Dice curve
        if train_dices_np:
            ax2.plot(train_dices_np, label='Training Dice')
        if val_dices_np:
            ax2.plot(val_dices_np, label='Validation Dice')
        ax2.set_title('Dice curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function - Execute model training"""
    print("Start steel defect detection model training...")
    
    # Load configuration from config.json
    import json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("Configuration loaded from config.json")
    except FileNotFoundError:
        print("Error: config.json not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing config.json: {e}")
        return
    
    # Load data
    train_df = pd.read_csv('data/train.csv')
    images_dir = 'data/train_images'
    
    # Create trainer
    trainer = SteelDefectTrainer(config)
    
    # Start training
    trainer.train(train_df, images_dir)
    
    # Plot training history
    trainer.plot_training_history()
    
    print("Model training completed!")

if __name__ == "__main__":
    main()
