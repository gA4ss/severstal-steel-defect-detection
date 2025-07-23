#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steel Defect Detection - Model Prediction Script
=================================================

This script loads a trained model and generates predictions on the test dataset,
outputting results in the competition submission format.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from model_training import UNetPlusPlus


class TestDataset(Dataset):
    """Dataset for test images"""
    
    def __init__(self, image_dir, image_ids, transform=None, image_size=(256, 1600)):
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # Handle both Path objects and strings
        if isinstance(image_id, Path):
            image_path = self.image_dir / image_id.name
        else:
            image_path = self.image_dir / image_id
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # Store original size (H, W)
        
        # Resize to target size
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, image_id, original_size


def get_test_transforms(image_size=(256, 1600)):
    """Get test transforms (normalization only)"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def custom_collate_fn(batch):
    """
    Custom collate function for TestDataset
    
    Args:
        batch: List of tuples (image, image_id, original_size)
        
    Returns:
        Tuple of (images_tensor, image_ids_list, original_sizes_list)
    """
    images, image_ids, original_sizes = zip(*batch)
    
    # Stack images into tensor
    images = torch.stack(images)
    
    # Keep image_ids and original_sizes as lists
    image_ids = list(image_ids)
    original_sizes = list(original_sizes)
    
    return images, image_ids, original_sizes


def rle_encode(mask):
    """
    Encode binary mask to RLE format
    
    Args:
        mask: 2D numpy array with 0s and 1s
        
    Returns:
        RLE encoded string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle_string, height, width):
    """Decode RLE string to binary mask"""
    if pd.isna(rle_string) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
        
    rle_numbers = [int(x) for x in rle_string.split()]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)
    
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in rle_pairs:
        start -= 1  # RLE is 1-indexed
        mask[start:start + length] = 1
        
    return mask.reshape((height, width), order='F')  # Fortran order for proper reshape

def post_process_mask(mask, min_area=100):
    """
    Post-process predicted mask
    
    Args:
        mask: 2D numpy array with predicted mask
        min_area: Minimum area threshold for connected components
        
    Returns:
        Post-processed mask
    """
    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    
    # Keep only components larger than min_area
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    
    return mask


def predict_batch(model, batch, device, threshold=0.5, tta=False):
    """
    Predict on a batch of images
    
    Args:
        model: Trained model
        batch: Batch of images (tensor) or dict with 'image' key
        device: Device to run inference on
        threshold: Threshold for binary prediction
        tta: Whether to use test-time augmentation
        
    Returns:
        Predicted masks
    """
    # Handle both tensor and dict inputs
    if isinstance(batch, dict):
        images = batch['image'].to(device)
    else:
        images = batch.to(device)
    
    with torch.no_grad():
        if tta:
            # Test-time augmentation: horizontal flip
            outputs = model(images)
            outputs_flip = model(torch.flip(images, dims=[3]))  # Flip horizontally
            outputs_flip = torch.flip(outputs_flip, dims=[3])   # Flip back
            outputs = (outputs + outputs_flip) / 2
        else:
            outputs = model(images)
        
        # Apply sigmoid and threshold
        probs = torch.sigmoid(outputs)
        masks = (probs > threshold).float()
    
    return masks.cpu().numpy(), probs.cpu().numpy()


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Initialize model
    model_config = config.get('model', {})
    model = UNetPlusPlus(
        encoder_name=model_config.get('encoder_name', 'efficientnet-b4'),
        encoder_weights=None,  # Don't load pretrained weights
        in_channels=3,
        classes=4
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle compiled model state dict
    model_state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
        # Remove _orig_mod. prefix for compiled models
        new_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # Remove '_orig_mod.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model_state_dict = new_state_dict
    
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    print(f"Best validation Dice: {checkpoint['dice_score']:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Steel Defect Detection - Model Prediction')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='data/test_images',
                        help='Path to test images directory')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission file path')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--min_area', type=int, default=100,
                        help='Minimum area for post-processing')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get image size from config
    model_config = config.get('model', {})
    image_size = (
        model_config.get('image_height', 256),
        model_config.get('image_width', 1600)
    )
    print(f"Image size: {image_size}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Get test image IDs
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")
    
    image_ids = sorted([f.name for f in test_dir.glob('*.jpg')])
    print(f"Found {len(image_ids)} test images")
    
    # Create test dataset and dataloader
    test_transforms = get_test_transforms(image_size)
    test_dataset = TestDataset(
        image_dir=test_dir,
        image_ids=image_ids,
        transform=test_transforms,
        image_size=image_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Prepare submission data
    submission_data = []
    
    print(f"\nStarting prediction on {len(image_ids)} test images...")
    print(f"Batch size: {args.batch_size}")
    print(f"Threshold: {args.threshold}")
    print(f"TTA: {'Enabled' if args.tta else 'Disabled'}")
    print(f"Post-processing min area: {args.min_area}")
    print("=" * 60)
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Get predictions
            masks, probs = predict_batch(
                model, batch, device, 
                threshold=args.threshold, 
                tta=args.tta
            )
            
            # Process each image in batch
            for i, image_id in enumerate(batch['image_id']):
                # Get mask for this image [num_classes, H, W]
                image_mask = masks[i]  # Shape: [4, H, W]
                
                # Process each class
                for class_id in range(4):
                    class_mask = image_mask[class_id]  # Shape: [H, W]
                    
                    # Post-process mask
                    if args.min_area > 0:
                        class_mask = post_process_mask(class_mask, args.min_area)
                    
                    # Resize back to original size (256, 1600) if needed
                    if class_mask.shape != (256, 1600):
                        class_mask = cv2.resize(
                            class_mask.astype(np.uint8), 
                            (1600, 256), 
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    # Encode to RLE
                    if class_mask.sum() > 0:
                        # Has defect
                        rle = rle_encode(class_mask)
                        submission_data.append({
                            'ImageId': image_id,
                            'EncodedPixels': rle,
                            'ClassId': class_id + 1  # Classes are 1-4
                        })
                    else:
                        # No defect for this class - skip (will be handled by default)
                        pass
                
                # Add default "no defect" entry if no defects found for this image
                has_any_defect = any(masks[i][class_id].sum() > 0 for class_id in range(4))
                if not has_any_defect:
                    submission_data.append({
                        'ImageId': image_id,
                        'EncodedPixels': '1 409600',  # Default "no defect" encoding
                        'ClassId': 0
                    })
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_data)
    
    # Ensure all test images are represented
    all_image_ids = set(image_ids)
    predicted_image_ids = set(submission_df['ImageId'].unique())
    missing_images = all_image_ids - predicted_image_ids
    
    # Add missing images with "no defect" prediction
    for image_id in missing_images:
        submission_df = pd.concat([
            submission_df,
            pd.DataFrame([{
                'ImageId': image_id,
                'EncodedPixels': '1 409600',
                'ClassId': 0
            }])
        ], ignore_index=True)
    
    # Sort by ImageId and ClassId
    submission_df = submission_df.sort_values(['ImageId', 'ClassId']).reset_index(drop=True)
    
    # Save submission file
    submission_df.to_csv(args.output, index=False)
    
    print(f"\nPrediction completed!")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Images with defects: {len(submission_df[submission_df['ClassId'] > 0]['ImageId'].unique())}")
    print(f"Images without defects: {len(submission_df[submission_df['ClassId'] == 0])}")
    print(f"Submission saved to: {args.output}")
    
    # Print some statistics
    class_counts = submission_df[submission_df['ClassId'] > 0]['ClassId'].value_counts().sort_index()
    print(f"\nDefect class distribution:")
    for class_id in range(1, 5):
        count = class_counts.get(class_id, 0)
        print(f"  Class {class_id}: {count} images")


# =============================================================================
# PUBLIC API FUNCTIONS FOR EXTERNAL USE
# =============================================================================

def predict_single_image(image_path, checkpoint_path, config_path=None, 
                         threshold=0.5, tta=False, device=None):
    """
    Predict defects for a single image
    
    Args:
        image_path (str): Path to the input image
        checkpoint_path (str): Path to the trained model checkpoint
        config_path (str, optional): Path to config.json. If None, uses default config
        threshold (float): Threshold for binary prediction (default: 0.5)
        tta (bool): Whether to use test-time augmentation (default: False)
        device (str, optional): Device to use ('cuda' or 'cpu'). If None, auto-detect
    
    Returns:
        dict: {
            'masks': numpy array of shape [4, H, W] - binary masks for each class
            'probabilities': numpy array of shape [4, H, W] - probability maps
            'rle_encoded': dict - RLE encoded masks for each class (if defects found)
            'has_defects': bool - whether any defects were detected
        }
    """
    import json
    from pathlib import Path
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load config
    if config_path is None:
        config_path = 'config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get image size from config
    model_config = config.get('model', {})
    image_size = (
        model_config.get('image_height', 256),
        model_config.get('image_width', 1600)
    )
    
    # Load model
    model = load_model(checkpoint_path, config, device)
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Apply transforms
    test_transforms = get_test_transforms(image_size)
    transformed = test_transforms(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        masks, probs = predict_batch(model, input_tensor, threshold=threshold, tta=tta)
    
    # Remove batch dimension
    masks = masks[0]  # [4, H, W]
    probs = probs[0]  # [4, H, W]
    
    # Resize back to original size
    masks_resized = np.zeros((4, *original_size), dtype=np.uint8)
    probs_resized = np.zeros((4, *original_size), dtype=np.float32)
    
    for class_idx in range(4):
        masks_resized[class_idx] = cv2.resize(
            masks[class_idx].astype(np.uint8), 
            (original_size[1], original_size[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        probs_resized[class_idx] = cv2.resize(
            probs[class_idx], 
            (original_size[1], original_size[0]), 
            interpolation=cv2.INTER_LINEAR
        )
    
    # Encode to RLE
    rle_encoded = {}
    has_defects = False
    
    for class_idx in range(4):
        mask = masks_resized[class_idx]
        if mask.sum() > 0:
            rle = rle_encode(mask)
            if rle:
                rle_encoded[f'class_{class_idx + 1}'] = rle
                has_defects = True
    
    return {
        'masks': masks_resized,
        'probabilities': probs_resized,
        'rle_encoded': rle_encoded,
        'has_defects': has_defects
    }


def predict_batch_images(image_paths, checkpoint_path, config_path=None,
                        threshold=0.5, tta=False, batch_size=8, device=None):
    """
    Predict defects for multiple images in batches
    
    Args:
        image_paths (list): List of paths to input images
        checkpoint_path (str): Path to the trained model checkpoint
        config_path (str, optional): Path to config.json. If None, uses default config
        threshold (float): Threshold for binary prediction (default: 0.5)
        tta (bool): Whether to use test-time augmentation (default: False)
        batch_size (int): Batch size for inference (default: 8)
        device (str, optional): Device to use ('cuda' or 'cpu'). If None, auto-detect
    
    Returns:
        dict: {
            'image_id': {
                'masks': numpy array of shape [4, H, W],
                'probabilities': numpy array of shape [4, H, W],
                'rle_encoded': dict,
                'has_defects': bool
            }, ...
        }
    """
    import json
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load config
    if config_path is None:
        config_path = 'config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get image size from config
    model_config = config.get('model', {})
    image_size = (
        model_config.get('image_height', 256),
        model_config.get('image_width', 1600)
    )
    
    # Load model
    model = load_model(checkpoint_path, config, device)
    
    # Create dataset and dataloader
    image_dir = Path(image_paths[0]).parent
    image_ids = [Path(p).name for p in image_paths]
    
    test_transforms = get_test_transforms(image_size)
    test_dataset = TestDataset(
        image_dir=image_dir,
        image_ids=image_ids,
        transform=test_transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn
    )
    
    results = {}
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, batch_image_ids, original_sizes) in enumerate(test_loader):
            images = images.to(device)
            
            # Predict batch
            masks, probs = predict_batch(model, images, device, threshold=threshold, tta=tta)
            
            # Process each image in batch
            for i, image_id in enumerate(batch_image_ids):
                original_size = original_sizes[i]
                
                # Convert original_size to integers (handle tensor/numpy types)
                if hasattr(original_size, 'item'):
                    # Handle single tensor
                    orig_h, orig_w = int(original_size[0].item()), int(original_size[1].item())
                elif isinstance(original_size, (list, tuple)):
                    # Handle list/tuple, convert each element
                    orig_h = int(original_size[0]) if hasattr(original_size[0], 'item') else int(original_size[0])
                    orig_w = int(original_size[1]) if hasattr(original_size[1], 'item') else int(original_size[1])
                else:
                    # Handle numpy array
                    orig_h, orig_w = int(original_size[0]), int(original_size[1])
                
                # Ensure positive integer dimensions
                orig_h, orig_w = max(1, orig_h), max(1, orig_w)
                
                # Resize back to original size
                masks_resized = np.zeros((4, orig_h, orig_w), dtype=np.uint8)
                probs_resized = np.zeros((4, orig_h, orig_w), dtype=np.float32)
                
                try:
                    for class_idx in range(4):
                        # Handle possible invalid dimensions or types
                        if masks[i][class_idx].sum() == 0 and np.all(masks[i][class_idx] == 0):
                            # If mask is all zeros, skip resize to avoid errors
                            continue
                            
                        # Safe resize with explicit type conversion
                        masks_resized[class_idx] = cv2.resize(
                            masks[i][class_idx].astype(np.uint8), 
                            (int(orig_w), int(orig_h)),  # Explicit integer conversion
                            interpolation=cv2.INTER_NEAREST
                        )
                        probs_resized[class_idx] = cv2.resize(
                            probs[i][class_idx].astype(np.float32),  # Ensure float32
                            (int(orig_w), int(orig_h)),  # Explicit integer conversion
                            interpolation=cv2.INTER_LINEAR
                        )
                except Exception as e:
                    print(f"Warning: Resize error for image {image_id}, class {class_idx}: {e}")
                    # Continue processing other images
                
                # Encode to RLE
                rle_encoded = {}
                has_defects = False
                
                for class_idx in range(4):
                    try:
                        mask = masks_resized[class_idx]
                        # Ensure mask is binary (0 or 1) for proper RLE encoding
                        if mask.sum() > 0:
                            # Binarize mask to ensure clean encoding
                            binary_mask = (mask > 0).astype(np.uint8)
                            rle = rle_encode(binary_mask)
                            if rle:
                                rle_encoded[f'class_{class_idx + 1}'] = rle
                                has_defects = True
                    except Exception as e:
                        print(f"Warning: RLE encoding error for image {image_id}, class {class_idx+1}: {e}")
                        # Continue processing other classes
                
                results[image_id] = {
                    'masks': masks_resized,
                    'probabilities': probs_resized,
                    'rle_encoded': rle_encoded,
                    'has_defects': has_defects
                }
    
    return results


def create_submission_from_test_dir(test_dir, checkpoint_path, output_path='submission.csv',
                                   config_path=None, threshold=0.5, tta=False, batch_size=8):
    """
    Create submission file from test directory
    
    Args:
        test_dir (str): Path to test images directory
        checkpoint_path (str): Path to trained model checkpoint
        output_path (str): Path to save submission CSV (default: 'submission.csv')
        config_path (str, optional): Path to config.json. If None, uses default config
        threshold (float): Threshold for binary prediction (default: 0.5)
        tta (bool): Whether to use test-time augmentation (default: False)
        batch_size (int): Batch size for inference (default: 8)
    
    Returns:
        str: Path to the created submission file
    """
    from pathlib import Path
    import pandas as pd
    
    # Get all test images
    test_dir = Path(test_dir)
    image_paths = list(test_dir.glob('*.jpg'))
    
    if not image_paths:
        raise ValueError(f"No .jpg images found in {test_dir}")
    
    print(f"Found {len(image_paths)} test images")
    
    # Predict all images
    results = predict_batch_images(
        image_paths=image_paths,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        threshold=threshold,
        tta=tta,
        batch_size=batch_size
    )
    
    # Create submission dataframe
    submission_data = []
    
    for image_id, result in results.items():
        # Make sure we handle both full paths and just filenames
        image_name = Path(image_id).stem  # Remove .jpg extension
        
        for class_idx in range(1, 5):  # 1-indexed classes (1,2,3,4)
            class_key = f'class_{class_idx}'
            if class_key in result['rle_encoded']:
                rle = result['rle_encoded'][class_key]
            else:
                rle = ''  # No defect found
            
            submission_data.append({
                'ImageId': f'{image_name}.jpg',  # 完整文件名
                'EncodedPixels': rle,
                'ClassId': class_idx  # 分离的类别ID
            })
    
    # Save submission
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission saved to: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    
    # Print statistics
    defect_count = sum(1 for rle in submission_df['EncodedPixels'] if rle)
    print(f"Images with defects: {defect_count}/{len(submission_df)} ({defect_count/len(submission_df)*100:.1f}%)")
    
    return output_path

def visualize_sample_predictions(test_dir, submission_csv, output_dir='./visualizations', 
                                num_samples=8):
    """
    Visualize sample test images with predictions from submission CSV file
    
    Args:
        test_dir (str): Path to test images directory
        submission_csv (str): Path to submission CSV file with predictions
        output_dir (str): Directory to save visualization results
        num_samples (int): Number of sample images to visualize
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import random
    from pathlib import Path
    
    print(f"Visualizing {num_samples} sample test images with submission predictions...")
    
    # Get test images
    test_dir = Path(test_dir)
    image_paths = list(test_dir.glob('*.jpg'))
    
    if not image_paths:
        raise ValueError(f"No .jpg images found in {test_dir}")
    
    # Load submission CSV
    submission_df = pd.read_csv(submission_csv)
    print(f"Loaded submission file with {len(submission_df)} predictions")
    
    # Sample random images
    random.seed(42)
    sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(num_samples, 7, figsize=(35, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Test Images with Model Predictions (All 4 Classes)', fontsize=16, fontweight='bold')
    
    for idx, image_path in enumerate(sample_paths):
        try:
            # Load original image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # Display original image
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title(f'Original: {image_path.name}')
            axes[idx, 0].axis('off')
            
            # Get image name without extension for matching with submission
            image_name = image_path.stem  # Remove .jpg extension
            
            # Get predictions from submission CSV for this image
            masks = np.zeros((4, original_size[0], original_size[1]), dtype=np.uint8)
            
            for class_id in range(1, 5):  # Classes 1-4
                # Look for prediction in submission CSV
                # Format: ImageId matches image_name.jpg, ClassId matches class_id
                prediction_rows = submission_df[
                    (submission_df['ImageId'] == f"{image_name}.jpg") & 
                    (submission_df['ClassId'] == class_id)
                ]
                
                if not prediction_rows.empty and pd.notna(prediction_rows.iloc[0]['EncodedPixels']):
                    # Decode RLE prediction
                    rle_string = prediction_rows.iloc[0]['EncodedPixels']
                    mask = rle_decode(rle_string, original_size[0], original_size[1])
                    masks[class_id-1] = mask
            
            # Create combined prediction mask
            combined_mask = np.zeros(original_size, dtype=np.uint8)
            has_predictions = False
            
            for class_idx in range(4):
                if masks[class_idx].sum() > 0:
                    combined_mask += masks[class_idx] * (class_idx + 1)
                    has_predictions = True
            
            # Display combined prediction mask
            if has_predictions:
                axes[idx, 1].imshow(combined_mask, cmap='tab10', vmin=0, vmax=4)
                predicted_classes = [i+1 for i in range(4) if masks[i].sum() > 0]
                axes[idx, 1].set_title(f'Predictions (Classes: {predicted_classes})')
            else:
                axes[idx, 1].imshow(np.zeros(original_size), cmap='gray')
                axes[idx, 1].set_title('No Defects Predicted')
            axes[idx, 1].axis('off')
            
            # Display overlay
            overlay = image.copy()
            if has_predictions:
                colored_mask = plt.cm.tab10(combined_mask / 4.0)[:, :, :3]
                overlay = cv2.addWeighted(
                    overlay.astype(np.float32), 0.7, 
                    (colored_mask * 255).astype(np.float32), 0.3, 0
                )
            
            axes[idx, 2].imshow(overlay.astype(np.uint8))
            axes[idx, 2].set_title('Prediction Overlay')
            axes[idx, 2].axis('off')
            
            # Display individual class predictions (All Classes 1-4)
            for class_idx in range(4):  # Now show all 4 classes
                col_idx = 3 + class_idx
                if masks[class_idx].sum() > 0:
                    # Show binary mask (no probability available from CSV)
                    axes[idx, col_idx].imshow(masks[class_idx], cmap='Reds', vmin=0, vmax=1)
                    axes[idx, col_idx].set_title(f'Class {class_idx+1} (Detected)')
                else:
                    axes[idx, col_idx].imshow(np.zeros(original_size), cmap='gray')
                    axes[idx, col_idx].set_title(f'Class {class_idx+1} (None)')
                axes[idx, col_idx].axis('off')
                
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            # Fill row with error message
            for col_idx in range(7):  # Updated for 7-column layout
                axes[idx, col_idx].text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                                      ha='center', va='center', transform=axes[idx, col_idx].transAxes)
                axes[idx, col_idx].set_title(f'{image_path.name} - Error')
                axes[idx, col_idx].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = output_dir / 'test_predictions_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    return str(output_path)


if __name__ == '__main__':
    main()
