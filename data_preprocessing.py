"""
Data Preprocessing Module for Steel Defect Detection
"""

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class SteelDefectPreprocessor:
    """Steel Defect Data Preprocessor"""
    
    def __init__(self, data_dir="data", output_dir="preprocessed_data"):
        """
        Initialize preprocessor
        
        Args:
            data_dir (str): Original data directory
            output_dir (str): Preprocessed data output directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.train_csv_path = self.data_dir / "train.csv"
        self.train_images_dir = self.data_dir / "train_images"
        self.test_images_dir = self.data_dir / "test_images"
        
        # Load data
        self.train_df = pd.read_csv(self.train_csv_path)
        
    def analyze_data_quality(self):
        """Analyze data quality"""
        print("=" * 50)
        print("Data quality analysis")
        print("=" * 50)
        
        # Basic information
        print(f"\nBasic information:")
        print(f"   - Training records: {len(self.train_df):,}")
        print(f"   - Unique images: {self.train_df['ImageId'].nunique():,}")
        print(f"   - Defect classes: {self.train_df['ClassId'].nunique()}")
        
        # Missing value check
        print(f"\nMissing value check:")
        missing_info = self.train_df.isnull().sum()
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                print(f"   - {col}: {missing_count} missing values")
            else:
                print(f"   - {col}: No missing values")
                
        # Defect distribution
        print(f"\nDefect distribution:")
        defect_counts = self.train_df['ClassId'].value_counts().sort_index()
        total_defects = len(self.train_df[self.train_df['EncodedPixels'].notna()])
        
        for class_id, count in defect_counts.items():
            percentage = count / total_defects * 100
            print(f"   - Class {class_id}: {count:,} ({percentage:.1f}%)")
            
        # Image file check
        print(f"\nImage file check:")
        train_images = list(self.train_images_dir.glob("*.jpg"))
        test_images = list(self.test_images_dir.glob("*.jpg"))
        
        print(f"   - Training image files: {len(train_images)}")
        print(f"   - Test image files: {len(test_images)}")
        
        # Check for records without corresponding image files
        missing_images = []
        for image_id in self.train_df['ImageId'].unique():
            if not (self.train_images_dir / image_id).exists():
                missing_images.append(image_id)
                
        if missing_images:
            print(f"   WARNING: Missing image files: {len(missing_images)}")
        else:
            print(f"   OK: All records have corresponding image files")
            
        return {
            'total_records': len(self.train_df),
            'unique_images': self.train_df['ImageId'].nunique(),
            'defect_distribution': defect_counts.to_dict(),
            'missing_images': missing_images
        }
        
    def decode_rle(self, rle_string, height=256, width=1600):
        """Decode RLE encoded mask - CONSISTENT with training pipeline"""
        if pd.isna(rle_string):
            return np.zeros((height, width), dtype=np.float32)
            
        rle_numbers = [int(x) for x in rle_string.split()]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        
        mask = np.zeros(height * width, dtype=np.float32)
        for start, length in rle_pairs:
            start -= 1
            mask[start:start + length] = 1.0
            
        return mask.reshape((height, width), order='F')
        
    def encode_rle(self, mask):
        """Encode mask to RLE format"""
        pixels = mask.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
        
    def validate_rle_encoding(self, sample_size=100):
        """Validate RLE encoding accuracy"""
        print("\nValidating RLE encoding...")
        
        defective_records = self.train_df[self.train_df['EncodedPixels'].notna()]
        sample_records = defective_records.sample(min(sample_size, len(defective_records)))
        
        errors = 0
        for _, row in sample_records.iterrows():
            try:
                # Decode and encode
                original_rle = row['EncodedPixels']
                mask = self.decode_rle(original_rle)
                reconstructed_rle = self.encode_rle(mask)
                
                # Validate consistency (allow minor differences)
                original_mask = self.decode_rle(original_rle)
                reconstructed_mask = self.decode_rle(reconstructed_rle)
                
                if not np.array_equal(original_mask, reconstructed_mask):
                    errors += 1
                    
            except Exception as e:
                errors += 1
                
        print(f"   - Test samples: {len(sample_records)}")
        print(f"   - Encoding errors: {errors}")
        print(f"   - Accuracy: {(len(sample_records) - errors) / len(sample_records) * 100:.1f}%")
        
        return errors == 0
        
    def create_stratified_splits(self, n_splits=5, test_size=0.2, random_state=42):
        """Create stratified data splits"""
        print(f"\nCreating stratified data splits (K={n_splits})...")
        
        # Create labels for each image (multi-label)
        image_labels = {}
        for image_id in self.train_df['ImageId'].unique():
            image_df = self.train_df[self.train_df['ImageId'] == image_id]
            
            # Check if each class has defects
            labels = []
            for class_id in range(1, 5):
                class_df = image_df[image_df['ClassId'] == class_id]
                has_defect = len(class_df[class_df['EncodedPixels'].notna()]) > 0
                labels.append(int(has_defect))
                
            # Create combined label (for stratification)
            label_str = ''.join(map(str, labels))
            image_labels[image_id] = label_str
            
        # Convert to DataFrame
        label_df = pd.DataFrame([
            {'ImageId': img_id, 'Label': label}
            for img_id, label in image_labels.items()
        ])
        
        print(f"   - Label distribution:")
        label_counts = label_df['Label'].value_counts().sort_index()
        for label, count in label_counts.items():
            defect_classes = [i+1 for i, c in enumerate(label) if c == '1']
            print(f"     - {label} (Class{defect_classes}): {count} images")
            
        # Check for rare labels with insufficient samples for stratification
        rare_labels = label_counts[label_counts < 2].index.tolist()
        if rare_labels:
            print(f"   WARNING: Found {len(rare_labels)} rare label combinations with <2 samples")
            print(f"   Using random split instead of stratified split for robustness")
            
            # Use random split when stratification is not possible
            train_imgs, test_imgs = train_test_split(
                label_df['ImageId'].values,
                test_size=test_size,
                random_state=random_state
            )
        else:
            # Use stratified split when all labels have sufficient samples
            train_imgs, test_imgs = train_test_split(
                label_df['ImageId'].values,
                test_size=test_size,
                stratify=label_df['Label'].values,
                random_state=random_state
            )
        
        # Create K-fold cross-validation splits
        train_label_df = label_df[label_df['ImageId'].isin(train_imgs)]
        
        # Check if stratified K-fold is possible for training set
        train_label_counts = train_label_df['Label'].value_counts()
        train_rare_labels = train_label_counts[train_label_counts < n_splits].index.tolist()
        
        cv_splits = []
        if train_rare_labels:
            print(f"   WARNING: Some labels have <{n_splits} samples, using regular K-fold instead of stratified")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_label_df['ImageId'])):
                fold_train_imgs = train_label_df.iloc[train_idx]['ImageId'].values
                fold_val_imgs = train_label_df.iloc[val_idx]['ImageId'].values
                
                cv_splits.append({
                    'fold': fold_idx,
                    'train_images': fold_train_imgs.tolist(),
                    'val_images': fold_val_imgs.tolist()
                })
        else:
            # Use stratified K-fold when possible
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_label_df['ImageId'], train_label_df['Label'])):
                fold_train_imgs = train_label_df.iloc[train_idx]['ImageId'].values
                fold_val_imgs = train_label_df.iloc[val_idx]['ImageId'].values
                
                cv_splits.append({
                    'fold': fold_idx,
                    'train_images': fold_train_imgs.tolist(),
                    'val_images': fold_val_imgs.tolist()
                })
            
        print(f"   - Training images: {len(train_imgs)}")
        print(f"   - Test images: {len(test_imgs)}")
        print(f"   - K-fold splits: {n_splits} folds")
        
        # Save split information
        split_info = {
            'train_images': train_imgs.tolist(),
            'test_images': test_imgs.tolist(),
            'cv_splits': cv_splits,
            'n_splits': n_splits,
            'test_size': test_size
        }
        import json
        with open(self.output_dir / 'data_splits.json', 'w') as f:
            json.dump(split_info, f, indent=2)
            
        return split_info
        
    def analyze_defect_characteristics(self):
        """Analyze defect characteristics"""
        print(f"\nAnalyzing defect characteristics...")
        
        defect_stats = {}
        sample_size = 500
        
        for class_id in range(1, 5):
            print(f"\n   Analyzing class {class_id}:")
            
            class_df = self.train_df[
                (self.train_df['ClassId'] == class_id) & 
                (self.train_df['EncodedPixels'].notna())
            ]
            
            if len(class_df) == 0:
                continue
                
            # Sample analysis
            sample_df = class_df.sample(min(sample_size, len(class_df)))
            
            areas = []
            aspect_ratios = []
            positions = []
            
            for _, row in sample_df.iterrows():
                mask = self.decode_rle(row['EncodedPixels'])
                
                # Calculate area
                area = mask.sum()
                areas.append(area)
                
                # Calculate bounding box and aspect ratio
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    aspect_ratio = w / h if h > 0 else 0
                    aspect_ratios.append(aspect_ratio)
                    
                    # Calculate center position (normalized)
                    center_x = (x + w/2) / mask.shape[1]
                    center_y = (y + h/2) / mask.shape[0]
                    positions.append((center_x, center_y))
                    
            # Statistics
            stats = {
                'count': int(len(class_df)),
                'sample_count': int(len(sample_df)),
                'area_mean': float(np.mean(areas)) if areas else 0.0,
                'area_std': float(np.std(areas)) if areas else 0.0,
                'area_min': int(np.min(areas)) if areas else 0,
                'area_max': int(np.max(areas)) if areas else 0,
                'aspect_ratio_mean': float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
                'aspect_ratio_std': float(np.std(aspect_ratios)) if aspect_ratios else 0.0,
            }
            
            defect_stats[class_id] = stats
            
            print(f"     - Total count: {stats['count']}")
            print(f"     - Average area: {stats['area_mean']:.0f} +/- {stats['area_std']:.0f} pixels")
            print(f"     - Area range: {stats['area_min']} - {stats['area_max']} pixels")
            print(f"     - Average aspect ratio: {stats['aspect_ratio_mean']:.2f} +/- {stats['aspect_ratio_std']:.2f}")
            
        # Save statistics
        import json
        with open(self.output_dir / 'defect_characteristics.json', 'w') as f:
            json.dump(defect_stats, f, indent=2)
            
        return defect_stats
        
    def create_balanced_dataset(self, strategy='oversample'):
        """Create balanced dataset"""
        print(f"\nCreating balanced dataset (strategy: {strategy})...")
        
        # Analyze current distribution
        defect_counts = self.train_df['ClassId'].value_counts().sort_index()
        max_count = defect_counts.max()
        
        print(f"   Current distribution:")
        for class_id, count in defect_counts.items():
            print(f"     - Class {class_id}: {count}")
            
        balanced_df = self.train_df.copy()
        
        if strategy == 'oversample':
            # Oversample minority classes
            for class_id in defect_counts.index:
                current_count = defect_counts[class_id]
                if current_count < max_count:
                    # Number of samples needed
                    need_samples = max_count - current_count
                    
                    # Get samples of this class
                    class_samples = self.train_df[self.train_df['ClassId'] == class_id]
                    
                    # Oversample
                    additional_samples = class_samples.sample(
                        need_samples, 
                        replace=True, 
                        random_state=42
                    )
                    
                    balanced_df = pd.concat([balanced_df, additional_samples], ignore_index=True)
                    
        elif strategy == 'undersample':
            # Undersample majority classes
            min_count = defect_counts.min()
            
            balanced_dfs = []
            for class_id in defect_counts.index:
                class_samples = self.train_df[self.train_df['ClassId'] == class_id]
                
                if len(class_samples) > min_count:
                    # Random sampling
                    sampled = class_samples.sample(min_count, random_state=42)
                    balanced_dfs.append(sampled)
                else:
                    balanced_dfs.append(class_samples)
                    
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        # Save balanced dataset
        balanced_df.to_csv(self.output_dir / f'balanced_train_{strategy}.csv', index=False)
        
        # Display balanced distribution
        new_counts = balanced_df['ClassId'].value_counts().sort_index()
        print(f"\n   Balanced distribution:")
        for class_id, count in new_counts.items():
            print(f"     - Class {class_id}: {count}")
            
        return balanced_df
        
    def generate_preprocessing_report(self):
        """Generate preprocessing report"""
        print("\n" + "=" * 60)
        print("Generating preprocessing report")
        print("=" * 60)
        
        # Execute all analysis
        quality_info = self.analyze_data_quality()
        rle_valid = self.validate_rle_encoding()
        split_info = self.create_stratified_splits()
        defect_stats = self.analyze_defect_characteristics()
        
        # Create balanced dataset
        balanced_oversample = self.create_balanced_dataset('oversample')
        balanced_undersample = self.create_balanced_dataset('undersample')
        
        # Generate text report
        report_path = self.output_dir / 'preprocessing_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Steel Defect Detection - Data Preprocessing Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Data quality
            f.write("Data quality assessment:\n")
            f.write(f"   - Total records: {quality_info['total_records']:,}\n")
            f.write(f"   - Unique images: {quality_info['unique_images']:,}\n")
            f.write(f"   - RLE encoding validation: {'PASS' if rle_valid else 'FAIL'}\n")
            f.write(f"   - Missing image files: {len(quality_info['missing_images'])}\n\n")
            
            # Defect distribution
            f.write("Defect distribution:\n")
            for class_id, count in quality_info['defect_distribution'].items():
                f.write(f"   - Class {class_id}: {count:,}\n")
            f.write("\n")
            
            # Data split
            f.write("Data split:\n")
            f.write(f"   - Training images: {len(split_info['train_images'])}\n")
            f.write(f"   - Test images: {len(split_info['test_images'])}\n")
            f.write(f"   - Cross-validation folds: {len(split_info['cv_splits'])}\n\n")
            
            # Defect features
            f.write("Defect feature analysis:\n")
            for class_id, stats in defect_stats.items():
                f.write(f"   Class {class_id}:\n")
                f.write(f"     - Count: {stats['count']}\n")
                f.write(f"     - Average area: {stats['area_mean']:.0f} pixels\n")
                f.write(f"     - Average aspect ratio: {stats['aspect_ratio_mean']:.2f}\n")
            f.write("\n")
            
            # Preprocessing suggestions
            f.write("Preprocessing suggestions:\n")
            f.write("   - Use stratified sampling to ensure representative validation set\n")
            f.write("   - Consider class weights or oversampling for imbalanced handling\n")
            f.write("   - Apply appropriate data augmentation techniques\n")
            f.write("   - Use multi-scale training to enhance model robustness\n")
            
        print(f"\nPreprocessing report generated:")
        print(f"   - Report file: {report_path}")
        print(f"   - Data split: {self.output_dir}/data_splits.json")
        print(f"   - Defect features: {self.output_dir}/defect_characteristics.json")
        print(f"   - Balanced dataset: {self.output_dir}/balanced_train_*.csv")

def main():
    """Main function - Execute complete data preprocessing workflow"""
    print("Starting steel defect detection data preprocessing...")
    
    # Create preprocessor
    preprocessor = SteelDefectPreprocessor()
    
    # Generate complete preprocessing report
    preprocessor.generate_preprocessing_report()
    
    print("\nData preprocessing completed!")

if __name__ == "__main__":
    main()
