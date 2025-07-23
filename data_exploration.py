"""
Data Exploration Module for Steel Defect Detection
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SteelDefectDataExplorer:
    """Steel Defect Data Explorer"""
    
    def __init__(self, data_dir="data"):
        """
        Initialize the data explorer
        
        Args:
            data_dir (str): data directory
        """
        self.data_dir = Path(data_dir)
        self.train_images_dir = self.data_dir / "train_images"
        self.test_images_dir = self.data_dir / "test_images"
        self.train_csv_path = self.data_dir / "train.csv"
        self.sample_submission_path = self.data_dir / "sample_submission.csv"
        
        # Load data
        self.train_df = None
        self.sample_submission = None
        self.load_data()
        
    def load_data(self):
        """Load training data and sample submission file"""
        try:
            print("Loading data...")
            self.train_df = pd.read_csv(self.train_csv_path)
            self.sample_submission = pd.read_csv(self.sample_submission_path)
            print(f"Training data loaded: {len(self.train_df)} records")
            print(f"Sample submission file loaded: {len(self.sample_submission)} records")
        except Exception as e:
            print(f"Data loading failed: {e}")
            
    def explore_data_structure(self):
        """Explore data structure"""
        print("=" * 50)
        print("Explore data structure")
        print("=" * 50)
        
        # Training data basic information
        print("\n1. Training data basic information:")
        print(f"   - Data shape: {self.train_df.shape}")
        print(f"   - Column names: {list(self.train_df.columns)}")
        print(f"   - Data types:\n{self.train_df.dtypes}")
        
        # Display the first few rows of the training data
        print("\n2. Training data first 5 rows:")
        print(self.train_df.head())
        
        # Missing value check
        missing_values = self.train_df.isna().sum()
        print(missing_values)
        
        # Unique value statistics
        print("\n4. Unique value statistics:")
        print(f"   - Unique image count: {self.train_df['ImageId'].nunique()}")
        print(f"   - Defect classes: {sorted(self.train_df['ClassId'].unique())}")
        
        # Sample submission file information
        print("\n5. Sample submission file information:")
        print(f"   - Data shape: {self.sample_submission.shape}")
        print(f"   - Column names: {list(self.sample_submission.columns)}")
        
    def explore_defect_distribution(self):
        """Explore defect distribution"""
        print("\n" + "=" * 50)
        print("Defect distribution analysis")
        print("=" * 50)
        
        # Defect count distribution
        defect_counts = self.train_df['ClassId'].value_counts().sort_index()
        print("\n1. Defect count distribution:")
        for class_id, count in defect_counts.items():
            print(f"   - Defect class {class_id}: {count} defects")
            
        # Defect image statistics
        images_with_defects = self.train_df[self.train_df['EncodedPixels'].notna()]
        total_images = self.train_df['ImageId'].nunique()
        defective_images = images_with_defects['ImageId'].nunique()
        
        print(f"\n2. Defect image statistics:")
        print(f"   - Total image count: {total_images}")
        print(f"   - Defective image count: {defective_images}")
        print(f"   - Defective image ratio: {defective_images/total_images:.2%}")
        
        # Defect count per image
        defects_per_image = images_with_defects.groupby('ImageId')['ClassId'].count()
        print(f"\n3. Defect count per image:")
        print(defects_per_image.value_counts().sort_index())
        
        return defect_counts, defects_per_image
        
    def explore_image_properties(self, sample_size=100):
        """Explore image properties"""
        print("\n" + "=" * 50)
        print("Image property analysis")
        print("=" * 50)
        
        # Get image file list
        train_images = list(self.train_images_dir.glob("*.jpg"))
        test_images = list(self.test_images_dir.glob("*.jpg"))
        
        print(f"\n1. Image file statistics:")
        print(f"   - Training image count: {len(train_images)}")
        print(f"   - Test image count: {len(test_images)}")
        
        # Random sampling to analyze image dimensions
        sample_images = np.random.choice(train_images, min(sample_size, len(train_images)), replace=False)
        
        image_sizes = []
        image_stats = []
        
        print(f"\n2. Analyzing {len(sample_images)} sample images...")
        
        for img_path in sample_images:
            try:
                # Use PIL to load image
                img = Image.open(img_path)
                width, height = img.size
                image_sizes.append((width, height))
                
                # Convert to numpy array to analyze pixel statistics
                img_array = np.array(img)
                if len(img_array.shape) == 3:  # RGB image
                    mean_val = img_array.mean()
                    std_val = img_array.std()
                    image_stats.append((mean_val, std_val))
                    
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                
        # Analyze image dimensions
        if image_sizes:
            widths, heights = zip(*image_sizes)
            print(f"\n3. Image dimension statistics:")
            print(f"   - Width range: {min(widths)} - {max(widths)}")
            print(f"   - Height range: {min(heights)} - {max(heights)}")
            print(f"   - Most common dimension: {max(set(image_sizes), key=image_sizes.count)}")
            
        # Analyze pixel statistics
        if image_stats:
            means, stds = zip(*image_stats)
            print(f"\n4. Pixel value statistics:")
            print(f"   - Mean pixel value range: {min(means):.2f} - {max(means):.2f}")
            print(f"   - Pixel value standard deviation range: {min(stds):.2f} - {max(stds):.2f}")
            
        return image_sizes, image_stats
        
    def decode_rle(self, rle_string, height=256, width=1600):
        """
        Decode RLE encoded mask
        
        Args:
            rle_string (str): RLE encoded string
            height (int): Image height
            width (int): Image width
            
        Returns:
            np.ndarray: Binary mask
        """
        if pd.isna(rle_string):
            return np.zeros((height, width), dtype=np.uint8)
            
        rle_numbers = [int(x) for x in rle_string.split()]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        
        mask = np.zeros(height * width, dtype=np.uint8)
        for start, length in rle_pairs:
            start -= 1  # RLE编码从1开始
            mask[start:start + length] = 1
            
        return mask.reshape((height, width), order='F')  # Fortran order
        
    def analyze_defect_sizes(self, sample_size=500):
        """Analyze defect size distribution"""
        print("\n" + "=" * 50)
        print("Defect size analysis")
        print("=" * 50)
        
        # Get records with defects
        defective_records = self.train_df[self.train_df['EncodedPixels'].notna()]
        
        if len(defective_records) > sample_size:
            sample_records = defective_records.sample(sample_size)
        else:
            sample_records = defective_records
            
        defect_sizes = {}
        
        print(f"Analyzing {len(sample_records)} defect samples...")
        
        for _, row in sample_records.iterrows():
            class_id = row['ClassId']
            rle_string = row['EncodedPixels']
            
            # Decode mask and calculate area
            mask = self.decode_rle(rle_string)
            defect_area = mask.sum()
            
            if class_id not in defect_sizes:
                defect_sizes[class_id] = []
            defect_sizes[class_id].append(defect_area)
            
        # 统计各类缺陷大小
        print("\nDefect size statistics (pixels):")
        for class_id in sorted(defect_sizes.keys()):
            sizes = defect_sizes[class_id]
            print(f"Defect class {class_id}:")
            print(f"   - Number of defects: {len(sizes)}")
            print(f"   - Mean size: {np.mean(sizes):.0f}")
            print(f"   - Median size: {np.median(sizes):.0f}")
            print(f"   - Size range: {min(sizes)} - {max(sizes)}")
            
        return defect_sizes
        
    def generate_summary_report(self):
        """Generate data exploration summary report"""
        print("\n" + "=" * 60)
        print("Data exploration summary report")
        print("=" * 60)
        
        # Basic statistics
        total_records = len(self.train_df)
        total_images = self.train_df['ImageId'].nunique()
        defective_records = len(self.train_df[self.train_df['EncodedPixels'].notna()])
        defective_images = self.train_df[self.train_df['EncodedPixels'].notna()]['ImageId'].nunique()
        
        print(f"\nData set overview:")
        print(f"   * Total records: {total_records:,}")
        print(f"   * Total images: {total_images:,}")
        print(f"   * Defective records: {defective_records:,}")
        print(f"   * Defective images: {defective_images:,}")
        print(f"   * Defective image ratio: {defective_images/total_images:.1%}")
        
        # Defect class distribution
        defect_counts = self.train_df['ClassId'].value_counts().sort_index()
        print(f"\nDefect class distribution:")
        for class_id, count in defect_counts.items():
            percentage = count / defective_records * 100
            print(f"   * Class {class_id}: {count:,} ({percentage:.1f}%)")
            
        # Data quality assessment
        print(f"\nData quality assessment:")
        missing_pixels = self.train_df['EncodedPixels'].isna().sum()
        print(f"   * Defective records: {missing_pixels:,}")
        print(f"   * Data integrity: {(total_records - missing_pixels)/total_records:.1%}")
        
        print(f"\nRecommendations:")
        print(f"   * Data set size is moderate, suitable for deep learning training")
        print(f"   * Defect distribution is imbalanced, consider class balance strategies")
        print(f"   * Suggest using data augmentation techniques to increase sample diversity")
        
def main():
    """Main function - Execute complete data exploration workflow"""
    print("Starting steel defect detection data exploration...")
    
    # Create data explorer
    explorer = SteelDefectDataExplorer()
    
    # Execute various exploration analyses
    explorer.explore_data_structure()
    explorer.explore_defect_distribution()
    explorer.explore_image_properties()
    explorer.analyze_defect_sizes()
    explorer.generate_summary_report()
    
    print("\nData exploration completed!")

if __name__ == "__main__":
    main()
