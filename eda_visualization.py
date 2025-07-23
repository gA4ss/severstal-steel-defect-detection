"""
EDA Visualization Module for Steel Defect Detection
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

class SteelDefectEDA:
    """Steel Defect EDA Visualizer"""
    
    def __init__(self, data_dir="data", output_dir="eda_outputs"):
        """
        Initialize EDA visualizer
        
        Args:
            data_dir (str): Data directory path
            output_dir (str): Output directory path
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.train_images_dir = self.data_dir / "train_images"
        self.test_images_dir = self.data_dir / "test_images"
        self.train_csv_path = self.data_dir / "train.csv"
        
        # Load data
        self.train_df = pd.read_csv(self.train_csv_path)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def decode_rle(self, rle_string, height=256, width=1600):
        """Decode RLE encoded mask"""
        if pd.isna(rle_string):
            return np.zeros((height, width), dtype=np.uint8)
            
        rle_numbers = [int(x) for x in rle_string.split()]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        
        mask = np.zeros(height * width, dtype=np.uint8)
        for start, length in rle_pairs:
            start -= 1
            mask[start:start + length] = 1
            
        return mask.reshape((height, width), order='F')
        
    def plot_defect_distribution(self):
        """Generate defect distribution visualization"""
        print("Generating defect distribution visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Steel Defect Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Each class defect count distribution
        defect_counts = self.train_df['ClassId'].value_counts().sort_index()
        axes[0, 0].bar(defect_counts.index, defect_counts.values, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Each class defect count distribution')
        axes[0, 0].set_xlabel('Class ID')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(defect_counts.values):
            axes[0, 0].text(defect_counts.index[i], v + 50, str(v), ha='center')
            
        # 2. Has defect image ratio
        has_defect = self.train_df['EncodedPixels'].notna().sum()
        no_defect = self.train_df['EncodedPixels'].isna().sum()
        
        labels = ['Has defect', 'No defect']
        sizes = [has_defect, no_defect]
        colors = ['lightcoral', 'lightblue']
        
        axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Has defect image ratio')
        
        # 3. Each image defect count distribution
        defective_images = self.train_df[self.train_df['EncodedPixels'].notna()]
        defects_per_image = defective_images.groupby('ImageId')['ClassId'].count()
        
        axes[1, 0].hist(defects_per_image.values, bins=range(1, max(defects_per_image.values)+2), 
                       alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('Each image defect count distribution')
        axes[1, 0].set_xlabel('Defect count')
        axes[1, 0].set_ylabel('Image count')
        
        # 4. Defect class co-occurrence matrix
        # Create image-defect class matrix
        image_defects = defective_images.groupby('ImageId')['ClassId'].apply(list).reset_index()
        
        # Calculate co-occurrence matrix
        cooccurrence = np.zeros((4, 4))
        for defects in image_defects['ClassId']:
            for i in defects:
                for j in defects:
                    cooccurrence[i-1, j-1] += 1
                    
        # Draw heatmap
        sns.heatmap(cooccurrence, annot=True, fmt='.0f', cmap='Blues', 
                   xticklabels=[f'Class {i}' for i in range(1, 5)],
                   yticklabels=[f'Class {i}' for i in range(1, 5)],
                   ax=axes[1, 1])
        axes[1, 1].set_title('Defect class co-occurrence matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'defect_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_defect_size_analysis(self, sample_size=1000):
        """Analyze and visualize defect size distribution"""
        print("Analyzing defect size distribution...")
        
        # Sample defect records
        defective_records = self.train_df[self.train_df['EncodedPixels'].notna()]
        if len(defective_records) > sample_size:
            sample_records = defective_records.sample(sample_size, random_state=42)
        else:
            sample_records = defective_records
            
        # Calculate defect size
        defect_data = []
        for _, row in sample_records.iterrows():
            mask = self.decode_rle(row['EncodedPixels'])
            area = mask.sum()
            defect_data.append({
                'ClassId': row['ClassId'],
                'Area': area,
                'ImageId': row['ImageId']
            })
            
        defect_df = pd.DataFrame(defect_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Defect size analysis', fontsize=16, fontweight='bold')
        
        # 1. Each class defect size distribution box plot
        sns.boxplot(data=defect_df, x='ClassId', y='Area', ax=axes[0, 0])
        axes[0, 0].set_title('Each class defect size distribution')
        axes[0, 0].set_xlabel('Class ID')
        axes[0, 0].set_ylabel('Area (pixels)')
        
        # 2. Each class defect size histogram
        for class_id in sorted(defect_df['ClassId'].unique()):
            class_data = defect_df[defect_df['ClassId'] == class_id]['Area']
            axes[0, 1].hist(class_data, alpha=0.6, label=f'Class {class_id}', bins=30)
        axes[0, 1].set_title('Each class defect size histogram')
        axes[0, 1].set_xlabel('Area (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Defect size statistics table
        stats_data = []
        for class_id in sorted(defect_df['ClassId'].unique()):
            class_areas = defect_df[defect_df['ClassId'] == class_id]['Area']
            stats_data.append([
                f'Class {class_id}',
                len(class_areas),
                f'{class_areas.mean():.0f}',
                f'{class_areas.median():.0f}',
                f'{class_areas.std():.0f}',
                f'{class_areas.min()}-{class_areas.max()}'
            ])
            
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table = axes[1, 0].table(cellText=stats_data,
                                colLabels=['Class', 'Count', 'Mean', 'Median', 'Standard Deviation', 'Range'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 0].set_title('Defect size statistics table')
        
        # 4. Defect size vs class violin plot
        sns.violinplot(data=defect_df, x='ClassId', y='Area', ax=axes[1, 1])
        axes[1, 1].set_title('Defect size distribution density')
        axes[1, 1].set_xlabel('Class ID')
        axes[1, 1].set_ylabel('Area (pixels)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'defect_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return defect_df
        
    def visualize_sample_images(self, num_samples=8):
        """Visualize sample images and defect masks"""
        print("Visualizing sample images and defect masks...")
        
        # Get defective images sample
        defective_images = self.train_df[self.train_df['EncodedPixels'].notna()]
        sample_images = defective_images.groupby('ImageId').first().sample(num_samples, random_state=42)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        fig.suptitle('Visualizing sample images and defect masks', fontsize=16, fontweight='bold')
        
        for idx, (image_id, row) in enumerate(sample_images.iterrows()):
            # Load original image
            img_path = self.train_images_dir / image_id
            if img_path.exists():
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display original image
                axes[idx, 0].imshow(image)
                axes[idx, 0].set_title(f'Original image: {image_id}')
                axes[idx, 0].axis('off')
                
                # Get all defects for this image
                image_defects = defective_images[defective_images['ImageId'] == image_id]
                
                # Create combined mask
                combined_mask = np.zeros((256, 1600), dtype=np.uint8)
                class_masks = {}
                
                for _, defect_row in image_defects.iterrows():
                    class_id = defect_row['ClassId']
                    mask = self.decode_rle(defect_row['EncodedPixels'])
                    class_masks[class_id] = mask
                    combined_mask += mask * class_id
                    
                # Display combined mask
                axes[idx, 1].imshow(combined_mask, cmap='tab10', vmin=0, vmax=4)
                axes[idx, 1].set_title(f'Combined mask (classes: {list(class_masks.keys())})')
                axes[idx, 1].axis('off')
                
                # Display overlay image
                overlay = image.copy()
                if combined_mask.max() > 0:
                    # Resize mask to match image dimensions
                    mask_resized = cv2.resize(combined_mask, (image.shape[1], image.shape[0]))
                    colored_mask = plt.cm.tab10(mask_resized / 4.0)[:, :, :3]
                    overlay = cv2.addWeighted(overlay.astype(np.float32), 0.7, 
                                            (colored_mask * 255).astype(np.float32), 0.3, 0)
                    
                axes[idx, 2].imshow(overlay.astype(np.uint8))
                axes[idx, 2].set_title('Overlay display')
                axes[idx, 2].axis('off')
                
                # Display single class mask (if multiple classes exist)
                if len(class_masks) > 1:
                    # Display first class mask
                    first_class = list(class_masks.keys())[0]
                    axes[idx, 3].imshow(class_masks[first_class], cmap='Reds')
                    axes[idx, 3].set_title(f'Class {first_class} mask')
                else:
                    # Display unique class mask
                    class_id = list(class_masks.keys())[0]
                    axes[idx, 3].imshow(class_masks[class_id], cmap='Reds')
                    axes[idx, 3].set_title(f'Class {class_id} mask')
                axes[idx, 3].axis('off')
                
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_images_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_image_statistics(self, sample_size=200):
        """Analyze and visualize image statistics"""
        print("Analyzing image statistics...")
        
        # Randomly sample images
        train_images = list(self.train_images_dir.glob("*.jpg"))
        sample_images = np.random.choice(train_images, min(sample_size, len(train_images)), replace=False)
        
        image_stats = []
        for img_path in sample_images:
            try:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Calculate statistics
                height, width = img.shape[:2]
                mean_brightness = img.mean()
                std_brightness = img.std()
                
                # Calculate contrast (standard deviation)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                contrast = gray.std()
                
                image_stats.append({
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height,
                    'mean_brightness': mean_brightness,
                    'std_brightness': std_brightness,
                    'contrast': contrast
                })
            except Exception as e:
                continue
                
        stats_df = pd.DataFrame(image_stats)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Image statistics analysis', fontsize=16, fontweight='bold')
        
        # 1. Image size distribution
        axes[0, 0].scatter(stats_df['width'], stats_df['height'], alpha=0.6)
        axes[0, 0].set_title('Image size distribution')
        axes[0, 0].set_xlabel('Width')
        axes[0, 0].set_ylabel('Height')
        
        # 2. Aspect ratio distribution
        axes[0, 1].hist(stats_df['aspect_ratio'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Aspect ratio distribution')
        axes[0, 1].set_xlabel('Aspect ratio')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Brightness distribution
        axes[0, 2].hist(stats_df['mean_brightness'], bins=30, alpha=0.7, color='orange')
        axes[0, 2].set_title('Average brightness distribution')
        axes[0, 2].set_xlabel('Average brightness')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Contrast distribution
        axes[1, 0].hist(stats_df['contrast'], bins=30, alpha=0.7, color='red')
        axes[1, 0].set_title('Contrast distribution')
        axes[1, 0].set_xlabel('Contrast')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Brightness vs contrast scatter plot
        axes[1, 1].scatter(stats_df['mean_brightness'], stats_df['contrast'], alpha=0.6)
        axes[1, 1].set_title('Brightness vs contrast')
        axes[1, 1].set_xlabel('Average brightness')
        axes[1, 1].set_ylabel('Contrast')
        
        # 6. Statistics summary table
        summary_stats = stats_df.describe().round(2)
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        table = axes[1, 2].table(cellText=summary_stats.values,
                                rowLabels=summary_stats.index,
                                colLabels=summary_stats.columns,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Statistics summary')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'image_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_df
        
    def create_interactive_dashboard(self):
        """Create interactive data exploration dashboard"""
        print("Creating interactive dashboard...")
        
        # Prepare data
        defect_counts = self.train_df['ClassId'].value_counts().sort_index()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Defect class distribution', 'Has defect ratio', 'Number of defects per image', 'Defect size trend'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. Defect class distribution bar chart
        fig.add_trace(
            go.Bar(x=defect_counts.index, y=defect_counts.values, name="Defect count"),
            row=1, col=1
        )
        
        # 2. Has defect pie chart
        has_defect = self.train_df['EncodedPixels'].notna().sum()
        no_defect = self.train_df['EncodedPixels'].isna().sum()
        
        fig.add_trace(
            go.Pie(labels=['Has defect', 'No defect'], values=[has_defect, no_defect]),
            row=1, col=2
        )
        
        # 3. Number of defects per image histogram
        defective_images = self.train_df[self.train_df['EncodedPixels'].notna()]
        defects_per_image = defective_images.groupby('ImageId')['ClassId'].count()
        
        fig.add_trace(
            go.Histogram(x=defects_per_image.values, name="Number of defects per image"),
            row=2, col=1
        )
        
        # 4. Example scatter plot (here using class vs record index as example)
        sample_data = self.train_df.sample(1000)
        fig.add_trace(
            go.Scatter(x=sample_data.index, y=sample_data['ClassId'], 
                      mode='markers', name="Sample distribution"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Steel defect detection data exploration dashboard",
            showlegend=False,
            height=800
        )
        
        # Save as HTML file
        fig.write_html(self.output_dir / 'interactive_dashboard.html')
        print(f"Interactive dashboard saved to: {self.output_dir / 'interactive_dashboard.html'}")
        
        return fig
        
    def generate_eda_report(self):
        """Generate complete EDA report"""
        print("\n" + "=" * 60)
        print("Generating exploratory data analysis report")
        print("=" * 60)
        
        # Execute all analysis
        print("\n1. Defect distribution analysis...")
        self.plot_defect_distribution()
        
        print("\n2. Defect size analysis...")
        defect_df = self.plot_defect_size_analysis()
        
        print("\n3. Sample image visualization...")
        self.visualize_sample_images()
        
        print("\n4. Image statistics analysis...")
        stats_df = self.plot_image_statistics()
        
        print("\n5. Interactive dashboard...")
        self.create_interactive_dashboard()
        
        # Generate text report
        report_path = self.output_dir / 'eda_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Steel defect detection - Exploratory data analysis report\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write("Dataset basic information:\n")
            f.write(f"Total records: {len(self.train_df):,}\n")
            f.write(f"Total images: {self.train_df['ImageId'].nunique():,}\n")
            f.write(f"Defect records: {self.train_df['EncodedPixels'].notna().sum():,}\n")
            f.write(f"Defect classes: {self.train_df['ClassId'].nunique()}\n\n")
            
            # Defect distribution
            f.write("Defect class distribution:\n")
            defect_counts = self.train_df['ClassId'].value_counts().sort_index()
            for class_id, count in defect_counts.items():
                f.write(f"Class {class_id}: {count:,} ({count/len(self.train_df)*100:.1f}%)\n")
            f.write("\n")
            
            # Key findings
            f.write("Key findings:\n")
            f.write("- Dataset contains 4 different types of steel surface defects\n")
            f.write("- Defect distribution is imbalanced, consider sampling strategies\n")
            f.write("- Image sizes are relatively uniform, beneficial for model training\n")
            f.write("- Some images contain multiple types of defects\n\n")
            
            f.write("Recommendations:\n")
            f.write("- Use weighted loss function to handle class imbalance\n")
            f.write("- Apply data augmentation techniques to increase sample diversity\n")
            f.write("- Consider using stratified sampling to ensure representative validation set\n")
            f.write("- Implement multi-label classification strategy to handle multi-defect images\n")
            
        print(f"\nEDA report generated and saved to: {self.output_dir}")
        print(f"   - Chart files: {self.output_dir}/*.png")
        print(f"   - Interactive dashboard: {self.output_dir}/interactive_dashboard.html")
        print(f"   - Text report: {report_path}")

def main():
    """Main function - Execute complete EDA process"""
    print("Starting steel defect detection exploratory data analysis...")
    
    # Create EDA analyzer
    eda = SteelDefectEDA()
    
    # Generate complete report
    eda.generate_eda_report()
    
    print("\nExploratory data analysis completed!")

if __name__ == "__main__":
    main()
