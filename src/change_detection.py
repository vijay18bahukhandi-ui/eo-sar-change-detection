import numpy as np
import rasterio
from skimage import filters, segmentation
from sklearn.cluster import KMeans
from scipy import ndimage
import cv2
import json
import os

class ChangeDetector:
    def __init__(self, output_dir="./output/change_maps"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def image_differencing(self, img1, img2, threshold=0.2):
        """Basic image differencing change detection"""
        diff = np.abs(img1 - img2)
        change_mask = diff > threshold
        return change_mask, diff
    
    def ratio_method(self, img1, img2):
        """Ratio method for SAR change detection"""
        ratio = img1 / (img2 + 1e-10)  # Avoid division by zero
        log_ratio = np.log(ratio)
        return log_ratio
    
    def pca_change_detection(self, img1, img2):
        """PCA-based change detection"""
        # Stack images
        stacked = np.stack([img1.flatten(), img2.flatten()], axis=-1)
        
        # Center data
        mean = np.mean(stacked, axis=0)
        centered = stacked - mean
        
        # PCA
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Second component often shows changes
        changes = np.dot(centered, eigenvectors[:, 1])
        change_map = changes.reshape(img1.shape)
        
        return change_map
    
    def unsupervised_kmeans(self, diff_image, n_clusters=3):
        """Unsupervised change classification using K-means"""
        # Reshape for clustering
        X = diff_image.reshape(-1, 1)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Reshape back
        classified = labels.reshape(diff_image.shape)
        
        return classified, kmeans.cluster_centers_
    
    def morphological_cleaning(self, change_mask, min_size=10):
        """Clean change mask using morphological operations"""
        # Remove small objects
        cleaned = ndimage.binary_opening(change_mask, structure=np.ones((3,3)))
        cleaned = ndimage.binary_closing(cleaned, structure=np.ones((3,3)))
        
        # Remove very small regions
        labeled, num_features = ndimage.label(cleaned)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < min_size:
                cleaned[labeled == i] = 0
        
        return cleaned
    
    def calculate_ndvi(self, red_band, nir_band):
        """Calculate Normalized Difference Vegetation Index"""
        return (nir_band - red_band) / (nir_band + red_band + 1e-10)
    
    def calculate_ndbi(self, nir_band, swir_band):
        """Calculate Normalized Difference Built-up Index"""
        return (swir_band - nir_band) / (swir_band + nir_band + 1e-10)
    
    def detect_urban_changes(self, optical_t1, optical_t2, output_name):
        """Detect urban changes using optical imagery"""
        # Updated for 4-band data: [R, G, NIR, SWIR]
        # For NDBI we need SWIR and NIR bands
        nir_t1, swir_t1 = optical_t1[2], optical_t1[3]  # Band 2: NIR, Band 3: SWIR
        nir_t2, swir_t2 = optical_t2[2], optical_t2[3]
        
        # Calculate NDBI for built-up areas
        ndbi_t1 = self.calculate_ndbi(nir_t1, swir_t1)
        ndbi_t2 = self.calculate_ndbi(nir_t2, swir_t2)
        
        # Urban growth = increase in built-up areas
        urban_growth = (ndbi_t2 - ndbi_t1) > 0.1
        
        # Clean the result
        urban_growth_cleaned = self.morphological_cleaning(urban_growth)
        
        return urban_growth_cleaned
    
    def detect_vegetation_changes(self, optical_t1, optical_t2, output_name):
        """Detect vegetation changes using NDVI"""
        # Updated for 4-band data: [R, G, NIR, SWIR]
        # For NDVI we need Red and NIR bands
        red_t1, nir_t1 = optical_t1[0], optical_t1[2]  # Band 0: Red, Band 2: NIR
        red_t2, nir_t2 = optical_t2[0], optical_t2[2]
        
        # Calculate NDVI
        ndvi_t1 = self.calculate_ndvi(red_t1, nir_t1)
        ndvi_t2 = self.calculate_ndvi(red_t2, nir_t2)
        
        # Vegetation loss
        vegetation_loss = (ndvi_t1 - ndvi_t2) > 0.2
        vegetation_gain = (ndvi_t2 - ndvi_t1) > 0.2
        
        # Clean results
        loss_cleaned = self.morphological_cleaning(vegetation_loss)
        gain_cleaned = self.morphological_cleaning(vegetation_gain)
        
        return loss_cleaned, gain_cleaned
    
    def sar_change_detection(self, sar_t1, sar_t2, output_name):
        """SAR-based change detection using ratio method"""
        # Ratio method
        log_ratio = self.ratio_method(sar_t1, sar_t2)
        
        # Threshold for significant changes
        change_mask = np.abs(log_ratio) > 0.5
        
        # Clean result
        change_cleaned = self.morphological_cleaning(change_mask)
        
        return change_cleaned
    
    def cross_sensor_change_detection(self, optical_t1, optical_t2, sar_t1, sar_t2, output_name):
        """Cross-sensor change detection combining optical and SAR"""
        # Optical-based change
        optical_diff = np.abs(optical_t1 - optical_t2)
        optical_changes = optical_diff > 0.3
        
        # SAR-based change
        sar_log_ratio = self.ratio_method(sar_t1, sar_t2)
        sar_changes = np.abs(sar_log_ratio) > 0.4
        
        # Fused change detection (weighted combination)
        fused_changes = optical_changes.astype(float) * 0.6 + sar_changes.astype(float) * 0.4
        fused_binary = fused_changes > 0.5
        
        # Clean result
        fused_cleaned = self.morphological_cleaning(fused_binary)
        
        return fused_cleaned
    
    def save_change_map(self, change_map, output_path, profile):
        """Save change map with proper profile"""
        profile.update({
            'count': 1,
            'dtype': 'uint8'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(change_map.astype('uint8'), 1)
        
        return output_path
    
    def calculate_change_statistics(self, change_map, area_name, change_type):
        """Calculate statistics for change detection results"""
        total_pixels = change_map.size
        changed_pixels = np.sum(change_map)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        stats = {
            'area_name': area_name,
            'change_type': change_type,
            'total_pixels': int(total_pixels),
            'changed_pixels': int(changed_pixels),
            'change_percentage': float(change_percentage),
            'changed_area_km2': float(changed_pixels * 0.01)  # Approximate for 10m resolution
        }
        
        return stats

# Example usage
if __name__ == "__main__":
    detector = ChangeDetector()
    
    # Create sample data
    img1 = np.random.rand(100, 100)
    img2 = img1.copy()
    img2[30:50, 40:60] = 0.9  # Add changes
    
    # Test change detection
    change_mask, diff = detector.image_differencing(img1, img2)
    print(f"Detected {np.sum(change_mask)} changed pixels")