import numpy as np
import rasterio
from skimage import exposure
import cv2
from scipy import ndimage
import os

class EODataFusion:
    def __init__(self):
        self.fusion_methods = {}
    
    def pca_fusion(self, optical_data, sar_data):
        """PCA-based fusion of optical and SAR data"""
        print("Applying PCA fusion...")
        
        # Ensure both images have same dimensions
        min_shape = (min(optical_data.shape[0], sar_data.shape[0]),
                    min(optical_data.shape[1], sar_data.shape[1]))
        
        optical_resized = cv2.resize(optical_data, min_shape[::-1])
        sar_resized = cv2.resize(sar_data, min_shape[::-1])
        
        # Stack images for PCA
        stacked = np.stack([optical_resized, sar_resized], axis=-1)
        stacked_flat = stacked.reshape(-1, 2)
        
        # Perform PCA
        mean = np.mean(stacked_flat, axis=0)
        centered = stacked_flat - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Project using first principal component
        pca_component = np.dot(centered, eigenvectors[:, 0])
        fused = pca_component.reshape(min_shape)
        
        # Normalize
        fused = exposure.rescale_intensity(fused)
        
        return fused
    
    def brovey_fusion(self, optical_data, sar_data, alpha=0.5):
        """Brovey transform fusion"""
        print("Applying Brovey fusion...")
        
        # Resize to common dimensions
        min_shape = (min(optical_data.shape[0], sar_data.shape[0]),
                    min(optical_data.shape[1], sar_data.shape[1]))
        
        optical_resized = cv2.resize(optical_data, min_shape[::-1])
        sar_resized = cv2.resize(sar_data, min_shape[::-1])
        
        # Normalize
        optical_norm = optical_resized / (optical_resized.max() + 1e-10)
        sar_norm = sar_resized / (sar_resized.max() + 1e-10)
        
        # Brovey fusion
        fused = (optical_norm * sar_norm) / (optical_norm + sar_norm + 1e-10)
        fused = alpha * optical_norm + (1 - alpha) * fused
        
        return fused
    
    def wavelet_fusion(self, optical_data, sar_data):
        """Wavelet-based fusion"""
        print("Applying wavelet fusion...")
        
        # Simple wavelet-like fusion using Gaussian pyramid
        optical_blur = cv2.GaussianBlur(optical_data, (5, 5), 0)
        sar_blur = cv2.GaussianBlur(sar_data, (5, 5), 0)
        
        # High-pass components
        optical_high = optical_data - optical_blur
        sar_high = sar_data - sar_blur
        
        # Fusion: take average of low-pass, maximum of high-pass
        fused_low = (optical_blur + sar_blur) / 2
        fused_high = np.maximum(optical_high, sar_high)
        
        fused = fused_low + fused_high
        fused = exposure.rescale_intensity(fused)
        
        return fused
    
    def save_fused_image(self, fused_data, output_path, profile):
        """Save fused image with proper profile"""
        profile.update({
            'count': 1,
            'dtype': 'float32'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(fused_data.astype('float32'), 1)
        
        return output_path

# Example usage
if __name__ == "__main__":
    fusion = EODataFusion()
    
    # Example with sample data
    optical_sample = np.random.rand(100, 100)
    sar_sample = np.random.rand(100, 100)
    
    pca_fused = fusion.pca_fusion(optical_sample, sar_sample)
    brovey_fused = fusion.brovey_fusion(optical_sample, sar_sample)
    wavelet_fused = fusion.wavelet_fusion(optical_sample, sar_sample)
    
    print("Fusion methods tested successfully!")
