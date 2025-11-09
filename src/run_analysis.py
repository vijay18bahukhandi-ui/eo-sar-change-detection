# BETTER DEHRADUN ANALYSIS - CLEAR VISUALIZATION
import matplotlib.pyplot as plt
import numpy as np
import os

print("🚀 Creating Clear Dehradun Analysis Visualization...")

# Create output folder
os.makedirs('output', exist_ok=True)

# Create more realistic sample data
print("📊 Generating realistic satellite data...")

# Urban area - make it look like a real city
urban_2019 = np.zeros((100, 100))
# Add some urban patches (bright areas)
urban_2019[20:40, 20:40] = 0.8  # City center
urban_2019[60:80, 30:50] = 0.6  # Suburban area
# Add some vegetation (medium values)
urban_2019[40:60, 60:80] = 0.3
urban_2019[10:30, 70:90] = 0.4

# Urban 2023 - show growth
urban_2023 = urban_2019.copy()
urban_2023[70:90, 10:30] = 0.9  # NEW construction area
urban_2023[50:70, 70:90] = 0.8  # NEW urban expansion
urban_2023[30:50, 40:60] = 0.7  # Infill development

# Forest area - make it look like real forest
forest_2019 = np.ones((100, 100)) * 0.7  # Dense forest
# Add some variation
forest_2019[20:40, 20:60] = 0.9  # Very dense area
forest_2019[60:80, 40:70] = 0.8  # Medium density

# Forest 2023 - show deforestation
forest_2023 = forest_2019.copy()
forest_2023[30:50, 30:50] = 0.2  # DEFORESTATION area 1
forest_2023[70:85, 50:65] = 0.3  # DEFORESTATION area 2

# Detect changes
urban_growth = (urban_2023 - urban_2019) > 0.2
forest_loss = (forest_2019 - forest_2023) > 0.3

print("🎨 Creating clear visualization...")

# Create a professional-looking plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Main title
fig.suptitle('DEHRADUN CHANGE DETECTION ANALYSIS\n(2019 vs 2023)', 
             fontsize=16, fontweight='bold', y=0.95)

# URBAN ANALYSIS
# Urban 2019
im1 = axes[0,0].imshow(urban_2019, cmap='YlOrRd', vmin=0, vmax=1)
axes[0,0].set_title('URBAN AREA 2019\n(Bright = Built-up, Dark = Vegetation)', 
                   fontweight='bold', fontsize=12)
axes[0,0].set_ylabel('URBAN ANALYSIS', fontweight='bold', fontsize=14)
axes[0,0].set_xlabel('Pixel Coordinates')
plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)

# Urban 2023
im2 = axes[0,1].imshow(urban_2023, cmap='YlOrRd', vmin=0, vmax=1)
axes[0,1].set_title('URBAN AREA 2023\n(New construction visible)', 
                   fontweight='bold', fontsize=12)
axes[0,1].set_xlabel('Pixel Coordinates')
plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)

# Urban Growth
im3 = axes[0,2].imshow(urban_growth, cmap='Reds')
axes[0,2].set_title('URBAN GROWTH DETECTED\n(Red = New Development Areas)', 
                   fontweight='bold', fontsize=12)
axes[0,2].set_xlabel('Pixel Coordinates')
# Add text annotation for urban growth
urban_growth_pixels = np.sum(urban_growth)
axes[0,2].text(50, 10, f'Growth: {urban_growth_pixels} pixels', 
               ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# FOREST ANALYSIS
# Forest 2019
im4 = axes[1,0].imshow(forest_2019, cmap='Greens', vmin=0, vmax=1)
axes[1,0].set_title('FOREST AREA 2019\n(Bright = Dense Forest)', 
                   fontweight='bold', fontsize=12)
axes[1,0].set_ylabel('FOREST ANALYSIS', fontweight='bold', fontsize=14)
axes[1,0].set_xlabel('Pixel Coordinates')
plt.colorbar(im4, ax=axes[1,0], fraction=0.046, pad=0.04)

# Forest 2023
im5 = axes[1,1].imshow(forest_2023, cmap='Greens', vmin=0, vmax=1)
axes[1,1].set_title('FOREST AREA 2023\n(Dark patches = Deforestation)', 
                   fontweight='bold', fontsize=12)
axes[1,1].set_xlabel('Pixel Coordinates')
plt.colorbar(im5, ax=axes[1,1], fraction=0.046, pad=0.04)

# Forest Loss
im6 = axes[1,2].imshow(forest_loss, cmap='Reds')
axes[1,2].set_title('FOREST LOSS DETECTED\n(Red = Deforested Areas)', 
                   fontweight='bold', fontsize=12)
axes[1,2].set_xlabel('Pixel Coordinates')
# Add text annotation for forest loss
forest_loss_pixels = np.sum(forest_loss)
axes[1,2].text(50, 10, f'Loss: {forest_loss_pixels} pixels', 
               ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Remove ticks for cleaner look
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('output/clear_dehradun_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Calculate statistics
urban_area_km2 = urban_growth_pixels * 0.01  # Approximate conversion
forest_area_km2 = forest_loss_pixels * 0.01

print("\n📈 ANALYSIS RESULTS:")
print("="*50)
print(f"📍 URBAN GROWTH: {urban_growth_pixels} pixels ({urban_area_km2:.2f} km²)")
print(f"📍 FOREST LOSS: {forest_loss_pixels} pixels ({forest_area_km2:.2f} km²)")
print("="*50)
print("✅ Clear analysis saved as: output/clear_dehradun_analysis.png")
print("🎉 Now you can see the urban growth and deforestation clearly!")