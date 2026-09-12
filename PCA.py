import os
import numpy as np
from imageio.v2 import imread, imwrite

folder_path = 'Images'




import os
import numpy as np
from imageio.v2 import imread
from sklearn.decomposition import PCA

def get_pca():
    folder_path = 'Images'
    image_vectors = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = imread(img_path)
            
            flat_img = img.flatten()
            image_vectors.append(flat_img)


    X = np.array(image_vectors, dtype=np.float32)


    X_centered = X - np.mean(X, axis=0)

    n_components = 20  
    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X_centered)
    return X_pca

print(f"Original shape: {X_centered.shape}")  # e.g., (N, 10000)
print(f"Reduced shape: {X_pca.shape}")        # e.g., (N, 50)

import matplotlib.pyplot as plt

# 1. Select the principal component index to visualize (0 = PC1)
pc_index = 0

# Extract the chosen eigenvector and reshape it back to 2D image dimensions
# (Replace height and width with your actual image dimensions)
height, width = 200, 200  
eigenimage = pca.components_[pc_index].reshape(height, width)
mean_img = np.mean(X, axis=0).reshape(height, width)

# Reshape the mean image back to 2D

# 2. Get the minimum and maximum scores across all images for this component
min_score = X_pca[:, pc_index].min()
max_score = X_pca[:, pc_index].max()

# 3. Compute the three variation images
img_min = mean_img + (min_score * eigenimage)
img_avg = mean_img
img_max = mean_img + (max_score * eigenimage)

# 4. Display the images side-by-side
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img_min, cmap='gray')
axes[0].set_title(f'Min Score ({min_score:.2f})')
axes[0].axis('off')

axes[1].imshow(img_avg, cmap='gray')
axes[1].set_title('Average Image')
axes[1].axis('off')

axes[2].imshow(img_max, cmap='gray')
axes[2].set_title(f'Max Score ({max_score:.2f})')
axes[2].axis('off')

plt.suptitle(f'Variation along Principal Component {pc_index + 1}')
plt.tight_layout()
# plt.show()
#plt.savefig(f"PC{pc_index+1}.svg")



# Extract percentage of variance explained by each component
explained_variance = pca.explained_variance_ratio_ * 100
pc_labels = [f'PC{i+1}' for i in range(len(explained_variance))]

# Create figure
plt.figure(figsize=(10, 5))

# Plot individual bar chart
bars = plt.bar(pc_labels, explained_variance, color='skyblue', edgecolor='navy', alpha=0.8, label='Individual Variance')

# Overlay cumulative variance step line (optional but helpful)
cumulative_variance = np.cumsum(explained_variance)
plt.plot(pc_labels, cumulative_variance, color='crimson', marker='o', linewidth=2, label='Cumulative Variance')

# Aesthetics and annotations
plt.xlabel('Principal Components', fontsize=11)
plt.ylabel('Percentage of Variance Explained (%)', fontsize=11)
plt.title('Variance Explained by Top 20 Principal Components', fontsize=13, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='center right')

# Add percentage labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f"Bar_plot{n_components}.svg")

