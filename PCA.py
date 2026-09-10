import os
import numpy as np
from imageio.v2 import imread, imwrite

folder_path = 'Images'




import os
import numpy as np
from imageio.v2 import imread
from sklearn.decomposition import PCA

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
plt.savefig(f"PC{pc_index+1}.svg")