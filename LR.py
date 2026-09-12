# %% Preparation
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imageio.v2 import imread
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

height, width = 200, 200

SYNTHETIC_IMAGES_DIR = Path("synthetic_images")
SYNTHETIC_IMAGES_DIR.mkdir(exist_ok=True)

def get_ratings():
    """
    Plots a histogram of the ratings.

    Parameters:
    file (str): The path to the JSON file containing the ratings.
    """
    ratings_list = []
    for file in Path(__file__).parent.glob("s*.csv"):
        df = pd.read_csv(file)  # Read the CSV file to ensure it exists and is valid
        # Extract the ratings from the dictionary
        ratings_list.append(np.array(df['rating']))
        ratings_list.append(np.array(df['rating2']))
    
    rating_values = np.vstack(ratings_list)

    return rating_values

full_rating_values = get_ratings()
ratings = np.mean(full_rating_values, axis=0, keepdims=True).T
###########################
folder_path = 'Images'
image_vectors = []

for filename in sorted(os.listdir(folder_path)):
    if filename.lower().endswith('.jpg'):
        img_path = os.path.join(folder_path, filename)
        img = imread(img_path)
        
        flat_img = img.flatten()
        image_vectors.append(flat_img)


X = np.array(image_vectors, dtype=np.float32)

X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

n_components = 20  
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_centered)
###########################

model = LinearRegression()
sfs = SequentialFeatureSelector(model, direction='forward')
sfs.fit(X_pca, ratings)
# Get the selected features
selected_features = sfs.get_support(indices=True)
print(f"Selected features (indices): {selected_features}")
# Use selected features to generate synthetic images
selected_eigenvectors = pca.components_[selected_features]

# Use the selected eigenvectors on X_pca
X_pca_selected = X_pca[:, selected_features]
model.fit(X_pca_selected, ratings)
# print(model.coef_)
# print(model.intercept_)
synthetic_ratings = np.arange(0.5, 6.0, 0.5)
synthetic_image_dict = {}
# 4. Display the images side-by-side
fig, axes = plt.subplots( 1, len(synthetic_ratings), figsize=(4 * len(synthetic_ratings), 4))
for i, rating in enumerate(synthetic_ratings):
    # print(f"Generating synthetic image for rating: {rating}")
    synthetic_image_pca = np.zeros((1, n_components))
    # Generate synthetic image in PCA space
    alpha = (rating - model.intercept_) / np.linalg.norm(model.coef_, ord=2)**2
    # print(f"Synthetic PCA for rating {rating}: {alpha}")
    synthetic_image_pca[:, selected_features]= alpha * model.coef_
    # Transform back to original space
    synthetic_image_flat = pca.inverse_transform(synthetic_image_pca) + X_mean
    synthetic_image = synthetic_image_flat.reshape(200, 200, 1)  # Assuming original images are 200x200 with 1 channel
    synthetic_image_dict[rating] = synthetic_image
    axes[i].imshow(synthetic_image, cmap='gray')
    axes[i].set_title(f'{rating}')
    axes[i].axis('off')
plt.suptitle('Synthetic Images for Ratings', fontsize=16)
plt.tight_layout()
# plt.savefig('synthetic_images.svg')
plt.show(block=False)
plt.close()

def save_synthetic_images(synthetic_image_dict):
    for rating, synthetic_image in synthetic_image_dict.items():
        plt.imshow(synthetic_image, cmap='gray')
        plt.axis('off')
        plt.imsave(f'{SYNTHETIC_IMAGES_DIR}/{rating}.svg', synthetic_image.squeeze(), cmap='gray')
        plt.close()
# save_synthetic_images(synthetic_image_dict)

# %% Task 6
print("Task 6")
X_pca_selected_pred = model.predict(X_pca_selected)
min_pred, max_pred = X_pca_selected_pred.min(), X_pca_selected_pred.max()
print(f"Predicted ratings range: {min_pred:.2f} to {max_pred:.2f}")

# %% Task 7
print("Task 7")
# list(range(len(synthetic_ratings)))

shuffled_indices =  np.repeat(list(range(len(synthetic_ratings))), repeats=10, axis=0)
np.random.shuffle(shuffled_indices)
shuffleds_synthetic_ratings = synthetic_ratings[shuffled_indices]
ratings_json = []
user_ratings = []
participant = input("Your student number: ")
for progress, i in enumerate(shuffled_indices):
    synthetic_rating = synthetic_ratings[i]
    synthetic_image = synthetic_image_dict[synthetic_rating]
    plt.imshow(synthetic_image, cmap='gray')
    plt.axis('off')
    plt.show(block=False)

    try:
        user_rating = int(input(f"Progress: {progress+1}/{len(shuffled_indices)} - Enter a rating: "))
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        continue
    user_ratings.append(user_rating)
    ratings_json.append(
		{
			"synthetic_rating": synthetic_rating,
			"user_rating": user_rating
		}
	)
    plt.close()

spearman_corr, _ = spearmanr(shuffleds_synthetic_ratings, user_ratings)
print(f"Spearman correlation: {spearman_corr:.4f}")

# Plot boxplot
sns.boxplot(x=shuffleds_synthetic_ratings, y=user_ratings)
plt.title(f"Spearman correlation: {spearman_corr:.4f}")
plt.xlabel("Synthetic Ratings")
plt.ylabel("User Ratings")
plt.tight_layout()
plt.savefig(f"{participant}_experiment2_boxplot.svg")

with open(f"{participant}_experiment2.json", "w") as f:
    json.dump(
        {
            "spearman": f"{spearman_corr:.4f}",
            "ratings": ratings_json
        }, f, indent=4)