import os
import numpy as np
import pandas as pd
from imageio.v2 import imread
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

# 1. Load and process images
folder_path = 'Images'
image_vectors = []

# Sort filenames to ensure image order matches CSV row order
filenames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

for filename in filenames:
    img_path = os.path.join(folder_path, filename)
    img = imread(img_path)
    image_vectors.append(img.flatten())

X = np.array(image_vectors, dtype=np.float32)

# Subtract mean image and run PCA
X_centered = X - np.mean(X, axis=0)
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_centered)

# 2. Load ratings CSV and normalize target (y)
ratings_df = pd.read_csv('s224375.csv')

# Extract 2nd and 3rd columns (indices 1 & 2) and take their mean across rows
raw_ratings = ratings_df.iloc[:, [1, 2]].mean(axis=1).values

# Z-score normalization (mean=0, std=1)
y_ratings = (raw_ratings - np.mean(raw_ratings)) / np.std(raw_ratings)

# 3. Perform Forward Feature Selection with 5-fold CV
base_model = LinearRegression()
sfs = SequentialFeatureSelector(
    estimator=base_model,
    direction='forward',
    n_features_to_select='auto',
    tol=0.01,
    cv=5
)

sfs.fit(X_pca, y_ratings)

# 4. Extract selected PCs and fit final Linear Regression model
selected_features_mask = sfs.get_support()
selected_pc_indices = np.where(selected_features_mask)[0]

X_selected = X_pca[:, selected_features_mask]

final_model = LinearRegression()
final_model.fit(X_selected, y_ratings)

# 5. Output evaluation
r_squared = final_model.score(X_selected, y_ratings)

print(f"Selected PC indices (0-indexed): {selected_pc_indices}")
print(f"Total PCs selected: {len(selected_pc_indices)}")
print(f"Model Coefficients: {final_model.coef_}")
print(f"Model Intercept: {final_model.intercept_:.4f}")
print(f"Final Model R^2 score: {r_squared:.3f}")

from sklearn.model_selection import cross_val_score

predictions = final_model.predict(X_selected)
residuals = y_ratings - predictions

