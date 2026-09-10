import os
import numpy as np
from skimage.color import rgb2gray
from imageio.v2 import imread, imwrite

folder_path = 'Images'

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.jpg'):
        img_path = os.path.join(folder_path, filename)
        
        img = imread(img_path)
        
        gray_img = rgb2gray(img)
        
        gray_img_uint8 = (gray_img * 255).astype(np.uint8)
        
        imwrite(img_path, gray_img_uint8)