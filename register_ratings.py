import json
from pathlib import Path

import matplotlib.pyplot as plt

IMAGE_DIR = Path(__file__).parent / "images"

OUTPUT_FILE = Path(__file__).parent / "ratings.json"

image_files = sorted([f.name for f in IMAGE_DIR.glob('*') if f.is_file()])
print(f"Found {len(image_files)} image files in {IMAGE_DIR}.")
ratings = {}

for image_file in image_files:
	image = plt.imread(IMAGE_DIR / image_file)
	plt.imshow(image)
	plt.axis("off")
	plt.show(block=False)
	
	try:
		rating = int(input(f"Enter a rating for {image_file}: "))
	except ValueError:
		print("Invalid input. Please enter a valid integer.")
		continue
	ratings[image_file] = rating
	plt.close()

with open(OUTPUT_FILE, "w") as f:
	json.dump(ratings, f, indent=4)
