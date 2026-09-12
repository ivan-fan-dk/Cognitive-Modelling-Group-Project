# %% Task 8
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



height, width = 200, 200

ADAPTATION_TIME = 10
TEST_TIME = 1

SYNTHETIC_IMAGES_DIR = Path("synthetic_images")
SYNTHETIC_IMAGES_DIR.mkdir(exist_ok=True)

synthetic_images_path = sorted(SYNTHETIC_IMAGES_DIR.glob("*.png"), key=lambda x: float(x.stem))
synthetic_images = [plt.imread(str(path)) for path in synthetic_images_path]
test_image_path_list = [SYNTHETIC_IMAGES_DIR / f"{rating}.png" for rating in ["2.5", "3.0", "3.5"]]
print(f"Test image path set: {test_image_path_list}")
lowest_rating_image = synthetic_images[0]
highest_rating_image = synthetic_images[-1]

student_number = input("Please enter your student number: ")
user_ratings = []
for adaptation_label, adaption_image in [
    ("Low adaptation", lowest_rating_image),
    ("High adaptation", highest_rating_image),
]:
    for test_image_path in np.random.choice(np.array(test_image_path_list), size=3, replace=False):
        plt.imshow(adaption_image, cmap='gray')
        plt.axis('off')
        plt.title(f"Adaptation Image (After {ADAPTATION_TIME} seconds, Test image will appear for {TEST_TIME} seconds)")
        plt.plot(width / 2, height / 2, "ko", markersize=8)
        plt.show(block=False)
        plt.pause(ADAPTATION_TIME)
        plt.close()
        test_image = plt.imread(str(test_image_path))
        plt.imshow(test_image, cmap='gray')
        plt.title("Test Image")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(TEST_TIME)
        plt.close()

        while True:
            try:
                user_rating = float(input("Your rating for the test image (1-5): "))
            except ValueError:
                print("Please enter a number from 1 to 5.")
                continue
            if 1 <= user_rating <= 5:
                break
            print("Please enter a whole number from 1 to 5.")

        print(f"You rated the test image ({test_image_path.stem}) with a score of: {user_rating}")
        user_ratings.append({
            "test_rating": float(test_image_path.stem),
            "adaptation": adaptation_label,
            "user_rating": user_rating,
        })

# In your report, plot the ratings for the three test stimuli separately for the two adaptation conditions.
# For each test stimulus, compare the ratings obtained after adaptation to the two opposite endpoints.
# Determine whether the direction of the difference is consistent with the predicted perceptual after-
# effect.

fig, axes = plt.subplots(1, len(test_image_path_list), figsize=(15, 5), sharey=True)
for ax, test_image_path in zip(axes, test_image_path_list):
    test_rating = float(test_image_path.stem)
    condition_ratings = [
        [
            result["user_rating"]
            for result in user_ratings
            if result["test_rating"] == test_rating
            and result["adaptation"] == adaptation_label
        ]
        for adaptation_label in ("Low adaptation", "High adaptation")
    ]
    means = [
        np.mean(np.asarray(ratings, dtype=float)) if ratings else np.nan
        for ratings in condition_ratings
    ]
    ax.plot([0, 1], means, color="tab:blue", marker="o", label="Mean rating")
    ax.set_title(f"Test stimulus: {test_rating:g}")
    ax.set_xticks([0, 1], ["Low", "High"])
    ax.set_xlabel("Adaptation")
    ax.set_ylim(1, 6)
    ax.grid(axis="y", alpha=0.3)

    difference = means[0] - means[1]
    direction = "consistent" if difference > 0 else "not consistent"
    print(
        f"Test stimulus {test_rating:g}: low adaptation mean - high adaptation mean "
        f"= {difference:.2f} ({direction} with the predicted after-effect)."
    )

axes[0].set_ylabel("User rating (1-5)")
axes[0].legend()
fig.suptitle("Ratings after adaptation to opposite endpoints")
fig.tight_layout()
# plt.show()
plt.savefig(f"{student_number}_adaptation_ratings.svg")
