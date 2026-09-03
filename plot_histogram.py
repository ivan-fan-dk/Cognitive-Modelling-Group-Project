from pathlib import Path
import matplotlib.pyplot as plt
import json


def plot_histogram(file):
    """
    Plots a histogram of the ratings.

    Parameters:
    file (str): The path to the JSON file containing the ratings.
    """
    with open(file, "r") as f:
        sample_ratings = json.load(f)
        # Extract the ratings from the dictionary
        rating_values = list(sample_ratings.values())

        # Create a histogram
        plt.figure(figsize=(10, 6))
        plt.hist(rating_values, bins=range(min(rating_values), max(rating_values) + 2), edgecolor='black', alpha=0.7)
        
        # Add titles and labels
        plt.title('Histogram of Image Ratings')
        plt.xlabel('Ratings')
        plt.ylabel('Frequency')
        
        # Show the plot
        plt.grid(axis='y', alpha=0.75)
        plt.title(f'Histogram of Ratings from {file.name}')
        plt.show()

if __name__ == "__main__":
    for file in Path(__file__).parent.glob("ratings_*.json"):
        plot_histogram(file)