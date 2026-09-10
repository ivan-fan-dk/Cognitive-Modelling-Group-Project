import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd

def plot_histogram(file):
    """
    Plots a histogram of the ratings.

    Parameters:
    file (str): The path to the JSON file containing the ratings.
    """
    df = pd.read_csv(file)  # Read the CSV file to ensure it exists and is valid
    # Extract the ratings from the dictionary
    rating_values = list(df['rating'])
    rating_values.extend(df['rating2'])

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
    plt.savefig(f'histogram_{file.stem}.svg')

if __name__ == "__main__":
    for file in Path(__file__).parent.glob("s*.csv"):
        plot_histogram(file)