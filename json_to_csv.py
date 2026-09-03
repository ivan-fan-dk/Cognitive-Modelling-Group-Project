import json
import csv
import os

def json_to_csv(json_path, output_csv_path=None):
    """
    Reads a JSON file with filename-to-rating mappings and converts it to a CSV file.
    
    :param json_path: Path to the input .json file
    :param output_csv_path: Optional path for the output .csv file. 
                            If None, saves in the same directory as the JSON file.
    """
    # 1. Ensure the JSON file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Error: File not found at path: '{json_path}'")
    
    # 2. Determine output CSV path if not specified
    if output_csv_path is None:
        output_csv_path = os.path.splitext(json_path)[0] + ".csv"
        
    # 3. Read the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        ratings_data = json.load(f)
        
    # 4. Write to CSV file
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header row
        writer.writerow(["filename", "rating"])
        
        # Data rows
        for filename, rating in ratings_data.items():
            writer.writerow([filename, rating])
            
    print(f"Success! Converted '{json_path}' -> '{output_csv_path}' ({len(ratings_data)} rows).")

# ==========================================
# USAGE OPTIONS
# ==========================================

# Option A: Direct path specification
json_file_path = r"ratings_fabian.json"
# json_to_csv(json_file_path)

# Option B: Interactive prompt (asks you for the path when run)
if __name__ == "__main__":
    user_json_path = input("Enter the path to your JSON file: ").strip('"\'')
    json_to_csv(user_json_path)