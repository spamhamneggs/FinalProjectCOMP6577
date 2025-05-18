import pandas as pd
import argparse  # For handling command-line arguments


def extract_top_million_rows(input_csv_path: str, output_csv_path: str):
    """
    Reads the first 1 million rows from an input CSV file and writes them to an output CSV file.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    try:
        # Read the first 1,000,000 rows from the CSV file.
        # pandas.read_csv is highly optimized and uses vectorized operations internally for reading.
        print(f"Reading the first 1,000,000 rows from '{input_csv_path}'...")
        df = pd.read_csv(input_csv_path, nrows=500000)

        # Write the extracted DataFrame to a new CSV file.
        # pandas.to_csv is also optimized for speed.
        print(f"Writing the extracted rows to '{output_csv_path}'...")
        df.to_csv(
            output_csv_path, index=False
        )  # index=False prevents pandas from writing the DataFrame index as a column

        print("Successfully extracted and saved the first 1,000,000 rows.")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_csv_path}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract the first 1 million rows from a CSV file."
    )
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "output_file", type=str, help="Path to save the output CSV file."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    extract_top_million_rows(args.input_file, args.output_file)

    # Example usage from the command line:
    # python csv_slicer.py input.csv output_million.csv
