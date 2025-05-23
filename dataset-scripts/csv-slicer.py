import pandas as pd
import argparse  # For handling command-line arguments


def extract_random_half_million_rows(input_csv_path: str, output_csv_path: str):
    """
    Reads 500,000 random rows from an input CSV file and writes them to an output CSV file.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    try:
        # Read the entire CSV file.
        print(f"Reading the entire CSV file from '{input_csv_path}'...")
        df = pd.read_csv(input_csv_path)

        # Sample 500,000 random rows.
        print(f"Sampling 500,000 random rows...")
        df_sampled = df.sample(n=500000, random_state=42) if len(df) >= 500000 else df

        # Write the sampled DataFrame to a new CSV file.
        print(f"Writing the sampled rows to '{output_csv_path}'...")
        df_sampled.to_csv(
            output_csv_path, index=False
        )  # index=False prevents pandas from writing the DataFrame index as a column

        print("Successfully sampled and saved 500,000 random rows.")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_csv_path}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract 500,000 random rows from a CSV file."
    )
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "output_file", type=str, help="Path to save the output CSV file."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    extract_random_half_million_rows(args.input_file, args.output_file)

    # Example usage from the command line:
    # python csv_slicer.py input.csv output_sampled.csv
