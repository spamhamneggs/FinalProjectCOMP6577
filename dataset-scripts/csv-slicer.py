import argparse  # For handling command-line arguments

import pandas as pd


def extract_random_n_rows(input_csv_path: str, output_csv_path: str, n: int):
    """
    Reads n random rows from an input CSV file and writes them to an output CSV file.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_csv_path (str): The path where the output CSV file will be saved.
        n (int): Number of rows to sample.
    """
    try:
        print(f"Reading the entire CSV file from '{input_csv_path}'...")
        df = pd.read_csv(input_csv_path)

        print(f"Sampling {n} random rows...")
        df_sampled = df.sample(n=n, random_state=42) if len(df) >= n else df

        print(f"Writing the sampled rows to '{output_csv_path}'...")
        df_sampled.to_csv(output_csv_path, index=False)

        print(f"Successfully sampled and saved {min(n, len(df))} random rows.")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{input_csv_path}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract n random rows from a CSV file."
    )
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "output_file", type=str, help="Path to save the output CSV file."
    )
    parser.add_argument("n", type=int, help="Number of random rows to sample.")

    args = parser.parse_args()

    extract_random_n_rows(args.input_file, args.output_file, args.n)

    # Example usage from the command line:
    # python csv_slicer.py input.csv output_sampled.csv 100000
