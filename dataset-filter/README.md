# Dataset Filter Script

This script loads the `Roronotalt/bluesky-ten-million` dataset from Hugging Face, filters it to include only English language posts, and saves the filtered data to a CSV file.

## Description

The script performs the following main operations:

1. **Loads Data**: It fetches a specified dataset from Hugging Face. By default, it uses the "Roronotalt/bluesky-ten-million" dataset.
2. **Filters Data**: It converts the dataset into a pandas DataFrame and filters out entries that are not in English.
3. **Saves Data**: The resulting DataFrame, containing only English posts, is saved as a CSV file.

## Functionality

- **Directory Setup**: Creates `output` and `temp` directories if they don't already exist. The filtered dataset will be saved within the `output/dataset-filter` directory.
- **Logging**: Implements basic logging to track the script's progress and any potential issues.
- **Data Processing**:
  - Uses the `datasets` library to load data.
  - Uses the `pandas` library for data manipulation and filtering.
  - Specifically filters posts where the 'langs' field contains 'en'.

## Output

The script generates a CSV file named `bluesky_ten_million_english_only.csv` located in the `./output/dataset-filter/` directory relative to the script's parent directory. This file contains the posts from the original dataset that were identified as being in English.
