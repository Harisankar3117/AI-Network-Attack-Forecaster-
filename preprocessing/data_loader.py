import pandas as pd
from preprocessing.data_cleaner import clean_dataset

def load_dataset(file_path, chunksize=100000):

    print("\nLoading dataset...")
    print("File:", file_path)

    cleaned_chunks = []
    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            chunksize=chunksize,
            low_memory=False
        ),
        start=1
    ):

        print(f"Processing chunk {chunk_number}...")

        total_rows += len(chunk)

        # Clean current chunk
        cleaned_chunk = clean_dataset(chunk)

        if not cleaned_chunk.empty:
            cleaned_chunks.append(cleaned_chunk)

    if not cleaned_chunks:
        print("No valid data found!")
        return pd.DataFrame()

    # Combine cleaned chunks
    df = pd.concat(cleaned_chunks, ignore_index=True)

    print("\nDataset loaded successfully!")
    print("--------------------------------")
    print("Original rows :", total_rows)
    print("Cleaned rows  :", len(df))
    print("Columns       :", len(df.columns))
    print("--------------------------------")

    return df


if __name__ == "__main__":
    print("Data loader module is ready.")