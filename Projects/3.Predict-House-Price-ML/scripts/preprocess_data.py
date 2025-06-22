import pandas as pd
import os

def preprocess_house_data(input_path, output_path):
    print(f"Preprocessing data from: {input_path}")
    df = pd.read_csv(input_path)

    # Perform complex preprocessing steps here
    # Example:
    # df.drop_duplicates(inplace=True)
    # df['AgeOfHouse'] = 2025 - df['YearBuilt']
    # df.fillna(df.mean(numeric_only=True), inplace=True)

    # Select necessary columns for later training
    # (Or save the entire cleaned dataframe)
    processed_df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'LotArea', 'Neighborhood', 'SalePrice']].copy()
    processed_df.dropna(inplace=True) # Ensure no NaNs before saving

    processed_df.to_csv(output_path, index=False)
    print(f"Saved processed data at: {output_path}")

if __name__ == "__main__":
    input_csv = os.path.join('data', 'house_data.csv')
    output_csv = os.path.join('data', 'processed_house_data.csv') # Save back to the data directory
    preprocess_house_data(input_csv, output_csv)