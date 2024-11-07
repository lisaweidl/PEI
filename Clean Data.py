import pandas as pd

# Load the Excel file with the actual headers on the third row (index 2)
file_path = '/Users/lisa-marieweidl/Desktop/SEEP/WS 24:25/PEI 2/PEI Data News Outlets.xlsx'
df = pd.read_excel(file_path, header=2)

# Drop rows that are completely empty and columns with more than 90% missing values
df.dropna(how='all', inplace=True)  # Drop rows where all values are NaN
df.dropna(axis=1, thresh=len(df) * 0.1, inplace=True)  # Drop columns with >90% missing values

# Rename columns based on the actual header row
df.columns = ['article_id', 'title', 'content', 'news_outlet', 'keyword', 'publication_date', 'author_1', 'url']

# Drop the first row, which contains the old header information
df = df.drop(0).reset_index(drop=True)

# Convert 'publication_date' to datetime format
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')

# Save the cleaned data to a new Excel file
cleaned_file_path = '/Users/lisa-marieweidl/Desktop/SEEP/WS 24:25/PEI 2/PEI Cleaned Data.xlsx'
df.to_excel(cleaned_file_path, index=False)

print("Data cleaned and saved to", cleaned_file_path)
