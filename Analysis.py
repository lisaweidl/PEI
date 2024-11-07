import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# Load the model and tokenizer
model_name = 'oliverguhr/german-sentiment-bert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Read the Excel file
excel_path = '/Users/lisa-marieweidl/Desktop/SEEP/WS 24:25/PEI 2/PEI Cleaned Data.xlsx'
df = pd.read_excel(excel_path)

# Set the batch size
batch_size = 20  # You can change this value as needed

# Start timer
start_time = time.time()

# Initialize an empty list to store the results
results = []


# Define the classify_sentiment function
def classify_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Get the predicted class
    predicted_class_id = torch.argmax(logits, dim=1).item()
    # Map class ID to label
    predicted_class_label = model.config.id2label[predicted_class_id]
    return predicted_class_label


# Process the DataFrame in batches
for start_idx in range(0, len(df), batch_size):
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx].copy()

    # Apply classify_sentiment function to the batch
    batch_df['sentiment'] = batch_df['content'].apply(classify_sentiment)

    # Append the batch results to the results list
    results.append(batch_df[['article_id', 'news_outlet', 'sentiment']])

    # Print intermediate results
    print(f"Processed batch {start_idx // batch_size + 1}")
    print(batch_df['sentiment'].value_counts())
    print(batch_df[['article_id', 'news_outlet', 'sentiment']])
    print("\n")
    break

# Concatenate all batch results
final_df = pd.concat(results, ignore_index=True)

# End timer
end_time = time.time()
execution_time = end_time - start_time

# Save the specified columns to a CSV file
output_csv_path = 'sentiment_results.csv'  # You can specify your desired output path
final_df.to_csv(output_csv_path, index=False)

# Save the specified columns to an XLSX file
output_excel_path = 'sentiment_results.xlsx'  # You can specify your desired output path
final_df.to_excel(output_excel_path, index=False)

# Display the sentiment counts and the news outlets for the entire DataFrame
print("Final sentiment counts:")
print(final_df['sentiment'].value_counts())
print(final_df[['article_id', 'news_outlet', 'sentiment']])
print(f"Execution time: {execution_time:.2f} seconds")
