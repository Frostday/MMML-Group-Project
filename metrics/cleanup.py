import pandas as pd
import re

df = pd.read_csv('data/llava_quantized_fine_tuned_results.csv')

def extract_answer(text):
    match = re.search(r'Answer:\s*(.*)', text)
    if match:
        return match.group(1)
    return text

df['generated_answer'] = df['generated_answer'].apply(extract_answer).str.replace(r'### Question:.*', '', regex=True)

df.to_csv('data/llava_quantized_fine_tuned_results_cleaned.csv', index=False)
