import json
from pathlib import Path
import numpy as np
import pandas as pd

file_path = "test-A/SPIQA_testA.json"

with open(file_path, 'r') as file:
    text = json.load(file)

data = []
cols = ['paper', 'question', 'answer', 'reference']
for paper in text.keys():
    for question in text[paper]['qa']:
        data.append([paper, question['question'], question['answer'], question['reference']])

df = pd.DataFrame(data, columns=cols)
df['generated_answer'] = np.nan
df.to_csv('sample.csv', index=False)
