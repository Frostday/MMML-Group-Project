import json
import numpy as np
import pandas as pd

# file_path = "SPIQA_testA.json"
file_path = "SPIQA_val.json"

with open(file_path, 'r') as file:
    text = json.load(file)

data = []
cols = ['paper', 'question', 'answer', 'reference_figure', 'reference_figure_caption']
for paper in text.keys():
    for question in text[paper]['qa']:
        data.append([paper, question['question'], question['answer'], question['reference'], text[paper]['all_figures'][question['reference']]['caption']])

df = pd.DataFrame(data, columns=cols)
df['generated_answer'] = np.nan
# df.to_csv('sample.csv', index=False)
df.to_csv('data/sample_val.csv', index=False)
