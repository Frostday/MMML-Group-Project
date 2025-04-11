import os
import pandas as pd

images_path = "/Users/frostday/Downloads/SPIQA_train_val_Images/"
new_path = "/Users/frostday/Downloads/val_images/"

df = pd.read_csv("data/sample_val.csv")

for idx, val in df.iterrows():
    source = os.path.join(images_path, df.iloc[idx]['paper'], df.iloc[idx]['reference_figure'])
    destination = os.path.join(new_path, df.iloc[idx]['paper'], df.iloc[idx]['reference_figure'])
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    os.system(f"cp {source} {destination}")
