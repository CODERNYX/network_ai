import pandas as pd
import time
import numpy as np

def stream_data(file_path):

    df = pd.read_csv(file_path)

    df = df.sample(frac=1).reset_index(drop=True)

    for i in range(len(df)):

        row = df.iloc[i]

        traffic = row["Flow Bytes/s"]

        yield traffic, row

        time.sleep(0.2)
