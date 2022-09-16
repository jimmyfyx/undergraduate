import torch
import pandas as pd

df = pd.read_csv('PJME_hourly.csv')

df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)
if not df.index.is_monotonic:
    df = df.sort_index()
    
df = df.rename(columns={'PJME_MW': 'value'})
