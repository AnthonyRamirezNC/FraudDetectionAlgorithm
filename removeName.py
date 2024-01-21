import pandas as pd

dataset = pd.read_csv("creditCardData.csv")
print('read csv')
dataset = dataset.drop(labels=["nameOrig", "nameDest"], axis=1)

# Swap the last two columns
cols = dataset.columns.tolist()
cols[-1], cols[-2] = cols[-2], cols[-1]
dataset = dataset[cols]

print('writing to csv')
dataset.to_csv("creditCardDataCopy.csv", index=False)