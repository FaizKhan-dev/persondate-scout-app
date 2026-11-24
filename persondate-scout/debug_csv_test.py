import io
import pandas as pd

path = 'ner_data.csv'
with open(path, 'rb') as f:
    data = f.read()
print('raw bytes length:', len(data))
# Show the first 3 lines repr
lines = data.splitlines()
for i, line in enumerate(lines[:5]):
    print(i, repr(line))

print('\nTrying pandas read_csv default...')
df = pd.read_csv(path)
print('df shape default:', df.shape)
print(df.columns.tolist())

print('\nTrying pandas read_csv engine=python...')
df2 = pd.read_csv(path, engine='python')
print('df shape python engine:', df2.shape)
print(df2.columns.tolist())

print('\nTrying pandas read_csv sep=","...')
df3 = pd.read_csv(path, sep=',')
print('df shape sep comma:', df3.shape)
print(df3.columns.tolist())
