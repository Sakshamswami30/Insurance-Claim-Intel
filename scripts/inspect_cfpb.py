import pandas as pd

df = pd.read_csv("data/samples/cfpb_complaints.csv", nrows=5)

print("Shape:", df.shape)
print()
print("Columns:")
for i, col in enumerate(df.columns):
    print(f"  {i}. {col}")
print()
print("First row values:")
for col in df.columns:
    value = df.iloc[0][col]
    if isinstance(value, str) and len(value) > 100:
        value = value[:100] + "..."
    print(f"  {col}: {value}")