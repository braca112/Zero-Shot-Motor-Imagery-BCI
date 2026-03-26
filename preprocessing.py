import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load sample EEG data
df = pd.read_csv('../data/sample_eeg.csv')

# Standardize channels
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Save processed data
df_scaled.to_csv('../data/sample_eeg_scaled.csv', index=False)

print("Preprocessing done. Saved to data/sample_eeg_scaled.csv")
