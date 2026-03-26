import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load preprocessed EEG data
df = pd.read_csv('../data/sample_eeg_scaled.csv')

# Dummy labels for simulation: 0 = Left Hand, 1 = Right Hand
np.random.seed(42)
labels = np.random.choice([0,1], size=df.shape[0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

# Train a simple classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, '../data/dummy_zero_shot_model.pkl')

# Print training accuracy
accuracy = clf.score(X_test, y_test)
print(f"Dummy model trained. Test Accuracy: {accuracy:.2f}")
print("Model saved to data/dummy_zero_shot_model.pkl")
