import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load preprocessed EEG data
df = pd.read_csv('../data/sample_eeg_scaled.csv')

# Dummy labels (same as train_model.py)
np.random.seed(42)
labels = np.random.choice([0,1], size=df.shape[0])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

# Load trained model
clf = joblib.load('../data/dummy_zero_shot_model.pkl')

# Generate predictions
y_pred = clf.predict(X_test)

# Compute accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Left Hand","Right Hand"], yticklabels=["Left Hand","Right Hand"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Zero-Shot Demo")
plt.show()
