import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(train_path, test_path):
    # Load CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split features and labels
    X = train_df.drop('label', axis=1).values
    y = train_df['label'].values

    # Normalize pixel values
    X = X / 255.0
    test_data = test_df.values / 255.0

    # Reshape for CNN: (samples, 28, 28, 1)
    X = X.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y = to_categorical(y, num_classes=10)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_val, y_train, y_val, test_data

