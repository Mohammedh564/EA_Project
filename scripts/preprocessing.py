import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Preprocessing:
    def __init__(self):
        # Robust path handling
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.datapath = os.path.join(BASE_DIR, "..", "data", "raw", "data.csv")

        self.data = self.load_data(self.datapath)

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self):
        data = self.data.copy()

        # 1. Encode target
        le = LabelEncoder()
        data['diagnosis'] = le.fit_transform(data['diagnosis'])

        # 2. Drop unnecessary columns
        data = data.drop(columns=['id', 'Unnamed: 32'])

        # 3. Split features and target
        X = data.drop(columns=['diagnosis'])
        y = data['diagnosis']

        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Convert labels to numpy
        y_train = y_train.values
        y_test = y_test.values

        # 5. Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, scaler

    def save_data(self, X_train, X_test, y_train, y_test, scaler):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(BASE_DIR, "..", "data", "processed")

        # Create folder if not exists
        os.makedirs(save_path, exist_ok=True)

        np.save(os.path.join(save_path, "X_train.npy"), X_train)
        np.save(os.path.join(save_path, "X_test.npy"), X_test)
        np.save(os.path.join(save_path, "y_train.npy"), y_train)
        np.save(os.path.join(save_path, "y_test.npy"), y_test)

        joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))


if __name__ == "__main__":
    preprocessor = Preprocessing()

    X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess_data()

    preprocessor.save_data(X_train, X_test, y_train, y_test, scaler)

    print("✅ Data preprocessing completed and saved successfully.")