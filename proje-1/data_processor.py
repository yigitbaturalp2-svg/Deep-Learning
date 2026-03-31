import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    """
    Veri okuma, eksik veri analizi, kodlama (encoding) ve bölme işlemlerini yapan yardımcı sınıf.
    """
    def __init__(self, file_path, random_seed=42):
        self.file_path = file_path
        self.random_seed = random_seed
        self.target_col = 'Depression'
        
        self.feature_cols = [
            'Gender', 'Age', 'Academic Pressure', 'Study Satisfaction',
            'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
            'Study Hours', 'Financial Stress', 'Family History of Mental Illness'
        ]
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.X_train = None
        self.X_dev = None
        self.X_test = None
        self.y_train = None
        self.y_dev = None
        self.y_test = None
        self.num_classes = None
        self.input_size = None

    def _handle_missing_values(self, df):
        """
        Eksik verileri sayısal ve kategorik dağılıma göre doldurur.
        """
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                mode_opts = df[col].mode()
                if len(mode_opts) > 0:
                    df[col] = df[col].fillna(mode_opts[0])
                else:
                    df[col] = df[col].fillna("Bilinmiyor")
        return df

    def _encode_features(self, X):
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        else:
            X_encoded = X.copy()
        
        return X_encoded.astype(float)

    def process_data(self):
        df = pd.read_csv(self.file_path, low_memory=False)
        
        df = df.dropna(subset=[self.target_col])
        y_raw = df[self.target_col]
        X_raw = df[self.feature_cols].copy()
        
        X_raw = self._handle_missing_values(X_raw)
        
        y_encoded = self.label_encoder.fit_transform(y_raw)
        self.num_classes = len(self.label_encoder.classes_)
        
        X_encoded = self._encode_features(X_raw)
        
        # Eğitim, Doğrulama (Dev) ve Test Bölünmesi (%70, %15, %15)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_encoded, y_encoded, test_size=0.30, random_state=self.random_seed, stratify=y_encoded
        )
        X_dev, X_test, y_dev, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.random_seed, stratify=y_temp
        )
        
        # Ölçekleme (StandardScaler) 
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_dev = self.scaler.transform(X_dev)
        self.X_test = self.scaler.transform(X_test)
        
        self.y_train = y_train.reshape(-1, 1)
        self.y_dev = y_dev.reshape(-1, 1)
        self.y_test = y_test.reshape(-1, 1)
        
        self.input_size = self.X_train.shape[1]
        
        print("Veri ön işleme tamamlandı.")
        print(f"Sınıf Sayısı: {self.num_classes} | Özellik Sayısı: {self.input_size}")
        
        return self.X_train, self.y_train, self.X_dev, self.y_dev, self.X_test, self.y_test
