import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class WeatherEM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        print("Data loaded successfully. Shape:", self.data.shape)
        print("\nSample data:")
        print(self.data.head())
        
    def preprocess_data(self, features):
        # Pilih fitur yang relevan
        self.X = self.data[features].copy()
        
        # Handle missing values
        self.X.fillna(self.X.mean(), inplace=True)
        
        # Normalisasi data
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print("\nData after preprocessing:")
        print(pd.DataFrame(self.X_scaled, columns=features).head())
        
    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test = train_test_split(
            self.X_scaled, test_size=test_size, random_state=random_state
        )
        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        
    def fit_model(self):
        self.model = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=42
        )
        
        # E-step dan M-step
        self.model.fit(self.X_train)
        
        print("\nModel training completed.")
        print(f"Converged: {self.model.converged_}")
        print(f"Iterations: {self.model.n_iter_}")
        
    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model belum dilatih. Panggil fit_model() terlebih dahulu.")
            
        # Prediksi cluster untuk data training
        train_labels = self.model.predict(self.X_train)
        
        # Hitung silhouette score
        score = silhouette_score(self.X_train, train_labels)
        print(f"\nSilhouette Score: {score:.3f}")
        
        return score
        
    def predict_weather_cluster(self, new_data):
        # Preprocess data baru
        new_data_processed = new_data.copy()
        new_data_processed.fillna(self.X.mean(), inplace=True)
        new_data_scaled = self.scaler.transform(new_data_processed)
        
        # Prediksi
        cluster = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)
        
        return cluster, probabilities
        
    def visualize_clusters(self, feature1, feature2):
        """
        Visualisasi cluster untuk dua fitur
        
        Parameters:
        - feature1: Nama fitur pertama
        - feature2: Nama fitur kedua
        """
        if self.model is None:
            raise ValueError("Model belum dilatih. Panggil fit_model() terlebih dahulu.")
            
        # Dapatkan indeks fitur
        idx1 = self.X.columns.get_loc(feature1)
        idx2 = self.X.columns.get_loc(feature2)
        
        # Plot data points
        plt.figure(figsize=(10, 6))
        
        # Plot data training dengan warna sesuai cluster
        train_labels = self.model.predict(self.X_train)
        scatter = plt.scatter(
            self.X_train[:, idx1], 
            self.X_train[:, idx2], 
            c=train_labels, 
            cmap='viridis',
            alpha=0.6,
            label='Data points'
        )
        
        # Plot pusat cluster
        centers = self.model.means_
        plt.scatter(
            centers[:, idx1], 
            centers[:, idx2], 
            c='red', 
            marker='X', 
            s=200,
            label='Cluster centers'
        )
        
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title('Weather Microclimate Clusters')
        plt.legend()
        plt.colorbar(scatter, label='Cluster')
        plt.show()


# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi model
    weather_model = WeatherEM(n_components=4, max_iter=200)
    
    # Memuat data (ganti dengan path file Anda)
    # Contoh data: https://www.kaggle.com/datasets/mathijs/weather-data-in-netherlands
    try:
        weather_model.load_data('weather_data.csv')
    except FileNotFoundError:
        print("\nFile tidak ditemukan. Menggunakan data contoh...")
        # Buat data contoh jika file tidak ada
        np.random.seed(42)
        example_data = {
            'temperature': np.random.normal(25, 5, 1000),
            'humidity': np.random.normal(60, 15, 1000),
            'pressure': np.random.normal(1010, 10, 1000),
            'wind_speed': np.random.exponential(5, 1000),
            'precipitation': np.random.gamma(1, 2, 1000)
        }
        weather_model.data = pd.DataFrame(example_data)
        print("Menggunakan data contoh dengan shape:", weather_model.data.shape)
    
    # Pilih fitur yang relevan untuk prediksi cuaca mikro
    features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
    
    # Preprocessing data
    weather_model.preprocess_data(features)
    
    # Bagi data menjadi training dan testing set
    weather_model.train_test_split()
    
    # Latih model
    weather_model.fit_model()
    
    # Evaluasi model
    weather_model.evaluate_model()
    
    # Visualisasi cluster untuk dua fitur
    weather_model.visualize_clusters('temperature', 'humidity')
    
    # Contoh prediksi untuk data baru
    new_weather_data = pd.DataFrame([{
        'temperature': 22.5,
        'humidity': 65,
        'pressure': 1012,
        'wind_speed': 3.2,
        'precipitation': 0.0
    }])
    
    cluster, probs = weather_model.predict_weather_cluster(new_weather_data)
    print(f"\nPredicted cluster: {cluster[0]}")
    print("Cluster probabilities:")
    for i, prob in enumerate(probs[0]):
        print(f"Cluster {i}: {prob:.3f}")