import matplotlib
matplotlib.use('TkAgg')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from collections import Counter

sns.set(style='whitegrid')

def plot_class_distribution(y, labels, title):
    counts = pd.Series(y).value_counts().sort_index()
    valid_indices = counts.index[counts.index < len(labels)]
    counts = counts.loc[valid_indices]
    names = labels[counts.index]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=names, y=counts.values, hue=names, palette='viridis', legend=False)
    ax.set_title(title)
    ax.set_xlabel('Sınıf')
    ax.set_ylabel('Sayı')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

# Veri yükleme ve önişleme
def load_data():
    tracks_path = 'fma_metadata/tracks.csv'
    features_path = 'fma_metadata/features.csv'

    if not os.path.exists(tracks_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Gerekli veri dosyaları bulunamadı. '{tracks_path}' ve '{features_path}' dosyalarının mevcut olduğundan emin olun.")

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0,1])
    
    features = pd.read_csv(features_path, index_col=0, header=[0,1])  # Çok seviyeli başlıkla oku
    features = features.loc[:, features.columns.get_level_values(0) != 'statistics']  # 'statistics' sütunlarını kaldır
    features = features.astype(np.float32)  # Sayısal olmayan sütunları kaldırdıktan sonra float'a dönüştür

    features.index = features.index.astype(str)
    tracks.index = tracks.index.astype(str)

    genre_series = tracks[('track', 'genre_top')].dropna()
    common_index = features.index.intersection(genre_series.index)

    X = features.loc[common_index]
    y_labels = genre_series.loc[common_index]

    X = X.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    print('Veriler yüklendi ve önişlendi.')
    return X, y, label_encoder

if __name__ == '__main__':
    try:
        X, y, le = load_data()
        plot_class_distribution(y, le.classes_, 'Başlangıç Sınıf Dağılımı')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        plot_class_distribution(y_train, le.classes_, 'Eğitim Seti Dağılımı')
        print(f'Eğitim/test bölünmesi tamamlandı: X_train {X_train.shape}, X_test {X_test.shape}')

        # İlk dağılımı yazdır
        unique, counts = np.unique(y_train, return_counts=True)
        print("Eğitim Seti Dağılımı (ham sayılar):", dict(zip(unique, counts)))

        # Adım 1: En az temsil edilen sınıf için RandomOverSampler kullan
        print('\nAdım 1: Aşırı az temsil edilen sınıflar için RandomOverSampler uygulanıyor...')
        min_samples_threshold = 100  # BorderlineSMOTE için gereken minimum örnek sayısı
        ros = RandomOverSampler(sampling_strategy={3: min_samples_threshold}, random_state=42)
        X_partial, y_partial = ros.fit_resample(X_train, y_train)
        
        # Ara sonuçları göster
        unique_partial, counts_partial = np.unique(y_partial, return_counts=True)
        print("RandomOverSampler sonrası dağılım (ham sayılar):", dict(zip(unique_partial, counts_partial)))
        plot_class_distribution(y_partial, le.classes_, 'RandomOverSampler Sonrası')

        # Adım 2: Kalan sınıfları dengelemek için BorderlineSMOTE uygula
        print('\nAdım 2: Kalan sınıflar için BorderlineSMOTE uygulanıyor...')
        borderline_smote = BorderlineSMOTE(random_state=42)
        try:
            X_res, y_res = borderline_smote.fit_resample(X_partial, y_partial)
            print(f'Kombine örnekleme tamamlandı: X_res {X_res.shape}, y_res {y_res.shape}')
            
            # Son dağılımı yazdır ve göster
            unique_res, counts_res = np.unique(y_res, return_counts=True)
            print("Son Dağılım (ham sayılar):", dict(zip(unique_res, counts_res)))
            
            plot_class_distribution(y_res, le.classes_, 'Son Dengelenmiş Dağılım')
            plt.pause(0.1)

        except Exception as e:
            print(f'BorderlineSMOTE örnekleme başarısız oldu: {e} - kısmi örneklenmiş veri kullanılıyor')
            X_res, y_res = X_partial, y_partial
            plot_class_distribution(y_res, le.classes_, 'Kısmi Örnekleme (BorderlineSMOTE başarısız)')
            plt.pause(0.1)

        print("\nİşlem hattı tamamlandı. Yeniden örneklenmiş eğitim verisi (X_res, y_res) ve test verisi (X_test, y_test) hazır.")
        print("\nTüm dağılım grafikleri görünür olmalı. Örnekleme artırmanın etkilerini incelemek için grafikleri inceleyin.")
        
        # kapatana kadar pencereleri açık tut
        input("\nTüm diagram pencerelerini kapatmak için Enter'a basın...")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")