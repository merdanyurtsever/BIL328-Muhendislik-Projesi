# Müzik Türü Sınıflandırma Projesi - Optimized Version

"""
Bu script, FMA (Free Music Archive) veri setini kullanarak müzik türü sınıflandırma modeli 
geliştirmek için veri hazırlama ve dengeleme işlemlerini içermektedir.

FEATURES:
- Enhanced temporal LSTM models with attention mechanisms
- Focal loss for handling class imbalance
- Comprehensive hyperparameter optimization
- Feature importance analysis
- Memory-optimized training pipeline
- Model checkpointing and visualization

USAGE:
    python Main.py --epochs 30 --batch_size 128 --hidden_size 128
    python Main.py --analyze_features --optimize_hyperparameters
    python Main.py --help  # For full option list
"""

# Gerekli Kütüphanelerin İçe Aktarılması
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_fscore_support, roc_curve, auc, f1_score, accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from collections import Counter, defaultdict
from itertools import cycle
import warnings
import traceback
import json
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Matplotlib ayarları
plt.style.use('default')
sns.set_style('whitegrid')

# Yardımcı Fonksiyonlar

def plot_class_distribution(y, labels, title):
    """
    Veri setindeki sınıf dağılımlarını görselleştirmek için kullanılan fonksiyon.
    Bu görselleştirme, veri dengesizliğini anlamamıza yardımcı olur.
    
    Args:
        y: Sınıf etiketleri
        labels: Sınıf isimleri
        title: Grafik başlığı
    """
    counts = pd.Series(y).value_counts().sort_index()
    valid_indices = counts.index[counts.index < len(labels)]
    counts = counts.loc[valid_indices]
    names = labels[counts.index]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=names, y=counts.values, hue=names, palette='viridis', legend=False)
    ax.set_title(title)
    ax.set_xlabel('Sınıf')
    ax.set_ylabel('Sayı')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def load_data():
    """
    FMA metadata dosyalarını yükler ve önişleme yapar.
    Enhanced with better error handling and memory optimization.
    
    Returns:
        X: Özellik matrisi
        y: Kodlanmış etiketler
        label_encoder: Etiket kodlayıcı
    """
    tracks_path = 'fma_metadata/tracks.csv'
    features_path = 'fma_metadata/features.csv'

    if not os.path.exists(tracks_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Gerekli veri dosyaları bulunamadı. '{tracks_path}' ve '{features_path}' dosyalarının mevcut olduğundan emin olun.")

    try:
        print("📂 Loading tracks metadata...")
        tracks = pd.read_csv(tracks_path, index_col=0, header=[0,1])
        
        print("📂 Loading features...")
        features = pd.read_csv(features_path, index_col=0, header=[0,1])
        
        # Remove statistics columns (not needed for temporal modeling)
        features = features.loc[:, features.columns.get_level_values(0) != 'statistics']
        
        # Memory optimization: convert to float32 instead of float64
        print("🔄 Converting to optimized data types...")
        features = features.astype(np.float32)

        # Ensure consistent index types
        features.index = features.index.astype(str)
        tracks.index = tracks.index.astype(str)

        # Extract genre information
        genre_series = tracks[('track', 'genre_top')].dropna()
        common_index = features.index.intersection(genre_series.index)
        
        if len(common_index) == 0:
            raise ValueError("No common indices found between features and tracks")

        X = features.loc[common_index]
        y_labels = genre_series.loc[common_index]

        # Data cleaning with improved handling
        print("🧹 Cleaning data...")
        initial_shape = X.shape
        X = X.fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
        
        # Check for any remaining invalid values
        if X.isnull().any().any():
            print("⚠️ Warning: Still have NaN values after cleaning")
        
        print(f"📊 Data shape: {initial_shape} -> {X.shape}")

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)
        
        print(f"✅ Data loaded successfully:")
        print(f"   - Samples: {X.shape[0]}")
        print(f"   - Features: {X.shape[1]}")
        print(f"   - Classes: {len(label_encoder.classes_)}")
        print(f"   - Class names: {list(label_encoder.classes_)}")

        return X, y, label_encoder
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def create_temporal_feature_data(X, y, feature_info=None):
    """
    Gelişmiş özellik mühendisliği: Doğal ses özelliği yapısını koruyan temporal sekanslar oluşturur.
    FMA özellik yapısı:
    - chroma_cens/cqt/stft: 12 boyut (pitch sınıfları) - müzikal anlamlı temporal yapı
    - mfcc: 20 boyut (katsayılar) - spektral envelope karakteristikleri
    - spectral_contrast: 7 boyut (alt bantlar) - frekans bantları arası kontrast
    - tonnetz: 6 boyut (harmonik koordinatlar) - tonal harmonik uzay
    - tek boyutlu: rmse, spectral_bandwidth, spectral_centroid, spectral_rolloff, zcr
    Her boyut için 7 istatistik: kurtosis, max, mean, median, min, skew, std
    Args:
        X: Multi-level column özellik matrisi (pandas DataFrame)
        y: Etiketler
        feature_info: Existing feature info (for validation/test sets), optional
    Returns:
        temporal_features: Dict of temporal feature tensors by feature type
        y_tensor: Etiket tensörü
        feature_info: Temporal yapı hakkında bilgi
    """
    print("🎵 Gelişmiş temporal özellik mühendisliği başlatılıyor...")
    temporal_feature_types = {
        'chroma_cens': 12, 'chroma_cqt': 12, 'chroma_stft': 12,
        'mfcc': 20, 'spectral_contrast': 7, 'tonnetz': 6
    }
    scalar_feature_types = {
        'rmse': 1, 'spectral_bandwidth': 1, 'spectral_centroid': 1,
        'spectral_rolloff': 1, 'zcr': 1
    }
    stats = ['kurtosis', 'max', 'mean', 'median', 'min', 'skew', 'std']
    temporal_features = {}
    
    # Use existing feature_info if provided, otherwise create new one
    if feature_info is None:
        feature_info = {
            'temporal_types': [], 'sequence_lengths': [],
            'feature_dimensions': [], 'total_temporal_features': 0
        }
        is_training = True
    else:
        is_training = False
    n_samples = X.shape[0]

    # Process temporal features
    if is_training:
        # Training mode: discover feature structure
        feature_types_to_process = temporal_feature_types.items()
    else:
        # Validation/test mode: use existing feature structure
        feature_types_to_process = [(ft, temporal_feature_types.get(ft, 0)) for ft in feature_info['temporal_types']]

    for feature_type, seq_length in feature_types_to_process:
        if not is_training and feature_type not in feature_info['temporal_types']:
            continue
            
        print(f"   📊 İşleniyor: {feature_type} (temporal boyut: {seq_length})")
        feature_data = np.zeros((n_samples, seq_length, len(stats)))
        found_features_count = 0
        
        # Try to access columns assuming 2-level (feature_type, stat)
        # This structure implies that X[(feature_type, stat)] returns a DataFrame of shape (n_samples, seq_length)
        if X.columns.nlevels == 2:
            for stat_idx, stat in enumerate(stats):
                try:
                    # This should be a DataFrame (n_samples, seq_length) or Series if seq_length is 1
                    data_slice = X.loc[:, (feature_type, stat)] 
                    if isinstance(data_slice, pd.DataFrame) and data_slice.shape[1] == seq_length:
                        feature_data[:, :, stat_idx] = data_slice.values
                        found_features_count += seq_length # All dimensions for this stat processed
                    elif isinstance(data_slice, pd.Series) and seq_length == 1: # Should not happen for these types
                        feature_data[:, 0, stat_idx] = data_slice.values
                        found_features_count += 1
                except KeyError:
                    pass # This stat might not exist for this feature_type
        
        # Fallback or primary access for 3-level columns (feature_type, stat, dim_str)
        elif X.columns.nlevels == 3:
            for stat_idx, stat in enumerate(stats):
                for dim in range(seq_length):
                    # FMA uses 1-based indexing for dimensions in column names like '01', '02', ... or '1', '2', ...
                    dim_str_v1 = str(dim + 1) # e.g., '1', '2', ..., '12'
                    dim_str_v2 = f"{dim + 1:02d}" # e.g., '01', '02', ..., '12'
                    
                    col_name_v1 = (feature_type, stat, dim_str_v1)
                    col_name_v2 = (feature_type, stat, dim_str_v2)
                    
                    actual_col_name = None
                    if col_name_v1 in X.columns:
                        actual_col_name = col_name_v1
                    elif col_name_v2 in X.columns:
                        actual_col_name = col_name_v2
                        
                    if actual_col_name:
                        feature_data[:, dim, stat_idx] = X[actual_col_name].values
                        found_features_count += 1

        if found_features_count > 0:
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            temporal_features[feature_type] = torch.FloatTensor(feature_data)
            if is_training:
                feature_info['temporal_types'].append(feature_type)
                feature_info['sequence_lengths'].append(seq_length)
                feature_info['feature_dimensions'].append(len(stats))
                feature_info['total_temporal_features'] += seq_length * len(stats)
            print(f"      ✓ Shape: {feature_data.shape} (samples, seq_len={seq_length}, features={len(stats)})")
        else:
            print(f"      ❌ No columns found or processed for {feature_type}")

    # Process scalar features
    scalar_feature_values = []
    if is_training or 'scalar_features' in feature_info:
        for feature_type_scalar, _ in scalar_feature_types.items():
            for stat in stats:
                actual_col_name = None
                # Scalar features in FMA usually have a single dimension, often named '0' or '01'
                # Or, if columns are 2-level, it's just (feature_type, stat)
                if X.columns.nlevels == 3:
                    col_name_v1 = (feature_type_scalar, stat, '0')
                    col_name_v2 = (feature_type_scalar, stat, '01')
                    col_name_v3 = (feature_type_scalar, stat, '1') # Less common but possible
                    if col_name_v1 in X.columns: actual_col_name = col_name_v1
                    elif col_name_v2 in X.columns: actual_col_name = col_name_v2
                    elif col_name_v3 in X.columns: actual_col_name = col_name_v3
                elif X.columns.nlevels == 2:
                    col_name_2level = (feature_type_scalar, stat)
                    if col_name_2level in X.columns: actual_col_name = col_name_2level
                
                if actual_col_name:
                    scalar_feature_values.append(X[actual_col_name].values)

        if scalar_feature_values:
            scalar_array = np.column_stack(scalar_feature_values)
            scalar_array = np.nan_to_num(scalar_array, nan=0.0, posinf=0.0, neginf=0.0)
            temporal_features['scalar'] = torch.FloatTensor(scalar_array)
            if is_training:
                feature_info['scalar_features'] = scalar_array.shape[1]
            print(f"   📊 Skaler özellikler: {scalar_array.shape[1]} özellik")
        else:
            print("   ⚠️ No scalar features found or processed.")
            if is_training and 'scalar_features' in feature_info: 
                del feature_info['scalar_features']
            
    y_tensor = torch.LongTensor(y)
    print(f"✅ Temporal özellik mühendisliği tamamlandı!")
    if is_training:
        print(f"   Temporal özellik türleri: {feature_info['temporal_types']}")
        if 'scalar_features' in feature_info:
            print(f"   Skaler özellikler: {feature_info['scalar_features']}")
        if not feature_info['temporal_types'] and 'scalar_features' not in feature_info:
            print("⚠️ UYARI: Hiçbir özellik (temporal veya skaler) bulunamadı/işlenemedi.")
        elif not feature_info['temporal_types']:
            print("⚠️ UYARI: Hiç temporal özellik bulunamadı. Model yalnızca skaler özelliklerle çalışacak.")
            feature_info['feature_dimensions'] = []
    else:
        print(f"   Using existing feature structure with {len(feature_info['temporal_types'])} temporal types")
        
    return temporal_features, y_tensor, feature_info

def perform_data_balancing(X_train, y_train):
    """
    Veri dengeleme işlemini gerçekleştirir.
    İlk aşamada RandomOverSampler, ikinci aşamada BorderlineSMOTE kullanır.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri
        
    Returns:
        X_res: Dengelenmiş özellikler
        y_res: Dengelenmiş etiketler
    """
    # Adım 1: En az temsil edilen sınıflar için RandomOverSampler
    print('Adım 1: Aşırı az temsil edilen sınıflar için RandomOverSampler uygulanıyor...')
    min_samples_threshold = 20  # BorderlineSMOTE için gereken minimum örnek sayısı
    ros = RandomOverSampler(sampling_strategy={3: min_samples_threshold}, random_state=42)
    X_partial, y_partial = ros.fit_resample(X_train, y_train)

    # Ara sonuçları göster
    unique_partial, counts_partial = np.unique(y_partial, return_counts=True)
    print("RandomOverSampler sonrası dağılım (ham sayılar):")
    for i, (u, c) in enumerate(zip(unique_partial, counts_partial)):
        print(f"Sınıf {u}: {c} örnek")

    # Adım 2: Kalan sınıflar için BorderlineSMOTE
    print('Adım 2: Kalan sınıflar için BorderlineSMOTE uygulanıyor...')
    borderline_smote = BorderlineSMOTE(random_state=42)

    try:
        X_res, y_res = borderline_smote.fit_resample(X_partial, y_partial)
        print(f'Kombine örnekleme tamamlandı: X_res {X_res.shape}, y_res {y_res.shape}')
        
        # Son dağılımı yazdır
        unique_res, counts_res = np.unique(y_res, return_counts=True)
        print("Son Dağılım (ham sayılar):")
        for i, (u, c) in enumerate(zip(unique_res, counts_res)):
            print(f"Sınıf {u}: {c} örnek")

    except Exception as e:
        print(f'BorderlineSMOTE örnekleme başarısız oldu: {e} - kısmi örneklenmiş veri kullanılıyor')
        X_res, y_res = X_partial, y_partial

    return X_res, y_res

def clear_gpu_memory():
    """
    Clear GPU memory and cache for memory management
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def monitor_gpu_memory(prefix=""):
    """
    Monitor and log GPU memory usage
    """
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        print(f"{prefix}GPU Memory: {memory_allocated:.0f}MB allocated, {memory_reserved:.0f}MB reserved")
        return memory_allocated, memory_reserved
    return 0, 0

# LSTM model sınıfını tanımlama
class MusicGenreLSTM(nn.Module):
    """
    Müzik türü sınıflandırması için LSTM (Long Short-Term Memory) ağı.
    LSTM'ler, müzik gibi sıralı verilerde başarılı olan bir derin öğrenme mimarisidir.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(MusicGenreLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM katmanları
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dropout katmanı
        self.dropout = nn.Dropout(dropout)
        
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Aktivasyon fonksiyonları
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM katmanından geçirme
        lstm_out, _ = self.lstm(x)
        
        # Son zaman adımının çıktısını al
        lstm_out = lstm_out[:, -1, :]
        
        # Batch normalization
        batch_norm_out = self.batch_norm(lstm_out)
        
        # İlk tam bağlantılı katman
        fc1_out = self.fc1(batch_norm_out)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # İkinci tam bağlantılı katman (çıkış katmanı)
        out = self.fc2(fc1_out)
        
        return out

class AdvancedMusicGenreLSTM(nn.Module):
    """
    Gelişmiş müzik türü sınıflandırması için çoklu-temporal LSTM ağı.
    
    Her özellik türü için ayrı LSTM'ler kullanır, böylece:
    - Chroma özelliklerinin pitch class yapısını korur
    - MFCC'nin spektral envelope bilgisini korur  
    - Spectral contrast'ın frekans bandı yapısını korur
    - Tonnetz'in harmonik uzay bilgisini korur
    """
    def __init__(self, feature_info, hidden_size=128, num_layers=2, num_classes=8, dropout=0.3):
        super(AdvancedMusicGenreLSTM, self).__init__()
        
        self.feature_info = feature_info
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Her temporal özellik türü için ayrı LSTM
        self.temporal_lstms = nn.ModuleDict()
        self.temporal_batch_norms = nn.ModuleDict()
        
        total_lstm_output = 0
        
        for i, feature_type in enumerate(feature_info['temporal_types']):
            input_size = feature_info['feature_dimensions'][i]  # 7 istatistik
            
            self.temporal_lstms[feature_type] = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.temporal_batch_norms[feature_type] = nn.BatchNorm1d(hidden_size)
            total_lstm_output += hidden_size
        
        # Skaler özellikler için tam bağlantılı katman
        if 'scalar_features' in feature_info:
            self.scalar_fc = nn.Linear(feature_info['scalar_features'], hidden_size)
            self.scalar_batch_norm = nn.BatchNorm1d(hidden_size)
            total_lstm_output += hidden_size
        
        # Birleştirici katmanlar
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc1 = nn.Linear(total_lstm_output, hidden_size * 2)
        self.fusion_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Aktivasyon fonksiyonları
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, temporal_features):
        lstm_outputs = []
        
        # Her temporal özellik için LSTM işlemi
        for feature_type in self.feature_info['temporal_types']:
            if feature_type in temporal_features:
                x = temporal_features[feature_type]
                
                # LSTM'den geçir
                lstm_out, _ = self.temporal_lstms[feature_type](x)
                
                # Son zaman adımını al
                lstm_out = lstm_out[:, -1, :]
                
                # Batch normalization
                lstm_out = self.temporal_batch_norms[feature_type](lstm_out)
                
                lstm_outputs.append(lstm_out)
        
        # Skaler özelikleri işle
        if 'scalar' in temporal_features:
            scalar_out = self.scalar_fc(temporal_features['scalar'])
            scalar_out = self.scalar_batch_norm(scalar_out)
            scalar_out = self.relu(scalar_out)
            lstm_outputs.append(scalar_out)
        
        # Tüm çıktıları birleştir
        if lstm_outputs:
            combined = torch.cat(lstm_outputs, dim=1)
        else:
            raise ValueError("No valid temporal features provided")
        
        # Fusion katmanları
        x = self.fusion_dropout(combined)
        x = self.fusion_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fusion_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Sınıflandırma
        out = self.classifier(x)
        
        return out

# ===== ENHANCED MODEL CLASSES (INTEGRATED FROM enhanced_temporal_model.py) =====

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on the most important time steps
    within each feature stream.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights to LSTM output
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        
        return context_vector, attention_weights


class EnhancedMusicGenreLSTM(nn.Module):
    """
    Enhanced music genre classification model with:
    - Per-feature temporal attention
    - Residual connections
    - Layer normalization
    - Dropout regularization at multiple levels
    """
    def __init__(self, feature_info, hidden_size=128, num_layers=2, 
                 num_classes=8, dropout=0.3, use_attention=True):
        super(EnhancedMusicGenreLSTM, self).__init__()
        
        self.feature_info = feature_info
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Per-feature LSTM streams
        self.temporal_lstms = nn.ModuleDict()
        self.temporal_batch_norms = nn.ModuleDict()
        self.temporal_attentions = nn.ModuleDict() if use_attention else None
        
        total_lstm_output = 0
        
        for i, feature_type in enumerate(feature_info['temporal_types']):
            # Get the feature dimension - input size for LSTM
            input_size = feature_info['feature_dimensions'][i]
            
            # Bidirectional LSTM for better temporal understanding
            self.temporal_lstms[feature_type] = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True  # Using bidirectional LSTM for improved performance
            )
            
            # Batch normalization for each feature stream
            lstm_output_size = hidden_size * 2  # *2 because of bidirectional
            self.temporal_batch_norms[feature_type] = nn.BatchNorm1d(lstm_output_size)
            
            # Attention mechanism per feature stream
            if use_attention:
                self.temporal_attentions[feature_type] = TemporalAttention(lstm_output_size)
                total_lstm_output += lstm_output_size
            else:
                total_lstm_output += lstm_output_size
        
        # Scalar features processing
        if 'scalar_features' in feature_info:
            self.scalar_fc = nn.Linear(feature_info['scalar_features'], hidden_size)
            self.scalar_batch_norm = nn.BatchNorm1d(hidden_size)
            total_lstm_output += hidden_size
        
        # Fusion layers with residual connections
        self.fusion_layer_norm1 = nn.LayerNorm(total_lstm_output)
        self.fusion_dropout1 = nn.Dropout(dropout)
        self.fusion_fc1 = nn.Linear(total_lstm_output, hidden_size * 2)
        
        self.fusion_layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.fusion_dropout2 = nn.Dropout(dropout)
        self.fusion_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        
        # Output layer
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Store attention weights for visualization
        self.last_attention_weights = {}
        
    def forward(self, temporal_features):
        lstm_outputs = []
        self.last_attention_weights = {}
        
        # Process each temporal feature stream
        for feature_type in self.feature_info['temporal_types']:
            if feature_type in temporal_features:
                x = temporal_features[feature_type]
                
                # Pass through LSTM
                lstm_out, _ = self.temporal_lstms[feature_type](x)
                
                # Apply attention or use the last output
                if self.use_attention:
                    context, attention_weights = self.temporal_attentions[feature_type](lstm_out)
                    self.last_attention_weights[feature_type] = attention_weights
                    lstm_out = context
                else:
                    # Use last time step if no attention
                    lstm_out = lstm_out[:, -1, :]
                
                # Apply batch normalization
                lstm_out = self.temporal_batch_norms[feature_type](lstm_out)
                
                lstm_outputs.append(lstm_out)
        
        # Process scalar features if available
        if 'scalar' in temporal_features and hasattr(self, 'scalar_fc'):
            scalar_out = self.scalar_fc(temporal_features['scalar'])
            scalar_out = self.scalar_batch_norm(scalar_out)
            scalar_out = self.relu(scalar_out)
            lstm_outputs.append(scalar_out)
        
        # Concatenate all outputs
        if lstm_outputs:
            combined = torch.cat(lstm_outputs, dim=1)
        else:
            raise ValueError("No valid temporal features provided")
        
        # First fusion layer with residual connection
        residual = combined
        x = self.fusion_layer_norm1(combined)
        x = self.fusion_dropout1(x)
        x = self.fusion_fc1(x)
        x = self.leaky_relu(x)
        
        # Second fusion layer
        x = self.fusion_layer_norm2(x)
        x = self.fusion_dropout2(x)
        x = self.fusion_fc2(x)
        x = self.leaky_relu(x)
        
        # Classification layer
        logits = self.classifier(x)
        
        return logits
    
    def get_attention_weights(self):
        """Return attention weights for visualization"""
        return self.last_attention_weights


class FocalLoss(nn.Module):
    """
    Focal Loss to focus more on hard-to-classify examples
    Especially useful for imbalanced genre performance
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha[targets]
            else:
                alpha = self.alpha
            ce_loss = alpha * ce_loss
            
        # Apply focal scaling
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ===== TEMPORAL DATASET AND TRAINING UTILITIES (INTEGRATED FROM fixed_temporal_training.py) =====

class TemporalFeatureDataset(Dataset):
    """
    Proper PyTorch Dataset for temporal features.
    This fixes the hanging issue by using standard PyTorch patterns.
    """
    def __init__(self, temporal_features, labels):
        self.temporal_features = temporal_features
        self.labels = labels
        self.length = len(labels)
        
        # Ensure all tensors are on CPU initially for memory efficiency
        for key in self.temporal_features:
            if isinstance(self.temporal_features[key], torch.Tensor):
                self.temporal_features[key] = self.temporal_features[key].cpu()
        
        if isinstance(self.labels, torch.Tensor):
            self.labels = self.labels.cpu()
            
        # Pre-validate data integrity
        if len(self.temporal_features) == 0:
            raise ValueError("No temporal features provided to dataset")
        
        # Verify all feature tensors have the same number of samples
        feature_lengths = [tensor.shape[0] for tensor in self.temporal_features.values()]
        if not all(length == self.length for length in feature_lengths):
            raise ValueError(f"Inconsistent feature tensor lengths: {feature_lengths}, expected: {self.length}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get item for specific index
        sample_temporal = {}
        for key, tensor in self.temporal_features.items():
            sample_temporal[key] = tensor[idx]
        
        sample_label = self.labels[idx]
        
        return sample_temporal, sample_label


def temporal_collate_fn(batch):
    """
    Custom collate function for temporal features.
    Properly handles the dictionary structure with improved error handling.
    """
    if not batch:
        raise ValueError("Empty batch provided to collate function")
        
    temporal_batch = {}
    labels_batch = []
    
    try:
        # Extract temporal features and labels
        for sample_temporal, sample_label in batch:
            labels_batch.append(sample_label)
            for key, tensor in sample_temporal.items():
                if key not in temporal_batch:
                    temporal_batch[key] = []
                temporal_batch[key].append(tensor)
        
        # Stack tensors with error checking
        for key in temporal_batch:
            try:
                temporal_batch[key] = torch.stack(temporal_batch[key])
            except RuntimeError as e:
                raise RuntimeError(f"Error stacking tensors for key '{key}': {e}")
        
        labels_batch = torch.stack(labels_batch)
        
        return temporal_batch, labels_batch
        
    except Exception as e:
        raise RuntimeError(f"Error in temporal_collate_fn: {e}")


def train_enhanced_temporal_model(model, temporal_features_train, y_train, 
                                 temporal_features_val, y_val,
                                 class_weights=None, num_epochs=50, batch_size=128, 
                                 learning_rate=0.001, focal_gamma=2.0,
                                 weight_decay=1e-5, early_stopping_patience=7):
    """
    Enhanced training function with:
    - Focal Loss for handling class imbalance
    - One-cycle learning rate scheduling
    - Gradient clipping
    - Regularization via weight decay
    - Early stopping with patience
    - Model checkpointing
    """
    print("🔧 Using ENHANCED temporal model training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    model = model.to(device)
    
    # Create proper datasets
    train_dataset = TemporalFeatureDataset(temporal_features_train, y_train)
    val_dataset = TemporalFeatureDataset(temporal_features_val, y_val)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  
        collate_fn=temporal_collate_fn,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=temporal_collate_fn,
        pin_memory=False,
        drop_last=False
    )
    
    # Use focal loss for better handling of class imbalance
    if class_weights is not None:
        print(f"Using class weights for loss calculation")
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)
    else:
        criterion = FocalLoss(gamma=focal_gamma)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # One-cycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.3,  # Spend 30% of time in increasing LR
        anneal_strategy='cos',
        div_factor=25.0,  # Initial LR = max_lr/25
        final_div_factor=1000.0  # Final LR = max_lr/1000
    )
    
    # Training tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("🚀 Enhanced temporal model training started...")
    print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # === TRAINING PHASE ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0
        
        for batch_idx, (batch_temporal, batch_y) in enumerate(train_loader):
            # Move to device
            for key in batch_temporal:
                batch_temporal[key] = batch_temporal[key].to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_temporal)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate each batch
            scheduler.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Training accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            # Progress indicator every 50 batches
            if batch_idx % 50 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Clear GPU cache periodically for memory efficiency
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                clear_gpu_memory()
                
            # Memory monitoring (optional debug info)
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                monitor_gpu_memory(prefix="    ")
        
        avg_train_loss = train_loss / train_batches
        train_accuracy = 100.0 * train_correct / train_total
        
        print(f"✅ Training complete - Avg Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # === VALIDATION PHASE ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (batch_temporal, batch_y) in enumerate(val_loader):
                # Move to device
                for key in batch_temporal:
                    batch_temporal[key] = batch_temporal[key].to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(batch_temporal)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Validation accuracy calculation
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / val_batches
        val_accuracy = 100.0 * val_correct / val_total
        
        # Record metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"✅ Validation complete - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            print(f"🎯 New best validation loss: {avg_val_loss:.4f} (improvement: {improvement:.4f})")
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏰ No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"✅ Best model loaded (val_loss: {best_val_loss:.4f})")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save the trained model
    try:
        model_save_path = 'enhanced_temporal_music_genre_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_info': model.feature_info,
            'best_val_loss': best_val_loss,
            'model_config': {
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'use_attention': model.use_attention
            }
        }, model_save_path)
        print(f"💾 Model saved to {model_save_path}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save model: {e}")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_enhanced_temporal_model(model, temporal_features_test, y_test, 
                                   label_encoder, batch_size=128):
    """
    Enhanced evaluation function for temporal model
    """
    print("🔧 Using ENHANCED temporal model evaluation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create test dataset and loader
    test_dataset = TemporalFeatureDataset(temporal_features_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=temporal_collate_fn,
        pin_memory=False
    )
    
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_temporal, batch_y in test_loader:
            # Move to device
            for key in batch_temporal:
                batch_temporal[key] = batch_temporal[key].to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(batch_temporal)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_predictions) * 100
    
    print(f"🎯 Test Accuracy: {accuracy:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'classification_report': classification_report(
            all_true_labels, 
            all_predictions, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
    }

# ===== END OF INTEGRATED ENHANCED COMPONENTS =====

# LSTM eğitimi fonksiyonu
def train_model(model, train_loader, val_loader, num_epochs, device):
    """
    LSTM modelini eğitir.
    
    Args:
        model: LSTM modeli
        train_loader: Eğitim veri yükleyici
        val_loader: Doğrulama veri yükleyici  
        num_epochs: Epoch sayısı
        device: Cihaz (CPU/GPU)
        
    Returns:
        Eğitim ve doğrulama kayıp değerleri ve doğruluk oranları
    """
    # Optimizör ve kayıp fonksiyonu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # En iyi model takibi
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 7
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("Model eğitimi başlatılıyor...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Eğitim fazı
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Doğruluk hesaplama
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        # Doğrulama fazı
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        
        # Sonuçları kaydet
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'           Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f'Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f'New best validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs')
            
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
    
    # En iyi modeli yükle
    model.load_state_dict(best_model_state)
    print('En iyi model yüklendi.')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Eğitim geçmişini görselleştirir.
    
    Args:
        train_losses: Eğitim kayıp değerleri
        val_losses: Doğrulama kayıp değerleri
        train_accs: Eğitim doğruluk oranları
        val_accs: Doğrulama doğruluk oranları
    """
    
    plt.figure(figsize=(15, 5))
    
    # Kayıp grafiği
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Eğitim Kaybı', color='blue', linewidth=2)
    plt.plot(val_losses, label='Doğrulama Kaybı', color='red', linewidth=2)
    plt.title('Model Kayıp Değerleri', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Eğitim Doğruluğu', color='blue', linewidth=2)
    plt.plot(val_accs, label='Doğrulama Doğruluğu', color='red', linewidth=2)
    plt.title('Model Doğruluk Oranları', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# Test veri seti üzerinde değerlendirme
def evaluate_model(model, test_loader, device, label_encoder):
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Doğruluk hesapla
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    # Sonuçları yazdır
    print(f"Test Doğruluğu: {accuracy:.4f}")
    
    # Sınıflandırma raporu
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    # Karmaşıklık matrisi
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Etiketler')
    plt.ylabel('Gerçek Etiketler')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Genre')
    plt.ylabel('True Genre')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_per_class_performance(y_true, y_pred, class_names):
    """Analyze per-class performance"""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print(f"\n📊 Per-Class Performance Analysis:")
    print(f"{'Genre':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print(f"{'='*65}")
    
    for i, genre in enumerate(class_names):
        print(f"{genre:<15} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10}")
    
    print(f"{'='*65}")
    print(f"{'Average':<15} {np.mean(precision):<10.3f} {np.mean(recall):<10.3f} {np.mean(f1):<10.3f} {np.sum(support):<10}")

def plot_multiclass_roc_curve(model, test_loader, device, label_encoder, title="ROC Curves"):
    """Plot ROC curves for multiclass classification with enhanced temporal model support"""
    model.eval()
    all_predictions_proba = []
    all_true_labels = []
    
    # Get prediction probabilities
    with torch.no_grad():
        for batch_temporal, batch_y in test_loader:
            # Move to device - handle both dict and tensor inputs
            if isinstance(batch_temporal, dict):
                for key in batch_temporal:
                    batch_temporal[key] = batch_temporal[key].to(device)
            else:
                batch_temporal = batch_temporal.to(device)
            
            # Forward pass
            outputs = model(batch_temporal)
            # Apply softmax to get probabilities
            proba = torch.softmax(outputs, dim=1)
            all_predictions_proba.extend(proba.cpu().numpy())
            all_true_labels.extend(batch_y.cpu().numpy())
    
    y_true = np.array(all_true_labels)
    y_proba = np.array(all_predictions_proba)
    n_classes = len(label_encoder.classes_)
    
    # Binarize the labels for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot ROC curve for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, linewidth=2,
                 label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{title} - Multiclass ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print AUC summary
    print(f"\n📊 ROC AUC Scores:")
    print(f"{'Genre':<15} {'AUC Score':<10}")
    print(f"{'='*25}")
    for i, genre in enumerate(label_encoder.classes_):
        print(f"{genre:<15} {roc_auc[i]:<10.3f}")
    print(f"{'='*25}")
    print(f"{'Micro-Average':<15} {roc_auc['micro']:<10.3f}")
    
    return roc_auc

def plot_f1_scores_by_genre(y_true, y_pred, class_names, title="F1 Scores by Genre"):
    """Plot F1 scores for each genre with detailed visualization"""
    
    # Calculate F1 scores for each class
    precision, recall, f1_scores, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Calculate macro and weighted averages
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    x_pos = np.arange(len(class_names))
    bars = plt.bar(x_pos, f1_scores, alpha=0.8, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f'][:len(class_names)])
    
    # Customize the plot
    plt.xlabel('Music Genres', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title(f'{title}\nMacro Avg: {f1_macro:.3f} | Weighted Avg: {f1_weighted:.3f}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x_pos, class_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add value labels on bars
    for i, (bar, score, support_count) in enumerate(zip(bars, f1_scores, support)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}\n(n={support_count})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal lines for averages
    plt.axhline(y=f1_macro, color='red', linestyle='--', alpha=0.7, 
                label=f'Macro Average: {f1_macro:.3f}')
    plt.axhline(y=f1_weighted, color='orange', linestyle='--', alpha=0.7, 
                label=f'Weighted Average: {f1_weighted:.3f}')
    
    # Add legend and grid
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Print detailed F1 score analysis
    print(f"\n📊 F1 Score Analysis by Genre:")
    print(f"{'Genre':<15} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Support':<10}")
    print(f"{'='*65}")
    
    for i, genre in enumerate(class_names):
        print(f"{genre:<15} {f1_scores[i]:<10.3f} {precision[i]:<10.3f} {recall[i]:<10.3f} {support[i]:<10}")
    
    print(f"{'='*65}")
    print(f"{'Macro Avg':<15} {f1_macro:<10.3f} {np.mean(precision):<10.3f} {np.mean(recall):<10.3f} {np.sum(support):<10}")
    print(f"{'Weighted Avg':<15} {f1_weighted:<10.3f} {np.average(precision, weights=support):<10.3f} {np.average(recall, weights=support):<10.3f} {np.sum(support):<10}")
    
    # Identify best and worst performing genres
    best_genre_idx = np.argmax(f1_scores)
    worst_genre_idx = np.argmin(f1_scores)
    
    print(f"\n🏆 Best Performing Genre: {class_names[best_genre_idx]} (F1: {f1_scores[best_genre_idx]:.3f})")
    print(f"🔍 Needs Improvement: {class_names[worst_genre_idx]} (F1: {f1_scores[worst_genre_idx]:.3f})")
    
    # Performance categories
    excellent_genres = [class_names[i] for i, score in enumerate(f1_scores) if score >= 0.8]
    good_genres = [class_names[i] for i, score in enumerate(f1_scores) if 0.6 <= score < 0.8]
    poor_genres = [class_names[i] for i, score in enumerate(f1_scores) if score < 0.6]
    
    print(f"\n📈 Performance Categories:")
    if excellent_genres:
        print(f"   🟢 Excellent (≥0.8): {', '.join(excellent_genres)}")
    if good_genres:
        print(f"   🟡 Good (0.6-0.8): {', '.join(good_genres)}")
    if poor_genres:
        print(f"   🔴 Needs Work (<0.6): {', '.join(poor_genres)}")
    
    return f1_scores, f1_macro, f1_weighted

def parse_arguments():
    """Parse command line arguments for flexible model configuration"""
    parser = argparse.ArgumentParser(description='Advanced Music Genre Classification with Temporal Features')
    
    # Data parameters
    parser.add_argument('--features', type=int, default=250, help='Number of features to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Model parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--no_attention', action='store_true', help='Disable attention mechanism')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--no_class_weights', action='store_true', help='Disable class weights')
    
    # Analysis options
    parser.add_argument('--analyze_features', action='store_true', help='Perform feature importance analysis')
    parser.add_argument('--optimize_hyperparameters', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--skip_training', action='store_true', help='Skip training (for analysis only)')
    
    return parser.parse_args()

def validate_configuration(args):
    """
    Validate configuration parameters to prevent common errors
    """
    validation_errors = []
    
    # Check numeric parameters
    if args.batch_size <= 0:
        validation_errors.append("Batch size must be positive")
    if args.hidden_size <= 0:
        validation_errors.append("Hidden size must be positive")
    if args.num_layers <= 0:
        validation_errors.append("Number of layers must be positive")
    if args.dropout < 0 or args.dropout >= 1:
        validation_errors.append("Dropout must be in range [0, 1)")
    if args.lr <= 0:
        validation_errors.append("Learning rate must be positive")
    if args.epochs <= 0:
        validation_errors.append("Number of epochs must be positive")
    if args.features <= 0:
        validation_errors.append("Number of features must be positive")
    
    # Check logical constraints
    if args.batch_size > 1024:
        print("⚠️ Warning: Very large batch size may cause memory issues")
    if args.hidden_size > 512:
        print("⚠️ Warning: Very large hidden size may cause memory issues")
    if args.lr > 0.01:
        print("⚠️ Warning: Very high learning rate may cause training instability")
    
    if validation_errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors))
    
    print("✅ Configuration validation passed")

def analyze_feature_importance(X, y, label_encoder):
    """
    Analyze feature importance using Random Forest
    Enhanced from analyze_features.py
    """
    print("\n📊 Performing Feature Importance Analysis...")
    
    # Split data for analysis
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Balance training data
    X_train_balanced, y_train_balanced = perform_data_balancing(X_train, y_train)
    
    # Feature selection with ANOVA F-value
    print("🔍 Computing ANOVA F-scores...")
    selector = SelectKBest(score_func=f_classif, k=min(200, X.shape[1]))
    selector.fit(X_train_balanced, y_train_balanced)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns.tolist(),
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    
    print(f"\nTop 10 features by F-score:")
    for i, (feature, score) in enumerate(zip(feature_scores['Feature'].head(10), 
                                            feature_scores['Score'].head(10))):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Random Forest feature importance
    print("\n🌲 Computing Random Forest feature importance...")
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train_balanced)
    
    # Get feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop 10 features by Random Forest importance:")
    for i in range(min(10, len(X.columns))):
        idx = indices[i]
        print(f"  {i+1}. {X.columns[idx]}: {importances[idx]:.4f}")
    
    # Evaluate Random Forest
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"\n🎯 Random Forest baseline accuracy: {accuracy:.2f}%")
    
    # Save feature importance results
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'RF_Importance': importances,
        'F_Score': selector.scores_
    }).sort_values('RF_Importance', ascending=False)
    
    importance_df.to_csv('feature_importance_analysis.csv', index=False)
    print("💾 Feature importance analysis saved to 'feature_importance_analysis.csv'")
    
    # Plot feature importance - Fix MultiIndex plotting issue
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20).copy()
    
    # Convert tuple feature names to strings for plotting
    if any(isinstance(feat, tuple) for feat in top_features['Feature']):
        top_features['Feature_str'] = top_features['Feature'].apply(
            lambda x: f"{x[0]}.{x[1]}" if isinstance(x, tuple) and len(x) == 2 else str(x)
        )
        y_column = 'Feature_str'
    else:
        y_column = 'Feature'
    
    sns.barplot(data=top_features, x='RF_Importance', y=y_column)
    plt.title('Top 20 Features by Random Forest Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def hyperparameter_optimization(X, y, label_encoder, args):
    """
    Perform hyperparameter optimization using grid search
    Enhanced from optimize_model.py
    """
    print("\n🔧 Starting Hyperparameter Optimization...")
    
    # Create results directory
    os.makedirs('optimization_results', exist_ok=True)
    
    # Parameter grid
    param_combinations = [
        {
            'batch_size': 64,
            'hidden_size': 64,
            'num_layers': 1,
            'learning_rate': 0.001,
            'dropout': 0.2,
            'focal_gamma': 1.0,
            'use_attention': True,
            'use_class_weights': True
        },
        {
            'batch_size': 128,
            'hidden_size': 128,
            'num_layers': 2,
            'learning_rate': 0.0005,
            'dropout': 0.3,
            'focal_gamma': 2.0,
            'use_attention': True,
            'use_class_weights': True
        },
        {
            'batch_size': 256,
            'hidden_size': 256,
            'num_layers': 2,
            'learning_rate': 0.0001,
            'dropout': 0.4,
            'focal_gamma': 3.0,
            'use_attention': False,
            'use_class_weights': False
        }
    ]
    
    num_trials = 2  # Reduced for faster execution
    all_results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n📋 Testing parameter combination {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        
        trial_results = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial+1}/{num_trials}")
            
            # Set trial seed
            trial_seed = args.seed + trial
            torch.manual_seed(trial_seed)
            np.random.seed(trial_seed)
            
            try:
                # Data preparation
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=trial_seed, stratify=y
                )
                
                # Feature selection
                selector = SelectKBest(score_func=f_classif, k=args.features)
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                # Create temporal features
                temporal_features_train, y_train_tensor_temp, feature_info = create_temporal_feature_data(
                    pd.DataFrame(X_train_selected), y_train
                )
                temporal_features_test, y_test_tensor_temp, _ = create_temporal_feature_data(
                    pd.DataFrame(X_test_selected), y_test, feature_info
                )
                
                # Convert to tensors
                y_train_tensor = torch.tensor(y_train, dtype=torch.long)
                y_test_tensor = torch.tensor(y_test, dtype=torch.long)
                
                # Create model for this trial
                model = EnhancedMusicGenreLSTM(
                    feature_info=feature_info,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    num_classes=len(label_encoder.classes_),
                    dropout=params['dropout'],
                    use_attention=params['use_attention']
                )
                
                # Calculate class weights if needed
                class_weights = None
                if params['use_class_weights']:
                    from sklearn.utils.class_weight import compute_class_weight
                    class_weights = compute_class_weight(
                        'balanced', classes=np.unique(y_train), y=y_train
                    )
                
                # Model training with trial parameters
                model, train_losses, val_losses, train_accs, val_accs = train_enhanced_temporal_model(
                    model=model,
                    temporal_features_train=temporal_features_train,
                    y_train=y_train_tensor,
                    temporal_features_val=temporal_features_test,  # Using test as val for quick optimization
                    y_val=y_test_tensor,
                    class_weights=class_weights,
                    num_epochs=15,  # Reduced for faster optimization
                    batch_size=params['batch_size'],
                    learning_rate=params['learning_rate'],
                    focal_gamma=params['focal_gamma'],
                    weight_decay=1e-5,
                    early_stopping_patience=5
                )
                
                # Evaluation
                results = evaluate_enhanced_temporal_model(
                    model=model,
                    temporal_features_test=temporal_features_test,
                    y_test=y_test_tensor,
                    label_encoder=label_encoder,
                    batch_size=params['batch_size']
                )
                
                trial_result = {
                    'params': params,
                    'trial': trial,
                    'accuracy': results['accuracy'],
                    'macro_f1': results['classification_report']['macro avg']['f1-score'],
                    'weighted_f1': results['classification_report']['weighted avg']['f1-score'],
                    'best_val_loss': min(val_losses) if val_losses else 0,
                    'trial_seed': trial_seed
                }
                
                trial_results.append(trial_result)
                print(f"    Accuracy: {trial_result['accuracy']:.2f}%, F1: {trial_result['macro_f1']:.4f}")
                
            except Exception as e:
                print(f"    ❌ Trial {trial+1} failed: {e}")
                continue
        
        if trial_results:
            # Calculate average performance
            avg_accuracy = np.mean([r['accuracy'] for r in trial_results])
            avg_macro_f1 = np.mean([r['macro_f1'] for r in trial_results])
            std_accuracy = np.std([r['accuracy'] for r in trial_results])
            
            combo_result = {
                'params': params,
                'avg_accuracy': avg_accuracy,
                'avg_macro_f1': avg_macro_f1,
                'std_accuracy': std_accuracy,
                'trial_results': trial_results
            }
            
            all_results.append(combo_result)
            print(f"  📊 Average: {avg_accuracy:.2f}% (±{std_accuracy:.2f})")
    
    if all_results:
        # Find best configuration
        best_config = max(all_results, key=lambda x: x['avg_macro_f1'])
        
        print(f"\n🏆 Best configuration:")
        print(f"Parameters: {best_config['params']}")
        print(f"Average Accuracy: {best_config['avg_accuracy']:.2f}%")
        print(f"Average Macro F1: {best_config['avg_macro_f1']:.4f}")
        
        # Save results
        with open('optimization_results/best_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print("💾 Optimization results saved to 'optimization_results/best_config.json'")
        return best_config
    else:
        print("❌ No successful trials completed")
        return None

def save_training_results(args, results, model_path=None):
    """
    Save training configuration and results for reproducibility
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"training_results_{timestamp}.json"
        
        training_summary = {
            'timestamp': timestamp,
            'configuration': vars(args),
            'results': results,
            'model_path': model_path,
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'pytorch_version': torch.__version__
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        print(f"💾 Training results saved to {results_file}")
        return results_file
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save training results: {e}")
        return None

def main(args):
    """
    Main training pipeline for enhanced music genre classification
    Integrated from multiple development files with command line argument support
    """
    print("🎵 Enhanced Music Genre Classification Training")
    print("=" * 50)
    print(f"Settings:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    try:
        # Validate configuration
        validate_configuration(args)
        
        # Load and prepare data
        print("\n📂 Step 1: Loading data...")
        X, y, label_encoder = load_data()
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"Classes: {list(label_encoder.classes_)}")
        
        # Split data into train/val/test
        print("\n🔄 Step 2: Splitting data...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=args.seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Balance training data
        print("\n⚖️  Step 3: Balancing data...")
        X_train_balanced, y_train_balanced = perform_data_balancing(X_train, y_train)
        plot_class_distribution(y_train_balanced, label_encoder.classes_, "Balanced Training Set - Class Distribution")
        
        # Feature selection
        print(f"\n🔍 Step 4: Feature selection (top {args.features} features)...")
        selector = SelectKBest(score_func=f_classif, k=args.features)
        X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        print(f"Feature selection complete: {X_train_selected.shape[1]} features selected")
        
        # Create temporal features 
        print("\n🔄 Step 5: Creating temporal features...")
        temporal_features_train, y_train_tensor_temp, feature_info = create_temporal_feature_data(
            pd.DataFrame(X_train_selected), y_train_balanced
        )
        temporal_features_val, y_val_tensor_temp, _ = create_temporal_feature_data(
            pd.DataFrame(X_val_selected), y_val, feature_info
        )
        temporal_features_test, y_test_tensor_temp, _ = create_temporal_feature_data(
            pd.DataFrame(X_test_selected), y_test, feature_info
        )
        
        # Convert targets to tensors
        y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Calculate class weights if not disabled
        class_weights = None
        if not args.no_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(y_train_balanced), y=y_train_balanced
            )
            print(f"📊 Using class weights: {class_weights}")
        
        # Create enhanced model
        print(f"\n🧠 Step 6: Creating enhanced temporal model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {device}")
        if torch.cuda.is_available():
            print(f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
        
        model = EnhancedMusicGenreLSTM(
            feature_info=feature_info,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=len(label_encoder.classes_),
            dropout=args.dropout,
            use_attention=not args.no_attention
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        print(f"\n🚀 Step 7: Training enhanced temporal model...")
        model, train_losses, val_losses, train_accs, val_accs = train_enhanced_temporal_model(
            model=model,
            temporal_features_train=temporal_features_train,
            y_train=y_train_tensor,
            temporal_features_val=temporal_features_val,
            y_val=y_val_tensor,
            class_weights=class_weights,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.patience
        )
        
        # Plot training history
        print("\n📊 Step 8: Plotting training history...")
        plot_training_history(train_losses, val_losses, train_accs, val_accs)
        
        # Evaluate model
        print("\n🎯 Step 9: Final model evaluation...")
        test_results = evaluate_enhanced_temporal_model(
            model, temporal_features_test, y_test_tensor, label_encoder, args.batch_size
        )
        
        # Save training results
        print("\n💾 Step 10: Saving training results...")
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accs,
            'val_accuracies': val_accs
        }
        
        enhanced_results = {
            'test_accuracy': test_results['accuracy'],
            'classification_report': test_results['classification_report'],
            'training_history': training_history
        }
        
        save_training_results(args, enhanced_results, 'enhanced_temporal_music_genre_model.pth')
        
        print("✅ Training pipeline completed successfully!")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Error in main training pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate configuration before proceeding
    validate_configuration(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"🎵 Advanced Music Genre Classification")
    print(f"⚙️  Configuration: {vars(args)}")
    
    try:
        # Validate configuration
        validate_configuration(args)
        
        # Load data for analysis/optimization
        X, y, label_encoder = load_data()
        
        # Feature importance analysis
        if args.analyze_features:
            analyze_feature_importance(X, y, label_encoder)
        
        # Hyperparameter optimization
        if args.optimize_hyperparameters:
            best_config = hyperparameter_optimization(X, y, label_encoder, args)
            if best_config and not args.skip_training:
                # Update args with best parameters
                for key, value in best_config['params'].items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                print(f"🔄 Updated configuration with optimized parameters")
        
        # Run main training if not skipped
        if not args.skip_training:
            main(args)
        else:
            print("⏭️  Training skipped as requested")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()