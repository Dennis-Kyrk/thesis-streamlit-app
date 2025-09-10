import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
import warnings
import os
import time
import pickle
import hashlib
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Visual Data Flow",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS styling - Consolidated similar styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(45deg, #2e7d32, #66bb6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.4rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        line-height: 1.4;
    }
    
    /* Base card styles */
    .base-card {
        padding: 1rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem;
        text-align: center;
    }
    
    .data-flow-container {
        background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.3);
        text-align: center;
    }
    
    .data-points {
        background: linear-gradient(135deg, #0277bd 0%, #0288d1 100%);
        box-shadow: 0 4px 15px rgba(2, 119, 189, 0.3);
    }
    
    .xgboost-model {
        background: linear-gradient(135deg, #5e35b1 0%, #7e57c2 100%);
        box-shadow: 0 4px 15px rgba(94, 53, 177, 0.3);
    }
    
    .classification-result {
        background: linear-gradient(135deg, #616161 0%, #757575 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(97, 97, 97, 0.3);
    }
    
    .minority-class {
        background: linear-gradient(135deg, #c2185b 0%, #e91e63 100%);
        box-shadow: 0 4px 15px rgba(194, 24, 91, 0.3);
    }
    
    .majority-class {
        background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%);
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }
    
    .manual-inspection {
        background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%);
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
    }
    
    .flow-arrow {
        font-size: 2rem;
        color: #2e7d32;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .flow-arrow-horizontal {
        font-size: 2rem;
        color: #2e7d32;
        text-align: center;
        margin: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .flow-arrow-down {
        font-size: 2rem;
        color: #2e7d32;
        text-align: center;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .flow-arrow-diagonal {
        font-size: 1.5rem;
        color: #2e7d32;
        text-align: center;
        margin: 0.3rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .flow-container {
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .flow-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .flow-arrow-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0.5rem;
    }
    
         /* Classification result cards - common styles */
     .classification-card {
         padding: 0.8rem;
         border-radius: 10px;
         color: white;
         margin: 0.3rem;
         text-align: center;
     }
     
     .classification-card h5 {
         margin: 0;
         font-size: 1.2rem;
         font-weight: 600;
     }
     
     .classification-card h3 {
         margin: 0.3rem 0;
         font-size: 1.8rem;
         font-weight: 700;
     }
     
     .classification-card p {
         margin: 0;
         font-size: 1.1rem;
     }
     
     .classification-card small {
         font-size: 0.9rem;
         opacity: 0.9;
     }
    
    .correct-classification {
        background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%);
        box_shadow: 0 3px 12px rgba(46, 125, 50, 0.3);
    }
    
    .wrong-classification {
        background: linear-gradient(135deg, #f44336 0%, #ff5722 100%);
        box-shadow: 0 3px 12px rgba(244, 67, 54, 0.3);
    }
    
    .manual-breakdown {
        background: linear-gradient(135deg, #795548 0%, #8d6e63 100%);
        box-shadow: 0 3px 12px rgba(121, 85, 72, 0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #2e7d32;
        margin: 0.3rem 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #2e7d32;
    }
    
    .metric-card p {
        margin: 0.3rem 0 0 0;
        color: #666;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .navigation-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .sample-counter {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Tooltip Styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        line-height: 1.4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 1px solid #555;
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Tooltip positioning variants */
    .tooltip-right .tooltiptext {
        top: -5px;
        left: 110%;
        bottom: auto;
        margin-left: 0;
    }
    
    .tooltip-right .tooltiptext::after {
        top: 20px;
        left: -5px;
        border-color: transparent #333 transparent transparent;
    }
    
    .tooltip-left .tooltiptext {
        top: -5px;
        right: 110%;
        left: auto;
        bottom: auto;
        margin-left: 0;
    }
    
    .tooltip-left .tooltiptext::after {
        top: 20px;
        right: -5px;
        left: auto;
        border-color: transparent transparent transparent #333;
    }
    
    .tooltip-bottom .tooltiptext {
        top: 125%;
        bottom: auto;
    }
    
    .tooltip-bottom .tooltiptext::after {
        top: -5px;
        border-color: transparent transparent #333 transparent;
    }
    
    /* Sidebar tooltip positioning - force right positioning */
    .sidebar-tooltip {
        font-weight: bold;
    }
    
    .sidebar-tooltip .tooltiptext {
        top: -5px;
        left: 110%;
        right: auto;
        bottom: auto;
        margin-left: 0;
        width: 250px;
    }
    
    .sidebar-tooltip .tooltiptext::after {
        top: 20px;
        left: -5px;
        right: auto;
        border-color: transparent #333 transparent transparent;
    }
    
    /* Info icon styling */
    .info-icon {
        color: #2196f3;
        font-size: 0.9em;
        margin-left: 4px;
        vertical-align: middle;
        cursor: help;
    }
    
    .info-icon:hover {
        color: #1976d2;
    }
    
    /* Special tooltip for large content */
    .tooltip-large .tooltiptext {
        width: 400px;
        margin-left: -200px;
    }
    
    .tooltip-wide .tooltiptext {
        width: 500px;
        margin-left: -250px;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions to reduce redundancy
def create_div_card(class_name, content, extra_class=""):
    """Helper to create consistent div cards"""
    return f'<div class="{class_name} {extra_class}">{content}</div>'

def create_classification_card(title, count, percentage, subtitle, card_type="correct"):
    """Helper to create classification result cards"""
    class_name = f"{card_type}-classification classification-card"
    return f'''
    <div class="{class_name}">
        <h5>{title}</h5>
        <h3>{count:,}</h3>
        <p>{percentage:.1f}%</p>
    </div>
    '''

def create_metric_card(value, label):
    """Helper to create metric cards"""
    return f'''
    <div class="metric-card">
        <h3>{value:.3f}</h3>
        <p>{label}</p>
    </div>
    '''

def create_tooltip(text, tooltip_text, position="top", size="normal"):
    """Helper to create tooltips with info icons"""
    size_class = f"tooltip-{size}" if size != "normal" else ""
    return f'<span class="tooltip {position} {size_class}">{text}<span class="info-icon">ℹ️</span><span class="tooltiptext">{tooltip_text}</span></span>'

def create_sidebar_tooltip(text, tooltip_text, size="normal"):
    """Helper to create tooltips specifically for sidebar elements"""
    size_class = f"tooltip-{size}" if size != "normal" else ""
    return f'<span class="tooltip sidebar-tooltip {size_class}">{text}<span class="info-icon">ℹ️</span><span class="tooltiptext">{tooltip_text}</span></span>'

def display_tooltip(text, tooltip_text, position="top", size="normal"):
    """Display a tooltip directly using st.markdown"""
    tooltip_html = create_tooltip(text, tooltip_text, position, size)
    st.markdown(tooltip_html, unsafe_allow_html=True)

def create_navigation_controls(session_key, current_index, total_items, item_name="sample"):
    """Create reusable navigation controls"""
    st.markdown('<div class="navigation-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", key=f"prev_{session_key}", 
                    disabled=(current_index == 0),
                    use_container_width=True):
            st.session_state[f"{session_key}_index"] -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"<div class='sample-counter'>Sample {current_index + 1} of {total_items}</div>", 
                   unsafe_allow_html=True)
    
    with col3:
        if st.button("Next ➡️", key=f"next_{session_key}",
                    disabled=(current_index >= total_items - 1),
                    use_container_width=True):
            st.session_state[f"{session_key}_index"] += 1
            st.rerun()
    
    # Quick navigation slider
    col1, col2 = st.columns([3, 1])
    with col1:
        new_index = st.slider(f"Jump to {item_name}:", 
                             min_value=1, 
                             max_value=total_items,
                             value=current_index + 1,
                             key=f"{session_key}_slider",
                             label_visibility="collapsed")
    
    if new_index - 1 != current_index:
        st.session_state[f"{session_key}_index"] = new_index - 1
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    return col2

def get_dataset_hash():
    """Generate a hash of the dataset to ensure model compatibility"""
    try:
        if os.path.exists('wave_dataset.parquet'):
            # Get file modification time and size for hash
            stat = os.stat('wave_dataset.parquet')
            hash_input = f"{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    except Exception:
        pass
    return "unknown"

def save_pretrained_model(model_data, dataset_hash, trained_models):
    """Save all 5 trained XGBoost models from cross-validation and essential data"""
    try:
        # Save all 5 models from cross-validation
        model_files = []
        total_model_size = 0
        
        for fold, model in enumerate(trained_models):
            model_file = f'xgboost_model_fold{fold}_{dataset_hash}.json'
            model.save_model(model_file)
            model_files.append(model_file)
            total_model_size += os.path.getsize(model_file) / (1024 * 1024)  # MB
        
        # Save metadata separately
        metadata_file = f'model_metadata_{dataset_hash}.pkl'
        metadata = {
            # Core model data
            'avg_metrics': model_data['avg_metrics'],
            'feature_names': model_data['feature_names'],
            'imputer': model_data['imputer'],
            
            # Essential metadata
            'training_timestamp': model_data['training_timestamp'],
            'corrections_applied': model_data['corrections_applied'],
            'discards_applied': model_data['discards_applied'],
            'dataset_hash': dataset_hash,
            'model_version': '4.0',  # New version for 5-fold models
            
            # Status mapping (small)
            'group_to_status': model_data['group_to_status'],
            
            # Full cross-validation results (not just samples)
            'all_y_test': model_data['all_y_test'].tolist(),
            'all_y_pred_proba': model_data['all_y_pred_proba'].tolist(),
            'all_groups_test': model_data['all_groups_test'].tolist(),
            'all_test_indices': model_data['all_test_indices'].tolist(),
            'all_fold_metrics': model_data['all_fold_metrics'],
            'all_misclassified_data': model_data['all_misclassified_data']
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Get total file size for feedback
        metadata_size = os.path.getsize(metadata_file) / (1024 * 1024)  # MB
        total_size = total_model_size + metadata_size
        
        return True, total_size, total_model_size, metadata_size
    except Exception as e:
        st.warning(f"Could not save pretrained model: {str(e)}")
        return False, 0, 0, 0

def load_pretrained_model(dataset_hash):
    """Load all 5 trained XGBoost models and metadata from disk"""
    try:
        # Check if all 5 model files exist
        model_files = []
        total_model_size = 0
        for fold in range(5):
            model_file = f'xgboost_model_fold{fold}_{dataset_hash}.json'
            if not os.path.exists(model_file):
                return None
            model_files.append(model_file)
            total_model_size += os.path.getsize(model_file) / (1024 * 1024)  # MB
        
        metadata_file = f'model_metadata_{dataset_hash}.pkl'
        if not os.path.exists(metadata_file):
            return None
        
        # Load all 5 models
        models = []
        for model_file in model_files:
            model = xgb.XGBClassifier()
            model.load_model(model_file)
            models.append(model)
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Validate the loaded data (but be more lenient for GitHub deployment)
        if metadata.get('dataset_hash') != dataset_hash:
            # Hash mismatch is common in cloud deployments, continue anyway
            pass
        
        if metadata.get('model_version') != '4.0':
            st.warning("Pretrained model version mismatch, retraining...")
            return None
        
        # Get file sizes for feedback
        metadata_size = os.path.getsize(metadata_file) / (1024 * 1024)  # MB
        total_size = total_model_size + metadata_size
        
        return models, metadata, total_size, total_model_size, metadata_size
    except Exception as e:
        st.warning(f"Could not load pretrained model: {str(e)}")
        return None, None, 0, 0, 0

def train_single_model_for_pretrained(X_train, y_train, X_test, y_test):
    """Train a single XGBoost model for pretrained model purposes"""
    # Apply SMOTE
    smote_tomek = SMOTETomek(random_state=42)
    X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    
    # Train model with optimized parameters
    model = xgb.XGBClassifier(
        tree_method="hist",     # fast CPU builder
        max_depth=4,            # shallow trees -> smaller model
        max_bin=128,            # slightly coarser hist -> faster/smaller
        n_estimators=200,       # let early stopping pick best round
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,              # encourages pruning (smaller trees)
        reg_alpha=1.0,          # L1 -> fewer splits (smaller model)
        reg_lambda=1.0,         # L2 regularization
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
    )
    
    params = {'eval_set': [(X_train, y_train), (X_test, y_test)], 'verbose': False}
    try:
        model.fit(X_train, y_train, **params, early_stopping_rounds=20)
    except TypeError:
        model.fit(X_train, y_train, **params)
    
    return model

@st.cache_data
def load_real_data():
    """Load the actual parquet file and process sequences"""
    try:
        if os.path.exists('wave_dataset.parquet'):
            df = pd.read_parquet('wave_dataset.parquet')
            df['col1'] = df['col1'].fillna(0)
            df = df[df['col1'] != 0] 

            sequences = []
            labels = []
            metadata_features = []
            group_ids_list = [] 
            exclude_cols = ['col1', 'col2', 'label_1', 'label_2', 'label_3', 'group', 'label', 'status']
            metadata_columns = [col for col in df.columns if col not in exclude_cols] 

            statuses = []

            for seq_id, group in df.groupby('group'):
                group_ids_list.append(seq_id)
                sequences.append(group[['col1', 'col2']].values)
                labels.append(1 if group["label"].iloc[0] else 0)
                statuses.append(group["status"].iloc[0])   # 👈 add this
                metadata_features.append(
                    [int(group[col].iloc[0]) for col in metadata_columns if col != "label"]
                )      

            groups = np.array(group_ids_list)
            y = np.array(labels)
            metadata_features = np.array(metadata_features)
            statuses = np.array(statuses)   # 👈 convert to numpy

            return {
                'sequences': sequences,
                'labels': y,
                'groups': groups,
                'metadata_features': metadata_features,
                'metadata_columns': metadata_columns,
                'raw_df': df,
                'statuses': statuses   # 👈 add to return dict
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Import moved to function level to avoid module-level import warnings

def extract_advanced_features(sequences, groups):
    """Extract advanced features for quality control analysis"""
    from scipy import signal
    from scipy.stats import kurtosis, skew
    from scipy.fft import fft, fftfreq
    from scipy.ndimage import uniform_filter1d
    
    features_list = []
    
    for i, (group_id, sequence) in enumerate(zip(groups, sequences)):
        y = sequence[:, 1]  # col2 values
        # x = sequence[:, 0]  # col1 time values (not used in current implementation)
        
        if len(y) < 10:  # Skip very short sequences
            features_list.append([0] * 15)  # 15 features we'll extract
            continue
            
        features = []
        
        # 1. Outlier residuals vs smooth baseline
        try:
            # Smooth baseline using moving average
            window_size = max(3, len(y) // 10)
            smooth_baseline = uniform_filter1d(y, size=window_size, mode='nearest')
            residuals = y - smooth_baseline
            
            # Residual features
            max_abs_residual = np.max(np.abs(residuals))
            sorted_abs_residuals = np.sort(np.abs(residuals))[::-1]
            mean_top3_abs_residual = np.mean(sorted_abs_residuals[:3]) if len(sorted_abs_residuals) >= 3 else max_abs_residual
            
            # MAD-based outlier count
            mad = np.median(np.abs(residuals - np.median(residuals)))
            count_residual_gt_3MAD = np.sum(np.abs(residuals) > 3 * mad)
            
            features.extend([max_abs_residual, mean_top3_abs_residual, count_residual_gt_3MAD])
        except Exception:
            features.extend([0, 0, 0])
        
        # 2. Peak consistency & asymmetry
        try:
            # Find peaks in original and inverted signal
            peaks_pos, props_pos = signal.find_peaks(y, height=0)
            peaks_neg, props_neg = signal.find_peaks(-y, height=0)
            
            # Peak height statistics
            if len(peaks_pos) > 0:
                pos_heights = y[peaks_pos]
                std_pos_peak_height = np.std(pos_heights) if len(pos_heights) > 1 else 0
            else:
                std_pos_peak_height = 0
                
            if len(peaks_neg) > 0:
                neg_heights = -y[peaks_neg]
                std_neg_peak_depth = np.std(neg_heights) if len(neg_heights) > 1 else 0
            else:
                std_neg_peak_depth = 0
            
            # Prominence features
            all_peaks, all_props = signal.find_peaks(y, prominence=0)
            if len(all_peaks) > 0 and 'prominences' in all_props:
                prominences = all_props['prominences']
                max_prominence = np.max(prominences)
                median_prominence = np.median(prominences)
                prominence_ratio = max_prominence / median_prominence if median_prominence > 0 else 0
            else:
                max_prominence = 0
                median_prominence = 0
                prominence_ratio = 0
            
            # Peak asymmetry
            max_abs = np.max(np.abs(y))
            min_abs = np.min(np.abs(y))
            peak_asym = max_abs - min_abs
            peak_ratio = max_abs / min_abs if min_abs > 0 else 0
            
            features.extend([std_pos_peak_height, std_neg_peak_depth, max_prominence, 
                           median_prominence, prominence_ratio, peak_asym, peak_ratio])
        except Exception:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        # 3. Spectral leakage / high-frequency ratio
        try:
            # FFT analysis
            fft_vals = fft(y)
            freqs = fftfreq(len(y), d=1.0)  # Assuming unit time step
            
            # Calculate power spectral density
            psd = np.abs(fft_vals) ** 2
            
            # Estimate fundamental frequency (20 waves per series)
            f0 = 20 / len(y)
            f_cutoff = 3 * f0
            
            # High frequency ratio
            hf_mask = np.abs(freqs) > f_cutoff
            hf_energy = np.sum(psd[hf_mask])
            total_energy = np.sum(psd)
            hf_ratio = hf_energy / total_energy if total_energy > 0 else 0
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
            psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0
            
            features.extend([hf_ratio, spectral_entropy])
        except Exception:
            features.extend([0, 0])
        
        # 4. Periodicity robustness
        try:
            # Autocorrelation analysis
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            
            # Nominal period (20 waves per series)
            P = max(1, round(len(y) / 20))
            if P < len(autocorr):
                r_lagP = autocorr[P]
                r_lag1 = autocorr[1] if len(autocorr) > 1 else 1
                r_lagP_ratio = r_lagP / r_lag1 if r_lag1 != 0 else 0
            else:
                r_lagP_ratio = 0
            
            features.append(r_lagP_ratio)
        except Exception:
            features.append(0)
        
        # 5. Shape heavy-tail
        try:
            excess_kurt = kurtosis(y)
            skewness = skew(y)
            features.extend([excess_kurt, skewness])
        except Exception:
            features.extend([0, 0])
        
        features_list.append(features)
    
    return np.array(features_list)

@st.cache_data
def extract_features_from_sequences(data_dict):
    """Extract features from wave_dataset.parquet using tsfresh + advanced features"""
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    
    raw_df = data_dict["raw_df"].copy()
    
    # tsfresh needs: [id, time, value]
    ts_df = raw_df.rename(columns={
        "group": "id",
        "col1": "time",
        "col2": "value"
    })[["id", "time", "value"]]
    
    # extract tsfresh features
    features_df = extract_features(
        ts_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0
    )
    
    # drop NaNs and infs
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # drop constant columns
    nunique = features_df.nunique()
    features_df = features_df.drop(columns=nunique[nunique <= 1].index)
    
    # Extract advanced features
    sequences = data_dict["sequences"]
    groups = data_dict["groups"]
    advanced_features = extract_advanced_features(sequences, groups)
    
    # Create feature names for advanced features
    advanced_feature_names = [
        'max_abs_residual', 'mean_top3_abs_residual', 'count_residual_gt_3MAD',
        'std_pos_peak_height', 'std_neg_peak_depth', 'max_prominence', 
        'median_prominence', 'prominence_ratio', 'peak_asym', 'peak_ratio',
        'hf_ratio', 'spectral_entropy', 'r_lagP_ratio', 'excess_kurtosis', 'skewness'
    ]
    
    # Combine tsfresh and advanced features
    combined_features = np.hstack([features_df.values, advanced_features])
    combined_feature_names = list(features_df.columns) + advanced_feature_names
    
    # Create new dataframe with combined features
    combined_df = pd.DataFrame(combined_features, columns=combined_feature_names)
    
    # cache
    combined_df.to_parquet("combined_features_extracted.parquet")
    
    return combined_df.values, list(combined_df.columns)

@st.cache_data
def load_or_extract_features():
    if os.path.exists("combined_features_extracted.parquet"):
        features_df = pd.read_parquet("combined_features_extracted.parquet")
        return features_df.values, list(features_df.columns)
    
    data_dict = load_real_data()
    if data_dict is None:
        return None, None
    
    return extract_features_from_sequences(data_dict)


@st.cache_resource
def train_model_with_cross_validation(_corrections_count=0, _discards_count=0):
    """Train AI model using 5-fold stratified cross-validation"""
    
    # Get dataset hash for pretrained model compatibility
    dataset_hash = get_dataset_hash()
    
    # Try to load pretrained model if no corrections or discards are applied
    if _corrections_count == 0 and _discards_count == 0:
        # Try to find any pretrained model (ignore hash for now)
        import glob
        all_model_files = glob.glob("xgboost_model_fold*_*.json")
        all_metadata_files = glob.glob("model_metadata_*.pkl")
        
        if all_model_files and all_metadata_files:
            # Extract hash from the first model file
            first_model_file = all_model_files[0]
            file_hash = first_model_file.split('_')[-1].replace('.json', '')
            
            # Try to load with the hash from the file
            pretrained_result = load_pretrained_model(file_hash)
        else:
            pretrained_result = load_pretrained_model(dataset_hash)
        
        if pretrained_result is not None:
            models, metadata, total_size, model_size, metadata_size = pretrained_result
            
            # Load data for reconstruction
            data_dict = load_real_data()
            if data_dict is None:
                return None
            
            X_combined, combined_feature_names = load_or_extract_features()
            if X_combined is None:
                return None
            
            # Use the saved cross-validation results directly (no need to recalculate)
            model_data = {
                'model': models[0],  # Use first model as the main model
                'X_combined': X_combined,
                'feature_names': metadata['feature_names'],
                'all_y_test': np.array(metadata['all_y_test']),  # Loaded from saved results
                'all_y_pred_proba': np.array(metadata['all_y_pred_proba']),  # Loaded from saved results
                'all_groups_test': np.array(metadata['all_groups_test']),  # Loaded from saved results
                'all_test_indices': np.array(metadata['all_test_indices']),
                'all_fold_metrics': metadata['all_fold_metrics'],  # Loaded from saved results
                'avg_metrics': metadata['avg_metrics'],  # Loaded from saved results
                'all_misclassified_data': metadata['all_misclassified_data'],  # Loaded from saved results
                'imputer': metadata['imputer'],
                'data_info': data_dict,
                'statuses': data_dict['statuses'],
                'group_to_status': metadata['group_to_status'],
                'training_timestamp': metadata['training_timestamp'],
                'corrections_applied': metadata['corrections_applied'],
                'discards_applied': metadata['discards_applied']
            }
            
            return model_data
    
    # Show training status for full retraining
    if _corrections_count > 0 or _discards_count > 0:
        st.markdown("### 🔄 Retraining Model with Data Changes")
        st.info(f"Corrections: {_corrections_count} | Discarded: {_discards_count}")
    else:
        st.markdown("### 🚀 Initial Model Training")
        st.info("Training AI with 5-fold cross-validation...")
    
    data_dict = load_real_data()
    if data_dict is None:
        return None
    
    X_combined, combined_feature_names = load_or_extract_features()
    if X_combined is None:
        return None
    
    y = data_dict['labels'].copy()
    groups = data_dict['groups']
    
    # Create a mask for samples to keep (not discarded)
    keep_mask = np.ones(len(y), dtype=bool)
    
    # Apply discarded data if any exists
    if 'discarded_data' in st.session_state and st.session_state.discarded_data and _discards_count > 0:
        st.write(f"🗑️ Removing {len(st.session_state.discarded_data)} discarded samples...")
        
        discarded_count = 0
        for group_id_str in st.session_state.discarded_data:
            try:
                # Try as is (if it's already the right type)
                if str(group_id_str) in [str(g) for g in groups]:
                    idx = [i for i, g in enumerate(groups) if str(g) == str(group_id_str)]
                    if idx:
                        keep_mask[idx[0]] = False
                        discarded_count += 1
                        continue
                
                # If direct string match didn't work, try numeric conversion
                if groups.dtype == np.int64:
                    group_id_numeric = np.int64(group_id_str)
                elif groups.dtype == np.int32:
                    group_id_numeric = np.int32(group_id_str)
                elif groups.dtype == np.float64:
                    group_id_numeric = float(group_id_str)
                else:
                    group_id_numeric = int(group_id_str)
                    
                # Find the index of this group
                group_idx = np.where(groups == group_id_numeric)[0]
                if len(group_idx) > 0:
                    keep_mask[group_idx[0]] = False
                    discarded_count += 1
                    
            except Exception as e:
                st.warning(f"Could not discard group {group_id_str}: {str(e)}")
                
        st.write(f"✅ Successfully removed {discarded_count} samples from training")
        
        # Apply the mask to filter out discarded samples
        X_combined = X_combined[keep_mask]
        y = y[keep_mask]
        groups = groups[keep_mask]
    
    # Apply relabeled data if any exists (only to non-discarded samples)
    if 'relabeled_data' in st.session_state and st.session_state.relabeled_data and _corrections_count > 0:
        st.write(f"📝 Applying {len(st.session_state.relabeled_data)} label corrections...")
        
        applied_corrections = 0
        labels_changed = {'0_to_1': 0, '1_to_0': 0}
        failed_corrections = []
        
        for group_id_str, new_label in st.session_state.relabeled_data.items():
            # Skip if this sample was discarded
            if group_id_str in st.session_state.get('discarded_data', set()):
                continue
                
            try:
                # Try as is (if it's already the right type)
                if str(group_id_str) in [str(g) for g in groups]:
                    idx = [i for i, g in enumerate(groups) if str(g) == str(group_id_str)]
                    if idx:
                        old_label = y[idx[0]]
                        y[idx[0]] = new_label
                        applied_corrections += 1
                        
                        # Track label changes
                        if old_label == 0 and new_label == 1:
                            labels_changed['0_to_1'] += 1
                        elif old_label == 1 and new_label == 0:
                            labels_changed['1_to_0'] += 1
                        continue
                
                # If direct string match didn't work, try numeric conversion
                if groups.dtype == np.int64:
                    group_id_numeric = np.int64(group_id_str)
                elif groups.dtype == np.int32:
                    group_id_numeric = np.int32(group_id_str)
                elif groups.dtype == np.float64:
                    group_id_numeric = float(group_id_str)
                else:
                    group_id_numeric = int(group_id_str)
                    
                # Find the index of this group
                group_idx = np.where(groups == group_id_numeric)[0]
                if len(group_idx) > 0:
                    old_label = y[group_idx[0]]
                    y[group_idx[0]] = new_label
                    applied_corrections += 1
                    
                    # Track label changes
                    if old_label == 0 and new_label == 1:
                        labels_changed['0_to_1'] += 1
                    elif old_label == 1 and new_label == 0:
                        labels_changed['1_to_0'] += 1
                else:
                    failed_corrections.append(f"Group {group_id_str} not found")
                    
            except Exception as e:
                failed_corrections.append(f"Group {group_id_str}: {str(e)}")
                
        st.write(f"✅ Successfully applied {applied_corrections} corrections:")
        st.write(f"   • {labels_changed['0_to_1']} changed from Not Normal (0) to Normal (1)")
        st.write(f"   • {labels_changed['1_to_0']} changed from Normal (1) to Not Normal (0)")
        
        if failed_corrections and st.checkbox("Show failed corrections"):
            for failure in failed_corrections:
                st.write(f"❌ {failure}")
                
        st.divider()
    
    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_combined = imputer.fit_transform(X_combined)
    
    # Define cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Prepare lists to store results from each fold
    all_fold_metrics = []
    all_y_test = []
    all_y_pred_proba = []
    all_groups_test = []
    all_test_indices = []
    all_misclassified_data = []
    
    # Progress container
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Store models for saving later
    trained_models = []
    
    # Train model for each fold
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_combined, y)):
        status_text.text(f"📊 Training fold {fold+1}/5...")
        
        # Split data for this fold
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test_fold = y[train_idx], y[test_idx]
        groups_test_fold = groups[test_idx]
        
        # Train model for this fold
        model = xgb.XGBClassifier(
        tree_method="hist",     # fast CPU builder
        n_estimators=1000,      # let early stopping pick best round
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,              # encourages pruning (smaller trees)
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
    )
        
        params = {'eval_set': [(X_train, y_train), (X_test, y_test_fold)], 'verbose': False}
        try:
            model.fit(X_train, y_train, **params, early_stopping_rounds=20)
        except TypeError:
            model.fit(X_train, y_train, **params)
        
        # Store this trained model
        trained_models.append(model)
        
        # Predict on test set
        y_pred_proba_fold = model.predict_proba(X_test)
        
        # Calculate and store metrics for this fold
        minority_threshold = 0.6  # Default threshold
        majority_threshold = 0.6  # Default threshold
        metrics_fold = calculate_metrics_with_threshold(y_test_fold, y_pred_proba_fold, minority_threshold, majority_threshold)
        
        # Store results for this fold
        all_fold_metrics.append(metrics_fold)
        all_y_test.extend(y_test_fold)
        all_y_pred_proba.extend(y_pred_proba_fold)
        all_groups_test.extend(groups_test_fold)
        all_test_indices.extend(test_idx)
        
        # Collect all misclassified samples from this fold
        fold_misclassified = get_misclassified_samples(
            y_test_fold, y_pred_proba_fold, minority_threshold, majority_threshold, groups_test_fold
        )
        
        # Add fold information to the misclassified data
        if fold_misclassified['count'] > 0:
            fold_misclassified['fold'] = fold + 1
            all_misclassified_data.append(fold_misclassified)
        
        # Update progress
        progress_bar.progress((fold + 1) / 5)
    
    # Convert lists to numpy arrays
    all_y_test = np.array(all_y_test)
    all_y_pred_proba = np.array(all_y_pred_proba)
    all_groups_test = np.array(all_groups_test)
    all_test_indices = np.array(all_test_indices)
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Training complete!")
    time.sleep(0.5)
    
    # Clear progress indicators
    progress_container.empty()
    
    # Calculate average metrics across folds
    avg_metrics = {
        'coverage': np.mean([metrics['coverage'] for metrics in all_fold_metrics]),
        'accuracy': np.mean([metrics['accuracy'] for metrics in all_fold_metrics]),
        'f1_0': np.mean([metrics['f1_0'] for metrics in all_fold_metrics]),
        'f1_1': np.mean([metrics['f1_1'] for metrics in all_fold_metrics]),
        'classified': np.sum([metrics['classified'] for metrics in all_fold_metrics]),
        'manual_inspection': np.sum([metrics['manual_inspection'] for metrics in all_fold_metrics]),
        'tp': np.sum([metrics['tp'] for metrics in all_fold_metrics]),
        'tn': np.sum([metrics['tn'] for metrics in all_fold_metrics]),
        'fp': np.sum([metrics['fp'] for metrics in all_fold_metrics]),
        'fn': np.sum([metrics['fn'] for metrics in all_fold_metrics]),
    }
    
    st.success(f"""
    🎉 **Model training completed successfully!** 
    - Applied {len(st.session_state.get('relabeled_data', {}))} label corrections
    - Removed {len(st.session_state.get('discarded_data', set()))} discarded samples
    - Training timestamp: {pd.Timestamp.now().strftime('%H:%M:%S')}
    """)
    
    st.divider()

    # Build a quick lookup for group → status
    group_to_status = {
        g: s for g, s in zip(data_dict['groups'], data_dict['statuses'])
    }
    
    # Create result dictionary
    result = {
        'model': trained_models[0],  # Use first model as the main model
        'X_combined': X_combined,
        'feature_names': combined_feature_names,
        'all_y_test': all_y_test,
        'all_y_pred_proba': all_y_pred_proba,
        'all_groups_test': all_groups_test,
        'all_test_indices': all_test_indices,
        'all_fold_metrics': all_fold_metrics,
        'avg_metrics': avg_metrics,
        'all_misclassified_data': all_misclassified_data,
        'imputer': imputer,
        'data_info': data_dict,
        'statuses': data_dict['statuses'],        # 👈 add raw statuses
        'group_to_status': group_to_status,       # 👈 add lookup dict
        'training_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'corrections_applied': len(st.session_state.get('relabeled_data', {})),
        'discards_applied': len(st.session_state.get('discarded_data', set()))
    }

    # Save pretrained model if no corrections or discards were applied
    if _corrections_count == 0 and _discards_count == 0:
        success, total_size, model_size, metadata_size = save_pretrained_model(result, dataset_hash, trained_models)
        if success:
            st.info(f"💾 Pretrained XGBoost model saved ({total_size:.1f} MB total: {model_size:.1f} MB models + {metadata_size:.1f} MB metadata) - ready for instant loading!")

    return result

def calculate_metrics_with_threshold(y_true, y_pred_proba, minority_threshold, majority_threshold):
    """Calculate performance metrics for given double threshold system"""
    # Apply double threshold logic
    y_pred = np.full(len(y_true), -1)  # -1 = manual inspection
    
    # Classify based on class-specific thresholds
    minority_mask = y_pred_proba[:, 0] >= minority_threshold
    majority_mask = y_pred_proba[:, 1] >= majority_threshold
    
    y_pred[minority_mask] = 0
    y_pred[majority_mask & ~minority_mask] = 1
    
    # Get indices of samples sent to manual inspection
    manual_inspection_indices = np.where(y_pred == -1)[0]
    
    # Count manual inspection samples
    manual_inspection = np.sum(y_pred == -1)
    classified_samples = len(y_true) - manual_inspection
    
    if classified_samples == 0:
        return {
            'threshold_minority': minority_threshold, 'threshold_majority': majority_threshold,
            'accuracy': 0, 'precision_0': 0, 'recall_0': 0, 'precision_1': 0, 'recall_1': 0,
            'f1_0': 0, 'f1_1': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
            'manual_inspection': manual_inspection, 'classified': classified_samples,
            'coverage': 0, 'manual_inspection_indices': manual_inspection_indices
        }
    
    # Calculate confusion matrix only for classified samples
    classified_mask = y_pred != -1
    y_true_classified = y_true[classified_mask]
    y_pred_classified = y_pred[classified_mask]
    
    tn, fp, fn, tp = confusion_matrix(y_true_classified, y_pred_classified).ravel()
    
    accuracy = accuracy_score(y_true_classified, y_pred_classified)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    coverage = classified_samples / len(y_true)
    
    return {
        'threshold_minority': minority_threshold, 'threshold_majority': majority_threshold,
        'accuracy': accuracy, 'precision_0': precision_0, 'recall_0': recall_0,
        'precision_1': precision_1, 'recall_1': recall_1, 'f1_0': f1_0, 'f1_1': f1_1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'manual_inspection': manual_inspection,
        'classified': classified_samples, 'coverage': coverage, 'manual_inspection_indices': manual_inspection_indices
    }

def simulate_classification_split(total_data_points, y_true, y_pred_proba, minority_threshold, majority_threshold):
    """Simulate how 10k data points would be classified using double threshold system"""
    # Use actual data distribution
    actual_minority_ratio = np.sum(y_true == 0) / len(y_true)
    
    # Generate 10k points with same distribution as real data
    minority_true_count = int(total_data_points * actual_minority_ratio)
    majority_true_count = total_data_points - minority_true_count
    
    # Simulate predictions based on actual probability distributions
    minority_probs = y_pred_proba[y_true == 0]
    majority_probs = y_pred_proba[y_true == 1]
    
    # Sample probabilities for our 10k simulation
    np.random.seed(42)
    sim_minority_probs = np.random.choice(len(minority_probs), minority_true_count, replace=True)
    sim_majority_probs = np.random.choice(len(majority_probs), majority_true_count, replace=True)
    
    # Apply double threshold logic
    # For true minority samples (class 0)
    minority_sample_probs = minority_probs[sim_minority_probs]
    minority_classified_as_minority = np.sum(minority_sample_probs[:, 0] >= minority_threshold)
    minority_classified_as_majority = np.sum(minority_sample_probs[:, 1] >= majority_threshold)
    minority_to_manual = minority_true_count - minority_classified_as_minority - minority_classified_as_majority
    
    # For true majority samples (class 1)  
    majority_sample_probs = majority_probs[sim_majority_probs]
    majority_classified_as_majority = np.sum(majority_sample_probs[:, 1] >= majority_threshold)
    majority_classified_as_minority = np.sum(majority_sample_probs[:, 0] >= minority_threshold)
    majority_to_manual = majority_true_count - majority_classified_as_majority - majority_classified_as_minority
    
    # Total counts for each output branch
    total_classified_minority = minority_classified_as_minority + majority_classified_as_minority
    total_classified_majority = minority_classified_as_majority + majority_classified_as_majority
    total_manual_inspection = minority_to_manual + majority_to_manual
    
    return {
        'total_classified_minority': total_classified_minority,
        'total_classified_majority': total_classified_majority,
        'total_manual_inspection': total_manual_inspection,
        'true_minority_count': minority_true_count,
        'true_majority_count': majority_true_count,
        'correct_minority': minority_classified_as_minority,
        'correct_majority': majority_classified_as_majority,
        'wrong_minority': majority_classified_as_minority,
        'wrong_majority': minority_classified_as_majority,
        'manual_from_minority': minority_to_manual,
        'manual_from_majority': majority_to_manual,
        'minority_coverage': (minority_classified_as_minority + minority_classified_as_majority) / minority_true_count if minority_true_count > 0 else 0,
        'majority_coverage': (majority_classified_as_majority + majority_classified_as_minority) / majority_true_count if majority_true_count > 0 else 0,
        'overall_coverage': (total_classified_minority + total_classified_majority) / total_data_points
    }

def get_misclassified_samples(y_true, y_pred_proba, minority_threshold, majority_threshold, groups_test):
    """Identify misclassified samples and return their details ordered by confidence."""
    # Apply double threshold logic to get predictions
    y_pred = np.full(len(y_true), -1)
    
    minority_mask = y_pred_proba[:, 0] >= minority_threshold
    majority_mask = y_pred_proba[:, 1] >= majority_threshold
    
    y_pred[minority_mask] = 0
    y_pred[majority_mask & ~minority_mask] = 1
    
    # Find misclassified samples (excluding manual inspection)
    classified_mask = y_pred != -1
    y_true_classified = y_true[classified_mask]
    y_pred_classified = y_pred[classified_mask]
    groups_classified = groups_test[classified_mask]
    y_proba_classified = y_pred_proba[classified_mask]
    
    # Find indices where predictions don't match true labels
    misclassified_mask = y_true_classified != y_pred_classified
    
    if not np.any(misclassified_mask):
        return {
            'count': 0,
            'indices': [],
            'groups': [],
            'true_labels': [],
            'pred_labels': [],
            'confidences': [],
            'ordered_indices': []
        }
    
    # Ensure misclassified indices are within bounds
    misclassified_indices = np.where(classified_mask)[0][misclassified_mask]
    
    # Convert numpy types to native Python types to avoid issues
    misclassified_groups = groups_classified[misclassified_mask]
    if len(misclassified_groups) > 0 and hasattr(misclassified_groups[0], 'item'):
        misclassified_groups = [g.item() if hasattr(g, 'item') else g for g in misclassified_groups]
    
    misclassified_true = y_true_classified[misclassified_mask]
    misclassified_pred = y_pred_classified[misclassified_mask]
    
    # Calculate confidence scores
    confidences = np.zeros(len(misclassified_pred))
    for i, pred in enumerate(misclassified_pred):
        confidences[i] = y_proba_classified[misclassified_mask][i, int(pred)]
    
    # Order by confidence (highest to lowest)
    ordered_indices = np.argsort(confidences)[::-1]
    
    return {
        'count': len(misclassified_indices),
        'indices': misclassified_indices,
        'groups': misclassified_groups,
        'true_labels': misclassified_true,
        'pred_labels': misclassified_pred,
        'confidences': confidences,
        'ordered_indices': ordered_indices
    }

def plot_sequence(df, group_id, title=None, confidence=None):
    """Plot a time series sequence using Plotly"""
    sequence_data = df[df['group'] == group_id]
    
    if len(sequence_data) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No data found for this group ID",
            annotations=[
                dict(
                    text="No data available for the specified group ID",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )
            ]
        )
        return fig
    
    time = sequence_data['col1'].values
    measurements = sequence_data['col2'].values
    label = "Normal" if sequence_data['label'].iloc[0] == 1 else "Not Normal"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time, 
        y=measurements,
        mode='lines',
        name='Measurements',
        line=dict(color='#FF9800', width=2)
    ))
    
    if confidence is not None:
        confidence_text = f"<br><span style='font-size:12px'>Model Confidence: {confidence:.4f}</span>"
    else:
        confidence_text = ""
    
    if title:
        display_title = f"{title}{confidence_text}"
    else:
        display_title = f"Sequence {group_id} ({label}){confidence_text}"
        
    fig.update_layout(
        title={'text': display_title, 'xanchor': 'center', 'x': 0.5},
        xaxis_title="Time",
        yaxis_title="Measurement",
        height=400,
        plot_bgcolor='rgba(240,240,240,0.9)',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            showline=True,
            linecolor='rgba(70,70,70,0.2)',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            showline=True,
            linecolor='rgba(70,70,70,0.2)',
        )
    )
    
    return fig


def display_sample_info(group_id, true_label, pred_label, confidence, fold, is_relabeled, relabel_value=None, is_discarded=False, status=None):
    true_label_text = "Normal (1)" if true_label == 1 else "Not Normal (0)"
    pred_label_text = "Normal (1)" if pred_label == 1 else "Not Normal (0)"
    
    if true_label == 1:
        box_style = "border-left: 4px solid #2e7d32; background-color: #e8f5e9;"
    else:
        box_style = "border-left: 4px solid #c62828; background-color: #ffebee;"
    
    status_indicators = ""
    if is_relabeled:
        new_label_text = "Normal (1)" if relabel_value == 1 else "Not Normal (0)"
        status_indicators += f" | <span style='color: #1976d2;'>✓ Relabeled to {new_label_text}</span>"
    if is_discarded:
        status_indicators += " | <span style='color: #d32f2f;'>🗑️ Marked for Discard</span>"
    
    st.markdown(f"""
    <div style="{box_style} padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div><strong>True Label:</strong> {true_label_text}{status_indicators}</div>
            <div><strong>Predicted:</strong> {pred_label_text}</div>
            <div><strong>Confidence:</strong> {confidence:.4f}</div>
            <div><strong>Status:</strong> {status}</div> <!-- 👈 NEW -->
            <div><strong>Fold:</strong> {fold}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_relabel_buttons(group_id, fold, is_relabeled, is_discarded):
    """Display relabeling and discard buttons"""
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    
    with col1:
        if st.button("✅ Normal", key=f"relabel_normal_{group_id}_{fold}", 
                    disabled=is_discarded,
                    help="Relabel as Normal (1)"):
            st.session_state.relabeled_data[str(group_id)] = 1
            save_corrections()
            st.rerun()
    
    with col2:
        if st.button("❌ Not Normal", key=f"relabel_not_normal_{group_id}_{fold}",
                    disabled=is_discarded,
                    help="Relabel as Not Normal (0)"):
            st.session_state.relabeled_data[str(group_id)] = 0
            save_corrections()
            st.rerun()
    
    with col3:
        if not is_discarded:
            if st.button("🗑️ Discard", key=f"discard_{group_id}_{fold}",
                        type="secondary",
                        help="Mark this sample as abnormal data to remove from training"):
                st.session_state.discarded_data.add(str(group_id))
                # Remove from relabeled if it was there
                if str(group_id) in st.session_state.relabeled_data:
                    del st.session_state.relabeled_data[str(group_id)]
                save_corrections()
                save_discards()
                st.rerun()
    
    with col4:
        if is_relabeled and not is_discarded:
            if st.button("↩️ Undo Label", key=f"remove_relabel_{group_id}_{fold}",
                        help="Remove label correction"):
                del st.session_state.relabeled_data[str(group_id)]
                save_corrections()
                st.rerun()
        elif is_discarded:
            if st.button("♻️ Restore", key=f"restore_{group_id}_{fold}",
                        help="Remove from discard list"):
                st.session_state.discarded_data.remove(str(group_id))
                save_discards()
                st.rerun()

# Initialize session state for relabeled data (session-only for multi-user)
if 'relabeled_data' not in st.session_state:
    st.session_state.relabeled_data = {}

# Initialize session state for discarded data (session-only for multi-user)
if 'discarded_data' not in st.session_state:
    st.session_state.discarded_data = set()

if 'retrain_counter' not in st.session_state:
    st.session_state.retrain_counter = 0

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Initialize navigation state
if 'misclassified_index' not in st.session_state:
    st.session_state.misclassified_index = 0
    
if 'manual_inspection_index' not in st.session_state:
    st.session_state.manual_inspection_index = 0

def save_corrections():
    """Save corrections to session state only (no file persistence for multi-user)"""
    # For multi-user deployment, we don't save to files
    # Changes are only kept in the current session
    pass

def save_discards():
    """Save discarded samples to session state only (no file persistence for multi-user)"""
    # For multi-user deployment, we don't save to files
    # Changes are only kept in the current session
    pass

# Main Application
main_title_html = create_tooltip(
    "🚀 AI Double Threshold Visual Analyzer",
    "This application demonstrates an AI-powered quality control system that uses different confidence thresholds for different classes (Normal vs Defective). The system automatically classifies high-confidence cases and sends uncertain cases to manual inspection, optimizing the balance between automation and accuracy.",
    "top",
    "large"
)
st.markdown(f'<h1 class="main-header">{main_title_html}</h1>', unsafe_allow_html=True)


# Thesis Description Section
st.markdown("---")
research_title_html = create_tooltip(
    "📚 Research Overview",
    "This section provides context about the research thesis that this application supports. It explains the focus on industrial quality control automation using machine learning and the challenges of maintaining high accuracy while reducing manual inspection workload.",
    "top"
)
st.markdown(f"### {research_title_html}", unsafe_allow_html=True)

st.markdown("""
<div style="font-size: 1.2rem; line-height: 1.6; padding: 1.5rem; background-color: #f0f8ff; border-radius: 10px; border-left: 4px solid #2e7d32;">
<strong>Thesis Focus:</strong> This research explores the feasibility of automating quality control processes in industrial settings using machine learning. The study utilizes a dataset created from manual quality inspections to develop an AI system that can potentially reduce manual inspection workload while maintaining or even improving quality standards.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Check for required files
if not os.path.exists('wave_dataset.parquet'):
    st.error("❌ Missing 'wave_dataset.parquet' file")
    st.stop()

# Add data management section in sidebar (loaded early to prevent layout shifts)
st.sidebar.markdown("---")
sidebar_title_html = create_sidebar_tooltip(
    "🏷️ Data Management",
    "This section allows you to manage data corrections and discards. You can relabel samples that were incorrectly classified or mark samples as abnormal data to remove from training."
)
st.sidebar.markdown(f"### {sidebar_title_html}", unsafe_allow_html=True)
session_note_html = create_tooltip(
    "💾 Changes are session-only (reset on page refresh)",
    "All changes you make (relabeling or discarding samples) are stored only in your current browser session. When you refresh the page, these changes will be lost unless you export them first.",
    "top"
)
st.sidebar.caption(session_note_html, unsafe_allow_html=True)

# Show statistics
col1, col2 = st.sidebar.columns(2)
with col1:
    corrections_html = create_sidebar_tooltip(
        "Corrections",
        "Number of samples you have relabeled (changed from their original ground truth labels). These changes will be applied when you retrain the model."
    )
    st.markdown(f"**{corrections_html}**", unsafe_allow_html=True)
    st.metric("", len(st.session_state.relabeled_data))
with col2:
    discarded_html = create_tooltip(
        "Discarded",
        "Number of samples you have marked for removal from training. These samples will be excluded when you retrain the model.",
        "top"
    )
    st.markdown(f"**{discarded_html}**", unsafe_allow_html=True)
    st.metric("", len(st.session_state.discarded_data))

# Show details in expanders
if st.session_state.relabeled_data:
    with st.sidebar.expander("View Corrections"):
        for group_id_str, new_label in st.session_state.relabeled_data.items():
            label_text = "Normal (1)" if new_label == 1 else "Not Normal (0)"
            st.write(f"Group {group_id_str}: → {label_text}")

if st.session_state.discarded_data:
    with st.sidebar.expander("View Discarded"):
        for group_id_str in st.session_state.discarded_data:
            st.write(f"Group {group_id_str}")

# Export data changes
if st.session_state.relabeled_data or st.session_state.discarded_data:
    # Prepare combined data for export
    export_data = []
    
    # Add corrections
    for group_id_str, new_label in st.session_state.relabeled_data.items():
        export_data.append({
            'group_id': group_id_str,
            'action': 'relabel',
            'new_label': new_label
        })
    
    # Add discards
    for group_id_str in st.session_state.discarded_data:
        export_data.append({
            'group_id': group_id_str,
            'action': 'discard',
            'new_label': None
        })
    
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Export All Changes",
        data=csv,
        file_name=f"data_changes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# Debug: Show current state (moved after model_data is defined)

# Load and train model
try:
    corrections_count = len(st.session_state.relabeled_data)
    discards_count = len(st.session_state.discarded_data)
    model_data = train_model_with_cross_validation(_corrections_count=corrections_count, _discards_count=discards_count)
    
    if model_data is None:
        st.error("❌ Failed to load data or train model")
        st.stop()
    
    model = model_data['model']
    X_combined = model_data['X_combined']
    feature_names = model_data['feature_names']
    all_y_test = model_data['all_y_test']
    all_y_pred_proba = model_data['all_y_pred_proba']
    all_groups_test = model_data['all_groups_test']
    all_fold_metrics = model_data['all_fold_metrics']
    avg_metrics = model_data['avg_metrics']
    all_misclassified_data = model_data['all_misclassified_data']
    
    # Add retraining controls that depend on model data
    # Check if model needs retraining
    needs_retrain = False
    if (len(st.session_state.relabeled_data) > 0 and 
        model_data.get('corrections_applied', 0) < len(st.session_state.relabeled_data)):
        needs_retrain = True
    if (len(st.session_state.discarded_data) > 0 and 
        model_data.get('discards_applied', 0) < len(st.session_state.discarded_data)):
        needs_retrain = True
    
    if needs_retrain:
        st.sidebar.warning("⚠️ Model needs retraining to apply changes")
    
    if st.sidebar.button("🔄 Retrain AI", 
                        type="primary" if needs_retrain else "secondary",
                        help="Retrain the model with corrected labels and discarded samples"):
        st.session_state.retrain_counter += 1
        st.cache_resource.clear()
        st.rerun()
    
    # Clear buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🗑️ Clear Corrections"):
            st.session_state.relabeled_data = {}
            save_corrections()
            st.cache_resource.clear()
            st.rerun()
    with col2:
        if st.sidebar.button("🗑️ Clear Discards"):
            st.session_state.discarded_data = set()
            save_discards()
            st.cache_resource.clear()
            st.rerun()

    
    # Initialize default thresholds (will be set in the visualization)
    minority_threshold = 0.9
    majority_threshold = 0.5
    
    # Reset navigation indices when thresholds change
    if 'prev_minority_threshold' not in st.session_state:
        st.session_state.prev_minority_threshold = minority_threshold
    if 'prev_majority_threshold' not in st.session_state:
        st.session_state.prev_majority_threshold = majority_threshold
        
    if (st.session_state.prev_minority_threshold != minority_threshold or 
        st.session_state.prev_majority_threshold != majority_threshold):
        st.session_state.misclassified_index = 0
        st.session_state.manual_inspection_index = 0
        st.session_state.prev_minority_threshold = minority_threshold
        st.session_state.prev_majority_threshold = majority_threshold
    
    
    # Calculate real performance metrics
    current_metrics = calculate_metrics_with_threshold(all_y_test, all_y_pred_proba, minority_threshold, majority_threshold)
    
    # Simulate classification for 10k points with fixed thresholds (no user control)
    total_points = 10000
    fixed_minority_threshold = 0.5
    fixed_majority_threshold = 0.5
    simulation_results = simulate_classification_split(total_points, all_y_test, all_y_pred_proba, fixed_minority_threshold, fixed_majority_threshold)
    
    
    # Data Flow Visualization
    performance_title_html = create_tooltip(
        "📊 Basic Model Performance",
        "This section shows the basic performance of the AI model on 10,000 simulated data points. It demonstrates how many samples are correctly classified versus misclassified, revealing the fundamental accuracy of the model before any advanced techniques are applied.",
        "top"
    )
    st.markdown(f"### {performance_title_html}", unsafe_allow_html=True)
    
    # Check if there are unapplied changes
    unapplied_corrections = (len(st.session_state.relabeled_data) > 0 and 
                           model_data.get('corrections_applied', 0) < len(st.session_state.relabeled_data))
    unapplied_discards = (len(st.session_state.discarded_data) > 0 and 
                         model_data.get('discards_applied', 0) < len(st.session_state.discarded_data))
    
    if unapplied_corrections or unapplied_discards:
        st.warning(f"""
        ⚠️ **Attention**: You have unapplied changes:
        - {len(st.session_state.relabeled_data) - model_data.get('corrections_applied', 0)} new label corrections
        - {len(st.session_state.discarded_data) - model_data.get('discards_applied', 0)} new discarded samples
        
        Click **🔄 Retrain AI** in the sidebar to apply all changes.
        """)
    
    # Show training info if model was retrained
    if st.session_state.retrain_counter > 0 or len(st.session_state.relabeled_data) > 0 or len(st.session_state.discarded_data) > 0:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            st.success(f"""
            🔄 **Model Status**: Retrained {st.session_state.retrain_counter} time(s)
            """)
        with col2:
            if 'training_timestamp' in model_data:
                st.info(f"⏰ **Last Trained**: {model_data['training_timestamp']}")
        with col3:
            st.metric("Applied Corrections", model_data.get('corrections_applied', 0))
        with col4:
            st.metric("Applied Discards", model_data.get('discards_applied', 0))
    
    # Create vertical flow layout
    # Step 1: Input Data
    input_data_title = create_tooltip(
        "📥 Input Data",
        "This represents the total dataset of 10,000 data points that will be processed by the AI system. The data is split between Normal samples (majority class) and Defective samples (minority class) based on the actual distribution in your training data.",
        "top"
    )
    st.markdown(create_div_card('data-points base-card', f'''
        <h3>{input_data_title}</h3>
        <h2>{total_points:,}</h2>
        <p>(Normal {simulation_results['true_majority_count']:,} + Defect {simulation_results['true_minority_count']:,})</p>
    '''), unsafe_allow_html=True)
    
    # Arrow down from Input Data
    st.markdown('<div class="flow-arrow-down">⬇️</div>', unsafe_allow_html=True)
    
    # Step 2: XGBoost Model (Full Width)
    ai_model_title = create_tooltip(
        "🤖 AI Model",
        "This is the XGBoost machine learning model that processes each data point and outputs confidence scores for both Normal and Defective classes. The model uses advanced features extracted from time series data to make its predictions.",
        "top"
    )
    processing_text = create_tooltip(
        "Processing 10,000 data points through double threshold classification",
        "The model analyzes each data point and assigns probability scores. The double threshold system then uses these scores to decide: if confidence is high enough, classify automatically; otherwise, send to manual inspection.",
        "top"
    )
    st.markdown(create_div_card('xgboost-model base-card', f'''
        <h3>{ai_model_title}</h3>
        <p><strong>{processing_text}</strong></p>
    '''), unsafe_allow_html=True)
    
    # Two arrows pointing from XGBoost to each classification output
    
    # Create two arrows pointing from XGBoost to each output
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="flow-arrow-diagonal" style="font-size: 2rem;">↙️</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="flow-arrow-diagonal" style="font-size: 2rem;">↘️</div>', unsafe_allow_html=True)
    
    # Two classification outputs (only automatic classifications)
    col1, col2 = st.columns(2)
    
    with col1:
        majority_percentage = (simulation_results['total_classified_majority'] / total_points) * 100
        normal_title = create_tooltip(
            "🟢 Classified as Normal",
            "These are data points that the AI model classified as normal. In a real manufacturing setting, these would pass quality control and proceed to the next stage.",
            "top"
        )
        st.markdown(create_div_card('majority-class base-card', f'''
            <h3>{normal_title}</h3>
            <h2>{simulation_results['total_classified_majority']:,}</h2>
            <p>{majority_percentage:.1f}% of total data</p>
        '''), unsafe_allow_html=True)
    
    with col2:
        minority_percentage = (simulation_results['total_classified_minority'] / total_points) * 100
        defective_title = create_tooltip(
            "🔴 Classified as Defective",
            "These are data points that the AI model classified as defective. In a real manufacturing setting, these would be rejected or sent for rework.",
            "top"
        )
        st.markdown(create_div_card('minority-class base-card', f'''
            <h3>{defective_title}</h3>
            <h2>{simulation_results['total_classified_minority']:,}</h2>
            <p>{minority_percentage:.1f}% of total data</p>
        '''), unsafe_allow_html=True)
    
    # Classification Accuracy Breakdown for First Visualization
    
    # Create 4 columns for horizontal layout (4x1)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Majority Classifications - Correct
        st.markdown('<div class="flow-arrow-diagonal">↙️</div>', unsafe_allow_html=True)
        if simulation_results['total_classified_majority'] > 0:
            correct_majority_pct = (simulation_results['correct_majority'] / simulation_results['total_classified_majority'] * 100)
            correct_majority_title = create_tooltip(
                "✅ Correct (Normal)",
                "These are samples that were correctly classified as normal. In manufacturing, these represent proper quality control - good products that were correctly identified and allowed to proceed.",
                "top"
            )
            st.markdown(create_classification_card(
                correct_majority_title,
                simulation_results['correct_majority'],
                correct_majority_pct,
                "True Positives",
                "correct"
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    with col2:
        # Majority Classifications - Wrong
        st.markdown('<div class="flow-arrow-diagonal">↘️</div>', unsafe_allow_html=True)
        if simulation_results['total_classified_majority'] > 0:
            wrong_majority_pct = (simulation_results['wrong_majority'] / simulation_results['total_classified_majority'] * 100)
            wrong_majority_title = create_tooltip(
                "❌ Wrong (Normal)",
                "These are samples that were incorrectly classified as normal when they are actually defective. In manufacturing, these represent quality escapes - defective products that reach customers and can cause serious problems.",
                "top"
            )
            st.markdown(create_classification_card(
                wrong_majority_title,
                simulation_results['wrong_majority'],
                wrong_majority_pct,
                "False Negatives",
                "wrong"
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    with col3:
        # Minority Classifications - Correct
        st.markdown('<div class="flow-arrow-diagonal">↙️</div>', unsafe_allow_html=True)
        if simulation_results['total_classified_minority'] > 0:
            correct_minority_pct = (simulation_results['correct_minority'] / simulation_results['total_classified_minority'] * 100)
            correct_minority_title = create_tooltip(
                "✅ Correct (Defect)",
                "These are samples that were correctly classified as defective. In manufacturing, these represent proper quality control - defective products that were correctly identified and rejected.",
                "top"
            )
            st.markdown(create_classification_card(
                correct_minority_title,
                simulation_results['correct_minority'],
                correct_minority_pct,
                "True Negatives",
                "correct"
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    with col4:
        # Minority Classifications - Wrong
        st.markdown('<div class="flow-arrow-diagonal">↘️</div>', unsafe_allow_html=True)
        if simulation_results['total_classified_minority'] > 0:
            wrong_minority_pct = (simulation_results['wrong_minority'] / simulation_results['total_classified_minority'] * 100)
            wrong_minority_title = create_tooltip(
                "❌ Wrong (Defect)",
                "These are samples that were incorrectly classified as defective when they are actually normal. In manufacturing, these represent false alarms that waste resources and slow down production.",
                "top"
            )
            st.markdown(create_classification_card(
                wrong_minority_title,
                simulation_results['wrong_minority'],
                wrong_minority_pct,
                "False Positives",
                "wrong"
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    # Performance Analysis
    st.markdown("---")
    performance_analysis_title_html = create_tooltip(
        "📈 Performance Analysis",
        "This section analyzes the AI model's performance by examining accuracy rates for each class. It reveals critical insights about class imbalance and the need for a hybrid system with manual inspection.",
        "top"
    )
    st.markdown(f"### {performance_analysis_title_html}", unsafe_allow_html=True)
    
    # Calculate the performance percentages
    defective_correct_pct = (simulation_results['correct_minority'] / simulation_results['total_classified_minority'] * 100) if simulation_results['total_classified_minority'] > 0 else 0
    normal_correct_pct = (simulation_results['correct_majority'] / simulation_results['total_classified_majority'] * 100) if simulation_results['total_classified_majority'] > 0 else 0
    
    # Calculate error rates for both classes
    defective_error_rate = 100 - defective_correct_pct
    normal_error_rate = 100 - normal_correct_pct
    error_ratio = defective_error_rate / normal_error_rate if normal_error_rate > 0 else float('inf')
    
    st.markdown(f"""
    <div style="font-size: 1.2rem; line-height: 1.6; padding: 1.5rem; background-color: #ffebee; border-radius: 10px; border-left: 4px solid #d32f2f;">
    <strong>⚠️ Critical Quality Control Issue:</strong> The results reveal a significant performance imbalance between the two classes. While the AI model shows good performance on normal data ({normal_correct_pct:.1f}% accuracy), its performance on defective data ({defective_correct_pct:.1f}% accuracy) is insufficient for our industrial quality control standards.
    <br><br>
    <strong>The Magnitude of the Problem:</strong> The error rates show a {error_ratio:.1f}x performance difference:
    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
        <li><strong>Among classified defective samples:</strong> {defective_error_rate:.1f}% are actually normal (false positives)</li>
        <li><strong>Among classified normal samples:</strong> {normal_error_rate:.1f}% are actually defective (false negatives)</li>
    </ul>
    This means a normal product is <strong>{error_ratio:.1f} times more likely</strong> to be classified as a defect than a defect product being classified as normal.
    <br><br>
    <strong>The Critical Impact:</strong> The {defective_error_rate:.1f}% false positive rate means that nearly 1 in 3 products flagged as defective are actually good products. This <strong>excessive rejection of good products</strong> creates significant operational and financial challenges:
    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
        <li><strong>Production Delays:</strong> Good products sent for unnecessary rework or disposal</li>
        <li><strong>Increased Costs:</strong> Wasted materials, labor, and resources on false alarms</li>
        <li><strong>Reduced Throughput:</strong> Production lines slowed by excessive false rejections</li>
        <li><strong>Resource Waste:</strong> Valuable production capacity consumed by processing good products as defective</li>
        <li><strong>Customer Impact:</strong> Delayed deliveries due to reduced production efficiency</li>
    </ul>
    This performance gap necessitates the implementation of a hybrid system to optimize the balance between automation and accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    # Hybrid System with Thresholds and Manual Inspection
    st.markdown("---")
    hybrid_system_title_html = create_tooltip(
        "🔧 Hybrid System: Thresholds & Manual Inspection",
        "This section introduces the hybrid approach that combines AI automation with human expertise. By setting different confidence thresholds for each class and sending uncertain cases to manual inspection, we can improve overall system performance while maintaining quality standards.",
        "top"
    )
    st.markdown(f"### {hybrid_system_title_html}", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 1.2rem; line-height: 1.6; padding: 1.5rem; background-color: #f0f8ff; border-radius: 10px; border-left: 4px solid #2e7d32;">
    To address the performance imbalance, we implemented a hybrid system with adjustable confidence thresholds and manual inspection for uncertain cases:
    </div>
    """, unsafe_allow_html=True)
    
    # Input Data
    st.markdown(create_div_card('data-points base-card', f'''
        <h3>📥 Input Data</h3>
        <h2>{total_points:,}</h2>
        <p>(Normal {simulation_results['true_majority_count']:,} + Defect {simulation_results['true_minority_count']:,})</p>
    '''), unsafe_allow_html=True)
    
    # Arrow down from Input Data
    st.markdown('<div class="flow-arrow-down">⬇️</div>', unsafe_allow_html=True)
    
    # AI Model
    st.markdown(create_div_card('xgboost-model base-card', '''
        <h3>🤖 AI Model with Adjustable Thresholds</h3>
        <p><strong>Processing with confidence-based classification</strong></p>
    '''), unsafe_allow_html=True)
    
    # Three arrows pointing from XGBoost to each classification output
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="flow-arrow-diagonal" style="font-size: 2rem;">↙️</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="flow-arrow-down" style="font-size: 2rem;">⬇️</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="flow-arrow-diagonal" style="font-size: 2rem;">↘️</div>', unsafe_allow_html=True)
    
    # Threshold controls positioned right above the classification outputs
    threshold_title_html = create_tooltip(
        "🎛️ Adjust Classification Thresholds",
        "Use these sliders to adjust the confidence thresholds for each class. Higher thresholds mean more samples will be sent to manual inspection, but with higher accuracy for automatically classified samples.",
        "top"
    )
    st.markdown(f"### {threshold_title_html}", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #e3f2fd; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #2196f3;">
        <strong style="font-size: 1.1rem;">💡 Try sliding the thresholds and see how performance generally improves but at the cost of more manual inspection</strong><br>
        <span style="font-size: 1rem; color: #666;">The higher the threshold, the more manual inspection is needed.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Threshold sliders positioned above the correct columns
    col1, col2, col3 = st.columns(3)
    
    # Initialize session state for thresholds
    if 'minority_threshold' not in st.session_state:
        st.session_state.minority_threshold = 0.5
    if 'majority_threshold' not in st.session_state:
        st.session_state.majority_threshold = 0.5
    
    # Check if optimal thresholds button was clicked
    if st.session_state.get('optimal_thresholds_applied', False):
        st.session_state.minority_threshold = 0.98
        st.session_state.majority_threshold = 0.95
        st.session_state.optimal_thresholds_applied = False
        st.rerun()
    
    with col1:
        majority_threshold = st.slider(
            "🟢 Normal Threshold", 
            min_value=0.5, 
            max_value=0.999,
            value=st.session_state.majority_threshold, 
            step=0.001,
            format="%.3f",
            help="This threshold controls how confident the AI must be to automatically classify a sample as normal. Higher values mean fewer false negatives but more samples sent to manual inspection.",
            key="majority_slider"
        )
        st.session_state.majority_threshold = majority_threshold
    
    with col2:
        st.markdown("")  # Empty column for spacing
    
    with col3:
        minority_threshold = st.slider(
            "🔴 Defective Threshold", 
            min_value=0.5, 
            max_value=0.999,
            value=st.session_state.minority_threshold, 
            step=0.001,
            format="%.3f",
            help="This threshold controls how confident the AI must be to automatically classify a sample as defective. Higher values mean fewer false positives but more samples sent to manual inspection.",
            key="minority_slider"
        )
        st.session_state.minority_threshold = minority_threshold
    
    # Simulate classification with current user thresholds (calculated after sliders are set)
    hybrid_simulation = simulate_classification_split(total_points, all_y_test, all_y_pred_proba, minority_threshold, majority_threshold)
    
    # Three classification outputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        majority_percentage = (hybrid_simulation['total_classified_majority'] / total_points) * 100
        hybrid_normal_title = create_tooltip(
            f"🟢 Classified as Normal (> {majority_threshold:.1%} sure)",
            f"These samples were automatically classified as normal because the AI's confidence exceeded {majority_threshold:.1%}. They can proceed directly to the next production stage without human review.",
            "top"
        )
        st.markdown(create_div_card('majority-class base-card', f'''
            <h3>{hybrid_normal_title}</h3>
            <h2>{hybrid_simulation['total_classified_majority']:,}</h2>
            <p>{majority_percentage:.1f}% of total data</p>
        '''), unsafe_allow_html=True)
    
    with col2:
        manual_percentage = (hybrid_simulation['total_manual_inspection'] / total_points) * 100
        manual_title = create_tooltip(
            "🟠 Manual Inspection",
            "These samples have low confidence scores and require human inspection. This hybrid approach ensures uncertain cases get proper attention while maintaining efficiency for clear-cut decisions.",
            "top"
        )
        st.markdown(create_div_card('manual-inspection base-card', f'''
            <h3>{manual_title}</h3>
            <h2>{hybrid_simulation['total_manual_inspection']:,}</h2>
            <p>{manual_percentage:.1f}% of total data</p>
            <p><strong>Requires Human Review</strong></p>
        '''), unsafe_allow_html=True)
    
    with col3:
        minority_percentage = (hybrid_simulation['total_classified_minority'] / total_points) * 100
        hybrid_defective_title = create_tooltip(
            f"🔴 Classified as Defective (> {minority_threshold:.1%} sure)",
            f"These samples were automatically classified as defective because the AI's confidence exceeded {minority_threshold:.1%}. They can proceed directly to rejection/rework without human review.",
            "top"
        )
        st.markdown(create_div_card('minority-class base-card', f'''
            <h3>{hybrid_defective_title}</h3>
            <h2>{hybrid_simulation['total_classified_minority']:,}</h2>
            <p>{minority_percentage:.1f}% of total data</p>
        '''), unsafe_allow_html=True)
    
    # Classification Accuracy Breakdown for Hybrid System
    
    # Create 6 columns for horizontal layout (6x1)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        # Majority Classifications - Correct
        st.markdown('<div class="flow-arrow-diagonal">↙️</div>', unsafe_allow_html=True)
        if hybrid_simulation['total_classified_majority'] > 0:
            correct_majority_pct = (hybrid_simulation['correct_majority'] / hybrid_simulation['total_classified_majority'] * 100)
            hybrid_correct_majority_title = create_tooltip(
                "✅ Correct (Normal)",
                "These are samples that were correctly classified as normal with high confidence. The hybrid system allows these to be automatically approved without human review, improving efficiency.",
                "top"
            )
            st.markdown(create_classification_card(
                hybrid_correct_majority_title,
                hybrid_simulation['correct_majority'],
                correct_majority_pct,
                "True Positives",
                "correct"
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    with col2:
        # Majority Classifications - Wrong
        st.markdown('<div class="flow-arrow-diagonal">↘️</div>', unsafe_allow_html=True)
        if hybrid_simulation['total_classified_majority'] > 0:
            wrong_majority_pct = (hybrid_simulation['wrong_majority'] / hybrid_simulation['total_classified_majority'] * 100)
            hybrid_wrong_majority_title = create_tooltip(
                "❌ Wrong (Normal)",
                "These are samples that were incorrectly classified as normal with high confidence. The hybrid system's high threshold helps minimize these quality escapes, but some may still occur.",
                "top"
            )
            st.markdown(create_classification_card(
                hybrid_wrong_majority_title,
                hybrid_simulation['wrong_majority'],
                wrong_majority_pct,
                "False Negatives",
                "wrong"
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    with col3:
        # Manual Inspection - True Normal
        st.markdown('<div class="flow-arrow-diagonal">↙️</div>', unsafe_allow_html=True)
        if hybrid_simulation['total_manual_inspection'] > 0:
            manual_majority_pct = (hybrid_simulation['manual_from_majority'] / hybrid_simulation['total_manual_inspection'] * 100)
            manual_normal_title = create_tooltip(
                "📋 Normal",
                "These are samples that are actually normal but had low confidence scores, so they were sent to manual inspection. The hybrid system ensures these get proper human attention.",
                "top"
            )
            st.markdown(create_div_card('manual-breakdown classification-card', f'''
                <h5>{manual_normal_title}</h5>
                <h3>{hybrid_simulation['manual_from_majority']:,}</h3>
                <p>{manual_majority_pct:.1f}%</p>
            '''), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    with col4:
        # Manual Inspection - True Not Normal
        st.markdown('<div class="flow-arrow-diagonal">↘️</div>', unsafe_allow_html=True)
        if hybrid_simulation['total_manual_inspection'] > 0:
            manual_minority_pct = (hybrid_simulation['manual_from_minority'] / hybrid_simulation['total_manual_inspection'] * 100)
            manual_defect_title = create_tooltip(
                "📋 Defect",
                "These are samples that are actually defective but had low confidence scores, so they were sent to manual inspection. The hybrid system ensures these get proper human attention.",
                "top"
            )
            st.markdown(create_div_card('manual-breakdown classification-card', f'''
                <h5>{manual_defect_title}</h5>
                <h3>{hybrid_simulation['manual_from_minority']:,}</h3>
                <p>{manual_minority_pct:.1f}%</p>
            '''), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    with col5:
        # Minority Classifications - Correct
        st.markdown('<div class="flow-arrow-diagonal">↙️</div>', unsafe_allow_html=True)
        if hybrid_simulation['total_classified_minority'] > 0:
            correct_minority_pct = (hybrid_simulation['correct_minority'] / hybrid_simulation['total_classified_minority'] * 100)
            hybrid_correct_minority_title = create_tooltip(
                "✅ Correct (Defect)",
                "These are samples that were correctly classified as defective with high confidence. The hybrid system allows these to be automatically rejected without human review, improving efficiency.",
                "top"
            )
            st.markdown(create_classification_card(
                hybrid_correct_minority_title,
                hybrid_simulation['correct_minority'],
                correct_minority_pct,
                "True Negatives",
                "correct"
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    with col6:
        # Minority Classifications - Wrong
        st.markdown('<div class="flow-arrow-diagonal">↘️</div>', unsafe_allow_html=True)
        if hybrid_simulation['total_classified_minority'] > 0:
            wrong_minority_pct = (hybrid_simulation['wrong_minority'] / hybrid_simulation['total_classified_minority'] * 100)
            hybrid_wrong_minority_title = create_tooltip(
                "❌ Wrong (Defect)",
                "These are samples that were incorrectly classified as defective with high confidence. The hybrid system's high threshold helps minimize these false alarms, but some may still occur.",
                "top"
            )
            st.markdown(create_classification_card(
                hybrid_wrong_minority_title,
                hybrid_simulation['wrong_minority'],
                wrong_minority_pct,
                "False Positives",
                "wrong"
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No samples*")
    
    
    # Strategic Threshold Analysis
    st.markdown("---")
    strategic_title_html = create_tooltip(
        "🎯 Strategic Threshold Optimization",
        "This section provides insights into optimal threshold settings for manufacturing environments. It explains why balanced manual inspection workloads are important for maintaining quality standards and inspector performance.",
        "top"
    )
    st.markdown(f"### {strategic_title_html}", unsafe_allow_html=True)
    
    # Calculate current manual inspection split
    manual_defect_ratio = (hybrid_simulation['manual_from_minority'] / hybrid_simulation['total_manual_inspection'] * 100) if hybrid_simulation['total_manual_inspection'] > 0 else 0
    manual_normal_ratio = (hybrid_simulation['manual_from_majority'] / hybrid_simulation['total_manual_inspection'] * 100) if hybrid_simulation['total_manual_inspection'] > 0 else 0
    
    # Create a single integrated text box with inline button
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col2:
        # Working button above the text box - full width with custom styling
        st.markdown("""
        <style>
        .stButton > button {
            background-color: #9c27b0 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            border-radius: 6px !important;
        }
        .stButton > button:hover {
            background-color: #7b1fa2 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Set Thresholds", 
                    type="primary", 
                    use_container_width=True,
                    help="Apply the recommended thresholds (Defective: 0.98, Normal: 0.95) for balanced manual inspection workload"):
            st.session_state.optimal_thresholds_applied = True
            st.rerun()
        
        # Create the text with inline button
        st.markdown(f"""
        <div style="font-size: 1.2rem; line-height: 1.6; padding: 1.5rem; background-color: #f3e5f5; border-radius: 10px; border-left: 4px solid #9c27b0;">
        <strong>💡 Manufacturing Quality Control Insight:</strong> Click the button above to apply thresholds (Normal at 0.95, Defective at 0.98) and see how it achieves a more balanced manual inspection workload. 
        <br><br>
        <strong>Current Manual Inspection Split:</strong> {manual_defect_ratio:.1f}% defective samples, {manual_normal_ratio:.1f}% normal samples
        <br><br>
        <strong>Why This Matters in Manufacturing:</strong>
        <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
            <li><strong>Reduces Inspector Bias:</strong> When manual inspectors see a balanced mix (closer to 50/50), they're less likely to develop unconscious bias toward the majority class</li>
            <li><strong>Maintains Alertness:</strong> A balanced workload prevents inspectors from becoming complacent when seeing mostly normal parts</li>
            <li><strong>Improves Training:</strong> New inspectors get equal exposure to both defect types, leading to better overall performance</li>
            <li><strong>Quality Consistency:</strong> Reduces the risk of missing defects due to "normalcy bias" when the workload is heavily skewed toward normal samples</li>
        </ul>
        This approach helps maintain high-quality standards while optimizing the human-AI collaboration in production environments.
        </div>
        """, unsafe_allow_html=True)
    
    # Arrow for next step
    st.markdown('<div class="flow-arrow">⬇️</div>', unsafe_allow_html=True)
    
    
    # Get the raw data for visualizations
    raw_df = model_data['data_info']['raw_df']
    
    # Misclassified Samples Visualization
    misclassified_title_html = create_tooltip(
        "❌ Misclassified Samples (From 5-Fold Cross-Validation)",
        "This section shows samples that the AI model classified incorrectly during cross-validation testing. These are ordered by confidence level (highest to lowest) to help you identify the most confident mistakes first.",
        "top"
    )
    st.markdown(f"### {misclassified_title_html}", unsafe_allow_html=True)
    
    navigation_text_html = create_tooltip(
        "Navigate through misclassified data points ordered by confidence (highest to lowest)",
        "The samples are sorted by the AI's confidence in its incorrect prediction. This helps prioritize which mistakes to review first - start with high-confidence errors as they may indicate systematic issues.",
        "top"
    )
    st.markdown(f"**{navigation_text_html}**", unsafe_allow_html=True)
    
    tip_text_html = create_tooltip(
        "💡 **Tip:** You can relabel or discard any misclassified sample by clicking the buttons above each plot.",
        "Use the relabel buttons to correct the ground truth labels if you believe the AI was actually right. Use the discard button to mark samples as abnormal data that should be removed from training.",
        "top"
    )
    st.markdown(f"{tip_text_html}", unsafe_allow_html=True)
    
    # Recalculate misclassified samples for current threshold
    all_misclassified = []
    for fold in range(5):
        fold_indices = [i for i, data in enumerate(all_misclassified_data) if data.get('fold') == fold + 1]
        if fold_indices:
            fold_data = all_misclassified_data[fold_indices[0]]
            if fold_data['count'] > 0:
                fold_start_idx = fold * (len(all_y_test) // 5)
                fold_end_idx = (fold + 1) * (len(all_y_test) // 5)
                fold_y_test = all_y_test[fold_start_idx:fold_end_idx]
                fold_y_pred_proba = all_y_pred_proba[fold_start_idx:fold_end_idx]
                fold_groups_test = all_groups_test[fold_start_idx:fold_end_idx]
                
                misclassified = get_misclassified_samples(
                    fold_y_test, 
                    fold_y_pred_proba, 
                    minority_threshold, 
                    majority_threshold,
                    fold_groups_test
                )
                
                if misclassified['count'] > 0:
                    misclassified['fold'] = fold + 1
                    all_misclassified.append(misclassified)
    
    # Count total misclassified samples
    total_misclassified = sum(data['count'] for data in all_misclassified)
    
    if total_misclassified > 0:
        # Create a combined list of all misclassified samples
        combined_misclassified = {
            'groups': [],
            'true_labels': [],
            'pred_labels': [],
            'confidences': [],
            'folds': []
        }
        
        for fold_data in all_misclassified:
            for idx in fold_data['ordered_indices']:
                combined_misclassified['groups'].append(fold_data['groups'][idx])
                combined_misclassified['true_labels'].append(fold_data['true_labels'][idx])
                combined_misclassified['pred_labels'].append(fold_data['pred_labels'][idx])
                combined_misclassified['confidences'].append(fold_data['confidences'][idx])
                combined_misclassified['folds'].append(fold_data['fold'])
        
        # Sort all samples by confidence (highest to lowest)
        sorted_indices = np.argsort(combined_misclassified['confidences'])[::-1]
        
        # Show summary statistics (should match the breakdown cards above)
        false_positives = sum(1 for i in range(len(combined_misclassified['true_labels'])) 
                             if combined_misclassified['true_labels'][i] == 0 and combined_misclassified['pred_labels'][i] == 1)
        false_negatives = sum(1 for i in range(len(combined_misclassified['true_labels'])) 
                             if combined_misclassified['true_labels'][i] == 1 and combined_misclassified['pred_labels'][i] == 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_misclassified_metric_html = create_tooltip(
                "Total Misclassified",
                "Total number of samples that the AI model classified incorrectly during cross-validation testing with the current threshold settings.",
                "top"
            )
            st.markdown(f"**{total_misclassified_metric_html}**", unsafe_allow_html=True)
            st.metric("", total_misclassified)
        with col2:
            false_positives_metric_html = create_tooltip(
                "False Positives",
                "Samples that are actually normal but were incorrectly classified as defective. These represent unnecessary rejections in manufacturing.",
                "top"
            )
            st.markdown(f"**{false_positives_metric_html}**", unsafe_allow_html=True)
            st.metric("", false_positives)
        with col3:
            false_negatives_metric_html = create_tooltip(
                "False Negatives",
                "Samples that are actually defective but were incorrectly classified as normal. These represent quality escapes in manufacturing.",
                "top"
            )
            st.markdown(f"**{false_negatives_metric_html}**", unsafe_allow_html=True)
            st.metric("", false_negatives)
        
        st.info(f"Navigate through {total_misclassified} misclassified samples ordered by confidence (highest to lowest).")
        
        # Add note about potential small differences
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 0.8rem; border-radius: 6px; border-left: 3px solid #ff9800; margin: 0.5rem 0;">
            <small><strong>💡 Note:</strong> Small differences between breakdown cards and misclassified counts are normal. The breakdown shows simulated 10k points, while misclassified samples are from actual cross-validation data with current thresholds.</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation controls
        if st.session_state.misclassified_index >= len(sorted_indices):
            st.session_state.misclassified_index = 0
            
        nav_col2 = create_navigation_controls("misclassified", st.session_state.misclassified_index, len(sorted_indices))
        
        with nav_col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 0.5rem;'>
                <small style='color: #666;'>
                    Confidence: {combined_misclassified['confidences'][sorted_indices[st.session_state.misclassified_index]]:.4f} | 
                    Fold: {combined_misclassified['folds'][sorted_indices[st.session_state.misclassified_index]]}
                </small>
            </div>
            """, unsafe_allow_html=True)
        
        # Display current misclassified sample
        i = sorted_indices[st.session_state.misclassified_index]
        group_id = combined_misclassified['groups'][i]
        true_label = combined_misclassified['true_labels'][i]
        pred_label = combined_misclassified['pred_labels'][i]
        confidence = combined_misclassified['confidences'][i]
        fold = combined_misclassified['folds'][i]
        
        # Check if this sample has been relabeled or discarded
        is_relabeled = str(group_id) in st.session_state.relabeled_data
        is_discarded = str(group_id) in st.session_state.discarded_data
        relabel_value = st.session_state.relabeled_data.get(str(group_id)) if is_relabeled else None
        
        # Display sample info
        status = model_data['group_to_status'].get(group_id, "unknown")

        display_sample_info(
            group_id,
            true_label,
            pred_label,
            confidence,
            fold,
            is_relabeled,
            relabel_value,
            is_discarded,
            status=status
        )
        
        # Add relabeling and discard buttons
        display_relabel_buttons(group_id, fold, is_relabeled, is_discarded)
        
        # Plot the sequence
        fig = plot_sequence(
            raw_df, 
            group_id, 
            title=f"Group {group_id} (Fold {fold})",
            confidence=confidence
        )
        st.plotly_chart(fig, use_container_width=True, key=f"misclassified_{group_id}_{fold}")
    else:
        st.success("No misclassified samples found with current threshold settings.")
        
    

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.markdown("""
    **Troubleshooting:**
    - Ensure 'wave_dataset.parquet' exists
    - Run feature extraction script first
    - Check Python package installations
    """)