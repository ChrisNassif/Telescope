import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

def process_array_string(array_string):
    """Convert string representation of array to numpy array"""
    try:
        cleaned = array_string.strip('[]').replace('\n', ' ')
        values = [x.strip() for x in cleaned.split() if x.strip()]
        return np.array([float(x.rstrip(',')) for x in values])
    except Exception as e:
        print(f"Error processing array string: {e}")
        return np.array([])

def extract_fft_features(signal):
    """Extract FFT amplitude features from a signal"""
    # Compute FFT
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)[:len(signal)//2]  # Only take positive frequencies
    
    # Normalize magnitudes by signal length
    fft_magnitude = fft_magnitude / len(signal)
    
    return fft_magnitude

def main():
    file_path = 'experiment_results_per_token/smollm_360M_essay_dataset/raw_data.csv'
    print(f"Loading data from {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    
    # Process signals and extract features
    print("\nExtracting FFT features...")
    features = []
    labels = []
    signal_lengths = []
    
    for idx, row in df.iterrows():
        try:
            signal = process_array_string(row['telescope_perplexity_per_token'])
            if len(signal) > 0:
                signal_lengths.append(len(signal))
                features.append(extract_fft_features(signal))
                labels.append(row['y_labels'])
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Find median length for feature normalization
    median_length = int(np.median(signal_lengths))
    print(f"Median signal length: {median_length}")
    
    # Normalize feature vectors to same length
    X = []
    for feat in features:
        if len(feat) < median_length//2:
            # Pad with zeros
            X.append(np.pad(feat, (0, median_length//2 - len(feat))))
        else:
            # Truncate
            X.append(feat[:median_length//2])
    
    X = np.array(X)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("\nTraining classifier...")
    clf = LogisticRegression(max_iter=1000)
    
    # Perform cross-validation
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Train on full training set
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = clf.score(X_train_scaled, y_train)
    test_score = clf.score(X_test_scaled, y_test)
    
    print(f"\nTraining accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Print detailed classification report
    y_pred = clf.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot feature importance
    feature_importance = np.abs(clf.coef_[0])
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.fft.fftfreq(median_length)[:median_length//2], feature_importance)
    plt.title('FFT Amplitude Feature Importance')
    plt.xlabel('Frequency (cycles per token)')
    plt.ylabel('Absolute Coefficient Value')
    plt.tight_layout()
    
    os.makedirs('fft_plots', exist_ok=True)
    plt.savefig('fft_plots/feature_importance.png')
    plt.close()
    
    # Print top discriminative frequencies
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    freqs = np.fft.fftfreq(median_length)[:median_length//2]
    
    print("\nTop 10 discriminative frequencies:")
    for idx in top_indices:
        print(f"Frequency {freqs[idx]:.3f} cycles/token: importance = {feature_importance[idx]:.3f}")

if __name__ == "__main__":
    main()