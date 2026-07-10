import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
plots_dir = 'per_token_plots'
os.makedirs(plots_dir, exist_ok=True)

def save_plot(fig, filename):
    """Save plot to the plots directory"""
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)

def process_array_string(array_string):
    """Convert string representation of array to numpy array"""
    cleaned = array_string.strip('[]').replace('\n', ' ')
    values = [x.strip() for x in cleaned.split() if x.strip()]
    return np.array([float(x.rstrip(',')) for x in values])

def is_valid_perplexity_data(data_string):
    """Check if the perplexity data string is valid (not truncated)"""
    return '...' not in data_string

def analyze_signal_frequency(valid_data):
    """
    Analyze the frequency components of perplexity signals for each label.
    """
    label_data = {0: [], 1: []}
    sequence_lengths = []
    
    # First pass: calculate sequence lengths and overall statistics
    all_perplexities = []
    for _, row in valid_data.iterrows():
        try:
            perp_data = process_array_string(row['telescope_perplexity_per_token'])
            sequence_lengths.append(len(perp_data))
            all_perplexities.extend(perp_data)
        except:
            continue
    
    all_perplexities = np.array(all_perplexities)
    print(f"\nPerplexity Statistics:")
    print(f"Maximum: {np.max(all_perplexities):.2f}")
    print(f"Mean: {np.mean(all_perplexities):.2f}")
    print(f"Std Dev: {np.std(all_perplexities):.2f}")
    
    median_length = int(np.median(sequence_lengths))
    sample_rate = 1.0  # 1 sample/token
    
    # Collect and process sequences
    for label in [0, 1]:
        label_sequences = valid_data[valid_data['y_labels'] == label]
        
        for _, row in label_sequences.iterrows():
            try:
                perp_data = process_array_string(row['telescope_perplexity_per_token'])
                
                # Pad or truncate to median length
                if len(perp_data) < median_length:
                    perp_data = np.pad(perp_data, (0, median_length - len(perp_data)), 'constant', constant_values=np.mean(perp_data))
                elif len(perp_data) > median_length:
                    perp_data = perp_data[:median_length]
                
                # Apply window function to reduce spectral leakage
                window = np.hanning(len(perp_data))
                windowed_data = perp_data * window
                
                # Calculate FFT
                fft_result = np.fft.rfft(windowed_data)
                fft_magnitude = np.abs(fft_result) / (median_length/2)
                
                label_data[label].append(fft_magnitude)
                
            except Exception as e:
                continue
    
    # Calculate average frequency spectrum for each label
    avg_spectrum = {
        label: np.mean(np.array(spectra), axis=0) 
        for label, spectra in label_data.items()
    }
    
    freqs = np.fft.rfftfreq(median_length, d=1/sample_rate)
    
    # Define frequency bands
    low_freq_idx = freqs <= 0.1  # frequencies corresponding to periods of 10 tokens or more
    high_freq_idx = freqs > 0.1
    
    # Create visualization with four subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Full spectrum (linear scale)
    ax1 = plt.subplot(2, 2, 1)
    for label in [0, 1]:
        spectrum = avg_spectrum[label]
        ax1.plot(freqs, spectrum, label=f'Label {label}', alpha=0.7)
    ax1.set_title('Full Frequency Spectrum (Linear Scale)')
    ax1.set_xlabel('Frequency (cycles per token)')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Low frequency detail
    ax2 = plt.subplot(2, 2, 2)
    for label in [0, 1]:
        spectrum = avg_spectrum[label]
        ax2.plot(freqs[low_freq_idx], spectrum[low_freq_idx], label=f'Label {label}', alpha=0.7)
    ax2.set_title('Low Frequency Detail (≤ 0.1 cycles/token)')
    ax2.set_xlabel('Frequency (cycles per token)')
    ax2.set_ylabel('Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: High frequency detail
    ax3 = plt.subplot(2, 2, 3)
    for label in [0, 1]:
        spectrum = avg_spectrum[label]
        ax3.plot(freqs[high_freq_idx], spectrum[high_freq_idx], label=f'Label {label}', alpha=0.7)
    ax3.set_title('High Frequency Detail (> 0.1 cycles/token)')
    ax3.set_xlabel('Frequency (cycles per token)')
    ax3.set_ylabel('Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Log scale full spectrum
    ax4 = plt.subplot(2, 2, 4)
    for label in [0, 1]:
        spectrum = avg_spectrum[label]
        ax4.semilogy(freqs, spectrum + 1e-10, label=f'Label {label}', alpha=0.7)
    ax4.set_title('Full Frequency Spectrum (Log Scale)')
    ax4.set_xlabel('Frequency (cycles per token)')
    ax4.set_ylabel('Magnitude (log scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'perplexity_frequency_bands.png')
    
    # Print band analysis
    print("\nFrequency Band Analysis:")
    for label in [0, 1]:
        spectrum = avg_spectrum[label]
        
        # Calculate band statistics
        low_freq_spectrum = spectrum[low_freq_idx]
        high_freq_spectrum = spectrum[high_freq_idx]
        
        print(f"\nLabel {label}:")
        print(f"  Number of sequences: {len(label_data[label])}")
        print("\n  Low Frequency Band (≤ 0.1 cycles/token):")
        print(f"    Maximum magnitude: {np.max(low_freq_spectrum):.4f}")
        print(f"    Mean magnitude: {np.mean(low_freq_spectrum):.4f}")
        print(f"    Std Dev magnitude: {np.std(low_freq_spectrum):.4f}")
        print(f"    Total energy: {np.sum(low_freq_spectrum**2):.4f}")
        
        print("\n  High Frequency Band (> 0.1 cycles/token):")
        print(f"    Maximum magnitude: {np.max(high_freq_spectrum):.4f}")
        print(f"    Mean magnitude: {np.mean(high_freq_spectrum):.4f}")
        print(f"    Std Dev magnitude: {np.std(high_freq_spectrum):.4f}")
        print(f"    Total energy: {np.sum(high_freq_spectrum**2):.4f}")
        
        # Find the frequency with maximum magnitude in each band
        if len(low_freq_spectrum) > 0:
            max_low_freq_idx = np.argmax(low_freq_spectrum)
            max_low_freq = freqs[low_freq_idx][max_low_freq_idx]
            print(f"\n  Strongest low frequency: {max_low_freq:.4f} cycles/token")
            print(f"  (Period: {1/max_low_freq:.1f} tokens)")
        
        if len(high_freq_spectrum) > 0:
            max_high_freq_idx = np.argmax(high_freq_spectrum)
            max_high_freq = freqs[high_freq_idx][max_high_freq_idx]
            print(f"  Strongest high frequency: {max_high_freq:.4f} cycles/token")
            print(f"  (Period: {1/max_high_freq:.1f} tokens)")

    return median_length, avg_spectrum, freqs

def main():
    # Load and filter data
    file_path = 'experiment_results_per_token/smollm_360M_ghostbusters_essay_deepseek_dataset/raw_data.csv'
    df = pd.read_csv(file_path)
    valid_data = df[df['telescope_perplexity_per_token'].apply(is_valid_perplexity_data)].copy()

    print("\nDataset Summary:")
    print(f"Total sequences: {len(df)}")
    print(f"Valid sequences: {len(valid_data)} ({len(valid_data)/len(df)*100:.1f}%)")
    print("\nLabel distribution in valid data:")
    print(valid_data['y_labels'].value_counts())

    # Perform frequency analysis
    median_length, avg_spectrum, freqs = analyze_signal_frequency(valid_data)

    print(f"\nPlots have been saved to the '{plots_dir}' directory.")

if __name__ == "__main__":
    main()