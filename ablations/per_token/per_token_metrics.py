import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
plots_dir = 'per_token/per_token_plots'
os.makedirs(plots_dir, exist_ok=True)

def save_plot(fig, filename):
    """Save plot to the plots directory"""
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)

def is_valid_perplexity_data(data_string):
    """Check if the perplexity data string is valid (not truncated)"""
    return '...' not in data_string

def process_array_string(array_string):
    """Convert string representation of array to numpy array"""
    # Remove double brackets and clean string
    cleaned = array_string.strip('[]').replace('\n', ' ')
    # Split by spaces and remove empty strings
    values = [x.strip() for x in cleaned.split() if x.strip()]
    # Remove any trailing commas and convert to float
    return np.array([float(x.rstrip(',')) for x in values])

def analyze_perplexity_spikes(perplexity_values, prominence_threshold=0.5):
    """Analyzes spikes in perplexity scores"""
    # Basic statistics (always present)
    stats = {
        'sequence_length': len(perplexity_values),
        'mean_perplexity': np.mean(perplexity_values),
        'median_perplexity': np.median(perplexity_values),
        'std_perplexity': np.std(perplexity_values),
        'max_perplexity': np.max(perplexity_values),
        'min_perplexity': np.min(perplexity_values),
        'num_spikes': 0,
        'spike_density': 0,
        'mean_spike_height': 0,
        'mean_prominence': 0,
        'early_spikes': 0,
        'middle_spikes': 0,
        'late_spikes': 0,
        'mean_spike_distance': 0,
        'std_spike_distance': 0
    }
    
    # Find peaks
    peaks, properties = find_peaks(perplexity_values, prominence=prominence_threshold)
    
    # Update spike-related statistics if peaks exist
    if len(peaks) > 0:
        stats.update({
            'num_spikes': len(peaks),
            'spike_density': len(peaks) / len(perplexity_values),
            'mean_spike_height': np.mean(perplexity_values[peaks]),
            'mean_prominence': np.mean(properties['prominences'])
        })
        
        # Calculate spike positions
        relative_positions = peaks / len(perplexity_values)
        stats.update({
            'early_spikes': np.sum(relative_positions < 0.33),
            'middle_spikes': np.sum((relative_positions >= 0.33) & (relative_positions < 0.66)),
            'late_spikes': np.sum(relative_positions >= 0.66)
        })
        
        # Calculate inter-spike distances if more than one spike
        if len(peaks) > 1:
            spike_distances = np.diff(peaks)
            stats.update({
                'mean_spike_distance': np.mean(spike_distances),
                'std_spike_distance': np.std(spike_distances)
            })
    
    return stats, peaks, properties
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
plots_dir = 'per_token/per_token_plots'
os.makedirs(plots_dir, exist_ok=True)

def save_plot(fig, filename):
    """Save plot to the plots directory"""
    filepath = os.path.join(plots_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)

def is_valid_perplexity_data(data_string):
    """Check if the perplexity data string is valid (not truncated)"""
    return '...' not in data_string

def process_array_string(array_string):
    """Convert string representation of array to numpy array"""
    # Remove double brackets and clean string
    cleaned = array_string.strip('[]').replace('\n', ' ')
    # Split by spaces and remove empty strings
    values = [x.strip() for x in cleaned.split() if x.strip()]
    # Remove any trailing commas and convert to float
    return np.array([float(x.rstrip(',')) for x in values])

def analyze_perplexity_spikes(perplexity_values, prominence_threshold=0.5):
    """Analyzes spikes in perplexity scores"""
    # Basic statistics (always present)
    stats = {
        'sequence_length': len(perplexity_values),
        'mean_perplexity': np.mean(perplexity_values),
        'median_perplexity': np.median(perplexity_values),
        'std_perplexity': np.std(perplexity_values),
        'max_perplexity': np.max(perplexity_values),
        'min_perplexity': np.min(perplexity_values),
        'num_spikes': 0,
        'spike_density': 0,
        'mean_spike_height': 0,
        'mean_prominence': 0,
        'early_spikes': 0,
        'middle_spikes': 0,
        'late_spikes': 0,
        'mean_spike_distance': 0,
        'std_spike_distance': 0
    }
    
    # Find peaks
    peaks, properties = find_peaks(perplexity_values, prominence=prominence_threshold)
    
    # Update spike-related statistics if peaks exist
    if len(peaks) > 0:
        stats.update({
            'num_spikes': len(peaks),
            'spike_density': len(peaks) / len(perplexity_values),
            'mean_spike_height': np.mean(perplexity_values[peaks]),
            'mean_prominence': np.mean(properties['prominences'])
        })
        
        # Calculate spike positions
        relative_positions = peaks / len(perplexity_values)
        stats.update({
            'early_spikes': np.sum(relative_positions < 0.33),
            'middle_spikes': np.sum((relative_positions >= 0.33) & (relative_positions < 0.66)),
            'late_spikes': np.sum(relative_positions >= 0.66)
        })
        
        # Calculate inter-spike distances if more than one spike
        if len(peaks) > 1:
            spike_distances = np.diff(peaks)
            stats.update({
                'mean_spike_distance': np.mean(spike_distances),
                'std_spike_distance': np.std(spike_distances)
            })
    
    return stats, peaks, properties

def plot_perplexity_analysis(perplexity_data, peaks, label, threshold):
    """Create and save analysis plots"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Perplexity over token position with marked spikes
    ax1.plot(perplexity_data, label='Perplexity', alpha=0.7)
    ax1.scatter(peaks, perplexity_data[peaks], color='red', marker='x', s=100, label='Spikes')
    ax1.set_title(f'Perplexity Scores with Detected Spikes - Label {label}')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Perplexity')
    ax1.legend()
    
    # Plot 2: Distribution of spike heights
    if len(peaks) > 0:
        sns.histplot(perplexity_data[peaks], ax=ax2, bins=20)
        ax2.set_title(f'Distribution of Spike Heights - Label {label}')
        ax2.set_xlabel('Perplexity at Spike')
        ax2.set_ylabel('Count')
    
    plt.tight_layout()
    save_plot(fig, f'perplexity_spikes_label_{label}_threshold_{threshold}.png')
def plot_perplexity_analysis(perplexity_data, peaks, label, threshold):
    """Create and save analysis plots"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Perplexity over token position with marked spikes
    ax1.plot(perplexity_data, label='Perplexity', alpha=0.7)
    ax1.scatter(peaks, perplexity_data[peaks], color='red', marker='x', s=100, label='Spikes')
    ax1.set_title(f'Perplexity Scores with Detected Spikes - Label {label}')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Perplexity')
    ax1.legend()
    
    # Plot 2: Distribution of spike heights
    if len(peaks) > 0:
        sns.histplot(perplexity_data[peaks], ax=ax2, bins=20)
        ax2.set_title(f'Distribution of Spike Heights - Label {label}')
        ax2.set_xlabel('Perplexity at Spike')
        ax2.set_ylabel('Count')
    
    plt.tight_layout()
    save_plot(fig, f'perplexity_spikes_label_{label}_threshold_{threshold}.png')
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

    # Analyze each label separately
    thresholds = [0.5, 1.0, 2.0]
    labels = valid_data['y_labels'].unique()

    for threshold in thresholds:
        print(f"\nAnalysis with prominence threshold {threshold}:")
        print("=" * 80)
        
        label_stats = {}
        
        for label in labels:
            print(f"\nLabel: {label}")
            print("-" * 50)
            
            label_data = valid_data[valid_data['y_labels'] == label]
            print(f"Number of sequences for label {label}: {len(label_data)}")
            
            label_sequences_stats = []
            first_sequence_plotted = False
            
            for idx, row in label_data.iterrows():
                try:
                    perplexity_data = process_array_string(row['telescope_perplexity_per_token'])
                    if len(perplexity_data) > 0:
                        stats, peaks, _ = analyze_perplexity_spikes(perplexity_data, threshold)
                        label_sequences_stats.append(stats)
                        
                        # Plot first sequence for each label
                        if not first_sequence_plotted:
                            plot_perplexity_analysis(perplexity_data, peaks, label, threshold)
                            first_sequence_plotted = True
                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)[:100]}")
                    continue
            
            if not label_sequences_stats:
                print(f"No valid sequences found for label {label}")
                continue
            
            # Calculate and store averages
            avg_stats = {}
            for key in label_sequences_stats[0].keys():
                values = [s[key] for s in label_sequences_stats]
                avg_stats[key] = np.mean(values)
            
            label_stats[label] = avg_stats
            
            print(f"\nSequences analyzed: {len(label_sequences_stats)}")
            print(f"\nBasic Perplexity Statistics:")
            print(f"  Mean perplexity: {avg_stats['mean_perplexity']:.3f}")
            print(f"  Median perplexity: {avg_stats['median_perplexity']:.3f}")
            print(f"  Std perplexity: {avg_stats['std_perplexity']:.3f}")
            print(f"  Max perplexity: {avg_stats['max_perplexity']:.3f}")
            print(f"  Min perplexity: {avg_stats['min_perplexity']:.3f}")
            
            print(f"\nSpike Statistics:")
            print(f"  Number of spikes: {avg_stats['num_spikes']:.2f}")
            print(f"  Spike density: {avg_stats['spike_density']:.3f} spikes per token")
            print(f"  Mean spike height: {avg_stats['mean_spike_height']:.3f}")
            print(f"  Mean prominence: {avg_stats['mean_prominence']:.3f}")
            
            print("\nSpike Location Distribution (averages):")
            print(f"  Early sequence: {avg_stats['early_spikes']:.2f} spikes")
            print(f"  Middle sequence: {avg_stats['middle_spikes']:.2f} spikes")
            print(f"  Late sequence: {avg_stats['late_spikes']:.2f} spikes")
            
            if avg_stats['mean_spike_distance'] > 0:
                print(f"\nSpike Spacing:")
                print(f"  Average distance between spikes: {avg_stats['mean_spike_distance']:.2f} tokens")
                print(f"  Std dev of distances between spikes: {avg_stats['std_spike_distance']:.2f} tokens")

        # Create comparison plots
        if len(label_stats) > 1:
            # Plot perplexity comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ['mean_perplexity', 'spike_density', 'mean_spike_height']
            labels_list = list(label_stats.keys())
            x = np.arange(len(metrics))
            width = 0.35
            
            for i, label in enumerate(labels_list):
                values = [label_stats[label][metric] for metric in metrics]
                ax.bar(x + i*width, values, width, label=f'Label {label}')
            
            ax.set_ylabel('Value')
            ax.set_title(f'Comparison of Metrics (Threshold {threshold})')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(metrics)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_plot(fig, f'label_comparison_threshold_{threshold}.png')

    print(f"\nPlots have been saved to the '{plots_dir}' directory.")
def analyze_bidirectional_spikes(valid_data, threshold=0.5):
    """
    Analyze both upward and downward spikes in perplexity values.
    """
    # Store spike data for each label
    label_data = {0: {'up_spikes': [], 'down_spikes': [], 'sequences': []},
                 1: {'up_spikes': [], 'down_spikes': [], 'sequences': []}}
    
    for label in [0, 1]:
        label_sequences = valid_data[valid_data['y_labels'] == label]
        
        for _, row in label_sequences.iterrows():
            try:
                perplexity_data = process_array_string(row['telescope_perplexity_per_token'])
                if len(perplexity_data) > 0:
                    # Find upward spikes
                    up_peaks, _ = find_peaks(perplexity_data, prominence=threshold)
                    
                    # Find downward spikes (invert the signal)
                    down_peaks, _ = find_peaks(-perplexity_data, prominence=threshold)
                    
                    if len(up_peaks) > 0:
                        label_data[label]['up_spikes'].extend(perplexity_data[up_peaks])
                    if len(down_peaks) > 0:
                        label_data[label]['down_spikes'].extend(perplexity_data[down_peaks])
                    
                    # Store full sequence
                    label_data[label]['sequences'].extend(perplexity_data)
                    
            except Exception as e:
                continue
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prepare data for plotting
    plot_data = {'Label': [], 'Magnitude': [], 'Direction': []}
    
    for label in [0, 1]:
        up_spikes = np.array(label_data[label]['up_spikes'])
        down_spikes = np.array(label_data[label]['down_spikes'])
        median_perp = np.median(label_data[label]['sequences'])
        
        # Add upward spikes
        if len(up_spikes) > 0:
            magnitudes = up_spikes - median_perp
            plot_data['Label'].extend([f'Label {label}'] * len(magnitudes))
            plot_data['Magnitude'].extend(magnitudes)
            plot_data['Direction'].extend(['Up'] * len(magnitudes))
            
        # Add downward spikes
        if len(down_spikes) > 0:
            magnitudes = median_perp - down_spikes
            plot_data['Label'].extend([f'Label {label}'] * len(magnitudes))
            plot_data['Magnitude'].extend(magnitudes)
            plot_data['Direction'].extend(['Down'] * len(magnitudes))
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Plot upward spikes
    up_data = df[df['Direction'] == 'Up']
    if not up_data.empty:
        sns.boxplot(data=up_data, x='Label', y='Magnitude', ax=ax1, showfliers=False)
        ax1.set_title('Upward Spike Magnitudes\n(relative to median)')
        ax1.set_ylabel('Magnitude above median')
    
    # Plot downward spikes
    down_data = df[df['Direction'] == 'Down']
    if not down_data.empty:
        sns.boxplot(data=down_data, x='Label', y='Magnitude', ax=ax2, showfliers=False)
        ax2.set_title('Downward Spike Magnitudes\n(relative to median)')
        ax2.set_ylabel('Magnitude below median')
    
    # Plot histograms
    if not up_data.empty:
        for label in [0, 1]:
            label_up = up_data[up_data['Label'] == f'Label {label}']
            if not label_up.empty:
                sns.histplot(data=label_up, x='Magnitude', bins=50, alpha=0.5, 
                           label=f'Label {label}', ax=ax3)
        ax3.set_yscale('log')
        ax3.set_title('Distribution of Upward Spike Magnitudes')
        ax3.set_xlabel('Magnitude above median')
        ax3.set_ylabel('Count (log scale)')
        ax3.legend()
    
    if not down_data.empty:
        for label in [0, 1]:
            label_down = down_data[down_data['Label'] == f'Label {label}']
            if not label_down.empty:
                sns.histplot(data=label_down, x='Magnitude', bins=50, alpha=0.5, 
                           label=f'Label {label}', ax=ax4)
        ax4.set_yscale('log')
        ax4.set_title('Distribution of Downward Spike Magnitudes')
        ax4.set_xlabel('Magnitude below median')
        ax4.set_ylabel('Count (log scale)')
        ax4.legend()
    
    plt.tight_layout()
    save_plot(fig, f'bidirectional_spike_analysis_threshold_{threshold}.png')
    
    # Print statistics
    print("\nBidirectional Spike Analysis:")
    for label in [0, 1]:
        sequences = np.array(label_data[label]['sequences'])
        up_spikes = np.array(label_data[label]['up_spikes'])
        down_spikes = np.array(label_data[label]['down_spikes'])
        median_perp = np.median(sequences)
        
        print(f"\nLabel {label}:")
        print(f"  Total sequences analyzed: {len(sequences)}")
        print(f"  Median perplexity: {median_perp:.3f}")
        
        if len(up_spikes) > 0:
            print("\n  Upward Spikes:")
            print(f"    Count: {len(up_spikes)}")
            print(f"    Mean magnitude above median: {np.mean(up_spikes - median_perp):.3f}")
            print(f"    Max magnitude above median: {np.max(up_spikes - median_perp):.3f}")
        
        if len(down_spikes) > 0:
            print("\n  Downward Spikes:")
            print(f"    Count: {len(down_spikes)}")
            print(f"    Mean magnitude below median: {np.mean(median_perp - down_spikes):.3f}")
            print(f"    Max magnitude below median: {np.max(median_perp - down_spikes):.3f}")

def analyze_perplexity_distributions(valid_data):
    """
    Analyze whether Label 1's higher average comes from consistently higher scores
    or from extreme spikes.
    """
    # Store perplexity values for each label
    label_perplexities = {0: [], 1: []}
    
    # Collect all perplexity values for each label
    for label in [0, 1]:
        label_data = valid_data[valid_data['y_labels'] == label]
        
        for _, row in label_data.iterrows():
            try:
                perplexity_data = process_array_string(row['telescope_perplexity_per_token'])
                if len(perplexity_data) > 0:
                    label_perplexities[label].extend(perplexity_data)
            except Exception as e:
                continue
    
    # Convert to numpy arrays for analysis
    perp_0 = np.array(label_perplexities[0])
    perp_1 = np.array(label_perplexities[1])
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Box plot (without fliers to focus on main distribution)
    plot_data = []
    labels = []
    for label in [0, 1]:
        plot_data.extend(label_perplexities[label])
        labels.extend([f'Label {label}'] * len(label_perplexities[label]))
    
    sns.boxplot(x=labels, y=plot_data, ax=ax1, showfliers=False)
    ax1.set_title('Main Distribution of Perplexity Values\n(excluding outliers)')
    ax1.set_ylabel('Perplexity')
    
    # Plot 2: Histograms overlaid (log scale y-axis)
    sns.histplot(perp_0, bins=100, alpha=0.5, label='Label 0', ax=ax2)
    sns.histplot(perp_1, bins=100, alpha=0.5, label='Label 1', ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title('Histogram of Perplexity Values\n(log scale)')
    ax2.set_xlabel('Perplexity')
    ax2.set_ylabel('Count (log scale)')
    ax2.legend()
    
    # Plot 3: Cumulative distribution
    sorted_0 = np.sort(perp_0)
    sorted_1 = np.sort(perp_1)
    cumulative_0 = np.arange(1, len(sorted_0) + 1) / len(sorted_0)
    cumulative_1 = np.arange(1, len(sorted_1) + 1) / len(sorted_1)
    
    ax3.plot(sorted_0, cumulative_0, label='Label 0', alpha=0.7)
    ax3.plot(sorted_1, cumulative_1, label='Label 1', alpha=0.7)
    ax3.set_title('Cumulative Distribution')
    ax3.set_xlabel('Perplexity')
    ax3.set_ylabel('Cumulative Proportion')
    ax3.legend()
    
    # Plot 4: Zoom in on the tail (top 10%)
    idx_0 = int(0.9 * len(sorted_0))
    idx_1 = int(0.9 * len(sorted_1))
    
    ax4.plot(sorted_0[idx_0:], cumulative_0[idx_0:], label='Label 0', alpha=0.7)
    ax4.plot(sorted_1[idx_1:], cumulative_1[idx_1:], label='Label 1', alpha=0.7)
    ax4.set_title('Cumulative Distribution (Top 10%)')
    ax4.set_xlabel('Perplexity')
    ax4.set_ylabel('Cumulative Proportion')
    ax4.legend()
    
    plt.tight_layout()
    save_plot(fig, 'perplexity_distribution_analysis.png')
    
    # Calculate statistics
    print("\nDistribution Analysis:")
    for label, perp in [(0, perp_0), (1, perp_1)]:
        print(f"\nLabel {label}:")
        print(f"  Number of tokens: {len(perp)}")
        print(f"  Mean perplexity: {np.mean(perp):.3f}")
        print(f"  Median perplexity: {np.median(perp):.3f}")
        print(f"  75th percentile: {np.percentile(perp, 75):.3f}")
        print(f"  90th percentile: {np.percentile(perp, 90):.3f}")
        print(f"  95th percentile: {np.percentile(perp, 95):.3f}")
        print(f"  99th percentile: {np.percentile(perp, 99):.3f}")
        print(f"  Max perplexity: {np.max(perp):.3f}")
        
    # Calculate percentage of values above various thresholds
    thresholds = [np.mean(perp_0), np.percentile(perp_0, 75), np.percentile(perp_0, 90)]
    print("\nPercentage of values above thresholds:")
    for threshold in thresholds:
        pct_0 = np.mean(perp_0 > threshold) * 100
        pct_1 = np.mean(perp_1 > threshold) * 100
        print(f"\nThreshold {threshold:.3f}:")
        print(f"  Label 0: {pct_0:.1f}%")
        print(f"  Label 1: {pct_1:.1f}%")
        print(f"  Ratio (Label 1 / Label 0): {pct_1/pct_0:.2f}x")

    # Add specific analysis of the main body vs tails
    print("\nAnalysis of distribution segments:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        p0 = np.percentile(perp_0, p)
        p1 = np.percentile(perp_1, p)
        ratio = p1/p0
        print(f"\n{p}th percentile:")
        print(f"  Label 0: {p0:.3f}")
        print(f"  Label 1: {p1:.3f}")
        print(f"  Ratio (Label 1 / Label 0): {ratio:.2f}x")
if __name__ == "__main__":
    main()

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

    # Analyze each label separately
    thresholds = [10,12.5, 15,20]
    labels = valid_data['y_labels'].unique()

    for threshold in thresholds:
        print(f"\nAnalysis with prominence threshold {threshold}:")
        print("=" * 80)
        
        label_stats = {}
        
        for label in labels:
            print(f"\nLabel: {label}")
            print("-" * 50)
            
            label_data = valid_data[valid_data['y_labels'] == label]
            print(f"Number of sequences for label {label}: {len(label_data)}")
            
            label_sequences_stats = []
            first_sequence_plotted = False
            
            for idx, row in label_data.iterrows():
                try:
                    perplexity_data = process_array_string(row['telescope_perplexity_per_token'])
                    if len(perplexity_data) > 0:
                        stats, peaks, _ = analyze_perplexity_spikes(perplexity_data, threshold)
                        label_sequences_stats.append(stats)
                        
                        # Plot first sequence for each label
                        if not first_sequence_plotted:
                            plot_perplexity_analysis(perplexity_data, peaks, label, threshold)
                            first_sequence_plotted = True
                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)[:100]}")
                    continue
            
            if not label_sequences_stats:
                print(f"No valid sequences found for label {label}")
                continue
            
            # Calculate and store averages
            avg_stats = {}
            for key in label_sequences_stats[0].keys():
                values = [s[key] for s in label_sequences_stats]
                avg_stats[key] = np.mean(values)
            
            label_stats[label] = avg_stats
            
            print(f"\nSequences analyzed: {len(label_sequences_stats)}")
            print(f"\nBasic Perplexity Statistics:")
            print(f"  Mean perplexity: {avg_stats['mean_perplexity']:.3f}")
            print(f"  Median perplexity: {avg_stats['median_perplexity']:.3f}")
            print(f"  Std perplexity: {avg_stats['std_perplexity']:.3f}")
            print(f"  Max perplexity: {avg_stats['max_perplexity']:.3f}")
            print(f"  Min perplexity: {avg_stats['min_perplexity']:.3f}")
            
            print(f"\nSpike Statistics:")
            print(f"  Number of spikes: {avg_stats['num_spikes']:.2f}")
            print(f"  Spike density: {avg_stats['spike_density']:.3f} spikes per token")
            print(f"  Mean spike height: {avg_stats['mean_spike_height']:.3f}")
            print(f"  Mean prominence: {avg_stats['mean_prominence']:.3f}")
            
            print("\nSpike Location Distribution (averages):")
            print(f"  Early sequence: {avg_stats['early_spikes']:.2f} spikes")
            print(f"  Middle sequence: {avg_stats['middle_spikes']:.2f} spikes")
            print(f"  Late sequence: {avg_stats['late_spikes']:.2f} spikes")
            
            if avg_stats['mean_spike_distance'] > 0:
                print(f"\nSpike Spacing:")
                print(f"  Average distance between spikes: {avg_stats['mean_spike_distance']:.2f} tokens")
                print(f"  Std dev of distances between spikes: {avg_stats['std_spike_distance']:.2f} tokens")

        # Create comparison plots
        if len(label_stats) > 1:
            # Plot perplexity comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ['mean_perplexity', 'spike_density', 'mean_spike_height']
            labels_list = list(label_stats.keys())
            x = np.arange(len(metrics))
            width = 0.35
            
            for i, label in enumerate(labels_list):
                values = [label_stats[label][metric] for metric in metrics]
                ax.bar(x + i*width, values, width, label=f'Label {label}')
            
            ax.set_ylabel('Value')
            ax.set_title(f'Comparison of Metrics (Threshold {threshold})')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(metrics)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_plot(fig, f'label_comparison_threshold_{threshold}.png')
    analyze_bidirectional_spikes(valid_data, threshold)
    analyze_perplexity_distributions(valid_data)
    print(f"\nPlots have been saved to the '{plots_dir}' directory.")

if __name__ == "__main__":
    main()