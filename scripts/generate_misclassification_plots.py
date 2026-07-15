import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import yaml

PLOT_COLORS: List[str] = yaml.safe_load(open("config.yaml"))["plot_colors"]

def specialized_error_analysis(experiment_name: str) -> Optional[pd.DataFrame]:
    try:
        # Load the data
        df: pd.DataFrame = pd.read_csv(f"experiment_results/{experiment_name}/raw_data.csv")
        
        # Set global font size for all matplotlib plots
        plt.rcParams.update({
            'font.size': 32, 'axes.labelsize': 32, 'axes.titlesize': 28,
            'xtick.labelsize': 32, 'ytick.labelsize': 32,
            'legend.fontsize': 32, 'figure.titlesize': 28
        })
        
        # Create directory structure
        analyses_dir: str = "experiment_analyses"
        plots_dir: str = f"{analyses_dir}/misclassification_plots"
        experiment_plots_dir: str = f"{plots_dir}/{experiment_name}"
        
        # Create directories if they don't exist
        os.makedirs(analyses_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(experiment_plots_dir, exist_ok=True)
        
        # We'll assume you already have basic metrics, ROC curves, calibration plots
        print(f"\nSpecialized Error Analysis for {experiment_name} - Telescope Perplexity")
        
        # Determine optimal threshold based on your existing metrics
        # For this analysis, we'll use the median as an example
        threshold: float = float(df['telescope_perplexity'].median())
        df['predicted'] = (df['telescope_perplexity'] > threshold).astype(int)
        df['correct'] = (df['predicted'] == df['y_labels']).astype(int)
        
        # Add human-readable labels for clarity in plots
        df['prediction_label'] = df['predicted'].map({1: 'AI-generated', 0: 'Human-written'})
        df['actual_label'] = df['y_labels'].map({1: 'AI-generated', 0: 'Human-written'})
        df['correctness'] = df['correct'].map({1: 'Correct', 0: 'Incorrect'})
        
        # 1. Error magnitude distribution using standardized distances (z-scores)
        # Calculate raw error distance first
        df['error_distance_raw'] = np.abs(df['telescope_perplexity'] - threshold)
        
        # Calculate standard deviation of perplexity values
        perplexity_std: float = float(df['telescope_perplexity'].std())
        
        # Convert to standardized error distance (in number of standard deviations)
        df['error_distance'] = df['error_distance_raw'] / perplexity_std
        
        # Get misclassified examples
        misclassified: pd.DataFrame = df[df['correct'] == 0]
        max_error_distance: float = float(misclassified['error_distance'].max()) if len(misclassified) > 0 else 0.0
        
        # Format the max error distance consistently for all plots
        max_error_distance_str: str = f"{max_error_distance:.2f}"
        
        # Compare error distances between correctly and incorrectly classified examples
        plt.figure(figsize=(16, 10))
        
        # Make sure the actual_label column is properly created before using it
        df['actual_label'] = df['y_labels'].map({1: 'AI-generated', 0: 'Human-written'})
        
        # Create custom color palette for better visibility
        custom_palette: Dict[str, str] = {'AI-generated': PLOT_COLORS[3], 'Human-written': PLOT_COLORS[0]}
        
        # Create the histogram manually to ensure legend works properly
        label: str
        color: str
        for label, color in custom_palette.items():
            subset: pd.DataFrame = df[df['actual_label'] == label]
            sns.histplot(data=subset, x='error_distance', color=color, bins=20, kde=True, alpha=0.7, label=label)
        
        # Set all font sizes
        plt.title(f'Error Distance\n{experiment_name}', fontsize=28)  # Clean title without max distance
        plt.xlabel('Distance from Decision Threshold (in std devs)', fontsize=32)
        plt.ylabel('Count', fontsize=32)
        
        # Increase tick font sizes
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        
        # Create explicit legend with the correct text labels
        plt.legend(title='Text Type', title_fontsize=32, fontsize=32, loc='upper right', frameon=True, framealpha=0.9)
        
        # Adjust layout to make room for the larger text
        plt.tight_layout()
        plt.savefig(f"{experiment_plots_dir}/error_distance.png", dpi=300)
        
        # 2. Comparative percentile analysis
        # Check how telescope_perplexity values are distributed within their actual classes
        df_pos: pd.DataFrame = df[df['y_labels'] == 1]
        df_neg: pd.DataFrame = df[df['y_labels'] == 0]
        
        # Calculate percentile rank of each value within its actual class
        if len(df_pos) > 0:
            df.loc[df['y_labels'] == 1, 'class_percentile'] = df_pos['telescope_perplexity'].rank(pct=True) * 100
        if len(df_neg) > 0:
            df.loc[df['y_labels'] == 0, 'class_percentile'] = df_neg['telescope_perplexity'].rank(pct=True) * 100
        
        # Identify outliers within each class (examples that don't fit with their class)
        class_outliers: pd.DataFrame = df[((df['y_labels'] == 1) & (df['class_percentile'] < 10)) |
                                           ((df['y_labels'] == 0) & (df['class_percentile'] > 90))]
        
        print(f"\nFound {len(class_outliers)} class outliers (examples with unexpected perplexity for their class)")
        
        # Save class outliers for further inspection
        class_outliers.to_csv(f"{experiment_plots_dir}/class_outliers.csv", index=False)
        
        # 3. Error clusters by perplexity value
        plt.figure(figsize=(16, 10))
        
        # Create scatter plot of telescope_perplexity values with AI/Human labels
        scatter: Any = plt.scatter(df.index, df['telescope_perplexity'], c=df['y_labels'], cmap='coolwarm', alpha=0.8, s=100)
        
        plt.axhline(y=threshold, color='black', linestyle='--', linewidth=3, label='Threshold')
        
        # Add labels with larger font sizes
        plt.xlabel('Example Index', fontsize=32)
        plt.ylabel('Telescope Perplexity', fontsize=32)
        plt.title(f'Telescope Perplexity: AI vs Human\n{experiment_name}', fontsize=28)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements: List[Line2D] = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=PLOT_COLORS[3], markersize=20, label='AI-generated'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=PLOT_COLORS[0], markersize=20, label='Human-written'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=3, label='Threshold')
        ]
        
        plt.legend(handles=legend_elements, fontsize=32)
        
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        
        plt.tight_layout()
        plt.savefig(f"{experiment_plots_dir}/error_distribution.png", dpi=300)
        
        # 4. Content-based error analysis (if text column exists)
        if 'text' in df.columns:
            incorrect_df: pd.DataFrame = df[df['correct'] == 0]
            
            # Text length analysis for errors
            if len(incorrect_df) > 0:
                incorrect_df['text_length'] = incorrect_df['text'].apply(len)
                incorrect_df['word_count'] = incorrect_df['text'].apply(lambda x: len(str(x).split()))
                
                # Group by error type (FP vs FN)
                fps: pd.DataFrame = incorrect_df[incorrect_df['predicted'] == 1]
                fns: pd.DataFrame = incorrect_df[incorrect_df['predicted'] == 0]
                
                # Compare text length distributions
                plt.figure(figsize=(16, 10))
                if len(fps) > 0 and len(fns) > 0:
                    # Use larger linewidth for better visibility
                    sns.kdeplot(fps['text_length'], label='AI-generated (False Positives)', linewidth=4, color=PLOT_COLORS[3])
                    sns.kdeplot(fns['text_length'], label='Human-written (False Negatives)', linewidth=4, color=PLOT_COLORS[0])
                    
                    # Set larger font sizes
                    plt.xlabel('Text Length', fontsize=32)
                    plt.ylabel('Density', fontsize=32)
                    plt.title(f'Text Length Distribution by Text Type\n{experiment_name}', fontsize=28)
                    
                    # Increase legend and tick font sizes
                    plt.legend(fontsize=32)
                    plt.xticks(fontsize=32)
                    plt.yticks(fontsize=32)
                    
                    plt.tight_layout()
                    plt.savefig(f"{experiment_plots_dir}/error_length_distribution.png", dpi=300)
                
                # Basic content analysis - most common words in errors
                if len(incorrect_df) >= 10:  # Need enough samples for meaningful analysis
                    try:
                        # Extract common words from error cases
                        vectorizer: CountVectorizer = CountVectorizer(stop_words='english', max_features=100)
                        X: Any = vectorizer.fit_transform(incorrect_df['text'].fillna(''))
                        
                        # Get most common words
                        word_counts: np.ndarray = np.sum(X.toarray(), axis=0)
                        words: np.ndarray = vectorizer.get_feature_names_out()
                        word_freq: Dict[str, float] = dict(zip(words, word_counts))
                        
                        # Plot top words in errors
                        plt.figure(figsize=(18, 14))
                        top_words: Dict[str, float] = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:25])
                        
                        # Create horizontal bar plot with custom styling
                        bars: Any = plt.barh(list(top_words.keys()), list(top_words.values()), 
                                       color=PLOT_COLORS[0], edgecolor='black', linewidth=1.5, alpha=0.8)
                        
                        # Add value labels to each bar
                        bar: Any
                        for bar in bars:
                            width: float = float(bar.get_width())
                            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                                    f'{width:.0f}', ha='left', va='center', fontsize=28)
                        
                        # Set larger font sizes
                        plt.title(f'Most Common Words in Misclassified Examples\n{experiment_name}', 
                                 fontsize=28)  # Clean title without max distance
                        plt.xlabel('Frequency', fontsize=32)
                        plt.ylabel('Words', fontsize=32)
                        
                        # Increase tick font sizes
                        plt.xticks(fontsize=28)
                        plt.yticks(fontsize=28)
                        
                        plt.tight_layout()
                        plt.savefig(f"{experiment_plots_dir}/error_common_words.png", dpi=300)
                        
                        # Separate analysis for FP and FN if enough examples
                        if len(fps) >= 5 and len(fns) >= 5:
                            # Extract words from FPs
                            fp_vectorizer: CountVectorizer = CountVectorizer(stop_words='english', max_features=50)
                            fp_X: Any = fp_vectorizer.fit_transform(fps['text'].fillna(''))
                            fp_words: np.ndarray = fp_vectorizer.get_feature_names_out()
                            fp_counts: np.ndarray = np.sum(fp_X.toarray(), axis=0)
                            fp_word_freq: Dict[str, float] = dict(zip(fp_words, fp_counts))
                            
                            # Extract words from FNs
                            fn_vectorizer: CountVectorizer = CountVectorizer(stop_words='english', max_features=50)
                            fn_X: Any = fn_vectorizer.fit_transform(fns['text'].fillna(''))
                            fn_words: np.ndarray = fn_vectorizer.get_feature_names_out()
                            fn_counts: np.ndarray = np.sum(fn_X.toarray(), axis=0)
                            fn_word_freq: Dict[str, float] = dict(zip(fn_words, fn_counts))
                            
                            # Find distinctive words for each error type
                            all_words: Set[str] = set(fp_words).union(set(fn_words))
                            fp_distinctive: Dict[str, float] = {}
                            fn_distinctive: Dict[str, float] = {}
                            
                            word: str
                            for word in all_words:
                                fp_freq: float = fp_word_freq.get(word, 0) / len(fps)
                                fn_freq: float = fn_word_freq.get(word, 0) / len(fns)
                                
                                if fp_freq > 2 * fn_freq and fp_freq > 0.1:
                                    fp_distinctive[word] = fp_freq
                                if fn_freq > 2 * fp_freq and fn_freq > 0.1:
                                    fn_distinctive[word] = fn_freq
                            
                            # Save distinctive words
                            with open(f"{experiment_plots_dir}/distinctive_words.txt", 'w') as f:
                                f.write("Words distinctive of False Positives:\n")
                                for word, freq in sorted(fp_distinctive.items(), key=lambda x: x[1], reverse=True):
                                    f.write(f"{word}: {freq:.3f}\n")
                                
                                f.write("\nWords distinctive of False Negatives:\n")
                                for word, freq in sorted(fn_distinctive.items(), key=lambda x: x[1], reverse=True):
                                    f.write(f"{word}: {freq:.3f}\n")
                    except Exception as e:
                        print(f"Error in text analysis: {e}")
            
            # 5. Find ambiguous boundary examples
            boundary_width: float = 0.25  # Now in units of standard deviations
            boundary_examples: pd.DataFrame = df[(df['error_distance'] < boundary_width)]
            
            print(f"\nFound {len(boundary_examples)} examples near decision boundary (within {boundary_width} σ)")
            print(f"Accuracy on boundary examples: {boundary_examples['correct'].mean():.4f}")
            
            # Save boundary examples for manual inspection
            boundary_examples.to_csv(f"{experiment_plots_dir}/boundary_examples.csv", index=False)
            
            # 6. Systematic error patterns - clustering
            if len(df) > 20:  # Need enough samples for clustering
                try:
                    # Get numerical features for clustering
                    features: List[str] = ['telescope_perplexity']
                    if 'text_length' not in df.columns and 'text' in df.columns:
                        df['text_length'] = df['text'].apply(len)
                    
                    if 'text_length' in df.columns:
                        features.append('text_length')
                    
                    # Add any other metrics if available
                    col: str
                    for col in df.columns:
                        if col.endswith('_perplexity') and col != 'telescope_perplexity':
                            features.append(col)
                    
                    # Standardize features
                    X_cluster: pd.DataFrame = df[features].copy()
                    for col in X_cluster.columns:
                        X_cluster[col] = (X_cluster[col] - X_cluster[col].mean()) / X_cluster[col].std()
                    
                    # Find optimal number of clusters (max 5)
                    max_clusters: int = min(5, len(df) // 10)
                    inertias: List[float] = []
                    k: int
                    for k in range(2, max_clusters + 1):
                        kmeans: KMeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(X_cluster)
                        inertias.append(kmeans.inertia_)
                    
                    # Choose optimal k (simple elbow method)
                    k = 3  # Default
                    if len(inertias) > 1:
                        diffs: np.ndarray = np.diff(inertias)
                        if len(diffs) > 1 and abs(diffs[1]) < 0.5 * abs(diffs[0]):
                            k = 3
                        else:
                            k = 2
                    
                    # Cluster the data
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    df['cluster'] = kmeans.fit_predict(X_cluster)
                    
                    # Analyze error rates by cluster
                    cluster_stats: pd.DataFrame = df.groupby('cluster').agg({
                        'correct': ['mean', 'count'],
                        'telescope_perplexity': ['mean', 'std'],
                        'y_labels': ['mean', 'count']
                    })
                    
                    print("\nCluster-based Error Analysis:")
                    print(cluster_stats)
                    
                    # Save cluster stats
                    cluster_stats.to_csv(f"{experiment_plots_dir}/cluster_stats.csv")
                    
                    # Visualize clusters
                    plt.figure(figsize=(16, 12))
                    
                    # Define a better color palette
                    from matplotlib.colors import ListedColormap
                    cmap: ListedColormap = ListedColormap(PLOT_COLORS[:k])
                    
                    # Create the scatter plot with larger markers
                    scatter = plt.scatter(df['telescope_perplexity'], 
                               df.get('text_length', np.random.normal(size=len(df))),
                               c=df['cluster'], s=150, cmap=cmap, alpha=0.8, edgecolor='black', linewidth=0.5)
                    
                    # Add a threshold line
                    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=3, 
                               label=f'Decision Threshold ({threshold:.2f})')
                    
                    # Add colorbar with larger font
                    cbar: Any = plt.colorbar(scatter, label='Cluster')
                    cbar.ax.tick_params(labelsize=28)
                    cbar.set_label('Cluster', size=32)
                    
                    # Add text labels to identify AI vs Human regions
                    high_perp_y: float = float(df.get('text_length', np.random.normal(size=len(df))).max()) * 0.9
                    low_perp_y: float = float(df.get('text_length', np.random.normal(size=len(df))).max()) * 0.9
                    
                    plt.text(threshold * 1.2, high_perp_y, 'AI-generated region', 
                            fontsize=32, color=PLOT_COLORS[3],
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
                    
                    plt.text(threshold * 0.8, low_perp_y, 'Human-written region', 
                            fontsize=32, color=PLOT_COLORS[0], ha='right',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
                    
                    # Set larger font sizes
                    plt.xlabel('Telescope Perplexity', fontsize=32)
                    plt.ylabel('Text Length' if 'text_length' in df.columns else 'Random Jitter', fontsize=32)
                    plt.title(f'Error Clusters\n{experiment_name}', 
                             fontsize=28)  # Clean title without max distance
                    
                    # Increase tick font sizes
                    plt.xticks(fontsize=28)
                    plt.yticks(fontsize=28)
                    
                    # Add legend
                    plt.legend(fontsize=28)
                    
                    plt.tight_layout()
                    plt.savefig(f"{experiment_plots_dir}/error_clusters.png", dpi=300)
                except Exception as e:
                    print(f"Error in clustering analysis: {e}")
        
        # 7. Perplexity-based segmentation analysis
        # Create segments based on perplexity ranges
        segment_edges: np.ndarray = np.percentile(df['telescope_perplexity'], [0, 25, 50, 75, 100])
        segment_labels: List[str] = ['Very Low', 'Low', 'Medium', 'High']
        df['perplexity_segment'] = pd.cut(df['telescope_perplexity'], 
                                           bins=segment_edges, 
                                           labels=segment_labels)
        
        # Analyze error patterns by segment
        segment_analysis: pd.DataFrame = df.groupby('perplexity_segment').agg({
            'y_labels': ['mean', 'count'],
            'correct': ['mean', 'sum'],
            'telescope_perplexity': ['min', 'max', 'mean']
        })
        
        print("\nPerplexity Segment Analysis:")
        print(segment_analysis)
        
        # Create a visualization of segment-based performance
        plt.figure(figsize=(16, 10))
        segment_correctness: pd.Series = df.groupby('perplexity_segment')['correct'].mean()
        segment_counts: pd.Series = df.groupby('perplexity_segment').size()
        
        # Create the bar plot
        ax: plt.Axes = segment_correctness.plot(kind='bar', color=PLOT_COLORS[0], width=0.6, edgecolor='black', linewidth=2)
        
        # Add title and labels with larger font sizes
        plt.title(f'Classification Accuracy by Perplexity Segment\n{experiment_name}', 
                  fontsize=28)  # Clean title without max distance
        plt.ylabel('Accuracy', fontsize=32)
        plt.xlabel('Perplexity Segment', fontsize=32)
        
        # Add count labels on each bar with larger font size
        i: int
        v: float
        for i, v in enumerate(segment_correctness):
            ax.text(i, v + 0.02, f"n={segment_counts[i]}", ha='center', fontsize=32)
        
        # Increase tick font sizes
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(f"{experiment_plots_dir}/segment_accuracy.png", dpi=300)
        
        # 8. Find extreme misclassification cases
        sorted_errors: pd.DataFrame = misclassified.sort_values('error_distance', ascending=False)
        extreme_errors: pd.DataFrame = sorted_errors.head(10)
        
        print("\nExtreme Misclassification Cases:")
        print(f"Max telescope_perplexity distance for misclassified examples: {max_error_distance_str} σ")
        
        # Output details of worst errors
        if 'text' in extreme_errors.columns:
            for i, (_, row) in enumerate(extreme_errors.iterrows()):
                label: str = "positive" if row['y_labels'] == 1 else "negative"
                pred: str = "positive" if row['predicted'] == 1 else "negative"
                print(f"\nExample {i+1}: True label: {label}, Predicted: {pred}")
                print(f"telescope_perplexity: {row['telescope_perplexity']:.4f}, Distance: {row['error_distance']:.2f} σ")
                if 'text' in row:
                    print(f"Text preview: {str(row['text'])[:150]}...")
        
        # Save extreme error cases
        extreme_errors.to_csv(f"{experiment_plots_dir}/extreme_errors.csv", index=False)
        
        # 9. Misclassification rate by standardized perplexity value ranges
        plt.figure(figsize=(18, 12))
        
        # Create standardized perplexity values
        df['standardized_perplexity'] = (df['telescope_perplexity'] - df['telescope_perplexity'].mean()) / df['telescope_perplexity'].std()
        
        # Create perplexity bins based on standardized values
        n_bins: int = 10
        # Use more interpretable bin sizes in terms of standard deviations
        bins: np.ndarray = np.linspace(-3, 3, n_bins+1)
        df['perplexity_bin'] = pd.cut(df['standardized_perplexity'], bins=bins)
        
        # Calculate error rate by bin
        bin_stats: pd.DataFrame = df.groupby('perplexity_bin').agg({
            'correct': ['mean', 'count', 'sum'],
            'standardized_perplexity': 'mean'
        })
        bin_stats.columns = ['accuracy', 'count', 'correct_count', 'mean_perplexity']
        bin_stats['error_rate'] = 1 - bin_stats['accuracy']
        
        # Plot error rate by perplexity value
        plt.bar(range(len(bin_stats)), bin_stats['error_rate'], alpha=0.8, color=PLOT_COLORS[3], 
                width=0.7, edgecolor='black', linewidth=2)
        
        # Add error rate labels on top of each bar
        for i, (_, row) in enumerate(bin_stats.iterrows()):
            plt.text(i, row['error_rate'] + 0.03, f"{row['error_rate']:.2f}", ha='center', fontsize=32)
        
        # Add count labels to each bar
        for i, (_, row) in enumerate(bin_stats.iterrows()):
            plt.text(i, row['error_rate']/2, f"n={int(row['count'])}", ha='center', color='white', fontsize=32)
        
        # Update x-axis labels
        plt.xticks(
            range(len(bin_stats)), 
            [f"{b.left:.1f}σ to {b.right:.1f}σ" for b in bin_stats.index], 
            rotation=45, fontsize=32
        )
        
        # Set larger font size for other elements
        plt.xlabel('Standardized Perplexity Range (σ)', fontsize=32)
        plt.ylabel('Error Rate', fontsize=32)
        plt.title(f'Error Rate by Standardized Perplexity Range\n{experiment_name}', fontsize=28)
        plt.yticks(fontsize=32)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{experiment_plots_dir}/error_rate_by_perplexity.png", dpi=300)
        
        # 10. Create a summary file with key findings
        with open(f"{experiment_plots_dir}/error_analysis_summary.txt", 'w') as f:
            f.write(f"ERROR ANALYSIS SUMMARY FOR {experiment_name}\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Total examples: {len(df)}\n")
            f.write(f"Misclassified examples: {len(misclassified)} ({len(misclassified)/len(df)*100:.2f}%)\n")
            f.write(f"Max perplexity distance for misclassified: {max_error_distance_str} σ\n\n")
            
            f.write("ERROR TYPES:\n")
            if len(misclassified) > 0:
                false_positives: int = len(misclassified[misclassified['predicted'] == 1])
                false_negatives: int = len(misclassified[misclassified['predicted'] == 0])
                f.write(f"False positives: {false_positives} ({false_positives/len(misclassified)*100:.2f}% of errors)\n")
                f.write(f"False negatives: {false_negatives} ({false_negatives/len(misclassified)*100:.2f}% of errors)\n\n")
            
            f.write("BOUNDARY ANALYSIS:\n")
            if 'boundary_examples' in locals():
                f.write(f"Examples near decision boundary (within {boundary_width} σ): {len(boundary_examples)}\n")
                f.write(f"Accuracy on boundary examples: {boundary_examples['correct'].mean():.4f}\n\n")
            
            f.write("SEGMENT ANALYSIS:\n")
            f.write(segment_analysis.to_string() + "\n\n")
            
            f.write("TOP MISCLASSIFIED EXAMPLES:\n")
            for i, (_, row) in enumerate(extreme_errors.head(5).iterrows()):
                label = "positive" if row['y_labels'] == 1 else "negative"
                pred = "positive" if row['predicted'] == 1 else "negative"
                f.write(f"Example {i+1}: True: {label}, Pred: {pred}, Perp: {row['telescope_perplexity']:.4f}, Dist: {row['error_distance']:.2f} σ\n")
        
        # Save the updated dataframe with analysis columns
        analysis_results_path: str = f"{experiment_plots_dir}/error_analysis.csv"
        df.to_csv(analysis_results_path, index=False)
        
        print(f"\nDetailed error analysis saved to {experiment_plots_dir}")
        return df
    
    except Exception as e:
        print(f"\nError processing experiment {experiment_name}: {str(e)}")
        return None




if __name__ == "__main__":
    
    experiment_folder: str = "experiment_results"
    if not os.path.exists(experiment_folder):
        print(f"Error: Experiment folder '{experiment_folder}' not found")
    else:
        print(f"Processing experiments in {experiment_folder}...")
        experiment_count: int = 0
        success_count: int = 0
        
        # Get list of experiment directories
        try:
            experiments: List[str] = os.listdir(experiment_folder)
            experiment_name: str
            for experiment_name in experiments:
                if not experiment_name.startswith(""):
                    continue
                
                experiment_count += 1
                print(f"\n{'-'*50}")
                print(f"Processing experiment {experiment_count}: {experiment_name}")
                
                try:
                    result: Optional[pd.DataFrame] = specialized_error_analysis(experiment_name)
                    if result is not None:
                        success_count += 1
                except Exception as e:
                    print(f"Failed to process experiment {experiment_name}: {str(e)}")
                    
            print(f"\n{'-'*50}")
            print(f"Completed processing {experiment_count} experiments")
            print(f"Successfully analyzed: {success_count}")
            print(f"Failed: {experiment_count - success_count}")
        except Exception as e:
            print(f"Error accessing experiment directory: {str(e)}")