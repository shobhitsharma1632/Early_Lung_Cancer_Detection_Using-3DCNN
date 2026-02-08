import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.dsets import LunaDataset

def main():
    print("Measuring the time to iterate through the LunaDataset instance")

    Luna_data = LunaDataset()
    start_time = time.time()
    itr_time = start_time
    
    # Store timing data for analysis
    iteration_times = []
    nodule_times = []
    non_nodule_times = []
    series_times = defaultdict(list)
    
    for i, (candidate_t, pos_t, series_uid, center_irc) in enumerate(Luna_data):
        if i > 1000:
            break
            
        # Calculate iteration time
        current_time = time.time()
        iter_duration = current_time - itr_time
        iteration_times.append(iter_duration)
        
        # Categorize by nodule type
        is_nodule = bool(pos_t[1].item())
        if is_nodule:
            nodule_times.append(iter_duration)
        else:
            non_nodule_times.append(iter_duration)
            
        # Group by series_uid for caching analysis
        series_times[series_uid].append(iter_duration)
        
        print(f"Sample {i}: Series {series_uid}, Shape {candidate_t.shape}")
        print(f"Is nodule: {is_nodule}")
        print(f"Iteration Time: {iter_duration:.4f}s")
        
        itr_time = current_time

    total_time = time.time() - start_time
    print(f"\nTotal time taken: {total_time:.2f}s")
    
    # Analyze timing distributions
    analyze_timing_distribution(iteration_times, nodule_times, non_nodule_times, series_times)

def analyze_timing_distribution(iteration_times, nodule_times, non_nodule_times, series_times):
    """Analyze and visualize timing distributions"""
    
    print("\n" + "="*60)
    print("TIMING DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Overall statistics
    times_array = np.array(iteration_times)
    print(f"\nOverall Statistics ({len(iteration_times)} samples):")
    print(f"  Mean time: {np.mean(times_array):.4f}s")
    print(f"  Median time: {np.median(times_array):.4f}s")
    print(f"  Std deviation: {np.std(times_array):.4f}s")
    print(f"  Min time: {np.min(times_array):.4f}s")
    print(f"  Max time: {np.max(times_array):.4f}s")
    print(f"  95th percentile: {np.percentile(times_array, 95):.4f}s")
    print(f"  99th percentile: {np.percentile(times_array, 99):.4f}s")
    
    # Nodule vs Non-nodule comparison
    if nodule_times and non_nodule_times:
        print(f"\nNodule vs Non-nodule Comparison:")
        print(f"  Nodule samples ({len(nodule_times)}):")
        print(f"    Mean: {np.mean(nodule_times):.4f}s")
        print(f"    Median: {np.median(nodule_times):.4f}s")
        print(f"  Non-nodule samples ({len(non_nodule_times)}):")
        print(f"    Mean: {np.mean(non_nodule_times):.4f}s")
        print(f"    Median: {np.median(non_nodule_times):.4f}s")
    
    # Series caching analysis
    print(f"\nCaching Analysis:")
    series_first_times = []
    series_subsequent_times = []
    
    for series_uid, times in series_times.items():
        if len(times) > 1:
            series_first_times.append(times[0])  # First access (cache miss)
            series_subsequent_times.extend(times[1:])  # Subsequent access (cache hit)
    
    if series_first_times and series_subsequent_times:
        print(f"  First access (cache miss): {np.mean(series_first_times):.4f}s avg")
        print(f"  Subsequent access (cache hit): {np.mean(series_subsequent_times):.4f}s avg")
        print(f"  Cache speedup: {np.mean(series_first_times)/np.mean(series_subsequent_times):.1f}x")
    
    # Find outliers
    q75, q25 = np.percentile(times_array, [75, 25])
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    outliers = times_array[times_array > outlier_threshold]
    
    print(f"\nOutlier Analysis:")
    print(f"  Outlier threshold (Q3 + 1.5*IQR): {outlier_threshold:.4f}s")
    print(f"  Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier range: {np.min(outliers):.4f}s - {np.max(outliers):.4f}s")
    
    # Time distribution bins
    print(f"\nTime Distribution Histogram:")
    bins = [0, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
    bin_labels = ['<0.1s', '0.1-0.5s', '0.5-1.0s', '1.0-2.0s', '2.0-5.0s', '>5.0s']
    
    for i in range(len(bins)-1):
        count = np.sum((times_array >= bins[i]) & (times_array < bins[i+1]))
        percentage = (count / len(times_array)) * 100
        print(f"  {bin_labels[i]}: {count} samples ({percentage:.1f}%)")
    
    # Create visualizations
    create_timing_plots(iteration_times, nodule_times, non_nodule_times, series_times)

def create_timing_plots(iteration_times, nodule_times, non_nodule_times, series_times):
    """Create timing visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LunaDataset Iteration Timing Analysis', fontsize=16)
    
    # 1. Overall time distribution histogram
    axes[0, 0].hist(iteration_times, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Iteration Times')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(iteration_times), color='red', linestyle='--', label=f'Mean: {np.mean(iteration_times):.3f}s')
    axes[0, 0].legend()
    
    # 2. Time series plot
    axes[0, 1].plot(iteration_times, alpha=0.7, color='green')
    axes[0, 1].set_title('Iteration Times Over Sample Index')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].axhline(np.mean(iteration_times), color='red', linestyle='--', label=f'Mean: {np.mean(iteration_times):.3f}s')
    axes[0, 1].legend()
    
    # 3. Nodule vs Non-nodule comparison
    if nodule_times and non_nodule_times:
        axes[1, 0].hist([nodule_times, non_nodule_times], bins=30, alpha=0.7, 
                       label=['Nodules', 'Non-nodules'], color=['red', 'blue'])
        axes[1, 0].set_title('Timing Comparison: Nodules vs Non-nodules')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
    
    # 4. Box plot of timing distribution
    box_data = [iteration_times]
    box_labels = ['All Samples']
    
    if nodule_times and non_nodule_times:
        box_data = [nodule_times, non_nodule_times]
        box_labels = ['Nodules', 'Non-nodules']
    
    axes[1, 1].boxplot(box_data, labels=box_labels)
    axes[1, 1].set_title('Timing Distribution Box Plot')
    axes[1, 1].set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('timing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'timing_analysis.png'")

if __name__ == "__main__":
    main()