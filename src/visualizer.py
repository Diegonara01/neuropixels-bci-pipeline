import matplotlib.pyplot as plt
import numpy as np

def plot_pro_report(brain, features):
    """
    Generates a professional report including PCA space, 
    waveforms, ISI, and Raster Plot.
    """
    isi = np.diff(brain.spike_indices) / (brain.fs / 1000)
    violations = np.sum(isi < 1.5) / len(isi) * 100 
    
    fig = plt.figure(figsize=(16, 12))
    plt.suptitle(f"Neural Unit Analysis - Channel 10\nISI Violations (<1.5ms): {violations:.2f}%", fontsize=16)

    # 1. PCA Space
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(features[:,0], features[:,1], c=brain.clusters, cmap='tab10', s=15, alpha=0.6)
    ax1.set_title("Neural Classification (PCA)")
    
    # 2. Mean Waveforms
    ax2 = plt.subplot(2, 2, 2)
    for i in range(len(np.unique(brain.clusters))):
        avg = np.mean(brain.waveforms[brain.clusters == i], axis=0)
        ax2.plot(avg, label=f"Unit {i+1}", linewidth=2)
    ax2.set_title("Spike Morphology (Electrical Signature)")
    ax2.legend()
    
    # 3. ISI Histogram
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(isi, bins=50, range=(0, 100), color='forestgreen', edgecolor='black', alpha=0.8)
    ax3.axvline(1.5, color='red', linestyle='--', label='Refractory Limit')
    ax3.set_title("ISI Histogram (Biological Validation)")
    ax3.set_xlabel("Interval (ms)")

    # 4. Raster Plot
    ax4 = plt.subplot(2, 2, 4)
    ax4.eventplot([brain.spike_indices[brain.clusters == i] for i in range(len(np.unique(brain.clusters)))], 
                 colors=plt.cm.tab10(np.linspace(0, 1, 4)))
    ax4.set_title("Raster Plot: Firing Chronology")
    ax4.set_xlabel("Time (Samples)")
    ax4.set_yticklabels([f"Neu {i+1}" for i in range(len(np.unique(brain.clusters)))])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/pro_report.png', dpi=300)
    plt.show()
