
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import correlate

def get_paddle_history(spike_indices, clusters, seconds=5, fs=30000, fps=30):
    """
    Translates discrete neural spikes into continuous paddle movement.
    Neu 1 (Cluster 0) -> UP movement
    Neu 2 (Cluster 1) -> DOWN movement
    """
    total_frames = seconds * fps
    samples_per_frame = fs // fps
    
    # Isolate activity for control units
    up_spikes = spike_indices[clusters == 0]
    down_spikes = spike_indices[clusters == 1]
    
    paddle_y = 0
    history = []
    
    for f in range(total_frames):
        # Window-based integration of spike counts
        t_start, t_end = f * samples_per_frame, (f + 1) * samples_per_frame
        ups = np.sum((up_spikes >= t_start) & (up_spikes < t_end))
        downs = np.sum((down_spikes >= t_start) & (down_spikes < t_end))
        
        # Update position based on differential firing rate
        paddle_y += (ups - downs) * 4
        history.append(paddle_y)
        
    return np.array(history)

def calculate_metrics(ball_y, paddle_y, fs_sim=30):
    """
    Evaluates system performance using Pearson Correlation and Temporal Lag.
    Used to validate BCI control accuracy.
    """
    # Linear relationship score (-1 to 1)
    corr, _ = pearsonr(ball_y, paddle_y)
    
    # Calculate cross-correlation to detect signal latency
    cross_corr = correlate(paddle_y - np.mean(paddle_y), ball_y - np.mean(ball_y))
    lags = np.arange(-len(ball_y) + 1, len(ball_y))
    latency_ms = (lags[np.argmax(cross_corr)] / fs_sim) * 1000
    
    return corr, latency_ms
