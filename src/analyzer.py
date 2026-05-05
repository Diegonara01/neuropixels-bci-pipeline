
import s3fs
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class NeuroAnalyzer:
    """
    Class to handle Neuropixels data acquisition from AWS, 
    signal preprocessing, and spike sorting.
    """
    def __init__(self, fs=30000):
        self.fs = fs  # Sampling frequency (30kHz for Neuropixels)
        self.raw_data = None
        self.filtered_data = None
        self.spike_indices = None
        self.waveforms = None
        self.clusters = None

    def load_from_aws(self, s3_path, channel=10, seconds=5):
        """Connects to public S3 bucket and streams raw binary data."""
        fs_s3 = s3fs.S3FileSystem(anon=True)
        num_channels = 384  # Standard NP1.0 probe configuration
        samples_to_read = seconds * self.fs
        
        with fs_s3.open(s3_path, 'rb') as f:
            # Read interleaved int16 binary data
            raw_bytes = f.read(samples_to_read * num_channels * 2)
            data_interleaved = np.frombuffer(raw_bytes, dtype=np.int16)
            # Reshape and extract target channel
            self.raw_data = data_interleaved.reshape(-1, num_channels)[:, channel].astype(np.float32)

    def preprocess(self):
        """Applies a 3rd order Butterworth bandpass filter (300Hz - 3000Hz)."""
        sos = butter(3, [300, 3000], btype='bandpass', fs=self.fs, output='sos')
        self.filtered_data = sosfiltfilt(sos, self.raw_data)

    def detect_spikes(self, sigma_mult=3.8, refractory_ms=1.5):
        """
        Detects spikes using an adaptive threshold based on Median Absolute Deviation (MAD).
        Formula: Threshold = sigma_mult * (median(|x|) / 0.6745)
        """
        # Robust estimation of noise standard deviation
        sigma_n = np.median(np.abs(self.filtered_data)) / 0.6745
        thr = sigma_mult * sigma_n
        
        # Identify crossings and apply strict refractory period (1.5ms)
        idx = np.where(self.filtered_data < -thr)[0]
        if len(idx) == 0: return
        
        ref_samples = int(refractory_ms * (self.fs / 1000))
        clean = [idx[0]]
        for i in range(1, len(idx)):
            if idx[i] - clean[-1] > ref_samples:
                clean.append(idx[i])
        self.spike_indices = np.array(clean)

    def sort_spikes(self, n_clusters=4):
        """Extracts spike waveforms and classifies units using PCA and K-Means."""
        w = []
        pre, post = 20, 40 # Window size (samples)
        for i in self.spike_indices:
            if i > pre and i < len(self.filtered_data) - post:
                # Align waveform to local minimum
                win = self.filtered_data[i-5 : i+5]
                p = i - 5 + np.argmin(win)
                w.append(self.filtered_data[p-pre : p+post])
        
        self.waveforms = np.array(w)
        # Dimension reduction for feature extraction
        pca = PCA(n_components=3)
        feat = pca.fit_transform(self.waveforms)
        # Cluster identification for individual neuron units
        self.clusters = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit_predict(feat)
        return feat
