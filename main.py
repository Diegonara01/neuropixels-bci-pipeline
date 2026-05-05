
"""
Main Execution Script: Neuropixels BCI Control Pipeline.
Author: [Your Name]
Description: This script orchestrates data ingestion from AWS, spike sorting, 
and control-loop validation for biological computing research.
"""

from src.analyzer import NeuroAnalyzer
from src.decoder import get_paddle_history, calculate_metrics
import numpy as np
import matplotlib.pyplot as plt

# AWS S3 Path to Allen Institute Neuropixels Dataset
S3_PATH = 'aind-benchmark-data/ephys-compression/aind-np1/625749_2022-08-03_15-15-06_ProbeA/traces_cached_seg0.raw'

def main():
    print("--- Starting Neuropixels BCI Pipeline ---")
    
    # 1. Initialize Neural Processing Pipeline
    # We use a 30kHz sampling frequency as per Neuropixels 1.0 standards
    brain = NeuroAnalyzer(fs=30000)
    
    # 2. Data Acquisition (5 seconds of activity from Channel 10)
    print("Loading data from AWS S3...")
    brain.load_from_aws(S3_PATH, channel=10, seconds=5)
    
    # 3. Signal Preprocessing & Spike Detection
    print("Preprocessing signal and detecting spikes...")
    brain.preprocess()
    brain.detect_spikes(sigma_mult=3.8, refractory_ms=1.5)
    
    # 4. Spike Sorting (Clustering neurons)
    print("Classifying neural units via PCA + K-Means...")
    brain.sort_spikes(n_clusters=4)
    print(f"Detected {len(brain.spike_indices)} spikes. Units isolated: 4")
    
    # 5. Control-Loop Simulation
    print("Generating simulated Pong ball trajectory...")
    fps = 30
    seconds = 5
    time_axis = np.linspace(0, seconds, seconds * fps)
    # Simulated stimulus (Target Ball)
    ball_y = 30 * np.sin(2 * np.pi * 0.5 * time_axis)
    
    print("Decoding neural activity into paddle movement...")
    paddle_history = get_paddle_history(brain.spike_indices, brain.clusters, seconds=5)
    
    # 6. Performance Validation
    correlation, latency = calculate_metrics(ball_y, paddle_history)
    
    print("\n--- FINAL REPORT ---")
    print(f"System Accuracy (Correlation): {correlation*100:.2f}%")
    print(f"System Latency: {latency:.2f} ms")
    print("-------------------------------------")

if __name__ == "__main__":
    main()
