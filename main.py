"""
Main Execution Script: Neuropixels BCI Control Pipeline.
Author: Diego Narvaez
Description: Orchestrates data ingestion from AWS, spike sorting, 
and control-loop validation for biological computing research.
"""

import os
import numpy as np
from src.analyzer import NeuroAnalyzer
from src.decoder import get_paddle_history, calculate_metrics
from src.visualizer import plot_pro_report, plot_performance_analysis

# AWS S3 Path to Allen Institute Neuropixels Dataset
S3_PATH = 'aind-benchmark-data/ephys-compression/aind-np1/625749_2022-08-03_15-15-06_ProbeA/traces_cached_seg0.raw'

def main():
    print("--- Starting Neuropixels BCI Pipeline ---")
    
    # 0. Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 1. Initialize Neural Processing Pipeline (30kHz sampling)
    brain = NeuroAnalyzer(fs=30000)
    
    # 2. Data Acquisition (5 seconds from Channel 10)
    print("Streaming data from AWS S3...")
    brain.load_from_aws(S3_PATH, channel=10, seconds=5)
    
    # 3. Preprocessing & Spike Detection
    print("Filtering signal and detecting spikes (Adaptive Threshold)...")
    brain.preprocess()
    brain.detect_spikes(sigma_mult=3.8, refractory_ms=1.5)
    
    # 4. Spike Sorting (Unit Identification)
    print("Running PCA + K-Means clustering...")
    features = brain.sort_spikes(n_clusters=4)
    
    # 5. Control-Loop Simulation
    fps = 30
    seconds = 5
    time_axis = np.linspace(0, seconds, seconds * fps)
    ball_y = 30 * np.sin(2 * np.pi * 0.5 * time_axis) # Target Stimulus
    
    print("Decoding neural activity into paddle movement...")
    paddle_history = get_paddle_history(brain.spike_indices, brain.clusters, seconds=5)
    
    # 6. Performance Validation
    correlation, latency = calculate_metrics(ball_y, paddle_history)
    
    # 7. Professional Report Generation (English Labels)
    print("Generating Engineering Reports...")
    plot_pro_report(brain, features)
    plot_performance_analysis(time_axis, ball_y, paddle_history, correlation, latency)
    
    print("\n--- PIPELINE EXECUTION SUCCESSFUL ---")
    print(f"Final Accuracy: {correlation*100:.2f}% | System Latency: {latency:.2f} ms")

if __name__ == "__main__":
    main()
