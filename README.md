# Neuropixels BCI Pipeline: Open-Loop Control Analysis

This project focuses on the development of a signal processing infrastructure for Brain-Computer Interfaces (BCI) using high-density recordings from Neuropixels probes. The system integrates cloud-based data acquisition, digital signal processing, and neural pattern decoding to control a dynamic system.

## Technical Overview
The pipeline is designed to process extracellular signals at a sampling frequency of 30 kHz. The architecture enables the identification of individual neural units and the translation of their firing rates into spatial movement commands, allowing for the quantitative evaluation of the relationship between external stimuli and decoded biological responses.

## Tech Stack
*   **Language:** Python 3.x
*   **Data Acquisition:** AWS S3 (via `s3fs`) for accessing Allen Institute datasets.
*   **Signal Processing:** SciPy (implementation of 3rd order Butterworth filters).
*   **Classification (Spike Sorting):** Scikit-learn (Principal Component Analysis - PCA and K-Means Clustering).
*   **Visualization & Analysis:** Matplotlib for generating engineering performance and validation reports.

## System Architecture
The software is structured into modules to ensure data integrity and analysis scalability:

1.  **Analysis Module (`analyzer.py`):** Manages binary data loading, band-pass filtering (300-3000 Hz), and action potential detection via adaptive thresholds.
2.  **Decoding Module (`decoder.py`):** Translation algorithm that converts electrophysiological activity into spatial coordinates.
3.  **Visualization Module (`visualizer.py`):** Generates technical metrics, including Inter-Spike Interval (ISI) histograms and transfer analysis.
4.  **Orchestrator (`main.py`):** Manages sequential workflow execution and result validation.

## Results and Validation

### Neural Unit Validation
The system isolates independent units through classification in the principal component space. Biological validity is corroborated via Inter-Spike Interval (ISI) histograms, ensuring compliance with the 1.5 ms refractory period for the detected neurons.

### Performance Analysis (Transfer Analysis)
System fidelity is evaluated by comparing the stimulus trajectory (Input) with the decoded response (Output). The implementation includes a linear regression-based trendline to quantify correlation and measure system latency, providing a quantitative baseline for decoder optimization.

## Installation Instructions

1.  Install the required dependencies:
    ```bash
    pip install s3fs scikit-learn scipy matplotlib numpy
    ```
2.  Run the primary pipeline:
    ```bash
    python main.py
    ```

## Future Development Roadmap
*   Optimization of decoding algorithms to reduce systemic latency.
*   Implementation of Deep Learning models to improve spike sorting precision.
*   Expansion of analysis to include simultaneous recordings from multiple Neuropixels channels.
