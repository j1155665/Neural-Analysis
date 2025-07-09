# MP Rig Neuropixels Data Preprocessing Pipeline

**neuralpipeline** is a data preprocessing pipeline for Open Ephys data collected from Neuropixels NHP probes.

## Overview

This pipeline streamlines preprocessing steps essential for neural data analysis, specifically designed for experiments using Open Ephys and Neuropixels in non-human primates.

## Main Features

- **Data Merging:** Combines `.dat`, 'timestamps.py', and TTL files for unified processing.
- **Info Struct:** Create an info struct into `.mat` files.
- **Struct Creation:** Converts event, unit, and eye tracking data into one MATLAB `.mat` files.
- **Notebook Guides:** Step-by-step Jupyter notebooks guide you through the workflow.
- **Waveform Visualization (optional):** Tools to plot waveform shapes sorted by probe depth.

## Usage Instructions

1. **Create Recording Info:**
    - In the `notebook` folder, use the provided notebook to write `RecSectionInfo` and generate a `.mat` file containing your recording information.
2. **Run Main Processing:**
    - Go to the `offline_processing_openephys` folder.
    - Run `OpenEphysOfflinePrcessing` to execute the main preprocessing steps:
        - Merge `.dat` and TTL files.
        - Create the event struct first (required before the other steps).
        - Then, create unit and eye position structs (these are saved into the same `.mat` file as the event struct).
3. **Plotting Tools (optional):**
    - Use `plot_waveform.ipynb` in the `notebook` folder to visualize waveform shapes sorted by depth.

> **Important:**  
> Always create the recording info `.mat` file before generating event, unit, or eye position structs.

## Folder Structure

- `offline_processing_openephys/` — Main processing functions and scripts.
- `notebook/` — Jupyter notebooks documenting each step and providing visualization tools.
- `Archive/` — Preliminary code (for reference or legacy support).

## Author

Developed by **Yueh-Chen Chiang**  
Christopher Fetsche’s Lab
