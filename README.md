# NGS QC Pipeline (Hampel/MAD)

End-to-end pipeline for longitudinal quality control (QC) of next-generation sequencing (NGS) runs using robust statistical methods and an interactive Streamlit dashboard.

## Overview

This repository contains a proof-of-concept pipeline for monitoring NGS run quality over time.  
It combines:

- Data extraction and preprocessing from QC reports  
- Robust outlier detection using the Hampel filter with Median Absolute Deviation (MAD)  
- An interactive dashboard for visualization and analysis  

## Live Dashboard

https://huggingface.co/spaces/cfernandez3/ngs-qc-dashboard

## Repository Structure

- `app/` — Streamlit dashboard  
- `data_processing/` — Data extraction and preprocessing (PDF → dataframe)  
- `data/` — Example input data  

## Author

Casiana Fernandez-Bango
