Geant4 Simulations for MSc Thesis: NEBULA-Xplorer Optics and Physics Validation
This repository contains the complete set of Geant4 simulations and analysis scripts developed for the MSc thesis of Friso Eigenhuis. The project is divided into two primary components: a physics list validation (exp_val) and the main instrument simulation (thesis).

The simulations are built and tested with Geant4 version 11.3.1.

Project Overview
The overarching goal of this work is to accurately model the interaction of low-energy space protons with the NEBULA-Xplorer X-ray optics and predict the resulting performance degradation of its detector over a 5-year mission lifetime.

This is achieved in two distinct stages:

Physics Validation (exp_val): First, a dedicated simulation is used to validate different Geant4 physics models against published experimental data for proton scattering on gold surfaces at grazing angles. This step is crucial for selecting a physics list that accurately reproduces real-world interactions.

Instrument Simulation (thesis): Using the validated physics, a full-scale simulation of the NEBULA-Xplorer instrument is performed. This model is used to assess how different X-ray concentrator designs affect the proton flux and Total Ionizing Dose (TID) on the detector, ultimately allowing for a prediction of the instrument's end-of-life energy resolution.

Project Structure
The repository is organized into two main directories:

1. exp_val: Physics Validation Simulation
This simulation replicates the experimental setup of Diebold et al. (2015) to validate Geant4's ability to model soft proton scattering on gold-coated mirrors.

Key Features
Multi-Model Comparison: Compares results from three different physics configurations: Single Scattering (run_SS.mac), Multiple Scattering Option 3 (run_O3.mac), and Multiple Scattering Option 4 (run_O4.mac).
Experimental Data Integration: The angle_val_all.py script compares simulation output against experimental data from all_efficiencies_combined.csv.
Statistical Analysis: Performs a chi-squared (χ²) analysis to quantify the goodness-of-fit for each model and calculates a systematic correction factor (C_eta) to account for discrepancies between simulation and reality.
How to Use
Compile: Make a build folder in either the thesis or exp_val folder move inside folder and run cmake .. followed by make.
Run Simulation: Execute the compiled program with one of the provided macro files (e.g., ./exp_val run_O4.mac). The output .root files will be generated in the root subdirectories.
The Simulation can also be run without any macro which will open the visualisation

Analyze: Navigate to the scripts directory and run the main analysis script: python angle_val_all.py. This will generate comparison plots and print the χ² and C_eta results to the console.
2. thesis: NEBULA-Xplorer Instrument Simulation
This is the main simulation of the thesis, modeling the complete NEBULA-Xplorer instrument to evaluate the radiation hardness of different X-ray optic designs.

Key Features
Configurable Geometry: Easily switch between three mirror geometries via macro commands:
Single Paraboloid (SP)
Wolter-I (DPH)
Double-Conic approximation (DCC)
Component Toggles: The presence of a magnetic filter and a central opening in the mirror shell can be enabled or disabled at runtime.
Comprehensive Analysis: The analysis scripts calculate key performance metrics, including:
Proton fluence at the detector.
Total Ionizing Dose (TID) in krad(Si).
Predicted increase in detector leakage current and the corresponding degradation in FWHM energy resolution over the mission lifetime.
How to Use
Compile: Navigate to the build directory and run cmake .. followed by make.
Run Simulation: Execute the compiled program with a macro file (e.g., ./thesis run.mac). The run_all_separate.sh script is provided to automate running all 24 required configurations.
Analyze: Navigate to the scripts directory and run the primary analysis scripts in order:
These scripts will process the .root files, print summary tables, and generate all the final plots for the thesis.
General Setup and Requirements
Dependencies
Geant4: Version 11.3.1
ROOT: Version 6+
CMake: Version 3.8+
Python: Version 3.8+ with the following packages:
uproot
pandas
numpy
matplotlib
scipy
seaborn
You can install the required Python packages using the requirements.txt file in each scripts directory:

Data Management
This project uses Git LFS (Large File Storage) to manage the large .root data files required for analysis.

Input Data: Small input files (e.g., kappa.csv, all_efficiencies_combined.csv) are tracked directly by Git.
Simulation Output: All .root files are tracked by Git LFS. To clone this repository and download the data, you must have Git LFS installed.
License
This project is licensed under the MIT License. See the LICENSE.md file for details.

