# Simulation Lab

This repository contains Python Jupyter notebooks, data, and visual outputs for various Simulation Lab exercises. 

## Repository Structure

The repository is organized by lab sessions:

* **Lab_1/**: Simulation of Chemical Reactions
  * Contains notebooks (`1.ipynb` and `ACE080BCT055.ipynb`) that model chemical reaction kinetics.
  * Includes related data log (`industrial_reactor_log.txt`) and `output.txt`.

* **Lab_2/**: Generation of Random Numbers
  * Contains `lab2.ipynb`, which focuses on Pseudo-Random Number generation and testing uniformity using the Chi-Square Test.

* **Lab_3/**: PI Simulation
  * Contains `3.ipynb` which runs a Monte Carlo simulation (dropping points randomly) to estimate the value of $\pi$.
  * Includes output visualizations (`pi_estimate_20_drops.png`, `rain_drops.png`).

* **Lab_4/**: Markov Chain and Q-Learning
  * Contains `4.ipynb`, which introduces Markov Decision Processes (MDP) and demonstrates a Q-learning grid world simulation.
  * Includes the original handwritten guide notes (`lab4guide.jpeg`).

## Setup Instructions

A Python virtual environment (`.venv`) is recommended for managing dependencies (like `pandas`, `numpy`, etc.).

1. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```
2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
3. **Deactivate when finished**:
   ```bash
   deactivate
   ```
