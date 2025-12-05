# Rolling Shutter Image as an IMU

This repository implements a rolling shutter camera motion estimation pipeline that estimates angular and translational velocities by exploiting rolling shutter distortions in images. 

## Quick Start
The `run.ipynb` notebook contains the end-to-end pipeline for processing rolling shutter images and comparing results with IMU ground truth.

## Setup Instructions

Follow these steps to set up your environment and install the necessary packages:

1. **Create a Conda Virtual Environment**

    ```bash
    conda env create -f environment.yml
    ```

2. **Activate the Virtual Environment**

    ```bash
    conda activate rolling-shutter-image-as-an-imu
    ```

3. **Install the Package**

    ```bash
    pip install -e .
    ```

If you prefer to use pip, install the requirements using `requirements.txt`.

4. **Download Datasets (Git LFS)**

    This repository uses Git LFS for large dataset files. Make sure you have Git LFS installed:

    ```bash
    git lfs install
    git lfs pull
    ```

## Repository Structure

- **`rsaai/run.ipynb`**: Main notebook to run the complete pipeline
- **`rsaai/model/`**: Model implementations (RS-Diffusion, depth estimation)
- **`rsaai/util/`**: Utility functions for flow processing and visualization
- **`checkpoint/`**: Pre-trained model weights
- **`config/`**: Configuration files for models

## Acknowledgments

This code is adapted from:
- [RS-Diffusion](https://github.com/lhaippp/RS-Diffusion) - Rolling shutter optical flow prediction
- [Image-as-an-IMU](https://github.com/jerredchen/image-as-an-imu) - Motion estimation from images

