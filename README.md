# Data Center Hotspot Temperature Prediction

## Overview

This project predicts temperature fields in data centers using a deep learning model. The model takes design parameters such as power, velocity, and inlet temperature and outputs a predicted temperature field as an image. A small framework for quick predictions on vast design points which can be used for early design decisions
## 2D Model Rack-level
### Features

- Predicts 2D temperature distributions in Racks.
- Uses a fully connected neural network for image generation.
- Streamlit-based web app for easy user interaction.

### Model Architecture

The neural network consists of:

- Input: 3 design parameters (Power, Velocity, Temperature)
- Fully connected layers with LeakyReLU activation
- Sigmoid activation in the final layer to normalize output
- Output: RGB temperature field image (128x256)

### Data and Training

- Dataset: Generated from 2000 CFD simulations at different design points.
- Inputs: Power (W), Velocity (m/s), Temperature (°C).
- Outputs: Temperature field images.
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Training: PyTorch with GPU acceleration

## Deployment

- The model is deployed using **Streamlit**.
- The user inputs design parameters, and the model predicts the temperature field.
- Outputs are visualized as images.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Streamlit
- PIL (Pillow)

## Usage

1. Install dependencies:
   ```bash
   pip install torch torchvision numpy streamlit pillow
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Enter design parameters and click "Predict" to generate the temperature field.

## Future Improvements (Currently working on)

- Train with more diverse CFD simulations for better accuracy.
- Optimize model architecture for higher resolution images.
- Deploy as a cloud-based service for wider accessibility.

## Author

Gulzar1911

This work is part of my thesis "Deep Learning Enhanced CFD approach in predicting Hotspots in Datacenters". This work is initiated as a framework to predict temperature fields generated from CFD using deep learning models. First part includes CFD on a single rack and second part includes modelling whole datacenter with more parameters for design optimizations.

