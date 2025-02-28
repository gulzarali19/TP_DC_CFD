# Data Center Hotspot Temperature Prediction

## Overview

This project predicts temperature fields in data centers using a deep learning model. The model takes design parameters such as power, velocity, and inlet temperature and outputs a predicted temperature field as an image. A small framework for quick predictions on vast design points which can be used for early design decisions
## 2D Model Rack-level
### Features

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

This work is part of my thesis "Deep Learning Enhanced CFD approach in predicting Hotspots in Datacenters for increasing energy efficiency". This work proposed a framework from existing methodolgies of field prediction using deep learning models. Temperature fields generated from CFD were used as data for training deep learning models. First part includes CFD on a single rack and second part includes converting this method to 3D field predictions and ultimately modelling whole datacenter with more parameters for vast designspace explorations.

