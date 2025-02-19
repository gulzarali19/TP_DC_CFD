import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

# **Load the Trained Model**
class TemperatureFieldPredictor(nn.Module):
    def __init__(self, image_size=(128, 256)):  
        super().__init__()
        self.image_size = image_size  
        
        self.fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, image_size[0] * image_size[1] * 3),  # Output 3 channels (RGB)
            nn.Sigmoid()
        )

    def forward(self, design_points):
        output = self.fc(design_points)
        output = output.view(-1, 3, self.image_size[0], self.image_size[1])  # Reshape to RGB
        return output

# **Initialize Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemperatureFieldPredictor().to(device)
model.load_state_dict(torch.load("new_model.pth", map_location=device))  # Load trained model
model.eval()

# **Streamlit App UI**
st.title("Data Center Hotspot Prediction")
st.markdown("### Enter Design Parameters:")

# **User Inputs**
power = st.number_input("Power (W)", min_value=50.0, max_value=500.0, step=10.0)
velocity = st.number_input("Velocity (m/s)", min_value=0.1, max_value=5.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=15.0, max_value=45.0, step=0.5)

# **Predict Button**
if st.button("Predict Temperature Field"):
    with torch.no_grad():
        design_params = torch.tensor([[power, velocity, temperature]], dtype=torch.float32).to(device)
        predicted_image = model(design_params)

    # Convert to numpy for visualization
    predicted_np = predicted_image.cpu().numpy().squeeze()
    predicted_np = np.transpose(predicted_np, (1, 2, 0))  # Convert from (C, H, W) -> (H, W, C)
    predicted_np = np.clip(predicted_np, 0, 1)  # Normalize to [0,1]

    # Display Prediction
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(predicted_np, aspect=256/128)
    ax.axis("off")
    ax.set_title("Predicted Temperature Field")
    
    st.pyplot(fig)
