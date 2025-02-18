import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image  # Use PIL instead of OpenCV
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
power = st.number_input("Power (W)", min_value=100.0, max_value=2000.0, step=10.0)
velocity = st.number_input("Velocity (m/s)", min_value=1.0, max_value=3.0, step=0.5)
temperature = st.number_input("Temperature (°C)", min_value=15.0, max_value=25.0, step=0.5)

# **Predict Button**
if st.button("Predict Temperature Field"):
    with torch.no_grad():
        design_params = torch.tensor([[power, velocity, temperature]], dtype=torch.float32).to(device)
        predicted_image = model(design_params)

    # Convert to numpy for visualization
    predicted_np = predicted_image.cpu().numpy().squeeze()
    predicted_np = np.transpose(predicted_np, (1, 2, 0))  # Convert from (C, H, W) -> (H, W, C)
    predicted_np = np.clip(predicted_np, 0, 1)  # Normalize to [0,1]

    # Convert NumPy array to PIL Image
    predicted_image_pil = Image.fromarray((predicted_np * 255).astype(np.uint8))
    
    # Display Prediction
    st.image(predicted_image_pil, caption="Predicted Temperature Field", use_column_width=True)
