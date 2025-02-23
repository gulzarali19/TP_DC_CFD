import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image  

# **Normalization Function**
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# **De-normalization Function (if needed)**
def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# **Model A: TemperatureFieldPredictor**
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
            nn.Linear(512, image_size[0] * image_size[1] * 3),
            nn.Sigmoid()
        )

    def forward(self, design_points):
        output = self.fc(design_points)
        output = output.view(-1, 3, self.image_size[0], self.image_size[1])
        return output

# **Model B: ScalarToImageModel**
class ScalarToImageModel(nn.Module):
    def __init__(self, output_height=256, output_width=512):
        super(ScalarToImageModel, self).__init__()

        initial_h, initial_w = output_height // 64, output_width // 64

        self.fc = nn.Sequential(
            nn.Linear(3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * initial_h * initial_w),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.initial_h = initial_h
        self.initial_w = initial_w

        self.deconv_module = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, self.initial_h, self.initial_w)
        x = self.deconv_module(x)
        return x

# **Device Configuration**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Function to Load Model**
def load_model(model_name):
    if model_name == "Model A":
        model = TemperatureFieldPredictor().to(device)
        model.load_state_dict(torch.load("new_model.pth", map_location=device))
    elif model_name == "Model B":
        model = ScalarToImageModel(output_height=256, output_width=512).to(device)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    return model

# **Streamlit UI**
st.title("Data Center Hotspot Prediction")
st.markdown("### Select Model & Enter Design Parameters")

# **Model Selection Toggle**
model_selection = st.radio("Choose Prediction Model:", ["Model A", "Model B"])
model = load_model(model_selection)

# **User Inputs**
power = st.number_input("Power (W)", min_value=400.0, max_value=2000.0, step=10.0)
velocity = st.number_input("Velocity (m/s)", min_value=1.0, max_value=3.0, step=0.5)
temperature = st.number_input("Temperature (K)", min_value=290.0, max_value=300.0, step=0.5)

# **Normalize inputs if using Model B**
if model_selection == "Model B":
    power = normalize(power, 400, 2000)
    velocity = normalize(velocity, 1.0, 3.0)
    temperature = normalize(temperature, 290, 300)

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
    st.image(predicted_image_pil, caption=f"Predicted Temperature Field ({model_selection})", use_container_width=True)
