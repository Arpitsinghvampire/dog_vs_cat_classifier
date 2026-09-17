import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.nn.functional as F

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Define the model
# --------------------------------------------------

class DogvsCat_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3 , out_channels = 32 , kernel_size = 3 , padding = 'same')
        self.conv2 = nn.Conv2d(in_channels = 32 , out_channels = 64 , kernel_size = 3 , padding = 'same')
        self.pool = nn.MaxPool2d(kernel_size = 2 , stride = 2)
        self.linear_layer = nn.Linear(64*56*56 , 128)
        self.linear_layer2 = nn.Linear(128 ,1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear_layer(x))
        x = self.linear_layer2(x)
        return x
model = DogvsCat_classifier()


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image):

    # Resize image
    image = image.resize((224, 224))

    # Convert PIL image to NumPy array
    image = np.array(image)

    # Convert NumPy array to PyTorch tensor
    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    # Convert pixel values from 0-255 to 0-1
    image = image / 255.0

    # Change shape:
    # (224, 224, 3)
    #       ↓
    # (3, 224, 224)
    image = image.permute(2, 0, 1)

    # Add batch dimension:
    # (3, 224, 224)
    #       ↓
    # (1, 3, 224, 224)
    image = image.unsqueeze(0)

    # Send image to device
    image = image.to(device)

    # Make prediction
    with torch.no_grad():

        output = model(image)

        probability = torch.sigmoid(output)

        prediction = (probability >= 0.5).int().item()

    return prediction, probability.item()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("Cat vs Dog Differentiator")


# Upload image
file_uploaded = st.file_uploader(
    "Upload your image file",
    type=["jpeg", "jpg", "png"]
)


# --------------------------------------------------
# Display uploaded image
# --------------------------------------------------

if file_uploaded is not None:

    image = Image.open(
        file_uploaded
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image"
    )


    # --------------------------------------------------
    # Submit button
    # --------------------------------------------------

    if st.button("Submit"):

        prediction, probability = predict_image(image)

        st.write("Prediction:", prediction)

        st.write("Probability:", probability)

        # Display result
        if prediction == 0:

            st.success("Prediction: Cat")

        else:

            st.success("Prediction: Dog")
