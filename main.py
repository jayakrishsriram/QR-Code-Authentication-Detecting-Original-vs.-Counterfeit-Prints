import streamlit as st
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import io

# Define image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_names = ["First Print", "Second Print"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
@st.cache_resource
def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("vit_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("Image Classification: First Print vs Second Print")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    st.write(f"Prediction: **{class_names[predicted.item()]}**")
