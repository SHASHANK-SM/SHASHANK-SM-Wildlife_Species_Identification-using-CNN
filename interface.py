import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image
import os
st.set_page_config(page_title="Wildlife Species Identifier", page_icon="🐾", layout="wide")
# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(num_classes):
    weights = ResNet101_Weights.IMAGENET1K_V2
    model = resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load("best_resnet101.pth", map_location="cpu"))
    model.eval()
    return model

# Same transforms used during training/validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Load Class Names
# -------------------------------
train_data_dir = r"D:\Downloads\Split_dataset\train"
class_names = sorted(os.listdir(train_data_dir))  # ensure consistent ordering

model = load_model(num_classes=len(class_names))

# -------------------------------
# Streamlit UI
# -------------------------------


# Custom CSS for wildlife theme
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(135deg, #e9f5ec, #cbe4d8);
    }
    .stProgress > div > div > div > div {
        background-color: #2e8b57;
    }
    .species-card {
    padding: 10px;
    margin: 5px;
    background-color: #ffffffcc;
    border-radius: 12px;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.2);
    color: black;  /* <-- This makes the text black */
}

    </style>
""", unsafe_allow_html=True)

st.title("🐾 Wildlife Species Identification using CNN")
st.write("Upload an image of a wild animal, and let AI identify its species!")

# Sidebar Info
st.sidebar.header("🌿 Project Info")
st.sidebar.write("""
**Wildlife Species Identifier**  
Model: ResNet101 (Fine-tuned)  
Developer: Shashank S M  
Dataset: Custom Wildlife Dataset  
""")

uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Show uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", width=300)

        # Preprocess
        input_tensor = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get top prediction
        confidence, predicted_class = torch.max(probabilities, dim=0)
        st.subheader(f"Prediction: 🐆 **{class_names[predicted_class.item()]}**")
        st.progress(float(confidence.item()))

        # Show top-5 predictions
        st.write("### 🔝 Top-5 Predictions:")
        top5_prob, top5_class = torch.topk(probabilities, 5)
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.markdown(f"""
                <div class="species-card">
                    <b>{class_names[top5_class[i].item()]}</b><br>
                    {top5_prob[i].item()*100:.2f}%
                </div>
                """, unsafe_allow_html=True)
