import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image

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
# Streamlit UI
# -------------------------------
st.title("🐾 Wildlife Species Identification")
st.write("Upload an image to identify the species using ResNet101.")

# -------------------------------
# Load class names from dataset
# -------------------------------
from torchvision import datasets

train_data_dir = r"D:\Downloads\Split_dataset\train"
train_dataset = datasets.ImageFolder(train_data_dir)
class_names = train_dataset.classes

model = load_model(num_classes=len(class_names))


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)


    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Get top prediction
    confidence, predicted_class = torch.max(probabilities, dim=0)
    st.subheader(f"Prediction: **{class_names[predicted_class.item()]}**")
    st.write(f"Confidence: **{confidence.item()*100:.2f}%**")

    # Show top-5 predictions
    st.write("### Top-5 Predictions:")
    top5_prob, top5_class = torch.topk(probabilities, 5)
    for i in range(5):
        st.write(f"{class_names[top5_class[i].item()]}: {top5_prob[i].item()*100:.2f}%")
