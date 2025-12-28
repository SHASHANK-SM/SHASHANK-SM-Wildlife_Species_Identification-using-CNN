import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image
import os
import requests
import os
from typing import List, Dict
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env")

client = Groq(api_key=GROQ_API_KEY)

# Conversation structure (optional, can just send single messages)
class Conversation:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are a useful AI assistant."}
        ]

conversation = Conversation()

def generate_description_with_groq(user_message: str) -> str:
    # Append user message
    conversation.messages.append({"role": "user", "content": user_message})
    
    try:
        # Query Groq API
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation.messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,  # optional, allows streaming response
        )
        
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        
        # Append AI response to conversation (optional)
        conversation.messages.append({"role": "assistant", "content": response})
        
        return response
    
    except Exception as e:
        return f"Error calling Groq API: {e}"




# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Wildlife Species Identifier", page_icon="🐾", layout="wide")
st.title("🐾 Wildlife Species Identification using CNN")
st.write("Upload an image, AI will identify its species, and you can generate a description via Groq!")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(num_classes):
    weights = ResNet101_Weights.IMAGENET1K_V2
    model = resnet101(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("best_resnet101.pth", map_location="cpu"))
    model.eval()
    return model

# -------------------------------
# Define Transforms
# -------------------------------
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
class_names = sorted(os.listdir(train_data_dir))
model = load_model(num_classes=len(class_names))


# -------------------------------
# Upload Images
# -------------------------------
uploaded_files = st.file_uploader(
    "Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", width=300)

        # Model Prediction
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        confidence, predicted_class = torch.max(probabilities, dim=0)
        predicted_label = class_names[predicted_class.item()]
        st.subheader(f"Prediction: 🐆 **{predicted_label}**")
        st.progress(float(confidence.item()))

        # Top-5 Predictions
        st.write("### 🔝 Top-5 Predictions:")
        top5_prob, top5_class = torch.topk(probabilities, 5)
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.markdown(f"""
                <div style="
                    padding:10px;
                    margin:5px;
                    background-color:#ffffffcc;
                    border-radius:12px;
                    box-shadow:1px 1px 6px rgba(0,0,0,0.2);
                    text-align:center;
                    color:black;">
                    <b>{class_names[top5_class[i].item()]}</b><br>
                    {top5_prob[i].item()*100:.2f}%
                </div>
                """, unsafe_allow_html=True)

        # Editable message for description
        st.write(f"### A Small description about {predicted_label}")

        if st.button(f"Click to know more about {predicted_label}"):
            description = generate_description_with_groq(f"In 4–5 sentences, describe the {predicted_label} including its habitat, key physical traits, estimated population status, and one reason why it is important for the environment, reason for its endangerment and humans. Keep the tone inspiring, encouraging people to appreciate and protect this species.")
            st.info(description)

        
