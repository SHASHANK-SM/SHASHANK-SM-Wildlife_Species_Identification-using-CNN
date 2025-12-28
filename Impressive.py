import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image
import os
from dotenv import load_dotenv
from groq import Groq

# -------------------------------
# Load environment & Groq
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env")

client = Groq(api_key=GROQ_API_KEY)

# Conversation memory
class Conversation:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are a useful AI assistant."}
        ]

conversation = Conversation()

def generate_description_with_groq(user_message: str) -> str:
    conversation.messages.append({"role": "user", "content": user_message})
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation.messages,
            temperature=1,
            max_tokens=512,
            top_p=1,
            stream=True,
        )
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        conversation.messages.append({"role": "assistant", "content": response})
        return response
    except Exception as e:
        return f"Error calling Groq API: {e}"


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Wildlife Species Identifier", page_icon="🐾", layout="wide")

st.markdown(
    """
    <style>
    
    .prediction-box {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    text-align: center;
    animation: fadeIn 1s ease-in-out;
    color: black !important;
}


    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .species-title {
        font-size: 28px;
        font-weight: bold;
        color: #solid_black;  /* coral */

    }
    .top-card {
        padding: 10px;
        margin: 5px;
        background-color: #f0f8ff;
        border-radius: 12px;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.2);
        text-align: center;
        font-weight: bold;
        animation: fadeIn 1.2s ease-in-out;
        color: black !important; /* makes text stand out */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐾 Wildlife Species Identification using CNN")
st.write("Upload an image, and the AI will identify its species. You can also generate an inspiring description via Groq!")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(num_classes):
    weights = ResNet101_Weights.IMAGENET1K_V2
    model = resnet101(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("best_resnet101NEW.pth", map_location="cpu"))
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

        # Split layout (image left, prediction right)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

        with col2:
            # Model Prediction
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]

            confidence, predicted_class = torch.max(probabilities, dim=0)
            predicted_label = class_names[predicted_class.item()]

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="species-title">Prediction: 🐆 {predicted_label}</div>
                    <p><b>Confidence:</b> {confidence.item()*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(confidence.item()))

            # Top-5 Predictions
            st.subheader("🔝 Top-5 Predictions")
            cols = st.columns(5)
            top5_prob, top5_class = torch.topk(probabilities, 5)
            for i in range(5):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="top-card">
                            {class_names[top5_class[i].item()]}<br>
                            {top5_prob[i].item()*100:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Editable message for description
            st.write(f"###  A Small description about {predicted_label}")

            if st.button(f"✨Click to know more about {predicted_label}", key=predicted_label):
                description = generate_description_with_groq(
                    f"In 4–5 sentences, describe the {predicted_label} including its habitat, key physical traits, estimated population status, and one reason why it is important for the environment, reason for its endangerment and humans. Keep the tone inspiring, encouraging people to appreciate and protect this species."
                )
                st.success(description)
