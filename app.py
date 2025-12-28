import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from medical_resnet import MedicalCNN

# ===============================
# 1. Page Config
# ===============================
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection from Chest X-ray")
st.write("Deep Learning based medical decision support system")

# ===============================
# 2. Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 3. Load Model (SAME MODEL USED IN TRAINING)
# ===============================
@st.cache_resource
def load_model():
    model = MedicalCNN(num_classes=2)
    model.load_state_dict(
        torch.load("best_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

model = load_model()

# ===============================
# 4. Transforms (same as training)
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["Normal", "Pneumonia"]

# ===============================
# 5. Upload Image
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    # ===============================
    # 6. Prediction
    # ===============================
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]
    confidence = confidence.item() * 100

    # ===============================
    # 7. Result
    # ===============================
    st.subheader("🧪 Prediction Result")

    if predicted_class == "Pneumonia":
        st.error(f"⚠️ Pneumonia Detected\n\nConfidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Normal Case\n\nConfidence: {confidence:.2f}%")

    st.warning(
        "⚠️ This system is for educational purposes only and does not replace a medical professional."
    )

# ===============================
# 8. Footer
# ===============================
st.markdown("---")
st.caption("Developed using Deep Learning & Streamlit")
