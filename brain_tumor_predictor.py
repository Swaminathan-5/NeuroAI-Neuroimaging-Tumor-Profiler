import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Model and preprocessing parameters
MODEL_PATH = 'brain_tumor_resnet18.pth'
IMG_SIZE = 128
class_names = ['no', 'yes']  # Adjust if your class order is different

# Preprocessing for validation/prediction
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(128, 1)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image, model):
    img = val_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)
    return class_names[pred], prob

# Helper function to get gradient color based on probability
def get_gradient_color(prob):
    # prob: probability of tumor (0 to 1)
    if prob < 0.25:
        return '#2ecc40'  # Green
    elif prob < 0.5:
        return '#ffe066'  # Yellow
    elif prob < 0.75:
        return '#ffa502'  # Orange
    else:
        return '#ff4136'  # Red

# --- Streamlit UI ---
st.set_page_config(page_title='🧠 Smart Brain MRI Analyzer: AI-Powered Tumor Detection', page_icon='🧠', layout='centered')

# Sidebar
st.sidebar.title('🧠 Brain Tumor Detection')
st.sidebar.markdown('''
**Instructions:**
- Upload an MRI image (JPG, PNG).
- The model will analyze and predict if a brain tumor is present.
- For best results, use clear MRI scans.
''')
st.sidebar.info('Project by [Your Name]')

# Main header
st.title('🧠 Smart Brain MRI Analyzer: AI-Powered Tumor Detection')
st.markdown('<h4 style="text-align:center; color:#555;">Upload an MRI image to analyze for brain tumor presence.</h4>', unsafe_allow_html=True)

# Layout with columns
col1, col2, col3 = st.columns([1,2,1])

with col2:
    uploaded_file = st.file_uploader('Choose an MRI image...', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        with st.spinner('Analyzing image...'):
            model = load_model()
            pred_class, prob = predict_image(image, model)
        st.success('Analysis Complete!')
        grad_color = get_gradient_color(prob)
        if pred_class == 'no' and prob < 0.25:
            st.balloons()
            st.markdown(f'<h2 style="color:{grad_color};">No Tumor Detected 🎉</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:20px; color:{grad_color};">Probability: <b>{(1-prob):.2%}</b> no tumor</p>', unsafe_allow_html=True)
        elif grad_color == '#ffe066':  # Yellow
            st.markdown(f'<h2 style="color:{grad_color};">Possible Early Tumor Growth ⚠️</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:20px; color:{grad_color};">Probability: <b>{prob:.2%}</b> tumor</p>', unsafe_allow_html=True)
            st.info('A tumor may be growing gradually. Please consult a medical professional and schedule follow-up scans for close monitoring.')
        elif pred_class == 'yes':
            st.markdown(f'<h2 style="color:{grad_color};">Tumor Detected ⚠️</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:20px; color:{grad_color};">Probability: <b>{prob:.2%}</b> tumor</p>', unsafe_allow_html=True)
            st.warning('Consult a medical professional for further analysis.')
        else:
            st.markdown(f'<h2 style="color:{grad_color};">No Tumor Detected</h2>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:20px; color:{grad_color};">Probability: <b>{(1-prob):.2%}</b> no tumor</p>', unsafe_allow_html=True)

# Footer
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:#888;">Made with ❤️ using Streamlit & PyTorch</div>', unsafe_allow_html=True) 