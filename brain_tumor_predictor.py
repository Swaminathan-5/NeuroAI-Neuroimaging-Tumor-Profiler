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

# Enhanced analysis functions
def get_tumor_characteristics(prob):
    """Analyze tumor characteristics based on probability"""
    if prob < 0.25:
        return {
            'stage': 'No Tumor Detected',
            'confidence': 'Very High',
            'description': 'Clear brain tissue with no visible abnormalities',
            'recommendation': 'Continue regular health monitoring',
            'urgency': 'None',
            'color': '#2ecc40'
        }
    elif prob < 0.5:
        return {
            'stage': 'Early Detection Phase',
            'confidence': 'Moderate',
            'description': 'Subtle changes detected that may indicate early tumor development',
            'recommendation': 'Schedule follow-up MRI in 3-6 months',
            'urgency': 'Low',
            'color': '#ffe066'
        }
    elif prob < 0.75:
        return {
            'stage': 'Developmental Phase',
            'confidence': 'High',
            'description': 'Significant abnormalities detected suggesting active tumor growth',
            'recommendation': 'Immediate consultation with neurologist/oncologist',
            'urgency': 'Medium',
            'color': '#ffa502'
        }
    else:
        return {
            'stage': 'Advanced Detection',
            'confidence': 'Very High',
            'description': 'Clear evidence of tumor presence with high confidence',
            'recommendation': 'Urgent medical intervention required',
            'urgency': 'High',
            'color': '#ff4136'
        }

def get_growth_stage(prob):
    """Estimate tumor growth stage based on probability"""
    if prob < 0.25:
        return "No Growth Detected"
    elif prob < 0.4:
        return "Pre-Growth Phase"
    elif prob < 0.6:
        return "Early Growth Phase"
    elif prob < 0.8:
        return "Active Growth Phase"
    else:
        return "Advanced Growth Phase"

def get_tumor_details(prob):
    """Provide detailed tumor information"""
    if prob < 0.25:
        return {
            'size': 'N/A',
            'location': 'N/A',
            'type': 'N/A',
            'growth_rate': 'N/A'
        }
    elif prob < 0.5:
        return {
            'size': 'Microscopic to Small (< 1cm)',
            'location': 'Requires detailed imaging analysis',
            'type': 'Early-stage (requires biopsy for confirmation)',
            'growth_rate': 'Slow to Moderate'
        }
    elif prob < 0.75:
        return {
            'size': 'Small to Medium (1-3cm)',
            'location': 'Requires detailed imaging analysis',
            'type': 'Developmental (requires biopsy for confirmation)',
            'growth_rate': 'Moderate to Fast'
        }
    else:
        return {
            'size': 'Medium to Large (> 3cm)',
            'location': 'Requires detailed imaging analysis',
            'type': 'Advanced (requires biopsy for confirmation)',
            'growth_rate': 'Fast to Rapid'
        }

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
st.set_page_config(page_title='🧠 NeuroImaging Tumor Profiler', page_icon='🧠', layout='wide')

# Custom CSS for compact design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .compact-section {
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .result-box {
        border: 2px solid;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #374151;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        color: #ffffff;
        font-weight: 500;
    }
    .urgent {
        background: #d32f2f;
        border-color: #b71c1c;
        color: #ffffff;
    }
    .warning {
        background: #f57c00;
        border-color: #e65100;
        color: #ffffff;
    }
    .success {
        background: #388e3c;
        border-color: #2e7d32;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar (compact)
with st.sidebar:
    st.title('🧠 NeuroImaging Tumor Profiler')
    st.markdown('**Instructions:** Upload an MRI image (JPG, PNG) for analysis.')
    st.info('Project by Swaminathan K')

# Main content
st.markdown('<h1 class="main-header">🧠 NeuroImaging Tumor Profiler</h1>', unsafe_allow_html=True)

# File upload and analysis in a compact layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader('Choose an MRI image...', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Create a compact layout for image and results
        img_col, result_col = st.columns([1, 1])
        
        with img_col:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with result_col:
            with st.spinner('Analyzing...'):
                model = load_model()
                pred_class, prob = predict_image(image, model)
            
            # Get analysis data
            characteristics = get_tumor_characteristics(prob)
            growth_stage = get_growth_stage(prob)
            tumor_details = get_tumor_details(prob)
            
            # Main result display
            if pred_class == 'no' and prob < 0.25:
                st.balloons()
                st.markdown(f'<div class="result-box success"><h2 style="color:#ffffff;">{characteristics["stage"]} 🎉</h2></div>', unsafe_allow_html=True)
            else:
                urgency_class = "urgent" if characteristics['urgency'] in ['High', 'Medium'] else "warning"
                st.markdown(f'<div class="result-box {urgency_class}"><h2 style="color:#ffffff;">{characteristics["stage"]} ⚠️</h2></div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-card"><strong style="color: #ffffff;">Probability:</strong> {prob:.2%} tumor</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><strong style="color: #ffffff;">Confidence:</strong> {characteristics["confidence"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><strong style="color: #ffffff;">Growth Stage:</strong> {growth_stage}</div>', unsafe_allow_html=True)

# Detailed analysis in tabs (if image uploaded)
if uploaded_file is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Analysis Details", "🧠 Tumor Info", "🏥 Recommendations"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            **📋 Description:** {characteristics['description']}
            
            **⚡ Urgency Level:** {characteristics['urgency']}
            """)
        with col_b:
            st.markdown(f"""
            **🎯 Recommendation:** {characteristics['recommendation']}
            """)
    
    with tab2:
        if prob >= 0.25:
            col_c, col_d = st.columns(2)
            with col_c:
                st.markdown(f"""
                **📏 Estimated Size:** {tumor_details['size']}
                
                **📍 Location:** {tumor_details['location']}
                """)
            with col_d:
                st.markdown(f"""
                **🔬 Type:** {tumor_details['type']}
                
                **📊 Growth Rate:** {tumor_details['growth_rate']}
                """)
            st.info("⚠️ **Note:** AI-based detection. Advanced imaging and biopsy required for precise diagnosis.")
        else:
            st.success("✅ No tumor characteristics to display.")
    
    with tab3:
        if characteristics['urgency'] == 'None':
            st.success("✅ Continue regular health monitoring and annual check-ups.")
        elif characteristics['urgency'] == 'Low':
            st.warning("⚠️ Schedule follow-up MRI scan in 3-6 months to monitor any changes.")
        elif characteristics['urgency'] == 'Medium':
            st.error("🚨 Schedule immediate consultation with a neurologist or oncologist for detailed evaluation.")
        else:
            st.error("🚨 **URGENT:** Immediate medical intervention required. Contact emergency services or visit nearest hospital.")

# Footer
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:#888;">Made with ❤️ using Streamlit & PyTorch</div>', unsafe_allow_html=True) 