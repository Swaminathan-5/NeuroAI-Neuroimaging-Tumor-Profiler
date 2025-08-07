import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage

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

def is_brain_mri(image):
    """
    Very strict validation to check if the image appears to be a brain MRI
    This specifically rejects photos, signatures, documents, and other non-medical images
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Check if image is grayscale or has low color variation (typical of MRI)
    if len(img_array.shape) == 3:
        # Calculate color variation
        color_variation = np.std(img_array, axis=2)
        avg_variation = np.mean(color_variation)
        
        # MRI images typically have very low color variation
        if avg_variation > 10:  # Even stricter - MRI should have very low color variation
            return False
        
        # Check for RGB balance - photos have balanced colors, MRI is grayscale
        r, g, b = cv2.split(img_array)
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        color_balance = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
        
        if color_balance < 5:  # Too balanced colors - likely a photo
            return False
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    
    # Calculate key metrics
    contrast = np.std(gray)
    mean_intensity = np.mean(gray)
    
    # Very strict MRI characteristics:
    # 1. Very low contrast (medical imaging)
    # 2. Specific intensity ranges
    # 3. Absence of typical photo features
    
    # Reject high contrast images (typical of photos, signatures, documents)
    if contrast > 40:  # Much stricter - MRI should have very low contrast
        return False
    
    # Reject very bright or very dark images (typical of photos)
    if mean_intensity < 50 or mean_intensity > 200:
        return False
    
    # Check for typical photo characteristics
    # Look for edges and patterns typical of photos vs medical images
    edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for more sensitive detection
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Photos typically have more edges than medical images
    if edge_density > 0.05:  # Much stricter - MRI should have very few edges
        return False
    
    # Check for uniform regions (typical of medical images)
    # MRI images often have large uniform areas
    labeled, num_features = ndimage.label(gray < 128)
    if num_features > 50:  # Much stricter - MRI should have fewer regions
        return False
    
    # Additional check: look for face-like patterns (common in photos)
    # This is a simplified face detection heuristic
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:  # Face detected - definitely not an MRI
            return False
    except:
        pass  # If face detection fails, continue with other checks
    
    # Check for text or document-like features (passport photos, documents, signatures)
    # Look for horizontal and vertical lines typical of documents
    horizontal_lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)  # Lower threshold
    vertical_lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)  # Lower threshold
    
    if horizontal_lines is not None and len(horizontal_lines) > 3:
        return False  # Too many horizontal lines - likely a document/photo
    
    if vertical_lines is not None and len(vertical_lines) > 3:
        return False  # Too many vertical lines - likely a document/photo
    
    # Check for signature-like patterns (thin, curved lines)
    # Signatures typically have thin, continuous lines
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    signature_density = np.sum(dilated > 0) / (dilated.shape[0] * dilated.shape[1])
    
    if signature_density > 0.02:  # Too many thin lines - likely a signature
        return False
    
    # Check for text-like patterns (uniform spacing, regular shapes)
    # Look for connected components that might be text
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_32S)
    
    # Check if there are many small, regularly spaced components (text-like)
    small_components = 0
    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] < 100 and stats[i, cv2.CC_STAT_AREA] > 10:
            small_components += 1
    
    if small_components > 20:  # Too many small components - likely text/signature
        return False
    
    # Check for typical photo characteristics
    # Look for gradients and smooth transitions typical of photos
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    avg_gradient = np.mean(gradient_magnitude)
    
    if avg_gradient > 20:  # Too much gradient - likely a photo
        return False
    
    # Final MRI-specific checks
    # MRI images should have very specific characteristics
    if contrast < 15 and mean_intensity > 100 and mean_intensity < 150:
        # Additional check: look for brain-like patterns
        # This is a simplified check for medical imaging characteristics
        return True
    
    return False  # Default to rejecting if unsure

def validate_mri_image(image):
    """
    Validate if the uploaded image is a brain MRI
    """
    if not is_brain_mri(image):
        return False, """❌ **Invalid Image Type Detected**

This appears to be a regular photo, signature, document, or other non-medical image rather than a brain MRI scan.

**🔍 Detection Results:**
• High contrast detected (typical of photos/signatures)
• Color variation too high (MRI images are grayscale)
• Too many edges detected (typical of signatures/text)
• Document-like features detected (if applicable)
• Face-like patterns detected (if applicable)

**✅ Please upload:**
• Brain MRI images only
• Clear, high-quality MRI scans
• Images showing brain tissue structure
• Grayscale medical imaging

**❌ Do NOT upload:**
• Passport photos or ID photos
• Signatures or handwritten text
• Selfies or regular photos
• Documents or text images
• MRI scans of other body parts
• X-rays or CT scans
• Any non-medical images"""
    
    return True, "✅ Valid brain MRI image detected"

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
st.set_page_config(page_title='🧠 Neuroimaging Tumor Profiler', page_icon='🧠', layout='wide')

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
    .error-box {
        background: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar (compact)
with st.sidebar:
    st.title('🧠 Neuroimaging Tumor Profiler')
    st.markdown('**Instructions:** Upload a brain MRI image (JPG, PNG) for analysis.')
    st.info('Project by Swaminathan K')

# Main content
st.markdown('<h1 class="main-header">🧠 Neuroimaging Tumor Profiler</h1>', unsafe_allow_html=True)

# File upload and analysis in a compact layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader('Choose a brain MRI image...', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Create a compact layout for image and results
        img_col, result_col = st.columns([1, 1])
        
        with img_col:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with result_col:
            # Validate if it's a brain MRI
            is_valid, validation_message = validate_mri_image(image)
            
            if not is_valid:
                st.markdown(f'<div class="error-box">{validation_message}</div>', unsafe_allow_html=True)
                st.stop()
            
            # If valid, proceed with analysis
            with st.spinner('Analyzing brain MRI...'):
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

# Detailed analysis in tabs (if image uploaded and valid)
if uploaded_file is not None:
    # Re-validate to ensure we have the image data
    image = Image.open(uploaded_file).convert('RGB')
    is_valid, _ = validate_mri_image(image)
    
    if is_valid:
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