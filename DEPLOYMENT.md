# 🚀 Deployment Guide for Neuroimaging Tumor Profiler

## Prerequisites

- Python 3.8 or higher
- Git
- Streamlit Cloud account (or other deployment platform)

## Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd NeuroAI
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app locally:**
   ```bash
   streamlit run brain_tumor_predictor.py
   ```

## Deployment to Streamlit Cloud

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add deployment files"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the path to your app: `brain_tumor_predictor.py`
   - Click "Deploy"

## Important Notes

### Model Files
Make sure your trained model files are included in the repository:
- `brain_tumor_resnet18.pth` (or your model file)
- Update the `MODEL_PATH` in `brain_tumor_predictor.py` if needed

### Dependencies
The `requirements.txt` file includes all necessary packages:
- `torch` and `torchvision` for the AI model
- `opencv-python` for image processing
- `scipy` for scientific computing
- `streamlit` for the web interface

### Troubleshooting

**If you get "ModuleNotFoundError":**
- Ensure all dependencies are listed in `requirements.txt`
- Check that the model file path is correct
- Verify that all imports are available

**If the app loads but doesn't work:**
- Check the Streamlit Cloud logs for detailed error messages
- Ensure your model file is properly uploaded
- Verify that the image validation functions work correctly

## File Structure
```
NeuroAI/
├── brain_tumor_predictor.py      # Main Streamlit app
├── brain_tumor_resnet18.pth      # Trained model
├── requirements.txt               # Dependencies
├── .streamlit/config.toml        # Streamlit config
├── DEPLOYMENT.md                 # This file
└── README.md                     # Project documentation
```

## Support
If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are properly uploaded
3. Test locally before deploying
4. Ensure your model file is accessible 