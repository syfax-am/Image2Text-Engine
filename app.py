import base64
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BlipForConditionalGeneration, BlipProcessor

from batch_processor import process_batch_images
from utils import (check_nsfw_image, generate_caption, 
                   generate_seo_metadata, load_models, moderate_content)

# annoying warnings
warnings.filterwarnings('ignore')

APP_VERSION = "v1.0.0"
icon = Image.open("assets/logo.ico")

#basic page setup
st.set_page_config(
    page_title="Image2Text Engine",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
#some CSS magic
# =============================================
st.markdown("""
<style>
:root {
    --primary-blue: #476da3;
    --secondary-gray: #d5d7d6;
    --accent-dark: #2c3e50;
}

.header-container {
    text-align: center;
    margin-bottom: 2rem;
}

.logo {
    max-width: 200px;
    margin: 0 auto 1rem auto;
}

.title-text {
    font-size: 3rem;
    font-weight: 800;
    margin: 0;
}

.title-blue { color: #476da3; }
.title-gray { color: #d5d7d6; }

.sidebar-version {
    text-align: center;
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e0e0e0;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; }

.stTabs [data-baseweb="tab"] {
    height: 60px;
    white-space: pre-wrap;
    background-color: var(--secondary-gray);
    border-radius: 10px 10px 0px 0px;
    gap: 8px;
    padding: 15px 20px;
    font-weight: 700;
    font-size: 1.2rem;
    color: var(--accent-dark);
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary-blue);
    color: white;
}

.stButton>button {
    border-radius: 25px;
    border: 2px solid var(--primary-blue);
    background: var(--primary-blue);
    color: white;
    font-weight: 600;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background: white;
    color: var(--primary-blue);
    border: 2px solid var(--primary-blue);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(71, 109, 163, 0.3);
}

.stAlert {
    border-radius: 10px;
    border-left: 5px solid var(--primary-blue);
}

h1, h2, h3 {
    color: var(--primary-blue);
    font-weight: 700;
}

.sidebar-header {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0;
    color: var(--primary-blue);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.sidebar-header svg {
    width: 24px;
    height: 24px;
    fill: var(--primary-blue);
}
</style>
""", unsafe_allow_html=True)

# =============================================
# Get things ready - setup session state
# =============================================
if 'generated_captions' not in st.session_state:
    st.session_state.generated_captions = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'image_cleared' not in st.session_state:
    st.session_state.image_cleared = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'models_dict' not in st.session_state:
    st.session_state.models_dict = None
if 'processor_dict' not in st.session_state:
    st.session_state.processor_dict = None

def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

#try to show the logo, if not just use text
try:
    img_base64 = get_image_base64("assets/logo.png")
    st.markdown(f"""
    <div class="header-container">
        <img src="data:image/png;base64,{img_base64}" class="logo">
        <div class="title-text">
            <span class="title-blue">Image2Text</span>
            <span class="title-gray"> Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
except:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 3rem; font-weight: 800;">
            <span style="color: #476da3;">Image2Text</span>
            <span style="color: #d5d7d6;"> Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# sidebar stuff
# =============================================
with st.sidebar:
    st.markdown(f'<div class="sidebar-version">{APP_VERSION}</div>', unsafe_allow_html=True)
    
    # Pick your poison - which model to use
    model_choice = st.selectbox(
        "Generation Model",
        ["BLIP Large (Recommended)", "BLIP Base"],
        help="Choose the model for generating captions"
    )
    
    st.markdown("""
    <div class="sidebar-header">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"> <path d="M12,15.5A3.5,3.5 0 0,1 8.5,12A3.5,3.5 0 0,1 12,8.5A3.5,3.5 0 0,1 15.5,12A3.5,3.5 0 0,1 12,15.5M19.43,12.97C19.47,12.65 19.5,12.33 19.5,12C19.5,11.67 19.47,11.34 19.43,11L21.54,9.37C21.73,9.22 21.78,8.95 21.66,8.73L19.66,5.27C19.54,5.05 19.27,4.96 19.05,5.05L16.56,6.05C16.04,5.66 15.5,5.32 14.87,5.07L14.5,2.42C14.46,2.18 14.25,2 14,2H10C9.75,2 9.54,2.18 9.5,2.42L9.13,5.07C8.5,5.32 7.96,5.66 7.44,6.05L4.95,5.05C4.73,4.96 4.46,5.05 4.34,5.27L2.34,8.73C2.22,8.95 2.27,9.22 2.46,9.37L4.57,11C4.53,11.34 4.5,11.67 4.5,12C4.5,12.33 4.53,12.65 4.57,12.97L2.46,14.63C2.27,14.78 2.22,15.05 2.34,15.27L4.34,18.73C4.46,18.95 4.73,19.03 4.95,18.95L7.44,17.94C7.96,18.34 8.5,18.68 9.13,18.93L9.5,21.58C9.54,21.82 9.75,22 10,22H14C14.25,22 14.46,21.82 14.5,21.58L14.87,18.93C15.5,18.68 16.04,18.34 16.56,17.94L19.05,18.95C19.27,19.03 19.54,18.95 19.66,18.73L21.66,15.27C21.78,15.05 21.73,14.78 21.54,14.63L19.43,12.97Z"/> </svg>
        Settings
    </div>
    """, unsafe_allow_html=True)
    
    max_length = st.slider("Max caption length", 10, 100, 50,
        help="Maximum number of words in generated caption")
    
    use_beam_search = st.checkbox("Use Beam Search", value=True,
        help="Better quality captions (slower but more accurate)")
    
    if use_beam_search:
        num_beams = st.slider("Number of beams", 1, 5, 3,
            help="Higher values = better quality but slower generation")
    else:
        num_beams = 1
    
    temperature = st.slider("Creativity", 0.1, 1.0, 0.7,
        help="Lower = more predictable, Higher = more creative")
    
    st.markdown("""
    <div class="sidebar-header">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M16.36 14c.08-.66.14-1.32.14-2 0-.68-.06-1.34-.14-2h3.38c.16.64.26 1.31.26 2s-.1 1.36-.26 2m-5.15 5.56c.6-1.11 1.06-2.31 1.38-3.56h2.95a8.03 8.03 0 01-4.33 3.56zM14.34 14H9.66c-.1-.66-.16-1.32-.16-2 0-.68.06-1.35.16-2h4.68c.09.65.16 1.32.16 2 0 .68-.07 1.34-.16 2zM12 19.96c-.83-1.2-1.5-2.53-1.91-3.96h3.82c-.41 1.43-1.08 2.76-1.91 3.96zM8 8H5.08A7.923 7.923 0 019.4 4.44C8.8 5.55 8.35 6.75 8 8zm-2.92 8H8c.35 1.25.8 2.45 1.4 3.56A8.008 8.008 0 015.08 16zm-.82-2C4.1 13.36 4 12.69 4 12s.1-1.36.26-2h3.38c-.08.66-.14 1.32-.14 2 0 .68.06 1.34.14 2H4.26zM12 4.03c.83 1.2 1.5, 2.54 1.91 3.97h-3.82c.41-1.43 1.08-2.77 1.91-3.97zM18.92 8h-2.95a15.65 15.65 0 00-1.38-3.56c1.84.63 3.37 1.9 4.33 3.56zM12 2C6.47 2 2 6.5 2 12a10 10 0 0010 10a10 10 0 0010-10A10 10 0 0012 2z"/>
        </svg>
        SEO
    </div>
    """, unsafe_allow_html=True)
    
    auto_seo = st.checkbox("Auto SEO generation", value=True,
        help="Generate keywords and meta descriptions automatically")
    
    st.markdown("""
    <div class="sidebar-header">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/>
        </svg>
        Security
    </div>
    """, unsafe_allow_html=True)
    
    enable_nsfw_check = st.checkbox("Enable NSFW detection", value=True,
        help="Block inappropriate images before processing")
    enable_moderation = st.checkbox("Enable content moderation", value=True,
        help="Check generated captions for inappropriate content")

# =============================================
# looad the models, this might take a sec
# =============================================
@st.cache_resource(show_spinner=False)
def load_all_models():
    return load_models()

# Only load models once, then cache them
if not st.session_state.models_loaded:
    try:
        with st.spinner("Loading models... This may take a few seconds for the first time."):
            models_dict, processor_dict = load_all_models()
            st.session_state.models_dict = models_dict
            st.session_state.processor_dict = processor_dict
            st.session_state.models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# =============================================
# tabs
# =============================================
tab1, tab2 = st.tabs(["IMAGE → TEXT", "BATCH PROCESSING"])

with tab1:    
    uploaded_image = st.file_uploader(
        "Choose an image...", 
        type=["png", "jpg", "jpeg"],
        key="single_uploader"
    )
    
    #handle image state changes
    if uploaded_image is None and st.session_state.current_image is not None:
        st.session_state.current_image = None
        st.session_state.generated_captions = []
        st.session_state.image_cleared = True
        st.rerun()
    
    if uploaded_image and uploaded_image != st.session_state.current_image:
        st.session_state.current_image = uploaded_image
        st.session_state.generated_captions = []
        st.session_state.image_cleared = False
    
    if st.session_state.image_cleared:
        st.info("Image cleared. Upload a new image to continue.")
        st.session_state.image_cleared = False
    
    # If we have an image to process
    if st.session_state.current_image and not st.session_state.image_cleared:
        current_image = st.session_state.current_image
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(current_image).convert("RGB")
            st.image(image, width=280, caption="Uploaded Image", use_container_width=False)
            
            nsfw_detected = False
            if enable_nsfw_check:
                with st.spinner("Checking image safety..."):
                    try:
                        nsfw_score, nsfw_class = check_nsfw_image(image)
                        
                        if nsfw_score > 0.9:
                            st.error(f"NSFW content detected with {nsfw_score:.1%} confidence! Image processing blocked.")
                            st.session_state.current_image = None
                            nsfw_detected = True
                        elif nsfw_score > 0.7:
                            st.warning(f"Potential NSFW content detected ({nsfw_score:.1%} confidence). Proceed with caution.")
                        else:
                            st.success("Image safety check passed")
                    except Exception as e:
                        st.warning(f"NSFW check unavailable: {str(e)}")
        
        if nsfw_detected:
            st.stop()
        
        with col2:            
            with st.spinner("Generating caption..."):
                try:
                    #f figure out which model they actually picked
                    actual_model = "BLIP Large" if "Large" in model_choice else "BLIP Base"
                    
                    caption = generate_caption(
                        image, 
                        actual_model, 
                        st.session_state.models_dict, 
                        st.session_state.processor_dict,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature
                    )
                    
                    with st.expander("Caption", expanded=True):
                        st.markdown(f"**{caption}**")
                    
                    # Check if the caption is appropriate
                    if enable_moderation:
                        toxicity_score = moderate_content(caption)
                        if toxicity_score > 0.7:
                            st.error(f"Potentially toxic content detected (score: {toxicity_score:.2f})")
                        else:
                            st.success("Content moderation passed")
                    
                    # generate seo  if enabled
                    if auto_seo:
                        keywords, meta_desc, _ = generate_seo_metadata(caption)
                        with st.expander("SEO Optimization", expanded=True):
                            st.write(f"**Keywords:** {', '.join(keywords)}")
                            st.write(f"**Meta Description:** {meta_desc}")
                    
                    #Save results
                    st.session_state.generated_captions.append({
                        'image': current_image.name,
                        'caption': caption,
                        'model': actual_model,
                        'keywords': keywords if auto_seo else []
                    })
                    
                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")

with tab2:    
    st.info("""
    **Batch Processing Feature:** 
    Upload a ZIP file containing your images to process them in bulk and download the results.
    """)
    
    uploaded_zip = st.file_uploader(
        "Upload a ZIP file containing your images",
        type="zip",
        key="batch_uploader"
    )
    
    # show button disabled if no ZIP
    actual_model = "BLIP Large" if "Large" in model_choice else "BLIP Base"
    
    process_clicked = st.button("Start Batch Processing", 
                               disabled=uploaded_zip is None,
                               key="batch_process_btn")
    
    if uploaded_zip and process_clicked:
        with st.spinner("Processing images..."):
            try:
                results_df = process_batch_images(
                    uploaded_zip, 
                    actual_model, 
                    st.session_state.models_dict, 
                    st.session_state.processor_dict,
                    enable_seo=auto_seo,
                    enable_nsfw_check=enable_nsfw_check
                )
                
                st.success(f"{len(results_df)} images processed successfully!")
                st.dataframe(results_df, use_container_width=True)
                
                #   download the results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv,
                    file_name="batch_caption_results.csv",
                    mime="text/csv"
                )
                
                json_data = results_df.to_dict('records')
                st.download_button(
                    label="Download Results (JSON)",
                    data=json.dumps(json_data, indent=2),
                    file_name="batch_caption_results.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error during batch processing: {str(e)}")

# =============================================
# contact info
# =============================================
st.markdown("---")

email_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V4Zm2-1a1 1 0 0 0-1 1v.217l7 4.2 7-4.2V4a1 1 0 0 0-1-1H2Zm13 2.383-4.708 2.825L15 11.105V5.383Zm-.034 6.876-5.64-3.471L8 9.583l-1.326-.795-5.64 3.47A1 1 0 0 0 2 13h12a1 1 0 0 0 .966-.741ZM1 11.105l4.708-2.897L1 5.383v5.722Z"/></svg>"""

linkedin_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/></svg>"""

github_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>"""

portfolio_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M6.5 1A1.5 1.5 0 0 0 5 2.5V3H1.5A1.5 1.5 0 0 0 0 4.5v8A1.5 1.5 0 0 0 1.5 14h13a1.5 1.5 0 0 0 1.5-1.5v-8A1.5 1.5 0 0 0 14.5 3H11v-.5A1.5 1.5 0 0 0 9.5 1h-3zm0 1h3a.5.5 0 0 1 .5.5V3H6v-.5a.5.5 0 0 1 .5-.5zm1.886 6.914L15 7.151V12.5a.5.5 0 0 1-.5.5h-13a.5.5 0 0 1-.5-.5V7.15l6.614 1.764a1.5 1.5 0 0 0 .772 0zM1.5 4h13a.5.5 0 0 1 .5.5v1.616L8.129 6.948a.5.5 0 0 1-.258 0L1 6.116V4.5a.5.5 0 0 1 .5-.5z"/></svg>"""

with st.expander("**🟢 Interested in my skills? Let's work together!**", expanded=False):
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center;">
          <a href="mailto:syfaxaitmedjber@gmail.com?subject=Contact%20from%20Palette%20App&body=Hello%20Syfax,%0A%0AI%20am%20interested%20in%20your%20skills...">
            {email_icon}<br/>
            <strong>Email</strong>
          </a>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
          <a href="https://www.linkedin.com/in/syfax-ait-medjber/" target="_blank">
            {linkedin_icon}<br/>
            <strong>LinkedIn</strong>
          </a>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div style="text-align: center;">
          <a href="https://github.com/syfax-am/" target="_blank">
            {github_icon}<br/>
            <strong>GitHub</strong>
          </a>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div style="text-align: center;">
          <a href="https://my-portfolio.com" target="_blank">
            {portfolio_icon}<br/>
            <strong>Portfolio</strong>
          </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "*I'm passionate about Data Science and AI - especially NLP, Computer Vision, and model deployment. Always happy to connect, share ideas, or collaborate on meaningful projects.*"
    )