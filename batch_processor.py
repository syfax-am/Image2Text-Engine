import zipfile
import os
import tempfile
import pandas as pd
from PIL import Image
from utils import generate_caption, generate_seo_metadata, check_nsfw_image

# Setup logging
from logging_config import get_logger
logger = get_logger(__name__)

def process_batch_images(zip_file, model_choice, models_dict, processor_dict, **kwargs):

    logger.info(f"Starting batch processing using model: {model_choice}")

    results = []
    nsfw_blocked = 0
    
    #extract the zip to a temp folder.
    logger.info("Extracting ZIP file...")
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        logger.info(f"ZIP extracted to temporary directory: {temp_dir}")
        
        # go through all the files we just extracted
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    logger.info(f"Processing image: {file}")

                    try:
                        image = Image.open(image_path).convert('RGB')
                        logger.debug(f"Image loaded: {image_path}")
                        
                        #check safety
                        if kwargs.get('enable_nsfw_check', True):
                            nsfw_score, nsfw_class = check_nsfw_image(image)
                            logger.debug(f"NSFW score for {file}: {nsfw_score:.2f} ({nsfw_class})")

                            # block anything too spicy
                            if nsfw_score > 0.9:
                                logger.warning(f"Image {file} blocked due to NSFW content.")

                                results.append({
                                    'File': file,
                                    'Caption': '[BLOCKED] NSFW content detected',
                                    'Keywords': '',
                                    'Meta Description': '',
                                    'NSFW Score': f'{nsfw_score:.1%}',
                                    'Status': 'Blocked - NSFW'
                                })
                                nsfw_blocked += 1
                                continue
                        
                        #   Generate the caption
                        caption = generate_caption(
                            image, model_choice, models_dict, processor_dict
                        )
                        logger.info(f"Caption generated for {file}: {caption}")
                        
                        # SEO if enabled
                        if kwargs.get('enable_seo', True):
                            keywords, meta_desc, _ = generate_seo_metadata(caption)
                            logger.debug(f"SEO metadata for {file}: {keywords}, {meta_desc}")
                        else:
                            keywords, meta_desc = [], ""
                        
                        #results list
                        results.append({
                            'File': file,
                            'Caption': caption,
                            'Keywords': ', '.join(keywords),
                            'Meta Description': meta_desc,
                            'NSFW Score': f'{nsfw_score:.1%}' if kwargs.get('enable_nsfw_check', True) else 'N/A',
                            'Status': 'Success'
                        })
                        logger.info(f"Image {file} processed successfully.")
                    
                    except Exception as e:
                        logger.error(f"Error processing image {file}: {e}")

                        results.append({
                            'File': file,
                            'Caption': '',
                            'Keywords': '',
                            'Meta Description': '',
                            'NSFW Score': 'N/A',
                            'Status': f'Error: {str(e)}'
                        })
    
    #let us know if we blocked any naughty images
    if nsfw_blocked > 0:
        logger.warning(f"Blocked {nsfw_blocked} NSFW images during processing")
    
    logger.info("Batch processing completed.")
    return pd.DataFrame(results)