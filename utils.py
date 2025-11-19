import torch
import numpy as np
from PIL import Image
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import re
from typing import List, Tuple, Dict

#logging
from logging_config import get_logger
logger = get_logger(__name__)

# NLTK data we need
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# wnot using the POS tagger anymore, so no need to download it

# cache for loaded models
_MODELS_LOADED = False
_MODELS_DICT = {}
_PROCESSOR_DICT = {}

def load_models():
    global _MODELS_LOADED, _MODELS_DICT, _PROCESSOR_DICT
    
    if _MODELS_LOADED:
        logger.info("Models already loaded. Returning cached versions.")
        return _MODELS_DICT, _PROCESSOR_DICT
        
    logger.info("Loading models: BLIP Base, BLIP Large, similarity model, NSFW detector...")

    # load BLIP Base
    try:
        _MODELS_DICT["BLIP Base"] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _PROCESSOR_DICT["BLIP Base"] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        logger.info("BLIP Base loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading BLIP Base: {e}")
    
    # Load BLIP Large
    try:
        _MODELS_DICT["BLIP Large"] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        _PROCESSOR_DICT["BLIP Large"] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        logger.info("BLIP Large loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading BLIP Large: {e}")
    
    # lload sentence similarity model
    try:
        _MODELS_DICT["sentence_similarity"] = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SentenceTransformer similarity model loaded.")
    except Exception as e:
        logger.error(f"Error loading similarity model: {e}")
    
    # load NSFW model
    try:
        _MODELS_DICT["nsfw_detector"] = pipeline(
            "image-classification", 
            model="Falconsai/nsfw_image_detection"
        )
        logger.info("NSFW detector model loaded.")
    except Exception as e:
        logger.error(f"Error loading NSFW model: {e}")
    
    _MODELS_LOADED = True
    logger.info("All models loaded successfully.")
    return _MODELS_DICT, _PROCESSOR_DICT

def check_nsfw_image(image: Image.Image) -> Tuple[float, str]:
    """Check if an image contains NSFW content"""
    logger.info("Running NSFW detection...")
    try:
        models_dict, _ = load_models()
        nsfw_detector = models_dict.get("nsfw_detector")
        if not nsfw_detector:
            logger.warning("NSFW detector unavailable.")
            return 0.0, "Model not available"
        
        results = nsfw_detector(image)
        logger.debug(f"NSFW raw results: {results}")
        
        #we look for explicit content labels first
        for result in results:
            if result['label'] in ['nsfw', 'porn', 'adult', 'explicit']:
                logger.warning(f"NSFW content detected: {result['label']} ({result['score']:.2f})")
                return result['score'], result['label']
        
        # If no explicit content, check for safe labels
        for result in results:
            if result['label'] in ['safe', 'sfw', 'normal']:
                logger.info("Image classified as safe.")
                return 1 - result['score'], "safe"
        
        logger.warning("Unknown NSFW classification.")
        return 0.0, "unknown"
        
    except Exception as e:
        logger.error(f"NSFW detection error: {e}")
        return 0.0, "error"

def generate_caption(image, model_name, models_dict, processor_dict, max_length=50, num_beams=3, temperature=0.7):
    """Generate a caption for an image using the specified model"""
    logger.info(f"Generating caption with model: {model_name}")
    try:
        if model_name in ["BLIP Base", "BLIP Large"]:
            processor = processor_dict[model_name]
            model = models_dict[model_name]
            
            # prepare the image
            inputs = processor(images=image, return_tensors="pt")
            logger.debug("Input tensor prepared for caption generation.")
            
            #  Generate the caption
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Caption generated: {caption}")
        else:
            logger.error(f"Unsupported model: {model_name}")
            caption = "Model not supported"
            
        return caption.strip()
    
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return f"Generation error: {str(e)}"

def generate_seo_metadata(caption: str, max_keywords: int = 5) -> Tuple[List[str], str, float]:
    """Generate SEO keywords and meta description from a caption"""
    logger.info("Generating SEO metadata...")
    logger.debug(f"Caption received: {caption}")

    try:
        # simple method (without pos tagging)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', caption.lower())
        
        # Common words to ignore
        stop_words = {
            'with', 'this', 'that', 'there', 'their', 'about', 'would', 'could',
            'should', 'which', 'what', 'when', 'where', 'who', 'whom', 'have',
            'has', 'had', 'been', 'being', 'will', 'shall', 'may', 'might',
            'must', 'can', 'could', 'the', 'a', 'an', 'and', 'or', 'but', 'in',
            'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'are', 'was'
        }
        
        # filter out the  words
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count how often each word appears
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        #good for SEO
        seo_boost_words = {
            'professional', 'quality', 'high', 'best', 'premium', 'luxury',
            'modern', 'contemporary', 'beautiful', 'stunning', 'amazing',
            'excellent', 'perfect', 'ideal', 'ultimate', 'complete'
        }
        
        # Give SEO friendly words a boost
        for word in word_freq:
            if word in seo_boost_words:
                word_freq[word] *= 1.5
        
        # pickin top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        keywords = [kw[0] for kw in keywords]

        logger.info(f"SEO keywords: {keywords}")
        
        #create a meta description from the caption
        words_list = caption.split()
        if len(words_list) > 20:
            meta_desc = ' '.join(words_list[:20])
            # t ry to end at a sentence boundary
            if '.' in meta_desc:
                last_period = meta_desc.rfind('.')
                meta_desc = meta_desc[:last_period + 1]
            else:
                meta_desc += '...'
        else:
            meta_desc = caption
        
        # mke sure it's not too long for search engines
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + '...'
        
        logger.info("SEO metadata generated successfully.")
        return keywords, meta_desc, 0.0
        
    except Exception as e:
        logger.error(f"SEO metadata generation error: {e}")        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', caption.lower())
        word_freq = {}
        
        for word in words:
            if word not in ['with', 'this', 'that', 'there', 'their', 'about', 'would', 'could']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        keywords = [kw[0] for kw in keywords]
        
        words_list = caption.split()
        if len(words_list) > 15:
            meta_description = ' '.join(words_list[:15]) + '...'
        else:
            meta_description = caption
        
        return keywords, meta_description, 0.0

def moderate_content(text: str) -> float:
    """Check text for potentially toxic content"""
    logger.info("Running toxicity moderation...")
    try:
        #patterns that might indicate toxic content
        toxic_patterns = [
            r'\b(hate|violence|kill|attack|terror|abuse|hurt|harm)\b',
            r'\b(die|death|dead|murder|suicide)\b',
            r'\b(racist|sexist|homophobic|transphobic)\b',
            r'\b(nude|naked|porn|sexual|xxx)\b'
        ]
        
        text_lower = text.lower()
        toxicity_score = 0.0
        
        # Check for each toxic pattern
        for pattern in toxic_patterns:
            matches = re.findall(pattern, text_lower)
            toxicity_score += len(matches) * 0.15
        
        # penalty for having multiple toxic terms
        if toxicity_score > 0.3:
            toxicity_score += 0.2

        logger.info(f"Toxicity score: {toxicity_score}")
        return min(toxicity_score, 1.0)
    
    except Exception as e:
        logger.error(f"Toxicity moderation error: {e}")
        return 0.0