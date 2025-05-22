
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import re
# Import OpenAI client - make it conditional to handle environments without the package
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .data_utils import load_categories # Assuming this doesn't create circular dependency

# Default Ollama settings
DEFAULT_LLM_API_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "llama3.1:8b" # Example, ensure this is a valid model you have

# Default OpenAI settings
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

def _call_llm_api(prompt: str, operation_name: str = "LLM call") -> Optional[str]:
    """
    Call the LLM API based on the selected provider (Ollama or OpenAI)
    
    This function checks the session state for API configuration and uses the appropriate
    provider. Falls back to Ollama as the default provider.
    """
    provider = st.session_state.get("llm_provider", "ollama")
    
    if provider == "openai":
        api_key = st.session_state.get("openai_api_key", "")
        model_name = st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL)
        
        if not api_key:
            st.warning("OpenAI API key is not set. Please enter your API key in the sidebar.")
            return None
            
        if not OPENAI_AVAILABLE:
            st.error("OpenAI Python package is not installed. Please install it with 'pip install openai'")
            return None
            
        return _call_openai_api(prompt, api_key, model_name, operation_name)
    else: # Default to Ollama
        api_url = st.session_state.get("ollama_api_url", DEFAULT_LLM_API_URL)
        model_name = st.session_state.get("ollama_model", DEFAULT_LLM_MODEL)
        
        return _call_ollama_api(prompt, api_url, model_name, operation_name)

def strip_thinking_tags(text: Optional[str]) -> Optional[str]:
    """
    Strips content between <thinking> and </thinking> tags from model responses.
    Handles None input gracefully.
    """
    if text is None:
        return None
    import re
    # Remove content between <thinking> and </thinking> tags
    cleaned_text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    # Also remove any standalone tags that might be left
    cleaned_text = cleaned_text.replace('<thinking>', '').replace('</thinking>', '')
    # Trim any excessive whitespace that might result from removing sections
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text.strip()

def _call_ollama_api(prompt: str, api_url: str, model_name: str, operation_name: str) -> Optional[str]:
    """Call the Ollama API with the given prompt and model."""
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        print(f"Sending request to Ollama LLM ({model_name}) for {operation_name}...")
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        llm_response_text = data.get("response", "").strip()
        
        # Strip thinking tags from the response
        llm_response_text = strip_thinking_tags(llm_response_text)
        
        if not llm_response_text and prompt: # Check if prompt was non-empty
            print(f"Ollama LLM returned an empty response for {operation_name} with a non-empty prompt.")
        elif llm_response_text: # Only log success if we got a response
            print(f"Ollama LLM {operation_name} successful.")
        # If llm_response_text is empty and prompt was also empty, it's not an error.
        return llm_response_text
    except requests.exceptions.Timeout:
        print(f"Ollama LLM API request timed out during {operation_name}.")
        st.error(f"LLM request timed out. ({operation_name})")
    except requests.exceptions.RequestException as e:
        print(f"Ollama LLM API communication error during {operation_name}: {e}")
        st.error(f"LLM API error: {e} ({operation_name})")
    except Exception as e:
        print(f"An unexpected error occurred during Ollama LLM {operation_name}: {e}")
        st.error(f"Unexpected LLM error: {e} ({operation_name})")
    return None

def _call_openai_api(prompt: str, api_key: str, model_name: str, operation_name: str) -> Optional[str]:
    """Call the OpenAI API with the given prompt and model."""
    try:
        print(f"Sending request to OpenAI ({model_name}) for {operation_name}...")
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise, accurate summaries and follows formatting instructions precisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Consider making this configurable
            max_tokens=1000  # Consider making this configurable
        )
        
        llm_response_text = response.choices[0].message.content.strip()
        llm_response_text = strip_thinking_tags(llm_response_text) # Also strip for OpenAI
        
        if not llm_response_text and prompt:
            print(f"OpenAI returned an empty response for {operation_name} with a non-empty prompt.")
        elif llm_response_text:
            print(f"OpenAI {operation_name} successful.")
        
        return llm_response_text
    except Exception as e:
        print(f"An error occurred during OpenAI {operation_name}: {e}")
        st.error(f"OpenAI API error: {e} ({operation_name})")
        return None

def generate_summary_and_category(
    bucket_data: Dict[str, Any], 
    return_prompt: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Generates a summary and category for a given bucket of activity data.
    This is a simpler version, not allowing new category suggestions.

    Args:
        bucket_data: Dictionary containing 'titles' and optionally 'ocr_text'.
        return_prompt: If True, the generated prompt is also returned.

    Returns:
        A tuple (summary_text, category_id, prompt_text_or_none).
        prompt_text_or_none is the generated prompt if return_prompt is True, else None.
    """
    bucket_titles = bucket_data.get("titles", [])
    bucket_ocr_texts = bucket_data.get("ocr_text", [])
    
    # Ensure inputs are lists of strings
    titles_to_send = [str(t) for t in bucket_titles if t and str(t).strip()] if bucket_titles else []
    ocr_to_send = [str(o) for o in bucket_ocr_texts if o and str(o).strip()] if bucket_ocr_texts else []

    if not titles_to_send and not ocr_to_send:
        empty_prompt_msg = "No input titles or OCR text provided for summarization."
        return None, None, (empty_prompt_msg if return_prompt else None)

    categories = load_categories()
    prompt_text = "Please provide a concise summary of computer activity based on the following window titles and detected text fragments. Focus on the primary tasks or topics.\n\n"
    if titles_to_send:
        prompt_text += "Window Titles:\n" + "\n".join([f'- \"{t}\"' for t in titles_to_send]) + "\n\n"
    if ocr_to_send:
        prompt_text += "Detected Text (OCR Snippets):\n" + "\n".join([f'- \"{o}\"' for o in ocr_to_send]) + "\n\n"
    
    if categories:
        prompt_text += "Based on the activity, please categorize it into ONE of the following categories:\n\n"
        for cat in categories:
            prompt_text += f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
        prompt_text += "\nFirst, provide a concise summary of the activity.\nThen, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity."
    else:
        prompt_text += "Output only the summary text."

    llm_response = _call_llm_api(prompt_text, "bucket summarization and categorization")
    if not llm_response:
        return None, None, (prompt_text if return_prompt else None)

    summary_text = llm_response
    category_id = None
    if categories: # Only parse category if categories were part of the prompt
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            summary_text = parts[0].strip()
            category_text_segment = parts[1].strip()
            if category_text_segment: # Ensure there's text after CATEGORY:
                category_id = category_text_segment.split()[0] if category_text_segment.split() else None
                # Validate against known category IDs
                if category_id and category_id not in [cat.get("id", "") for cat in categories]:
                    category_id = None # Invalidate if not a known ID
    
    if return_prompt:
        return summary_text, category_id, prompt_text
    else:
        return summary_text, category_id, None


def generate_summary_from_raw_with_llm(
    bucket_titles: List[str],
    bucket_ocr_texts: List[str],
    allow_suggestions: bool = True,
    return_prompt: bool = False
) -> Tuple[str, str, str, Optional[str]]:
    """
    Generates a summary, category ID, and optionally a new category suggestion from raw titles and OCR text.
    This is the primary function for processing raw activity data with LLM.

    Args:
        bucket_titles: List of window titles.
        bucket_ocr_texts: List of OCR text snippets.
        allow_suggestions: Whether to prompt the LLM for new category suggestions.
        return_prompt: If True, the generated prompt is also returned.

    Returns:
        A tuple (summary_text, category_id, suggested_category, prompt_text_or_none).
        prompt_text_or_none is the generated prompt if return_prompt is True, else None.
    """
    titles_to_send = [str(t) for t in bucket_titles if t and str(t).strip()] if bucket_titles else []
    ocr_to_send = [str(o) for o in bucket_ocr_texts if o and str(o).strip()] if bucket_ocr_texts else []
    
    if not titles_to_send and not ocr_to_send:
        print("No raw titles or OCR text available for LLM processing")
        empty_prompt_msg = "No input titles or OCR text provided for LLM processing."
        return "", "", "", (empty_prompt_msg if return_prompt else None)

    categories = load_categories()
    prompt_text = "Please provide a concise summary of computer activity based on the following window titles and detected text fragments. Focus on the primary tasks or topics.\n\n"
    if titles_to_send:
        prompt_text += "Window Titles:\n" + "\n".join([f'- \"{t}\"' for t in titles_to_send]) + "\n\n"
    if ocr_to_send:
        prompt_text += "Detected Text (OCR Snippets):\n" + "\n".join([f'- \"{o}\"' for o in ocr_to_send]) + "\n\n"

    if categories:
        prompt_text += "Based on the activity, please categorize it into ONE of the following categories:\n\n"
        for cat in categories:
            prompt_text += f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
        if allow_suggestions:
            prompt_text += "\nIf none of these categories fit well, you can suggest a new category instead."
            prompt_text += "\n\nFirst, provide a concise summary of the activity."
            prompt_text += "\nThen, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity, or 'none' if no categories fit well."
            prompt_text += "\nIf you answered 'none', then on a new line after 'SUGGESTION:', provide a suggested new category name and short description in the format 'name | description'."
        else:
            prompt_text += "\nFirst, provide a concise summary of the activity. Then, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity."
    else: # No categories defined
        prompt_text += "Output only the summary text."

    llm_response = _call_llm_api(prompt_text, "raw data summarization and categorization")
    if not llm_response:
        return "", "", "", (prompt_text if return_prompt else None)

    summary_text = llm_response
    category_id = ""
    suggested_category = ""

    if categories: # Only parse category if categories were part of the prompt
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            summary_text = parts[0].strip()
            category_text_segment = parts[1].strip()
            
            suggestion_parts = category_text_segment.split("SUGGESTION:")
            if allow_suggestions and len(suggestion_parts) > 1:
                actual_category_text = suggestion_parts[0].strip()
                suggested_category = suggestion_parts[1].strip()
            else:
                actual_category_text = category_text_segment
            
            # Extract category ID
            if actual_category_text.split(): # Check if there's any text to split
                category_id = actual_category_text.split()[0]
            else: # No text after CATEGORY: or after SUGGESTION:
                category_id = ""
            
            if category_id.lower() == "none":
                category_id = "" # Explicitly 'none' means no existing category fits
            else:
                # Validate against known category IDs
                valid_cat_ids = [cat.get("id", "") for cat in categories]
                if category_id not in valid_cat_ids:
                    category_id = "" # If invalid, treat as uncategorized
    
    if return_prompt:
        return summary_text, category_id, suggested_category, prompt_text
    else:
        return summary_text, category_id, suggested_category, None


def refine_summary_with_llm(
    original_summary: str,
    user_feedback: str,
    current_category_id: str = "",
    allow_suggestions: bool = True,
    return_prompt: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Refines an existing summary using user feedback and LLM, optionally re-categorizing.

    Args:
        original_summary: The current summary text.
        user_feedback: User's input for refinement.
        current_category_id: The current category ID of the activity.
        allow_suggestions: Whether to prompt for new category suggestions.
        return_prompt: If True, the generated prompt is also returned.

    Returns:
        A tuple (refined_summary, new_category_id, suggested_category, prompt_text_or_none).
    """
    categories = load_categories()
    current_category_name = "Uncategorized"
    if current_category_id:
        cat_match = next((cat.get("name", "Unknown") for cat in categories if cat.get("id") == current_category_id), "Uncategorized")
        current_category_name = cat_match

    prompt = f"""Current summary of activity:
"{original_summary}"

User's feedback or additional details:
"{user_feedback}"

Based on the current summary and the user's input, please provide an improved and concise summary.
If the user's input suggests a complete rewrite, then generate that new summary.
Focus on integrating the user's points accurately.
"""
    if categories:
        prompt += f"\nThe current activity is categorized as: {current_category_name} ({current_category_id or 'N/A'})\n\n"
        prompt += "Available categories:\n"
        for cat in categories:
            prompt += f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
        if allow_suggestions:
            prompt += "\nIf none of these categories fit well, you can suggest a new category instead."
            prompt += "\n\nFirst, provide the updated summary."
            prompt += f"\nThen, on a new line after 'CATEGORY:', provide the category ID that best matches (either keep '{current_category_id or 'current'}' or suggest a different one if appropriate), or 'none' if no categories fit well."
            prompt += "\nIf you answered 'none', then on a new line after 'SUGGESTION:', provide a suggested new category name and short description in the format 'name | description'."
        else:
            prompt += "\nFirst, provide the updated summary. Then, on a new line after 'CATEGORY:', provide the category ID that best matches (either keep the current one or suggest a different one if appropriate)."
    else: # No categories defined
        prompt += "\nOutput only the new summary text."

    llm_response = _call_llm_api(prompt, "summary refinement and re-categorization")
    if not llm_response:
        return None, current_category_id, "", (prompt if return_prompt else None)

    refined_summary = llm_response
    new_category_id = current_category_id # Default to keeping current category
    suggested_category = ""

    if categories: # Only parse category if categories were part of the prompt
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            refined_summary = parts[0].strip()
            category_text_segment = parts[1].strip()

            suggestion_parts = category_text_segment.split("SUGGESTION:")
            if allow_suggestions and len(suggestion_parts) > 1:
                actual_category_text = suggestion_parts[0].strip()
                suggested_category = suggestion_parts[1].strip()
            else:
                actual_category_text = category_text_segment
            
            parsed_category_id = actual_category_text.split()[0] if actual_category_text.split() else ""

            if parsed_category_id.lower() == "none":
                new_category_id = "" # Explicit 'none' means remove current category or suggest new
            elif parsed_category_id: # If a category ID was provided
                valid_cat_ids = [cat.get("id", "") for cat in categories]
                if parsed_category_id in valid_cat_ids:
                    new_category_id = parsed_category_id # Update to the new valid ID
                # If parsed_category_id is not 'none' and not valid, new_category_id remains current_category_id
    
    if return_prompt:
        return refined_summary, new_category_id, suggested_category, prompt
    else:
        return refined_summary, new_category_id, suggested_category, None


# Function to get available OpenAI models (list of common ones)
def get_available_openai_models() -> List[str]:
    """Return a list of commonly used OpenAI models, can be expanded."""
    return [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4-turbo-preview", # Alias often points to latest turbo preview
        "gpt-4-0125-preview", 
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k", # If still relevant
    ]

# Function to get available Ollama models (attempts to query the API)
def get_available_ollama_models(api_url: str = DEFAULT_LLM_API_URL) -> Tuple[List[str], Dict[str, str]]:
    """
    Fetch available models from Ollama API with detailed information.
    
    Args:
        api_url (str): The Ollama API URL (e.g., http://localhost:11434/api/generate)
        
    Returns:
        tuple: (model_names, model_display_info) where:
            - model_names is a list of model identifiers (e.g., "llama3:8b")
            - model_display_info maps model names to formatted display info (e.g., "8B (Llama)")
    """
    model_names: List[str] = []
    model_display_info: Dict[str, str] = {}
    
    # Extract base URL (remove /api/generate or similar if present)
    base_url = api_url.split('/api/')[0] if '/api/' in api_url else api_url
    tags_url = f"{base_url}/api/tags" # Endpoint to list local models
    
    try:
        response = requests.get(tags_url, timeout=5) # Increased timeout slightly
        if response.status_code == 200:
            data = response.json()
            models_data = data.get("models", [])
            
            for model_spec in models_data:
                name = model_spec.get("name")
                if not name: continue # Skip if no name
                
                # Skip embedding models more reliably
                if "embed" in name.lower() or "embedding" in name.lower() or (model_spec.get("details", {}).get("family", "") == "bert"): # Bert is often for embeddings
                    continue
                    
                details = model_spec.get("details", {})
                parameter_size = details.get("parameter_size", "") # e.g., "8B"
                family = details.get("family", "") # e.g., "llama"
                
                model_names.append(name)
                
                display_parts = []
                if parameter_size: display_parts.append(parameter_size)
                # Add family if it's not already obvious from the model name itself
                if family and family.lower() not in name.lower():
                    display_parts.append(f"({family.capitalize()})")
                
                if display_parts:
                    model_display_info[name] = " ".join(display_parts)
            
            # Sort models: by primary family (llama, mistral, gemma), then by size (smaller first)
            def get_sort_key(model_name_full: str):
                size_val = float('inf')
                # Try to extract size like 3b, 8B, 70b from name or display_info
                # More robust size extraction from model name itself first
                match_name = re.search(r'(\d+\.?\d*)[bB]', model_name_full)
                if match_name:
                    size_val = float(match_name.group(1))
                else: # Fallback to display info if present
                    info_str = model_display_info.get(model_name_full, "")
                    match_info = re.search(r'(\d+\.?\d*)[bB]', info_str)
                    if match_info:
                        size_val = float(match_info.group(1))

                family_prio = 3 # Default for others
                if "llama" in model_name_full.lower(): family_prio = 0
                elif "mistral" in model_name_full.lower(): family_prio = 1
                elif "gemma" in model_name_full.lower(): family_prio = 2
                
                return (family_prio, size_val, model_name_full) # Sort by name as tertiary

            model_names.sort(key=get_sort_key)
            
            if not model_names: # Fallback if API returns empty or only filtered models
                print("No suitable Ollama models found via API, returning defaults.")
                default_models = ["llama3:8b", "mistral:7b", "gemma:7b"] # Common defaults
                return (default_models, {m: "" for m in default_models}) # Empty display info for defaults
                
            return model_names, model_display_info
        else:
            print(f"Error fetching Ollama models from {tags_url}: HTTP {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print(f"Timeout connecting to Ollama API at {tags_url}")
    except requests.exceptions.ConnectionError:
        print(f"Connection error to Ollama API at {tags_url}. Is Ollama running?")
    except Exception as e:
        print(f"Error connecting to Ollama API or parsing models: {e}")
        
    # Fallback to a minimal list of common models if API fails
    print("Falling back to default Ollama model list.")
    default_models = ["llama3:8b", "mistral:7b"] # Or just ["llama3:latest"]
    return (default_models, {m: "" for m in default_models})


# Function to test LLM API connection
def test_llm_connection() -> Tuple[bool, str]:
    """Test the connection to the currently selected LLM provider"""
    provider = st.session_state.get("llm_provider", "ollama")
    test_prompt = "Please respond with just the word 'connected'." # Simple, non-creative prompt
    
    if provider == "openai":
        api_key = st.session_state.get("openai_api_key", "")
        model_name = st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL)
        
        if not api_key: return False, "OpenAI API key is not set"
        if not OPENAI_AVAILABLE: return False, "OpenAI package not installed"
        
        try:
            response = _call_openai_api(test_prompt, api_key, model_name, "connection test")
            if response and "connect" in response.lower():
                return True, f"Connected to OpenAI ({model_name})"
            else:
                return False, f"Failed to get a valid response from OpenAI. Got: '{str(response)[:50]}...'"
        except Exception as e: # Catching broad exception from _call_openai_api if it raises
            return False, f"OpenAI API error: {str(e)[:100]}" # str(e) in case e is not a string
    else: # Ollama
        api_url = st.session_state.get("ollama_api_url", DEFAULT_LLM_API_URL)
        model_name = st.session_state.get("ollama_model", DEFAULT_LLM_MODEL)
        
        try:
            # First, try to list models as a more basic connectivity check to the Ollama server
            base_url_ollama = api_url.split('/api/')[0] if '/api/' in api_url else api_url
            tags_check_url = f"{base_url_ollama}/api/tags"
            tags_response = requests.get(tags_check_url, timeout=3) # Short timeout for this check
            if tags_response.status_code != 200:
                return False, f"Ollama API error at {tags_check_url} (status {tags_response.status_code}). Is Ollama server running and accessible?"
                
            # If tags endpoint is OK, then test generation with the selected model
            response = _call_ollama_api(test_prompt, api_url, model_name, "connection test")
            if response and "connect" in response.lower():
                return True, f"Connected to Ollama ({model_name})"
            else:
                # Check if model exists if response is bad
                available_models, _ = get_available_ollama_models(api_url)
                if model_name not in available_models:
                    return False, f"Ollama model '{model_name}' not found or not responding. Valid response not received. Available: {available_models[:3]}..."
                return False, f"Failed to get a valid response from Ollama ({model_name}). Got: '{str(response)[:50]}...'"
        except Exception as e:
            return False, f"Ollama API error: {str(e)[:100]}"
        

def get_openai_models_from_api(api_key: str) -> List[str]:
    """
    Fetches the list of available models from OpenAI API using the provided API key.
    Filters for chat/completion models. Returns sorted list.
    """
    if not OPENAI_AVAILABLE or not api_key:
        return []
        
    try:
        client = OpenAI(api_key=api_key)
        models_response = client.models.list()
        all_model_ids = [model.id for model in models_response.data]
        
        # Filter for models likely usable for chat/completion.
        # This is a heuristic based on common naming patterns.
        # Prefixes for models we are interested in:
        gpt_prefixes = ("gpt-4", "gpt-3.5-turbo", "ft:gpt-") # Include fine-tuned models
        # Exclude models clearly for other purposes by keywords in their ID:
        excluded_keywords = ["embedding", "similarity", "edit", "instruct", "whisper", "tts", "dall-e", "vision", "image", "audio", "deprecated"]

        filtered_models = [
            model_id for model_id in all_model_ids
            if model_id.startswith(gpt_prefixes) and
               not any(keyword in model_id for keyword in excluded_keywords)
        ]
        
        # Sort them: e.g., gpt-4o > gpt-4-turbo > gpt-4 > gpt-3.5-turbo, newer dates first
        def sort_key_openai(model_id: str):
            score = 100 # Default for less common
            if "gpt-4o" in model_id: score = 0
            elif "gpt-4-turbo" in model_id: score = 10
            elif model_id.startswith("gpt-4"): score = 20 # Broader gpt-4
            elif "gpt-3.5-turbo" in model_id: score = 30
            elif "ft:gpt-4" in model_id: score = 40 # Fine-tuned later
            elif "ft:gpt-3.5" in model_id: score = 50

            # Date part for secondary sort (newer first, so negative timestamp)
            # Matches YYYYMMDD or MMYYDD (common in previews)
            date_val = 0 
            match = re.search(r'(\d{4}\d{2}\d{2})|(\d{6})', model_id)
            if match:
                date_str = match.group(1) or match.group(2)
                if len(date_str) == 6: # e.g. 012523 -> 20230125
                    # Heuristic for YYMMDD, might need adjustment for older models if any
                    year_prefix = "20" 
                    date_str_full = year_prefix + date_str[4:6] + date_str[0:2] + date_str[2:4] # YYMMDD -> YYYYMMDD
                else:
                    date_str_full = date_str
                try:
                    date_val = -datetime.strptime(date_str_full, "%Y%m%d").timestamp()
                except ValueError: pass # Ignore if not a valid date

            return (score, date_val, model_id)

        sorted_models = sorted(filtered_models, key=sort_key_openai)
        
        if not sorted_models:
            print("No suitable OpenAI models found after filtering.")
            return get_available_openai_models() # Fallback to static list
            
        return sorted_models
    except Exception as e:
        print(f"Error fetching or processing OpenAI models from API: {e}")
        # Fallback to a static list if API call fails or parsing error
        return get_available_openai_models() 