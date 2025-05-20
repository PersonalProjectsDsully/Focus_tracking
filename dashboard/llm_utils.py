import json
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

from .data_utils import load_categories

LLM_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1:8b"


def _call_llm_api(prompt: str, operation_name: str = "LLM call") -> Optional[str]:
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    try:
        print(f"Sending request to LLM ({LLM_MODEL}) for {operation_name}...")
        response = requests.post(LLM_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        llm_response_text = data.get("response", "").strip()
        if not llm_response_text and prompt:
            print(f"LLM returned an empty response for {operation_name}.")
        else:
            print(f"LLM {operation_name} successful.")
        return llm_response_text
    except requests.exceptions.Timeout:
        print(f"LLM API request timed out during {operation_name}.")
    except requests.exceptions.RequestException as e:
        print(f"LLM API communication error during {operation_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during LLM {operation_name}: {e}")
    return None

def generate_summary_and_category(bucket_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    bucket_titles = bucket_data.get("titles", [])
    bucket_ocr_texts = bucket_data.get("ocr_text", [])
    if not bucket_titles and not bucket_ocr_texts:
        return None, None

    categories = load_categories()
    prompt_text = "Please provide a concise summary of computer activity based on the following window titles and detected text fragments. Focus on the primary tasks or topics.\n\n"
    if bucket_titles:
        prompt_text += "Window Titles:\n" + "\n".join([f'- \"{t}\"' for t in bucket_titles]) + "\n\n"
    if bucket_ocr_texts:
        prompt_text += "Detected Text (OCR Snippets):\n" + "\n".join([f'- \"{o}\"' for o in bucket_ocr_texts]) + "\n\n"
    if categories:
        prompt_text += "Based on the activity, please categorize it into ONE of the following categories:\n\n"
        for cat in categories:
            prompt_text += f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
        prompt_text += "\nFirst, provide a concise summary of the activity.\nThen, on a new line after 'CATEGORY:', provide ONLY the category ID that best matches the activity."
    else:
        prompt_text += "Output only the summary text."

    llm_response = _call_llm_api(prompt_text, "bucket summarization")
    if not llm_response:
        return None, None

    summary_text = llm_response
    category_id = None
    if categories:
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            summary_text = parts[0].strip()
            category_text = parts[1].strip()
            if category_text:
                category_id = category_text.split()[0]
                if category_id not in [cat.get("id", "") for cat in categories]:
                    category_id = None
    return summary_text, category_id

def generate_summary_from_raw_with_llm(bucket_titles: List[str], bucket_ocr_texts: List[str], allow_suggestions: bool = True) -> Tuple[str, str, str]:
    titles_to_send = [str(t) for t in bucket_titles if t and str(t).strip()] if bucket_titles else []
    ocr_to_send = [str(o) for o in bucket_ocr_texts if o and str(o).strip()] if bucket_ocr_texts else []
    text_parts = list(set(titles_to_send + ocr_to_send))
    if not text_parts:
        print("No raw titles or OCR text available in this bucket")
        return "", "", ""

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
    else:
        prompt_text += "Output only the summary text."

    llm_response = _call_llm_api(prompt_text, "raw data summarization")
    if not llm_response:
        return "", "", ""

    summary_text = llm_response
    category_id = ""
    suggested_category = ""
    if categories:
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            summary_text = parts[0].strip()
            category_text = parts[1].strip()
            suggestion_parts = category_text.split("SUGGESTION:")
            if len(suggestion_parts) > 1:
                category_text = suggestion_parts[0].strip()
                suggested_category = suggestion_parts[1].strip()
            category_id = category_text.split()[0] if category_text.split() else ""
            if category_id.lower() == "none":
                category_id = ""
            else:
                valid_cat_ids = [cat.get("id", "") for cat in categories]
                if category_id not in valid_cat_ids:
                    category_id = ""
    return summary_text, category_id, suggested_category

def refine_summary_with_llm(original_summary: str, user_feedback: str, current_category_id: str = "", allow_suggestions: bool = True) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
        prompt += f"\nThe current activity is categorized as: {current_category_name} ({current_category_id})\n\n"
        prompt += "Available categories:\n"
        for cat in categories:
            prompt += f"- {cat.get('name')} ({cat.get('id')}): {cat.get('description')}\n"
        if allow_suggestions:
            prompt += "\nIf none of these categories fit well, you can suggest a new category instead."
            prompt += "\n\nFirst, provide the updated summary."
            prompt += f"\nThen, on a new line after 'CATEGORY:', provide the category ID that best matches (either keep '{current_category_id}' or suggest a different one if appropriate), or 'none' if no categories fit well."
            prompt += "\nIf you answered 'none', then on a new line after 'SUGGESTION:', provide a suggested new category name and short description in the format 'name | description'."
        else:
            prompt += "\nFirst, provide the updated summary. Then, on a new line after 'CATEGORY:', provide the category ID that best matches (either keep the current one or suggest a different one if appropriate)."
    else:
        prompt += "\nOutput only the new summary text."

    llm_response = _call_llm_api(prompt, "refinement")
    if not llm_response:
        return None, current_category_id, ""

    refined_summary = llm_response
    category_id = current_category_id
    suggested_category = ""
    if categories:
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            refined_summary = parts[0].strip()
            category_text = parts[1].strip()
            suggestion_parts = category_text.split("SUGGESTION:")
            if len(suggestion_parts) > 1:
                category_text = suggestion_parts[0].strip()
                suggested_category = suggestion_parts[1].strip()
            category_id = category_text.split()[0] if category_text.split() else current_category_id
            if category_id.lower() == "none":
                category_id = ""
            else:
                valid_cat_ids = [cat.get("id", "") for cat in categories]
                if category_id not in valid_cat_ids and category_id != current_category_id:
                    category_id = current_category_id
    return refined_summary, category_id, suggested_category

def process_activity_data(bucket_titles: List[str], bucket_ocr_texts: List[str], allow_suggestions: bool = True, update_bucket: bool = False, bucket_data: Optional[Dict[str, Any]] = None, bucket_file_path: Optional[str] = None, bucket_index: Optional[int] = None) -> Tuple[str, str, str]:
    titles_to_send = [str(t) for t in bucket_titles if t and str(t).strip()] if bucket_titles else []
    ocr_to_send = [str(o) for o in bucket_ocr_texts if o and str(o).strip()] if bucket_ocr_texts else []
    text_parts = list(set(titles_to_send + ocr_to_send))
    if not text_parts:
        print("No raw titles or OCR text available to generate a summary")
        return "", "", ""

    if update_bucket and (bucket_data is None or bucket_file_path is None or bucket_index is None):
        print("Warning: Cannot update bucket - missing required parameters")
        update_bucket = False

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
    else:
        prompt_text += "Output only the summary text."

    llm_response = _call_llm_api(prompt_text, "activity processing")
    if not llm_response:
        return "", "", ""

    summary_text = llm_response
    category_id = ""
    suggested_category = ""
    if categories:
        parts = llm_response.split("CATEGORY:")
        if len(parts) > 1:
            summary_text = parts[0].strip()
            category_text = parts[1].strip()
            suggestion_parts = category_text.split("SUGGESTION:")
            if len(suggestion_parts) > 1:
                category_text = suggestion_parts[0].strip()
                suggested_category = suggestion_parts[1].strip()
            category_id = category_text.split()[0] if category_text.split() else ""
            if category_id.lower() == "none":
                category_id = ""
            else:
                valid_cat_ids = [cat.get("id", "") for cat in categories]
                if category_id not in valid_cat_ids:
                    category_id = ""

    if update_bucket and bucket_data is not None and bucket_file_path and bucket_index is not None:
        try:
            if summary_text:
                bucket_data["summary"] = summary_text
            if category_id:
                bucket_data["category_id"] = category_id
            with open(bucket_file_path, "r", encoding="utf-8") as f:
                all_buckets = json.load(f)
            if 0 <= bucket_index < len(all_buckets):
                all_buckets[bucket_index] = bucket_data
                with open(bucket_file_path, "w", encoding="utf-8") as f:
                    json.dump(all_buckets, f, indent=2)
                print(f"Successfully updated bucket at index {bucket_index}")
            else:
                print(f"Error: bucket index {bucket_index} out of range (0-{len(all_buckets)-1})")
        except Exception as e:
            print(f"Error updating bucket: {e}")

    return summary_text, category_id, suggested_category
