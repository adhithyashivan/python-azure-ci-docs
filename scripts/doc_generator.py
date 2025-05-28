import os
import requests  # For Confluence API AND Hugging Face API
import json
import sys
from requests.auth import HTTPBasicAuth
import time

# --- Configuration ---

# For Hugging Face Inference API:
# Set via GitHub Secrets: secrets.HF_API_TOKEN
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
# Set via GitHub Actions env var, e.g., "google/flan-t5-base" or "google/flan-t5-large"
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "google/flan-t5-base")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# For Confluence:
# Set via GitHub Secrets: secrets.CONFLUENCE_URL
CONFLUENCE_BASE_URL = os.environ.get("CONFLUENCE_URL")
# Set via GitHub Secrets: secrets.CONFLUENCE_EMAIL
CONFLUENCE_USER_EMAIL = os.environ.get("CONFLUENCE_EMAIL")
# Set via GitHub Secrets: secrets.CONFLUENCE_API_TOKEN
CONFLUENCE_API_TOKEN = os.environ.get("CONFLUENCE_API_TOKEN")
# Set via GitHub Secrets: secrets.CONFLUENCE_SPACE_KEY
CONFLUENCE_SPACE_KEY = os.environ.get("CONFLUENCE_SPACE_KEY")

# Code and Documentation Structure:
# Set via GitHub Actions env var, e.g., 'app'
CODE_ROOT_DIR_RELATIVE_PATH = os.environ.get("CODE_ROOT_PATH", "app")
# Set via GitHub Actions env var, e.g., "${{ github.repository }} - Automated Docs"
ROOT_DOC_PROJECT_TITLE = os.environ.get(
    "ROOT_DOC_TITLE", "Project Documentation")


# --- Helper Functions ---

def get_ai_documentation_hf(file_content, file_path):
    """Generates documentation for a Python file using Hugging Face Inference API for Flan-T5."""
    if not HF_API_TOKEN:
        print(
            f"Hugging Face API Token (HF_API_TOKEN) not set. Skipping AI documentation for {file_path}.")
        return f"h2. Error: Configuration Issue\n\nHF_API_TOKEN not set. Could not generate documentation for {file_path}."

    # Construct the final API URL using the environment variable
    # This is just for clarity, it's the same as the global HF_API_URL
    current_hf_api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    # Print the URL being used
    print(f"    DEBUG: Constructed HF API URL: {current_hf_api_url}")

    instruction = (
        "Generate detailed technical documentation in Confluence Wiki Markup format for the following Python code. "
        "The documentation should include:\n"
        "1. *File Overview:* A concise summary of the file's purpose.\n"
        "2. *Classes (if any):* For each class, describe its purpose, key attributes, and methods (purpose, parameters, returns).\n"
        "3. *Functions (if any, not part of a class):* For each function, describe its purpose, parameters, and returns.\n"
        "Use Confluence Wiki Markup: h1. Title, h2. Subtitle, *bold*, _italic_, * list item."
    )
    prompt_template = (
        f"{instruction}\n\n"
        f"Python file: {file_path}\n"
        f"```python\n{file_content}\n```\n\n"
        f"Confluence Wiki Markup Documentation:"
    )

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    # Add Content-Type header, which is good practice for POST requests with JSON body
    headers["Content-Type"] = "application/json"

    # Print headers (mask most of token)
    print(
        f"    DEBUG: Request Headers (Token partially masked): Authorization: Bearer hf_...{HF_API_TOKEN[-4:] if HF_API_TOKEN and len(HF_API_TOKEN) > 4 else 'TOKEN_INVALID_OR_SHORT'}")

    payload = {
        "inputs": prompt_template,
        "parameters": {
            "max_new_tokens": 1536,
            "min_new_tokens": 50,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
        },
        "options": {
            "wait_for_model": True,
            "use_gpu": False
        }
    }
    # Print the payload
    print(f"    DEBUG: Payload being sent: {json.dumps(payload, indent=2)}")

    max_retries = 4
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            print(
                f"  Requesting documentation from Hugging Face ({HF_MODEL_ID}) for {file_path} (Attempt {attempt+1}/{max_retries})...")
            # Use current_hf_api_url
            response = requests.post(
                current_hf_api_url, headers=headers, json=payload, timeout=180)

            # Print status code
            print(f"    DEBUG: Response Status Code: {response.status_code}")
            # Try to print response text regardless of status code for debugging, if it's not too large
            try:
                response_text_preview = response.text[:500] + "..." if len(
                    response.text) > 500 else response.text
                print(
                    f"    DEBUG: Response Text Preview: {response_text_preview}")
            except Exception as e_text:
                print(f"    DEBUG: Could not get response text: {e_text}")

            if response.status_code == 200:
                # ... (rest of the success handling code from previous version) ...
                result = response.json()
                if isinstance(result, list) and result and "generated_text" in result[0]:
                    generated_doc = result[0]["generated_text"].strip()
                    if generated_doc:
                        print(
                            f"    Successfully received documentation from Hugging Face for {file_path}")
                        return generated_doc
                    else:
                        print(
                            f"    Hugging Face returned empty 'generated_text' for {file_path}. Response: {result}")
                        return f"h2. Notice: No Documentation Generated\n\nThe AI model returned an empty response for {file_path}."
                else:
                    print(
                        f"    Hugging Face response for {file_path} has unexpected format. Response: {result}")
                    return f"h2. Error: AI Model Response Issue\n\nUnexpected response format from Hugging Face for {file_path}. Check model compatibility or API changes."

            elif response.status_code == 429:
                print(
                    f"    Hugging Face Rate Limit Error for {file_path}. Retrying in {retry_delay}s...")
            elif response.status_code == 503:
                print(
                    f"    Hugging Face Model Unavailable (503) for {file_path}. Model: {HF_MODEL_ID}. Retrying in {retry_delay}s...")
            # elif response.status_code == 404: # Handled by the generic else for now
            #     print(f"    Hugging Face Model Not Found (404) for {file_path}. Model: {HF_MODEL_ID}. This is unexpected for standard models.")
            #     # For 404, retrying usually won't help if the URL or model ID is wrong.
            #     # But let's keep the retry for now to see if it's intermittent.
            else:
                print(
                    f"    Error calling Hugging Face API for {file_path}. Status: {response.status_code}.")
                if attempt == max_retries - 1:
                    return f"h2. Error: AI API Call Failed\n\nHugging Face API call failed for {file_path} with status {response.status_code} after multiple retries. Response Preview: {response_text_preview if 'response_text_preview' in locals() else 'N/A'}"

        except requests.exceptions.Timeout:
            print(
                f"    Request to Hugging Face timed out for {file_path} (Attempt {attempt+1}/{max_retries}).")
        except requests.exceptions.RequestException as e:
            print(
                f"    Request Exception calling Hugging Face API for {file_path}: {e}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, 60)
        else:
            print(
                f"Failed to get documentation from Hugging Face for {file_path} after {max_retries} retries.")
            return f"h2. Error: AI Processing Failed\n\nFailed to retrieve documentation from Hugging Face for {file_path} after multiple retries due to API issues or timeouts."

    return f"h2. Error: Unknown AI Processing Issue\n\nAn unknown error occurred while trying to generate documentation for {file_path}."


def get_confluence_page_id_and_version(title, space_key):
    """Checks if a page with the given title exists and returns its ID and version number."""
    if not CONFLUENCE_BASE_URL or not CONFLUENCE_USER_EMAIL or not CONFLUENCE_API_TOKEN:
        print(
            f"Confluence API credentials or URL not set. Cannot search for page '{title}'.")
        return None, None

    api_url = f"{CONFLUENCE_BASE_URL}/rest/api/content"
    auth = HTTPBasicAuth(CONFLUENCE_USER_EMAIL, CONFLUENCE_API_TOKEN)
    headers = {"Accept": "application/json"}
    params = {"title": title, "spaceKey": space_key, "expand": "version"}

    try:
        response = requests.get(api_url, headers=headers,
                                params=params, auth=auth, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            page_id = results[0]["id"]
            version_number = results[0]['version']['number']
            print(
                f"Found existing page '{title}' (ID: {page_id}, Version: {version_number})")
            return page_id, version_number
        print(f"Page '{title}' not found in space '{space_key}'.")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error checking Confluence page '{title}': {e}")
        if hasattr(response, 'text'):
            print(f"Response content: {response.text}")
        return None, None


def create_or_update_confluence_page(title, body_content, space_key, parent_id=None):
    """Creates or updates a Confluence page."""
    if not CONFLUENCE_BASE_URL or not CONFLUENCE_USER_EMAIL or not CONFLUENCE_API_TOKEN:
        print(
            f"Confluence API credentials or URL not set. Cannot publish page '{title}'.")
        return None

    api_url_base = f"{CONFLUENCE_BASE_URL}/rest/api/content"
    auth = HTTPBasicAuth(CONFLUENCE_USER_EMAIL, CONFLUENCE_API_TOKEN)
    headers = {"Accept": "application/json",
               "Content-Type": "application/json"}

    page_id, current_version = get_confluence_page_id_and_version(
        title, space_key)

    data = {
        "type": "page",
        "title": title,
        "space": {"key": space_key},
        "body": {
            "wiki": {  # Using Confluence Wiki Markup
                "value": body_content,
                "representation": "wiki"
            }
        }
    }

    if parent_id:
        data["ancestors"] = [{"id": str(parent_id)}]

    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            if page_id:
                data["id"] = page_id
                data["version"] = {"number": current_version + 1}
                api_url_specific = f"{api_url_base}/{page_id}"
                response = requests.put(api_url_specific, headers=headers, data=json.dumps(
                    data), auth=auth, timeout=30)
                action_taken = "Updating"
            else:
                api_url_specific = api_url_base
                response = requests.post(
                    api_url_specific, headers=headers, data=json.dumps(data), auth=auth, timeout=30)
                action_taken = "Creating"

            print(
                f"{action_taken} page '{title}' (Attempt {attempt+1}/{max_retries})...")
            response.raise_for_status()

            new_page_id = response.json().get("id")
            print(
                f"Successfully {action_taken.lower()} Confluence page: '{title}' (ID: {new_page_id})")
            return new_page_id

        except requests.exceptions.HTTPError as e:
            print(
                f"HTTP Error {action_taken.lower()} Confluence page '{title}': {e.response.status_code} - {e.response.reason}")
            print(f"Response content: {e.response.text}")
            if e.response.status_code == 409:  # Conflict
                page_id, current_version = get_confluence_page_id_and_version(
                    title, space_key)
                if page_id is None:
                    page_id = None
                    current_version = None
                    continue
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(
                    f"Failed to {action_taken.lower()} page '{title}' after {max_retries} attempts.")
                return None
        except requests.exceptions.RequestException as e:
            print(
                f"Request Error {action_taken.lower()} Confluence page '{title}': {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(
                    f"Failed to {action_taken.lower()} page '{title}' after {max_retries} attempts.")
                return None
    return None


def process_directory_recursively(current_dir_abs_path, space_key, current_confluence_parent_id, code_base_dir_abs_path):
    """Recursively processes directories and files to generate documentation."""
    print(f"Processing directory: {current_dir_abs_path}")
    items_in_dir = sorted(os.listdir(current_dir_abs_path))

    for item_name in items_in_dir:
        item_abs_path = os.path.join(current_dir_abs_path, item_name)
        item_relative_to_code_base = os.path.relpath(
            item_abs_path, code_base_dir_abs_path)

        if os.path.isdir(item_abs_path):
            dir_page_title = f"{ROOT_DOC_PROJECT_TITLE}: {item_relative_to_code_base.replace(os.sep, ' / ')}"
            print(f"  Creating/Updating directory page: '{dir_page_title}'")
            dir_content = f"h1. Directory: {item_name}\n\nDocumentation for modules and subdirectories within '{item_name}'."
            new_parent_id_for_children = create_or_update_confluence_page(
                dir_page_title, dir_content, space_key, current_confluence_parent_id)

            if new_parent_id_for_children:
                process_directory_recursively(
                    item_abs_path, space_key, new_parent_id_for_children, code_base_dir_abs_path)
            else:
                print(
                    f"  Skipping subdirectory '{item_abs_path}' due to error creating/updating its Confluence page.")

        elif item_name.endswith(".py"):
            print(f"  Processing Python file: {item_abs_path}")
            file_page_title = f"{ROOT_DOC_PROJECT_TITLE}: {item_relative_to_code_base}"
            try:
                with open(item_abs_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                if not file_content.strip():
                    print(f"  Skipping empty file: {item_abs_path}")
                    continue

                print(
                    f"    Generating AI documentation for: '{file_page_title}'")
                ai_doc_content = get_ai_documentation_hf(
                    file_content, item_relative_to_code_base)  # MODIFIED

                if ai_doc_content:
                    print(
                        f"    Publishing documentation to Confluence for: '{file_page_title}'")
                    create_or_update_confluence_page(
                        file_page_title, ai_doc_content, space_key, current_confluence_parent_id)
                else:
                    print(
                        f"    No documentation content generated by AI for {item_abs_path}")
            except Exception as e:
                print(f"  Error processing file {item_abs_path}: {e}")
        else:
            print(f"  Skipping non-Python file or non-directory: {item_name}")


# --- Main Execution ---
if __name__ == "__main__":
    # Basic check for essential configs
    if not all([HF_API_TOKEN, CONFLUENCE_BASE_URL, CONFLUENCE_USER_EMAIL, CONFLUENCE_API_TOKEN, CONFLUENCE_SPACE_KEY]):
        print("CRITICAL ERROR: One or more essential environment variables for API access are missing.")
        missing_vars = [var for var, val in [
            ("HF_API_TOKEN", HF_API_TOKEN),
            ("CONFLUENCE_BASE_URL", CONFLUENCE_BASE_URL),
            ("CONFLUENCE_USER_EMAIL", CONFLUENCE_USER_EMAIL),
            ("CONFLUENCE_API_TOKEN", CONFLUENCE_API_TOKEN),
            ("CONFLUENCE_SPACE_KEY", CONFLUENCE_SPACE_KEY)
        ] if not val]
        print(f"Missing: {', '.join(missing_vars)}")
        sys.exit(1)

    print("Starting documentation generation process using Hugging Face Flan-T5...")
    print(f"  Hugging Face Model ID: {HF_MODEL_ID}")
    print(f"  Project Title for Docs: {ROOT_DOC_PROJECT_TITLE}")
    print(f"  Confluence Space Key: {CONFLUENCE_SPACE_KEY}")
    print(f"  Relative Code Path to Document: '{CODE_ROOT_DIR_RELATIVE_PATH}'")

    project_root_page_body = f"h1. {ROOT_DOC_PROJECT_TITLE}\n\nThis page is the root for automatically generated documentation for the project (using Hugging Face Model: {HF_MODEL_ID}). It covers code found in the '{CODE_ROOT_DIR_RELATIVE_PATH}' directory."

    print(
        f"Creating/Updating the main project documentation page: '{ROOT_DOC_PROJECT_TITLE}'")
    project_root_confluence_page_id = create_or_update_confluence_page(
        ROOT_DOC_PROJECT_TITLE,
        project_root_page_body,
        CONFLUENCE_SPACE_KEY
    )

    if project_root_confluence_page_id:
        print(
            f"Successfully created/updated project root page. ID: {project_root_confluence_page_id}")
        github_workspace = os.environ.get("GITHUB_WORKSPACE", ".")
        code_to_document_abs_path = os.path.abspath(
            os.path.join(github_workspace, CODE_ROOT_DIR_RELATIVE_PATH))
        print(f"Absolute path to document: {code_to_document_abs_path}")

        if os.path.isdir(code_to_document_abs_path):
            process_directory_recursively(
                code_to_document_abs_path,
                CONFLUENCE_SPACE_KEY,
                project_root_confluence_page_id,
                code_to_document_abs_path
            )
            print("Documentation generation process completed.")
        else:
            print(
                f"Error: Code path '{code_to_document_abs_path}' (from '{CODE_ROOT_DIR_RELATIVE_PATH}') is not a valid directory.")
            sys.exit(1)
    else:
        print(
            f"CRITICAL ERROR: Failed to create or update main project documentation page: '{ROOT_DOC_PROJECT_TITLE}'. Aborting.")
        sys.exit(1)
