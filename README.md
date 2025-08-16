import io
import json
from typing import List, Dict, Any
from google.cloud import vision
from google.cloud.vision import types

# You must set GOOGLE_APPLICATION_CREDENTIALS env var pointing to your service account json key file before running this.


def analyze_image(image_content: bytes) -> Dict[str, Any]:
    """
    Sends image bytes to Google Cloud Vision API to detect labels and
    safe search properties to identify cleanliness issues.

    Returns raw Vision API response data needed for further processing.
    """
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=image_content)

    # Perform label detection to identify objects like dirt, stains, bed sheets, washroom items
    label_response = client.label_detection(image=image)
    labels = label_response.label_annotations

    # Perform safe search detection to detect inappropriate or unsafe content (optional)
    safe_search_response = client.safe_search_detection(image=image)
    safe_search = safe_search_response.safe_search_annotation

    return {
        "labels": labels,
        "safe_search": safe_search,
    }


def evaluate_cleanliness(vision_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate cleanliness based on labels detected and safe search data.

    Assigns issues detected and calculates a cleanliness score (0-100).

    Issues considered:
    - Dirt, Dust, Stains
    - Bed Sheets (if dirty)
    - Toilet, Washroom issues (mold, unclean)
    """

    labels = vision_data["labels"]
    safe_search = vision_data["safe_search"]

    issues_detected: List[str] = []

    # Keywords indicating potential problems
    problem_keywords = {
        "dirt": ["dirt", "dust", "mud", "soil", "grime"],
        "stain": ["stain", "blemish", "spot", "smudge"],
        "unclean_bedsheet": ["bed sheet", "bedsheet", "bedspread", "pillowcase"],
        "washroom_issues": ["toilet", "bathroom", "washroom", "sink", "mold", "mildew", "fungus"],
    }

    # Lowercase label descriptions for easier matching
    label_descriptions = [label.description.lower() for label in labels]

    # Check for dirt/stain-related issues
    for issue, keywords in problem_keywords.items():
        for keyword in keywords:
            if any(keyword in desc for desc in label_descriptions):
                if issue == "unclean_bedsheet":
                    # Presence of bed sheets isn't an issue itself, need to infer if unclean
                    # We'll assume if "dirty" or "stain" also appear near bed sheet labels, mark issue.
                    if any(
                        ("dirty" in desc or "stain" in desc or "blemish" in desc)
                        for desc in label_descriptions
                    ):
                        issues_detected.append("Unclean bedsheets detected")
                    else:
                        # No evidence of unclean bed sheets
                        pass
                elif issue == "washroom_issues":
                    # Presence of mold/mildew/fungus is issue
                    if keyword in ["mold", "mildew", "fungus"]:
                        issues_detected.append("Unhygienic washroom conditions detected")
                    else:
                        # Just presence of washroom items isn't an issue, skip
                        pass
                else:
                    if issue == "dirt":
                        issues_detected.append("Dirt or dust detected")
                    elif issue == "stain":
                        issues_detected.append("Stains detected")
                break  # Found keyword for this issue

    # Evaluate safe search results (optional, could indicate dirtiness)
    # We consider 'adult', 'medical', 'spoof', 'violence' as irrelevant here.

    # Assign cleanliness score based on issues detected
    if not issues_detected:
        cleanliness_score = 95  # Very clean
        recommendation = "Room is clean and ready for guests."
    else:
        # Deduct points per issue (simplified)
        score_deduction = 0
        if "Dirt or dust detected" in issues_detected:
            score_deduction += 30
        if "Stains detected" in issues_detected:
            score_deduction += 25
        if "Unclean bedsheets detected" in issues_detected:
            score_deduction += 30
        if "Unhygienic washroom conditions detected" in issues_detected:
            score_deduction += 40

        cleanliness_score = max(0, 100 - score_deduction)
        recommendation = "Please address the detected cleanliness issues before hosting guests."

    return {
        "Cleanliness_Score": cleanliness_score,
        "Issues_Detected": issues_detected,
        "Recommendation": recommendation,
    }


def process_uploaded_image(file_path: str) -> None:
    """
    Main function to read image file, analyze it and print the final JSON output.
    """
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()

    vision_data = analyze_image(content)
    result = evaluate_cleanliness(vision_data)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI-powered Cleanliness Verification for Airbnb")
    parser.add_argument("image_path", help="Path to the room or washroom image file")

    args = parser.parse_args()

    process_uploaded_image(args.image_path)
