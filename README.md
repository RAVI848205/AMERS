import io
import os
import base64
import json
from typing import List, Dict, Any

from google.cloud import vision
import googlemaps
import cv2
import numpy as np
import requests

# Initialize clients
vision_client = vision.ImageAnnotatorClient()
gmaps_client = googlemaps.Client(key='YOUR_GOOGLE_MAPS_API_KEY')


# ---------------- CLEANLINESS CHECK ----------------

def analyze_cleanliness(images: List[str]) -> Dict[str, Any]:
    issues = []
    score = 100

    dirty_keywords = ['stain', 'dirt', 'mess', 'trash', 'clutter', 'mold', 'urine', 'blood', 'dust']
    flagged_labels = []

    for image_path in images:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        response = vision_client.label_detection(image=image)
        labels = [label.description.lower() for label in response.label_annotations]

        for label in labels:
            if any(keyword in label for keyword in dirty_keywords):
                issues.append(f"Issue in {os.path.basename(image_path)}: {label}")
                flagged_labels.append(label)

    if issues:
        score = max(0, 100 - len(issues) * 10)

    recommendation = "Ensure proper cleaning before next guest." if issues else "Cleanliness meets Airbnb standards."

    return {
        "Score": score,
        "Issues": issues,
        "Recommendation": recommendation
    }


# ---------------- PRIVACY CHECK ----------------

def detect_suspicious_devices(images: List[str]) -> Dict[str, Any]:
    suspicious_keywords = ['camera', 'lens', 'surveillance', 'recorder', 'smoke detector', 'infrared']
    suspicious_items = []
    risk_score = 0

    for image_path in images:
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        response = vision_client.label_detection(image=image)
        labels = [label.description.lower() for label in response.label_annotations]

        for label in labels:
            if any(keyword in label for keyword in suspicious_keywords):
                suspicious_items.append(f"Suspicious item in {os.path.basename(image_path)}: {label}")

    if suspicious_items:
        risk_score = min(100, len(suspicious_items) * 20)

    recommendation = "Review the property for privacy-invading devices." if suspicious_items else "No obvious privacy risks detected."

    return {
        "Risk_Score": risk_score,
        "Suspicious_Items": suspicious_items,
        "Recommendation": recommendation
    }


# ---------------- AUTHENTICITY CHECK ----------------

def verify_property_authenticity(images: List[str], lat: float, lng: float) -> Dict[str, Any]:
    authenticity_score = 100
    detected_issues = []

    # Get Street View image from Google
    street_view_url = f"https://maps.googleapis.com/maps/api/streetview?size=600x400&location={lat},{lng}&key=YOUR_GOOGLE_MAPS_API_KEY"
    street_img_resp = requests.get(street_view_url)

    if street_img_resp.status_code != 200:
        return {
            "Authenticity_Score": 0,
            "Detected_Issues": ["Unable to fetch Street View data."],
            "Recommendation": "Verify the location or check if address exists."
        }

    street_view_img = np.asarray(bytearray(street_img_resp.content), dtype=np.uint8)
    street_view_img = cv2.imdecode(street_view_img, cv2.IMREAD_COLOR)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(street_view_img, cv2.COLOR_BGR2GRAY), None)

    match_found = False

    for image_path in images:
        uploaded_img = cv2.imread(image_path)
        if uploaded_img is None:
            continue

        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 10:
                match_found = True
                break

    if not match_found:
        authenticity_score = 40
        detected_issues.append("Uploaded images do not match Google Street View at given coordinates.")
        recommendation = "Request additional proof or schedule an inspection."
    else:
        recommendation = "Property images match location; likely authentic."

    return {
        "Authenticity_Score": authenticity_score,
        "Detected_Issues": detected_issues,
        "Recommendation": recommendation
    }


# ---------------- MAIN EVALUATION FUNCTION ----------------

def evaluate_airbnb_safety(cleanliness_images: List[str],
                           privacy_images: List[str],
                           authenticity_images: List[str],
                           latitude: float,
                           longitude: float) -> Dict[str, Any]:

    result = {
        "Cleanliness_Check": analyze_cleanliness(cleanliness_images),
        "Privacy_Check": detect_suspicious_devices(privacy_images),
        "Authenticity_Check": verify_property_authenticity(authenticity_images, latitude, longitude)
    }

    return result


# ---------------- SAMPLE TEST CASE ----------------

if __name__ == '__main__':
    # These images should be real paths or use mocked examples in production/testing
    CLEAN_IMAGES = ['test_images/bedroom.jpg', 'test_images/washroom.jpg']
    PRIVACY_IMAGES = ['test_images/room_corner.jpg']
    AUTH_IMAGES = ['test_images/property_front.jpg']
    LATITUDE = 37.7749
    LONGITUDE = -122.4194

    output = evaluate_airbnb_safety(
        cleanliness_images=CLEAN_IMAGES,
        privacy_images=PRIVACY_IMAGES,
        authenticity_images=AUTH_IMAGES,
        latitude=LATITUDE,
        longitude=LONGITUDE
    )

    print(json.dumps(output, indent=2))
🔄 Sample JSON Output
json
Copy code
{
  "Cleanliness_Check": {
    "Score": 80,
    "Issues": [
      "Issue in washroom.jpg: stain",
      "Issue in bedroom.jpg: clutter"
    ],
    "Recommendation": "Ensure proper cleaning before next guest."
  },
  "Privacy_Check": {
    "Risk_Score": 20,
    "Suspicious_Items": [
      "Suspicious item in room_corner.jpg: camera"
    ],
    "Recommendation": "Review the property for privacy-invading devices."
  },
  "Authenticity_Check": {
    "Authenticity_Score": 100,
    "Detected_Issues": [],
    "Recommendation": "Property images match location; likely authentic."
  }
}

