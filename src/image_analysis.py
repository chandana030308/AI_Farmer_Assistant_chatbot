from functools import lru_cache
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from transformers import pipeline

from src.data_processing import DATA_PATH, get_target_column, load_data
from src.prediction import predict_crop


MODEL_NAME = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def get_image_model():
    return pipeline(
        "zero-shot-image-classification",
        model=MODEL_NAME,
    )


def get_crop_names():
    df = load_data(DATA_PATH)
    target = get_target_column(df)

    crops = (
        df[target]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )

    return crops


def analyze_visual_condition(image):
    model = get_image_model()

    labels = [
        "healthy green crop",
        "dry or stressed crop",
        "yellowing crop",
        "crop with visible leaf damage",
        "healthy soil",
        "dry soil",
        "wet soil",
        "dark fertile-looking soil",
        "sandy-looking soil",
    ]

    results = model(image, candidate_labels=labels)

    observations = []

    for result in results[:3]:
        label = result["label"]
        score = result["score"] * 100

        observations.append(
            f"- {label.title()} ({score:.1f}% visual similarity)"
        )

    return observations


def get_image_crop_scores(image):
    model = get_image_model()
    crop_names = get_crop_names()

    if not crop_names:
        return []

    results = model(
        image,
        candidate_labels=crop_names,
    )

    return [
        (result["label"], result["score"] * 100)
        for result in results[:5]
    ]


def combine_predictions(numerical_result, image_scores):
    numerical_scores = dict(numerical_result.get("top_predictions", []))
    image_scores_dict = dict(image_scores)

    all_crops = set(numerical_scores) | set(image_scores_dict)

    combined = []

    for crop in all_crops:
        numerical_score = numerical_scores.get(crop, 0)
        image_score = image_scores_dict.get(crop, 0)

        # Numerical farm data is given more importance than image similarity.
        combined_score = (
            numerical_score * 0.70
            + image_score * 0.30
        )

        combined.append(
            (crop, combined_score, numerical_score, image_score)
        )

    combined.sort(key=lambda x: x[1], reverse=True)

    return combined[:3]


def analyze_crop_soil_image(
    image_file,
    temperature,
    humidity,
    ph,
    nitrogen,
    phosphorus,
    potassium,
    rainfall,
    location,
    soil_type,
    irrigation,
    season,
):
    if image_file is None:
        raise ValueError("Please upload or capture an image.")

    image_bytes = image_file.getvalue()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Use the existing numerical crop model.
    inputs = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    numerical_result = predict_crop(inputs)

    # Analyze visible image information locally.
    visual_observations = analyze_visual_condition(image)

    # Compare the image with crop names from the agriculture dataset.
    image_scores = get_image_crop_scores(image)

    # Combine both sources.
    combined = combine_predictions(
        numerical_result,
        image_scores,
    )

    if not combined:
        raise ValueError("Could not generate crop recommendations.")

    best_crop = combined[0][0]

    lines = []

    lines.append("## 🌾 Crop & Soil Analysis")
    lines.append("")

    lines.append("### 👁️ Visual Observation")
    lines.extend(visual_observations)
    lines.append("")

    lines.append("### 🌱 Best Crop")
    lines.append(f"**{best_crop.title()}**")
    lines.append("")

    lines.append("### 🏆 Top 3 Crop Recommendations")

    for index, (crop, score, numerical, image_score) in enumerate(
        combined,
        start=1,
    ):
        lines.append(
            f"{index}. **{crop.title()}** — {score:.1f}% combined suitability"
        )

    lines.append("")

    lines.append("### 📊 Recommendation Basis")
    lines.append(
        f"- Temperature: {temperature} °C"
    )
    lines.append(
        f"- Humidity: {humidity} %"
    )
    lines.append(
        f"- Soil pH: {ph}"
    )
    lines.append(
        f"- Nitrogen (N): {nitrogen}"
    )
    lines.append(
        f"- Phosphorus (P): {phosphorus}"
    )
    lines.append(
        f"- Potassium (K): {potassium}"
    )
    lines.append(
        f"- Rainfall: {rainfall} mm"
    )
    lines.append(
        f"- Location: {location}"
    )
    lines.append(
        f"- Soil Type: {soil_type}"
    )
    lines.append(
        f"- Irrigation: {irrigation}"
    )
    lines.append(
        f"- Season: {season}"
    )

    lines.append("")
    lines.append("### 🌿 Basic Cultivation Guidance")

    lines.append(
        f"- Select good-quality seeds suitable for **{best_crop.title()}**."
    )
    lines.append(
        "- Maintain appropriate irrigation according to soil moisture and weather."
    )
    lines.append(
        "- Monitor the crop regularly for pests, diseases, and nutrient deficiencies."
    )
    lines.append(
        "- Follow local agricultural recommendations for fertilizer application."
    )

    lines.append("")
    lines.append("### ⚠️ Important Limitation")
    lines.append(
        "The image analysis identifies visual patterns and does not accurately "
        "measure laboratory soil properties such as exact pH, nitrogen, phosphorus, "
        "or potassium. For reliable soil measurements, use a proper soil test."
    )

    return "\n".join(lines)