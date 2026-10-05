"""
AI Farmer Assistant - Streamlit user interface.
Run with:  python -m streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make sure the project root is importable (so "from src..." always works)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot import FarmerChatbot, get_api_key
from src.image_analysis import analyze_crop_soil_image
from src.data_processing import get_feature_stats, get_target_column, load_data
from src.prediction import get_model_bundle, predict_crop
from src.weather import get_weather, weather_to_text

# ---------------------------------------------------------------------------
# Page setup (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Farmer Assistant", page_icon="🌾", layout="wide")


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load_data():
    return load_data()


@st.cache_resource(show_spinner=False)
def cached_model_bundle():
    return get_model_bundle()


def init_session_state():
    """Create the variables we want to remember between reruns."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "weather" not in st.session_state:
        st.session_state.weather = None
    if "prediction" not in st.session_state:
        st.session_state.prediction = None


def get_chatbot():
    """Create the OpenAI chatbot once per user session."""
    if st.session_state.chatbot is None:
        st.session_state.chatbot = FarmerChatbot()
    return st.session_state.chatbot


def build_context():
    """Collect weather / prediction info so the chatbot can use it."""
    parts = []
    if st.session_state.weather:
        parts.append("Latest weather the farmer checked:\n" + weather_to_text(st.session_state.weather))
    if st.session_state.prediction:
        pred = st.session_state.prediction
        parts.append(
            f"Latest crop recommendation from the ML model: {pred['best_crop']} "
            f"(inputs used: {pred['inputs']})"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_chatbot():
    st.header("💬 Farming Chatbot")
    st.caption("Ask anything about crops, soil, fertilizers, pests, irrigation, or seasons.")

    if not get_api_key():
        st.error(
            "OPENAI_API_KEY was not found. Make sure your .env file is in the project "
            "root folder and contains the line: OPENAI_API_KEY=your_key"
        )
        return

    # Show the previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Type your farming question here...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = get_chatbot().ask(question, build_context())
                except Exception as error:  # show a friendly message instead of crashing
                    answer = f"Sorry, something went wrong: {error}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages and st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.chatbot = None
        st.rerun()


def page_crop_recommendation():
    st.header("🌱 Crop Recommendation")
    st.caption("Enter your soil and climate values to get the best crop suggestion.")

    try:
        df = cached_load_data()
        bundle = cached_model_bundle()
    except Exception as error:
        st.error(f"Could not load data or model: {error}")
        return

    if bundle.get("note"):
        st.info(bundle["note"])

    stats = get_feature_stats(df)
    features = bundle["features"]

    # Pre-fill temperature / humidity from the weather page if available
    weather_defaults = {}
    if st.session_state.weather:
        weather_defaults = {
            "temperature": st.session_state.weather["temperature"],
            "humidity": st.session_state.weather["humidity"],
        }

    inputs = {}
    with st.form("crop_form"):
        columns = st.columns(3)
        for index, feature in enumerate(features):
            info = stats.get(feature) or stats.get(feature.lower()) or {"min": 0.0, "max": 1000.0, "mean": 0.0}
            low = 0.0 if info["min"] >= 0 else info["min"] * 2
            high = info["max"] * 2 if info["max"] > 0 else info["max"] + 1.0
            if high <= low:
                high = low + 1.0

            default = float(info["mean"])
            for key, value in weather_defaults.items():
                if key in feature.lower():
                    default = float(value)
            default = min(max(default, low), high)

            with columns[index % 3]:
                inputs[feature] = st.number_input(
                    feature,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(round(default, 2)),
                    step=0.1,
                    key=f"input_{feature}_{round(default, 2)}",
                )
        submitted = st.form_submit_button("🌾 Recommend Crop")

    if submitted:
        try:
            result = predict_crop(inputs)
            st.session_state.prediction = {
                "best_crop": result["best_crop"],
                "inputs": inputs,
                "top": result["top_predictions"],
            }
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    prediction = st.session_state.prediction
    if prediction:
        st.success(f"✅ Recommended crop: **{str(prediction['best_crop']).title()}**")
        if prediction["top"]:
            st.subheader("Top suggestions")
            for crop, percent in prediction["top"]:
                st.write(f"**{str(crop).title()}** - {percent:.1f}% confidence")
                st.progress(min(max(percent / 100.0, 0.0), 1.0))

        if get_api_key() and st.button("🤖 Get growing tips from AI"):
            with st.spinner("Asking AI..."):
                try:
                    tips = get_chatbot().ask(
                        f"Give me short, practical tips to grow {prediction['best_crop']} "
                        "(sowing time, irrigation, fertilizer, common pests).",
                        build_context(),
                    )
                    st.markdown(tips)
                except Exception as error:
                    st.error(f"AI error: {error}")

def page_crop_soil_scanner():
    st.header("📷 Crop & Soil Scanner")
    st.caption(
        "Upload a crop or soil photo and provide farm details to find suitable crops."
    )

    st.subheader("📸 Upload Crop or Soil Photo")

    upload_option = st.radio(
        "Choose how to provide the photo:",
        ["📁 Upload from Explorer", "📷 Use Camera"],
        horizontal=True,
    )

    image = None

    if upload_option == "📁 Upload from Explorer":
        image = st.file_uploader(
            "Choose a crop or soil image",
            type=["jpg", "jpeg", "png"],
        )

    else:
        image = st.camera_input("Take a photo of your crop or soil")

    if image:
        st.image(
            image,
            caption="Selected crop/soil image",
            use_container_width=True,
        )

        st.success("✅ Image uploaded successfully.")

        st.subheader("🌱 Farm & Soil Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=25.0,
                step=0.5,
            )

            humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
            )

            ph = st.number_input(
                "🧪 Soil pH",
                min_value=0.0,
                max_value=14.0,
                value=6.5,
                step=0.1,
            )

        with col2:
            nitrogen = st.number_input(
                "🌿 Nitrogen (N)",
                min_value=0.0,
                max_value=300.0,
                value=90.0,
                step=1.0,
            )

            phosphorus = st.number_input(
                "🌿 Phosphorus (P)",
                min_value=0.0,
                max_value=300.0,
                value=40.0,
                step=1.0,
            )

            potassium = st.number_input(
                "🌿 Potassium (K)",
                min_value=0.0,
                max_value=300.0,
                value=40.0,
                step=1.0,
            )

        with col3:
            rainfall = st.number_input(
                "🌧️ Rainfall (mm)",
                min_value=0.0,
                max_value=3000.0,
                value=200.0,
                step=1.0,
            )

            location = st.text_input(
                "📍 Location",
                value="Hyderabad",
            )

            soil_type = st.selectbox(
                "🌱 Soil Type",
                [
                    "Loamy",
                    "Clay",
                    "Sandy",
                    "Silty",
                    "Black Soil",
                    "Red Soil",
                    "Alluvial Soil",
                    "Other",
                ],
            )

        col4, col5 = st.columns(2)

        with col4:
            irrigation = st.selectbox(
                "💦 Irrigation Availability",
                [
                    "Available",
                    "Limited",
                    "Rain-fed",
                ],
            )

        with col5:
            season = st.selectbox(
                "🗓️ Season",
                [
                    "Kharif",
                    "Rabi",
                    "Zaid",
                    "Summer",
                    "Winter",
                    "Monsoon",
                    "Other",
                ],
            )

        st.divider()

        if st.button(
            "🔍 Analyze & Recommend Best Crop",
            type="primary",
            use_container_width=True,
        ):
            

            with st.spinner(
                "🤖 Analyzing image and farm conditions..."
            ):
                try:
                    result = analyze_crop_soil_image(
                        image_file=image,
                        temperature=temperature,
                        humidity=humidity,
                        ph=ph,
                        nitrogen=nitrogen,
                        phosphorus=phosphorus,
                        potassium=potassium,
                        rainfall=rainfall,
                        location=location,
                        soil_type=soil_type,
                        irrigation=irrigation,
                        season=season,
                    )

                    st.success("🌾 Analysis completed!")

                    st.subheader("🌾 Crop Recommendation")

                    st.markdown(result)

                except Exception as error:
                    st.error(
                        f"Analysis failed: {error}"
                    )


def page_weather():
    st.header("⛅ Weather")
    st.caption("Live weather and 5-day forecast (powered by Open-Meteo, no key needed).")

    city = st.text_input("Enter a city or village name", value="Hyderabad")
    if st.button("Get Weather"):
        with st.spinner("Fetching weather..."):
            try:
                st.session_state.weather = get_weather(city)
            except Exception as error:
                st.session_state.weather = None
                st.error(str(error))

    data = st.session_state.weather
    if data:
        st.subheader(f"📍 {data['location']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{data['temperature']} °C")
        col2.metric("Humidity", f"{data['humidity']} %")
        col3.metric("Wind", f"{data['wind_speed']} km/h")
        col4.metric("Rain now", f"{data['precipitation']} mm")
        st.write(f"**Condition:** {data['description']}")

        st.subheader("5-day forecast")
        st.dataframe(data["forecast"], use_container_width=True, hide_index=True)
        st.info("This weather is also shared with the chatbot and pre-fills the crop form.")


def page_dataset():
    st.header("📊 Dataset Explorer")
    try:
        df = cached_load_data()
    except Exception as error:
        st.error(f"Could not load dataset: {error}")
        return

    st.write(f"Rows: **{len(df)}**, Columns: **{len(df.columns)}**")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(df.describe(), use_container_width=True)

    target = get_target_column(df)
    if target in df.columns:
        st.subheader(f"Count of each '{target}'")
        st.bar_chart(df[target].value_counts())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
init_session_state()

st.title("🌾 AI Farmer Assistant")

st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 46px;
        margin: 4px 0;
        padding: 0.65rem 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        text-align: left;
        transition: background-color 0.2s ease, border-color 0.2s ease,
                    transform 0.2s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(126, 231, 135, 0.65);
        transform: translateX(3px);
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(126, 231, 135, 0.16);
        border-color: #7ee787;
        color: inherit;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 🌾 Navigation")

navigation_items = [
    ("chatbot", "💬 Farming Chatbot"),
    ("crop_recommendation", "🌱 Crop Recommendation"),
    ("crop_soil_scanner", "📷 Crop & Soil Scanner"),
    ("weather", "⛅ Weather"),
    ("dataset", "📊 Dataset Explorer"),
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "chatbot"

for page_key, page_label in navigation_items:
    if st.sidebar.button(
        page_label,
        key=f"navigate_{page_key}",
        type="primary" if st.session_state.selected_page == page_key else "secondary",
        use_container_width=True,
    ):
        st.session_state.selected_page = page_key

page = st.session_state.selected_page

if page == "chatbot":
    page_chatbot()
elif page == "crop_recommendation":
    page_crop_recommendation()
elif page == "crop_soil_scanner":
    page_crop_soil_scanner()
elif page == "weather":
    page_weather()
elif page == "dataset":
    page_dataset()