"""
Weather using the free Open-Meteo API (no API key required).
Docs: https://open-meteo.com/
"""

import pandas as pd
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 15  # seconds

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _find_location(city):
    """Convert a place name into latitude/longitude."""
    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results")
    except requests.RequestException as error:
        raise RuntimeError(f"Could not reach the weather service: {error}")

    if not results:
        raise ValueError(f"Could not find a place named '{city}'. Try a nearby city.")

    place = results[0]
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    label = ", ".join(part for part in parts if part)
    return place["latitude"], place["longitude"], label


def get_weather(city):
    """Return current weather and a 5-day forecast for a city."""
    city = (city or "").strip()
    if not city:
        raise ValueError("Please enter a city name.")

    latitude, longitude, label = _find_location(city)

    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 5,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        raise RuntimeError(f"Could not fetch the weather forecast: {error}")

    current = data["current"]
    daily = data["daily"]

    forecast = pd.DataFrame(
        {
            "Date": daily["time"],
            "Min Temp (°C)": daily["temperature_2m_min"],
            "Max Temp (°C)": daily["temperature_2m_max"],
            "Rain (mm)": daily["precipitation_sum"],
        }
    )

    return {
        "location": label,
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "precipitation": current["precipitation"],
        "wind_speed": current["wind_speed_10m"],
        "description": WEATHER_CODES.get(current["weather_code"], "Unknown"),
        "forecast": forecast,
    }


def weather_to_text(weather):
    """Turn a weather dictionary into plain text for the AI chatbot."""
    lines = [
        f"Location: {weather['location']}",
        f"Condition: {weather['description']}",
        f"Temperature: {weather['temperature']} °C",
        f"Humidity: {weather['humidity']} %",
        f"Wind speed: {weather['wind_speed']} km/h",
        "5-day forecast:",
    ]
    for _, row in weather["forecast"].iterrows():
        lines.append(
            f"  {row['Date']}: {row['Min Temp (°C)']}-{row['Max Temp (°C)']} °C, "
            f"rain {row['Rain (mm)']} mm"
        )
    return "\n".join(lines)