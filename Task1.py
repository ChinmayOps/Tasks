#API INTEGRATION AND DATA VISUALIZATION

import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# API Configuration
API_KEY = "ce019f214672ec93e0c50fa7abef262d"  # Get from https://home.openweathermap.org/users/sign_up
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
CITIES = ["London", "New York", "Tokyo", "Dubai", "Sydney"]

def get_weather_data():
    """Fetch weather data for multiple cities"""
    weather_data = []
    
    for city in CITIES:
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }
        
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data.append({
                "City": city,
                "Temperature": data["main"]["temp"],
                "Humidity": data["main"]["humidity"],
                "Wind Speed": data["wind"]["speed"],
                "Cloud Cover": data["clouds"]["all"]
            })
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {city}: {str(e)}")
    
    return pd.DataFrame(weather_data)

def create_visualizations(df):
    """Create multiple weather visualizations"""
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    plt.suptitle("Current Weather Conditions", fontsize=16, y=1.02)

    # Temperature Comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x="City", y="Temperature", data=df, palette="coolwarm")
    plt.title("Temperature Comparison (°C)")
    plt.ylim(df["Temperature"].min()-2, df["Temperature"].max()+2)

    # Humidity Distribution
    plt.subplot(2, 2, 2)
    plt.pie(df["Humidity"], labels=df["City"], autopct="%1.1f%%",
            colors=sns.color_palette("Blues"), startangle=90)
    plt.title("Humidity Distribution (%)")

    # Wind Speed Analysis
    plt.subplot(2, 2, 3)
    sns.lineplot(x="City", y="Wind Speed", data=df, marker="o", color="green")
    plt.title("Wind Speed (m/s)")
    plt.ylim(0, df["Wind Speed"].max()+2)

    # Cloud Cover vs Temperature
    plt.subplot(2, 2, 4)
    sns.scatterplot(x="Cloud Cover", y="Temperature", data=df,
                    hue="City", s=200, palette="viridis")
    plt.title("Cloud Cover vs Temperature")
    plt.xlabel("Cloud Cover (%)")

    plt.tight_layout()
    plt.savefig("weather_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # Get and process data
    weather_df = get_weather_data()
    
    if not weather_df.empty:
        print("Fetched Weather Data:")
        print(weather_df)
        create_visualizations(weather_df)
    else:
        print("Failed to fetch weather data")


