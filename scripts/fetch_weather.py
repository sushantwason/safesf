#!/usr/bin/env python3
"""
Fetch historical weather data for San Francisco
Using Open-Meteo API (free, no API key needed)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_weather_data(start_date: str, end_date: str, output_file: str):
    """
    Fetch historical weather data from Open-Meteo API
    
    Args:
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format
        output_file: Path to save parquet file
    """
    
    # San Francisco coordinates
    lat = 37.7749
    lon = -122.4194
    
    logger.info(f"Fetching weather data for SF from {start_date} to {end_date}")
    
    # Open-Meteo Historical Weather API
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "precipitation_hours",
            "windspeed_10m_max"
        ],
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'temp_max_f': data['daily']['temperature_2m_max'],
        'temp_min_f': data['daily']['temperature_2m_min'],
        'temp_mean_f': data['daily']['temperature_2m_mean'],
        'precipitation_inches': data['daily']['precipitation_sum'],
        'rain_inches': data['daily']['rain_sum'],
        'precipitation_hours': data['daily']['precipitation_hours'],
        'wind_max_mph': data['daily']['windspeed_10m_max']
    })
    
    # Add derived features
    df['is_rainy'] = df['rain_inches'] > 0.1
    df['is_cold'] = df['temp_mean_f'] < 55  # Cold for SF
    df['is_hot'] = df['temp_mean_f'] > 70   # Hot for SF
    
    logger.info(f"Fetched {len(df)} days of weather data")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Rainy days: {df['is_rainy'].sum()} ({df['is_rainy'].sum()/len(df)*100:.1f}%)")
    
    # Save to parquet
    df.to_parquet(output_file, index=False)
    logger.info(f"✅ Saved weather data to {output_file}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch historical weather data for SF')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='data/weather_history.parquet', help='Output parquet file')
    
    args = parser.parse_args()
    
    fetch_weather_data(args.start_date, args.end_date, args.output)
