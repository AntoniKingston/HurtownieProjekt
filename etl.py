import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm
def add_event_category_id(df):
    df['EVENT_CATEGORY_ID'] = df['EVENT_TYPE'].map({
        'Tornado': 0,
        'Funnel Cloud': 0,
        'Hail': 1,
        'Heavy Rain': 1,
        'Thunderstorm Wind': 1,
        'Hurricane': 2,
        'Marine Hurricane/Typhoon': 2,
        'Tropical Storm': 2,
        'Winter Storm': 3,
        'Blizzard': 3,
        'Heavy Snow': 3,
        'Ice Storm': 3,
        'Dust Storm': 4,
        'Dense Fog': 4
    })
    return df

def create_at_least_hour_separated_begin_and_end_datetimes(df):
    # Ensure string formatting
    df['BEGIN_YEARMONTH'] = df['BEGIN_YEARMONTH'].astype(str)
    df['BEGIN_DAY'] = df['BEGIN_DAY'].astype(str).str.zfill(2)
    df['BEGIN_TIME'] = df['BEGIN_TIME'].astype(str).str.zfill(4)

    # Parse into datetime
    dt = pd.to_datetime(
        df['BEGIN_YEARMONTH'] + df['BEGIN_DAY'] + df['BEGIN_TIME'],
        format='%Y%m%d%H%M'
    )

    # Check if datetime is already on the hour
    on_the_hour = dt.dt.minute == 0

    # Subtract 1 hour if already on the hour
    dt_adjusted = dt.where(~on_the_hour, dt - pd.Timedelta(hours=1))

    # Now round down (though at this point they should already be rounded)
    df['BEGIN_DATETIME'] = dt_adjusted.dt.floor('H')

    # Similar stuff for END_DATETIME but without possibility of time shift
    df['END_YEARMONTH'] = df['END_YEARMONTH'].astype(str)
    df['END_DAY'] = df['END_DAY'].astype(str).str.zfill(2)
    df['END_TIME'] = df['END_TIME'].astype(str).str.zfill(4)


    dt = pd.to_datetime(
        df['END_YEARMONTH'] + df['END_DAY'] + df['END_TIME'],
        format='%Y%m%d%H%M'
    )

    df['END_DATETIME'] = dt.dt.ceil('H')

    return df



def preprocess_weather_data(weather_events_df):
    event_types = ['Heavy Snow', 'Thunderstorm Wind', 'Hail', 'Funnel Cloud', 'Heavy Rain', 'Tornado', 'Winter Storm', 'Blizzard', 'Dust Storm', 'Dense Fog', 'Ice Storm', 'Tropical Storm', 'Hurricane', 'Marine Hurricane/Typhoon']
    weather_events_df = weather_events_df[weather_events_df['EVENT_TYPE'].isin(event_types)]
    weather_events_df = weather_events_df[-(weather_events_df['BEGIN_LAT'].isna() | weather_events_df['BEGIN_LON'].isna() | weather_events_df['END_LAT'].isna() | weather_events_df['END_LON'].isna())]
    columns_to_select = ['BEGIN_DATETIME', 'END_DATETIME', 'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON', 'EVENT_TYPE']
    weather_events_df = create_at_least_hour_separated_begin_and_end_datetimes(weather_events_df)
    weather_events_df = weather_events_df[columns_to_select]
    weather_events_df = add_event_category_id(weather_events_df)
    return weather_events_df




def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # Radius of earth in kilometers
    r = 6371
    return r * c

def weather_airport_join(weather_df, airport_df, max_distance_km):
    """
    Perform complex join between weather events and airports.

    Parameters:
    - weather_df: DataFrame with weather events
    - airport_df: DataFrame with airport information
    - max_distance_km: Maximum distance in km to consider airports nearby

    Returns:
    - DataFrame with joined weather data points and nearby airports
    """

    # Step 1: Create hourly weather data points
    weather_points = []

    print("Step 1: Creating hourly weather data points...")
    for idx, row in tqdm(weather_df.iterrows(), total=len(weather_df), desc="Processing weather events"):
        begin_dt = row['BEGIN_DATETIME']
        end_dt = row['END_DATETIME']

        # Calculate duration in hours
        duration_hours = int((end_dt - begin_dt).total_seconds() / 3600)

        # Create hourly points (including start and end points)
        for hour_offset in range(duration_hours + 1):
            current_dt = begin_dt + timedelta(hours=hour_offset)

            # Calculate interpolation factor (0 to 1)
            if duration_hours == 0:
                t = 0  # Single point case
            else:
                t = hour_offset / duration_hours

            # Interpolate coordinates
            current_lat = row['BEGIN_LAT'] + t * (row['END_LAT'] - row['BEGIN_LAT'])
            current_lon = row['BEGIN_LON'] + t * (row['END_LON'] - row['BEGIN_LON'])

            # Create data point
            weather_point = {
                'ORIGINAL_INDEX': idx,
                'DATETIME': current_dt,
                'LATITUDE': current_lat,
                'LONGITUDE': current_lon,
                'EVENT_TYPE': row['EVENT_TYPE'],
                'EVENT_CATEGORY_ID': row['EVENT_CATEGORY_ID'],
                'BEGIN_DATETIME': row['BEGIN_DATETIME'],
                'END_DATETIME': row['END_DATETIME'],
                'HOUR_OFFSET': hour_offset,
                'INTERPOLATION_FACTOR': t
            }

            weather_points.append(weather_point)

    # Convert to DataFrame
    weather_points_df = pd.DataFrame(weather_points)
    print(f"Created {len(weather_points_df)} weather data points from {len(weather_df)} events")

    # Step 2: Join with airports based on distance
    joined_data = []

    print("\nStep 2: Finding nearby airports for each weather data point...")
    for wp_idx, wp_row in tqdm(weather_points_df.iterrows(), total=len(weather_points_df), desc="Processing weather points"):
        wp_lat = wp_row['LATITUDE']
        wp_lon = wp_row['LONGITUDE']

        # Find nearby airports
        for ap_idx, ap_row in airport_df.iterrows():
            ap_lat = ap_row['LATITUDE']
            ap_lon = ap_row['LONGITUDE']

            # Calculate distance
            distance = haversine_distance(wp_lat, wp_lon, ap_lat, ap_lon)

            # If within specified distance, create joined row
            if distance <= max_distance_km:
                joined_row = {
                    # Weather point columns
                    'ORIGINAL_WEATHER_INDEX': wp_row['ORIGINAL_INDEX'],
                    'DATETIME': wp_row['DATETIME'],
                    'WEATHER_LATITUDE': wp_row['LATITUDE'],
                    'WEATHER_LONGITUDE': wp_row['LONGITUDE'],
                    'EVENT_TYPE': wp_row['EVENT_TYPE'],
                    'EVENT_CATEGORY_ID': wp_row['EVENT_CATEGORY_ID'],
                    'BEGIN_DATETIME': wp_row['BEGIN_DATETIME'],
                    'END_DATETIME': wp_row['END_DATETIME'],
                    'HOUR_OFFSET': wp_row['HOUR_OFFSET'],
                    'INTERPOLATION_FACTOR': wp_row['INTERPOLATION_FACTOR'],

                    # Airport columns
                    'AIRPORT_ID': ap_row['AIRPORT_ID'],
                    'DISPLAY_AIRPORT_NAME': ap_row['DISPLAY_AIRPORT_NAME'],
                    'AIRPORT_COUNTRY_NAME': ap_row['AIRPORT_COUNTRY_NAME'],
                    'AIRPORT_LATITUDE': ap_row['LATITUDE'],
                    'AIRPORT_LONGITUDE': ap_row['LONGITUDE'],

                    # Distance
                    'DISTANCE_KM': distance
                }

                joined_data.append(joined_row)

    # Convert to DataFrame and return
    result_df = pd.DataFrame(joined_data)
    print(f"\nCompleted! Found {len(result_df)} weather point-airport combinations within {max_distance_km}km")

    # Sort by original weather index, then by hour offset, then by distance
    if not result_df.empty:
        print("Sorting results...")
        result_df = result_df.sort_values([
            'ORIGINAL_WEATHER_INDEX',
            'HOUR_OFFSET',
            'DISTANCE_KM'
        ]).reset_index(drop=True)

    return result_df

# Example usage function
# def example_usage():
#     """
#     Example of how to use the weather_airport_join function
#     """
#     # Sample weather data
#     weather_data = {
#         'BEGIN_DATETIME': [
#             pd.Timestamp('2024-01-01 10:00:00'),
#             pd.Timestamp('2024-01-01 14:00:00')
#         ],
#         'END_DATETIME': [
#             pd.Timestamp('2024-01-01 13:00:00'),
#             pd.Timestamp('2024-01-01 16:00:00')
#         ],
#         'BEGIN_LAT': [40.0, 41.0],
#         'BEGIN_LON': [-74.0, -73.0],
#         'END_LAT': [40.5, 41.5],
#         'END_LON': [-73.5, -72.5],
#         'EVENT_TYPE': ['THUNDERSTORM', 'HEAVY_RAIN'],
#         'EVENT_CATEGORY_ID': [1, 2]
#     }
#
#     # Sample airport data
#     airport_data = {
#         'AIRPORT_ID': ['JFK', 'LGA', 'EWR'],
#         'DISPLAY_AIRPORT_NAME': ['John F. Kennedy International', 'LaGuardia', 'Newark Liberty International'],
#         'AIRPORT_COUNTRY_NAME': ['USA', 'USA', 'USA'],
#         'LATITUDE': [40.6413, 40.7769, 40.6895],
#         'LONGITUDE': [-73.7781, -73.8740, -74.1745]
#     }
#
#     weather_df = pd.DataFrame(weather_data)
#     airport_df = pd.DataFrame(airport_data)
#
#     # Perform the join with 50km maximum distance
#     result = weather_airport_join(weather_df, airport_df, max_distance_km=50)
#
#     print("Sample result:")
#     print(result.head())
#     print(f"\nTotal joined rows: {len(result)}")
#
#     return result