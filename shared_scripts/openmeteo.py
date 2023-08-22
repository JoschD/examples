""" 
OpenMeteo Forecast Extractor
----------------------------

Extract historical and forecast data from OpenMeteo and show it in a plotly graph.

Required packages:
openmeteo-py, pandas, plotly
"""
from typing import Sequence
from openmeteo_py import OWmanager
from openmeteo_py.Hourly.HourlyForecast import HourlyForecast
from openmeteo_py.Hourly.HourlyHistorical import HourlyHistorical
from openmeteo_py.Options.ForecastOptions import ForecastOptions
from openmeteo_py.Options.HistoricalOptions import HistoricalOptions
from openmeteo_py.Utils.constants import celsius, kmh, mm, iso8601 
import pytz
import pandas as pd

import plotly.graph_objects as go

from dataclasses import dataclass
from datetime import datetime

@dataclass
class Location:
    """Class to store information about a location and its weather data."""
    name: str
    latitude: float
    longitude: float
    timezone: pytz.timezone = pytz.utc
    data: pd.DataFrame = None

    def copy(self):
        return Location(self.name, self.latitude, self.longitude, self.timezone, self.data)


GENEVA = Location("Geneva", 46.2052193, 6.1471942, pytz.timezone('Europe/Zurich'))
BERN = Location("Bern", 46.9546812, 7.3125359, pytz.timezone('Europe/Zurich'))

color_order = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def get_color(idx: int, alpha: float = 1.) -> str:
    """Get color at index `idx` in the color cycle in plotly-format.

    Args:
        idx (int): Color index   
        alpha (float, optional): Alpha transparency of the color. Defaults to 1.

    Returns:
        str: String representation of the color for plotly in rgba format. 
    """
    color = color_order[idx % len(color_order)]
    return 'rgba({}, {}, {}, {})'.format(*color, alpha)



def iso8601format(dt: datetime) -> str:
    """Not really iso8601, but the format used by OpenMeteo.

    Args:
        dt (datetime): Python datetime object.

    Returns:
        str: String representation of the datetime as expected by OpenMeteo.
    """
    # return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return dt.strftime('%Y-%m-%d')


def check_error(data: dict) -> None:
    """Check returned data for Errors.

    Args:
        data (dict): Returned (from json converted) dict object from OpenMeteo.         
    """
    try:
        if data["error"]:
            raise Exception(f"Failed to load meteo data. Reason: {data['reason']:s}")
    except KeyError:
        pass


def get_forecast(location: Location) -> pd.DataFrame:
    """Return the forecast data for the given location.

    Args:
        location (Location): Location description of the place you want the forecast from.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the requested forecast data.
    """
    hourly = HourlyHistorical()
    hourly.temperature_2m()

    options = ForecastOptions(
        latitude=location.latitude,
        longitude=location.longitude,
        current_weather=False,
        temperature_unit=celsius,
        windspeed_unit=kmh,
        precipitation_unit=mm,
        timeformat=iso8601,
        timezone=location.timezone,
        past_days=30,
        forecast_days=16,
    )

    mgr = OWmanager(
        options=options,
        api=OWmanager.forecast,
        hourly=hourly
    )

    meteo = mgr.get_data()
    check_error(meteo)
    df = pd.DataFrame(meteo["hourly"])
    df = df.set_index("time", drop=True)
    df = df.rename(columns={"temperature_2m": "forecast"})
    return df
    

def get_historical(location: Location, start_date: datetime) -> pd.DataFrame:
    """Return the historical data for the given location.

    Args:
        location (Location): Location description of the place you want the forecast from.
        start_date (datetime): Start date of the historical data. 
        The end date will be the closest historical data available to the current date.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the requested historical data.
    """
    hourly = HourlyForecast()
    hourly.temperature_2m()

    options = HistoricalOptions(
        latitude=location.latitude,
        longitude=location.longitude,
        current_weather=False,
        temperature_unit = celsius,
        windspeed_unit = kmh,
        precipitation_unit = mm,
        timeformat = iso8601,
        timezone = location.timezone,
        start_date=iso8601format(start_date),
        end_date=iso8601format(datetime.now()),
    )

    mgr = OWmanager(
        options=options,
        api=OWmanager.historical,
        hourly=hourly
    )

    meteo = mgr.get_data()
    check_error(meteo)
    df = pd.DataFrame(meteo["hourly"])
    df = df.set_index("time", drop=True)
    df = df.rename(columns={"temperature_2m": "historical"})

    return df


def get_24h_average(series: pd.Series) -> pd.Series:
    """Average the Series over 24h.

    Args:
        series (pd.Series): Series containing hourly data. 

    Returns:
        pd.Series: Rolling average of the given data over 24 entries.
    """
    return series.rolling(24).mean()


def plot(locations: Sequence[Location]):
    """Create a plot of the given locations, containing already the results of 
    the OpenMeteo forecast and historical data request.

    Args:
        locations (Sequence[Location]): Location objects with stored data as columns in a dataframe.
        start_date (datetime): Start date of the historical data. 
    """

    fig = go.Figure()
    fig.add_hline(y=25, line_width=1, line_dash="dash", line_color="red")
    fig.add_hline(y=27, line_width=1, line_dash="dash", line_color="purple")
    fig.add_vline(x=iso8601format(datetime.now()), line_width=1, line_dash="dash", line_color="black")

    for idx, location in enumerate(locations):
        df = location.data
        forecast_24 = get_24h_average(df.forecast)
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df.forecast, 
            name=f"forecast",
            legendgroup=location.name,
            legendgrouptitle_text=location.name,
            mode='lines', 
            line=dict(color=get_color(idx, 0.2)),
        ))
        fig.add_trace(go.Scatter(
            x=forecast_24.index, 
            y=forecast_24, 
            name="forecast av24h", 
            legendgroup=location.name,
            mode='lines', 
            line=dict(dash="dash", color=get_color(idx)),
        ))
        
        historical_24 = get_24h_average(df.historical)
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df.historical, 
            name="historical", 
            legendgroup=location.name,
            mode='lines',
            line=dict(color=get_color(idx, 0.4))
        ))
        fig.add_trace(go.Scatter(
            x=historical_24.index, 
            y=historical_24, 
            name="historical av24h", 
            legendgroup=location.name,
            mode='lines', 
            line=dict(color=get_color(idx))
        ))
        
    fig.update_layout(
        title=f"Temperature OpenMeteo",
        xaxis_title="Date",
        yaxis_title="Temperature [°C]",
    )
    fig.update_xaxes(minor=dict(showgrid=True, dtick="D1", tick0=df.historical.index[0], gridcolor="#fff"))
    fig.show()



if __name__ == "__main__":
    """Gather forecast and historical data and generate the plot."""
    start_date = datetime(2023, 6, 1)

    locations = [
        # BERN.copy(), 
        GENEVA.copy()
    ]

    for location in locations:
        df_hist = get_historical(location, start_date)
        df_forecast = get_forecast(location)
        location.data = pd.concat([df_hist, df_forecast], axis=1)

    plot(locations)
