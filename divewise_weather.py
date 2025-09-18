# divewise_weather.py
# requirements: requests>=2.28
from __future__ import annotations

import requests
from datetime import datetime, date
from typing import Any, Dict, Optional, List


class OpenMeteoError(RuntimeError):
    """Raised when Open-Meteo returns an error or the response is invalid."""


class DiveWiseWeather:
    """
    Lightweight client for Open-Meteo weather & marine data.

    Endpoints:
      - Weather: https://api.open-meteo.com/v1/forecast
      - Marine : https://marine-api.open-meteo.com/v1/marine
    """

    WEATHER_BASE = "https://api.open-meteo.com/v1/forecast"
    MARINE_BASE = "https://marine-api.open-meteo.com/v1/marine"

    # WMO code → simple text label (subset most useful for divers)
    WMO_CODE = {
        0: "Clear",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    def __init__(self, timeout: int = 15, include_marine: bool = True):
        """
        :param timeout: HTTP timeout in seconds.
        :param include_marine: If True, also fetch marine (wave) data when available.
        """
        self.timeout = timeout
        self.include_marine = include_marine

    # ---------------------------
    # Public API
    # ---------------------------

    def get_realtime(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get real-time conditions for a coordinate.
        """
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "precipitation",
                    "rain",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                    "wind_direction_10m",
                    "weather_code",
                ]
            ),
            "timezone": "auto",
        }
        weather = self._get_json(self.WEATHER_BASE, weather_params)

        result: Dict[str, Any] = {
            "coord": {"lat": lat, "lon": lon},
            "weather": self._parse_current_weather(weather),
        }

        if self.include_marine:
            marine_params = {
                "latitude": lat,
                "longitude": lon,
                "current": ",".join(
                    [
                        "wave_height",
                        "wave_direction",
                        "wave_period",
                        "wind_wave_height",
                        "wind_wave_direction",
                        "wind_wave_period",
                        "swell_wave_height",
                        "swell_wave_direction",
                        "swell_wave_period",
                    ]
                ),
                "timezone": "auto",
            }
            marine = self._get_json(self.MARINE_BASE, marine_params, allow_fail=True)
            if marine:
                result["marine"] = self._parse_current_marine(marine)

        return result

    def get_forecast_for_date(
            self, lat: float, lon: float, date_str: str
    ) -> Dict[str, Any]:
        """
        Get a daily summary + hourly series for a specific date (local time).
        """
        try:
            target = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError("date must be in 'YYYY-MM-DD' format") from e

        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": target.isoformat(),
            "end_date": target.isoformat(),
            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "rain_sum",
                    "windspeed_10m_max",
                    "windgusts_10m_max",
                    "winddirection_10m_dominant",
                    "uv_index_max",
                    "shortwave_radiation_sum",
                    "sunrise",
                    "sunset",
                ]
            ),
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "apparent_temperature",
                    "precipitation",
                    "rain",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                    "wind_direction_10m",
                    "weather_code",
                ]
            ),
            "timezone": "auto",
        }
        weather = self._get_json(self.WEATHER_BASE, weather_params)

        out: Dict[str, Any] = {
            "coord": {"lat": lat, "lon": lon},
            "date": target.isoformat(),
            "daily": self._parse_daily_weather(weather),
            "hourly": self._select_hourly_for_date(weather, target),
        }

        if self.include_marine:
            marine_params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": target.isoformat(),
                "end_date": target.isoformat(),
                "hourly": ",".join(
                    [
                        "wave_height",
                        "wave_direction",
                        "wave_period",
                        "wind_wave_height",
                        "swell_wave_height",
                    ]
                ),
                "timezone": "auto",
            }
            marine = self._get_json(self.MARINE_BASE, marine_params, allow_fail=True)
            if marine:
                out["marine_daily"] = self._aggregate_marine_day(marine)

        return out

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _get_json(
            self, url: str, params: Dict[str, Any], allow_fail: bool = False
    ) -> Optional[Dict[str, Any]]:
        try:
            r = requests.get(url, params=params, timeout=self.timeout)
            if r.status_code != 200:
                if allow_fail:
                    return None
                raise OpenMeteoError(f"HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            if isinstance(data, dict) and data.get("reason"):
                if allow_fail:
                    return None
                raise OpenMeteoError(data["reason"])
            return data
        except requests.RequestException as e:
            if allow_fail:
                return None
            raise OpenMeteoError(str(e)) from e

    def _parse_current_weather(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # keep as instance method (uses self.WMO_CODE)
        current = payload.get("current", {})
        return {
            "time": current.get("time"),
            "temperature_2m_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "relative_humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "cloud_cover_pct": current.get("cloud_cover"),
            "wind_speed_10m_kmh": current.get("wind_speed_10m"),
            "wind_gusts_10m_kmh": current.get("wind_gusts_10m"),
            "wind_direction_10m_deg": current.get("wind_direction_10m"),
            "weather_code": current.get("weather_code"),
            "weather_text": self.WMO_CODE.get(current.get("weather_code")),
        }

    @staticmethod
    def _parse_current_marine(payload: Dict[str, Any]) -> Dict[str, Any]:
        current = payload.get("current", {})
        return {
            "time": current.get("time"),
            "wave_height_m": current.get("wave_height"),
            "wave_direction_deg": current.get("wave_direction"),
            "wave_period_s": current.get("wave_period"),
            "wind_wave_height_m": current.get("wind_wave_height"),
            "wind_wave_direction_deg": current.get("wind_wave_direction"),
            "wind_wave_period_s": current.get("wind_wave_period"),
            "swell_wave_height_m": current.get("swell_wave_height"),
            "swell_wave_direction_deg": current.get("swell_wave_direction"),
            "swell_wave_period_s": current.get("swell_wave_period"),
        }

    @staticmethod
    def _parse_daily_weather(payload: Dict[str, Any]) -> Dict[str, Any]:
        daily = payload.get("daily", {})

        def first(arr_name: str) -> Optional[Any]:
            arr = daily.get(arr_name) or []
            return arr[0] if arr else None

        return {
            "temperature_max_c": first("temperature_2m_max"),
            "temperature_min_c": first("temperature_2m_min"),
            "precipitation_sum_mm": first("precipitation_sum"),
            "rain_sum_mm": first("rain_sum"),
            "windspeed_10m_max_kmh": first("windspeed_10m_max"),
            "windgusts_10m_max_kmh": first("windgusts_10m_max"),
            "winddirection_10m_dominant_deg": first("winddirection_10m_dominant"),
            "uv_index_max": first("uv_index_max"),
            "shortwave_radiation_sum_MJ_m2": first("shortwave_radiation_sum"),
            "sunrise_local": first("sunrise"),
            "sunset_local": first("sunset"),
        }

    @staticmethod
    def _aggregate_marine_day(payload: Dict[str, Any]) -> Dict[str, Any]:
        hourly = payload.get("hourly", {})

        def arr(name: str) -> List[Optional[float]]:
            return hourly.get(name) or []

        def safe_max(a: List[Optional[float]]) -> Optional[float]:
            vals = [x for x in a if isinstance(x, (int, float))]
            return max(vals) if vals else None

        def safe_min(a: List[Optional[float]]) -> Optional[float]:
            vals = [x for x in a if isinstance(x, (int, float))]
            return min(vals) if vals else None

        return {
            "wave_height_max_m": safe_max(arr("wave_height")),
            "wave_height_min_m": safe_min(arr("wave_height")),
            "wind_wave_height_max_m": safe_max(arr("wind_wave_height")),
            "swell_wave_height_max_m": safe_max(arr("swell_wave_height")),
            "wave_period_max_s": safe_max(arr("wave_period")),
        }

    def _select_hourly_for_date(
            self, payload: Dict[str, Any], target_day: date
    ) -> List[Dict[str, Any]]:
        hourly = payload.get("hourly", {})
        times: List[str] = hourly.get("time") or []
        result = []

        def val(name: str, t: int) -> Optional[Any]:
            arr = hourly.get(name) or []
            return arr[t] if t < len(arr) else None

        for i, ts in enumerate(times):
            try:
                dt = datetime.fromisoformat(ts)
            except ValueError:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.date() == target_day:
                result.append(
                    {
                        "time": ts,
                        "temperature_2m_c": val("temperature_2m", i),
                        "apparent_temperature_c": val("apparent_temperature", i),
                        "precipitation_mm": val("precipitation", i),
                        "rain_mm": val("rain", i),
                        "cloud_cover_pct": val("cloud_cover", i),
                        "wind_speed_10m_kmh": val("wind_speed_10m", i),
                        "wind_gusts_10m_kmh": val("wind_gusts_10m", i),
                        "wind_direction_10m_deg": val("wind_direction_10m", i),
                        "weather_code": val("weather_code", i),
                        "weather_text": self.WMO_CODE.get(val("weather_code", i)),
                    }
                )
        return result
