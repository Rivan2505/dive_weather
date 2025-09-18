# app.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Literal, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from divewise_weather import DiveWiseWeather, OpenMeteoError  # reuse your class as-is


# =========================
# Safety policy & risk core
# =========================

FogPolicy = Literal["score", "warn", "caution"]

@dataclass(frozen=True)
class SafetyPolicy:
    # hard stops
    thunderstorm_hard_stop: bool = True

    # wind thresholds (km/h)
    wind_mod: float = 20.0
    wind_strong: float = 30.0
    gust_mod: float = 35.0
    gust_strong: float = 45.0

    # precip thresholds (mm/hr)
    precip_mod: float = 2.0
    precip_heavy: float = 5.0

    # waves (m)
    wave_moderate: float = 1.0
    wave_elevated: float = 1.5
    wave_high: float = 2.0

    # swell (m) and special long-period note (s)
    swell_notable: float = 1.5
    swell_large: float = 2.0
    long_period_s: float = 10.0

    # scoring weights
    score_fog_realtime: int = 10
    score_fog_forecast: int = 20
    score_wind_strong: int = 40
    score_wind_mod: int = 15
    score_precip_heavy: int = 10
    score_precip_mod: int = 5
    score_wave_high: int = 50
    score_wave_elevated: int = 20
    score_wave_moderate: int = 8
    score_swell_large: int = 40
    score_swell_notable: int = 15
    score_surge_long_period: int = 10
    score_missing_waves: int = 5

    # classification bands
    band_caution_min: int = 20
    band_unsafe_min: int = 60


class RiskEngine:
    @staticmethod
    def _map_status(score: int, hard_unsafe: bool, policy: SafetyPolicy) -> str:
        if hard_unsafe:
            return "Unsafe"
        if score >= policy.band_unsafe_min:
            return "Unsafe"
        if score >= policy.band_caution_min:
            return "Caution"
        return "Safe"

    @staticmethod
    def _dedupe(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @staticmethod
    def assess_realtime(
        reading: Dict[str, Any],
        policy: SafetyPolicy,
        fog_policy: FogPolicy = "score",
    ) -> Dict[str, Any]:
        wx = (reading or {}).get("weather", {}) or {}
        sea = (reading or {}).get("marine", {}) or {}

        reasons: List[str] = []
        tips: List[str] = []
        score = 0
        hard_unsafe = False

        code = wx.get("weather_code")
        wind = wx.get("wind_speed_10m_kmh") or 0.0
        gust = wx.get("wind_gusts_10m_kmh") or 0.0
        precip = (wx.get("precipitation_mm") or 0.0) + (wx.get("rain_mm") or 0.0)

        # Thunderstorm
        if policy.thunderstorm_hard_stop and code in (95, 96, 99):
            reasons.append("Thunderstorm in area (lightning risk).")
            hard_unsafe = True

        # Fog
        if code in (45, 48):
            reasons.append("Fog reduces surface visibility — higher boat collision risk.")
            tips.append("Use dive flag + SMB; delay until fog clears or add surface support.")
            if fog_policy == "score":
                score += policy.score_fog_realtime
            elif fog_policy == "caution":
                score = max(score, policy.band_caution_min)

        # Wind
        if wind >= policy.wind_strong or gust >= policy.gust_strong:
            reasons.append(f"Strong wind/gusts ({wind:.0f}/{gust:.0f} km/h).")
            tips.append("Expect chop; consider sheltered site or postponing.")
            score += policy.score_wind_strong
        elif wind >= policy.wind_mod or gust >= policy.gust_mod:
            reasons.append(f"Moderate wind/gusts ({wind:.0f}/{gust:.0f} km/h).")
            tips.append("Prefer protected entry/exit; monitor drift.")
            score += policy.score_wind_mod

        # Precip
        if precip >= policy.precip_heavy:
            reasons.append(f"Heavy rain ({precip:.1f} mm/hr) may reduce vis/runoff.")
            score += policy.score_precip_heavy
        elif precip >= policy.precip_mod:
            reasons.append(f"Moderate rain ({precip:.1f} mm/hr).")
            score += policy.score_precip_mod

        # Sea state
        wave_h = sea.get("wave_height_m")
        swell_h = sea.get("swell_wave_height_m")
        swell_p = sea.get("swell_wave_period_s")

        if isinstance(wave_h, (int, float, float)):
            if wave_h > policy.wave_high:
                reasons.append(f"High waves ({wave_h:.2f} m) — unsafe entries likely.")
                tips.append("Postpone or choose very sheltered site.")
                score += policy.score_wave_high
            elif wave_h > policy.wave_elevated:
                reasons.append(f"Elevated waves ({wave_h:.2f} m).")
                tips.append("Assess surge at entry; plan alternate exits.")
                score += policy.score_wave_elevated
            elif wave_h > policy.wave_moderate:
                reasons.append(f"Moderate waves ({wave_h:.2f} m).")
                score += policy.score_wave_moderate
        else:
            reasons.append("Wave height unavailable — incomplete sea-state assessment.")
            score += policy.score_missing_waves

        if isinstance(swell_h, (int, float)) and isinstance(swell_p, (int, float)):
            if swell_h >= policy.swell_large:
                reasons.append(f"Large swell ({swell_h:.2f} m @ {swell_p:.0f}s).")
                score += policy.score_swell_large
            elif swell_h >= policy.swell_notable:
                reasons.append(f"Notable swell ({swell_h:.2f} m @ {swell_p:.0f}s).")
                score += policy.score_swell_notable
            elif swell_h >= 1.0 and swell_p >= policy.long_period_s:
                reasons.append(f"Long-period swell ({swell_h:.2f} m @ {swell_p:.0f}s) → surge risk.")
                score += policy.score_surge_long_period

        # Comfort (non-scoring)
        air_c = wx.get("temperature_2m_c")
        if isinstance(air_c, (int, float)) and air_c < 16:
            tips.append("Cold air — consider thicker wetsuit, hood, gloves.")

        status = RiskEngine._map_status(score, hard_unsafe, policy)
        return {
            "status": status,
            "score": 100 if hard_unsafe else int(score),
            "reasons": RiskEngine._dedupe(reasons),
            "tips": RiskEngine._dedupe(tips),
        }

    @staticmethod
    def assess_forecast(
        forecast: Dict[str, Any],
        policy: SafetyPolicy,
        fog_policy: FogPolicy = "score",
    ) -> Dict[str, Any]:
        hourly = forecast.get("hourly", []) or []
        marine = forecast.get("marine_daily", {}) or {}

        reasons: List[str] = []
        tips: List[str] = []
        score = 0
        hard_unsafe = False

        if not hourly:
            return {
                "status": "Unknown",
                "score": None,
                "reasons": ["No hourly forecast available."],
                "tips": [],
            }

        # Scan hours (worst hour contributes)
        for h in hourly:
            code = h.get("weather_code")
            wind = h.get("wind_speed_10m_kmh") or 0.0
            gust = h.get("wind_gusts_10m_kmh") or 0.0
            precip = (h.get("precipitation_mm") or 0.0) + (h.get("rain_mm") or 0.0)

            if policy.thunderstorm_hard_stop and code in (95, 96, 99):
                reasons.append("Thunderstorm forecast (lightning risk).")
                hard_unsafe = True
                break

            if code in (45, 48):
                reasons.append("Fog possible at some hours (surface visibility risk).")
                if fog_policy == "score":
                    score += policy.score_fog_forecast  # stricter than realtime
                elif fog_policy == "caution":
                    score = max(score, policy.band_caution_min)

            if wind >= policy.wind_strong or gust >= policy.gust_strong:
                reasons.append(f"Strong winds forecast ({wind:.0f}/{gust:.0f} km/h).")
                score += policy.score_wind_strong
            elif wind >= policy.wind_mod or gust >= policy.gust_mod:
                reasons.append(f"Moderate winds forecast ({wind:.0f}/{gust:.0f} km/h).")
                score += policy.score_wind_mod

            if precip >= policy.precip_heavy:
                reasons.append(f"Heavy rain forecast ({precip:.1f} mm/hr).")
                score += policy.score_precip_heavy
            elif precip >= policy.precip_mod:
                reasons.append(f"Moderate rain forecast ({precip:.1f} mm/hr).")
                score += policy.score_precip_mod

        # Marine (daily max/summary)
        wave_h = marine.get("wave_height_max_m")
        swell_h = marine.get("swell_wave_height_max_m")

        if isinstance(wave_h, (int, float)):
            if wave_h > policy.wave_high:
                reasons.append(f"High waves up to {wave_h:.2f} m.")
                score += policy.score_wave_high
            elif wave_h > policy.wave_elevated:
                reasons.append(f"Elevated waves up to {wave_h:.2f} m.")
                score += policy.score_wave_elevated
            elif wave_h > policy.wave_moderate:
                reasons.append(f"Moderate waves up to {wave_h:.2f} m.")
                score += policy.score_wave_moderate
        else:
            reasons.append("Wave height (forecast) unavailable — incomplete sea state.")
            score += policy.score_missing_waves

        if isinstance(swell_h, (int, float)):
            if swell_h >= policy.swell_large:
                reasons.append(f"Swell up to {swell_h:.2f} m (surge risk).")
                score += policy.score_swell_large
            elif swell_h >= policy.swell_notable:
                reasons.append(f"Swell up to {swell_h:.2f} m.")
                score += policy.score_swell_notable

        status = RiskEngine._map_status(score, hard_unsafe, policy)
        return {
            "status": status,
            "score": 100 if hard_unsafe else int(score),
            "reasons": RiskEngine._dedupe(reasons),
            "tips": RiskEngine._dedupe(tips),
        }


# ============
# API schemas
# ============

class RealtimeRequest(BaseModel):
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    fog_policy: FogPolicy = Field("score", description="'score'|'warn'|'caution'")
    include_marine: bool = Field(True, description="Fetch marine data if available")
    timeout: int = Field(15, ge=1, le=60)


class ForecastRequest(BaseModel):
    lat: float
    lon: float
    date: str = Field(..., description="YYYY-MM-DD (local)")
    fog_policy: FogPolicy = Field("score", description="'score'|'warn'|'caution'")
    include_marine: bool = Field(True)
    timeout: int = Field(15, ge=1, le=60)

    @validator("date")
    def _validate_date(cls, v: str) -> str:
        try:
            _ = date.fromisoformat(v)
        except ValueError:
            raise ValueError("date must be YYYY-MM-DD")
        return v


class SafetyResponse(BaseModel):
    status: Literal["Safe", "Caution", "Unsafe", "Unknown"]
    score: Optional[int]
    reasons: List[str]
    tips: List[str]


class RealtimeResponse(BaseModel):
    coord: Dict[str, float]
    weather: Dict[str, Any]
    marine: Optional[Dict[str, Any]] = None
    safety: SafetyResponse


class ForecastResponse(BaseModel):
    coord: Dict[str, float]
    date: str
    daily: Dict[str, Any]
    hourly: List[Dict[str, Any]]
    marine_daily: Optional[Dict[str, Any]] = None
    safety: SafetyResponse


# =========
# FastAPI
# =========

app = FastAPI(title="DiveWise Weather API", version="1.0.0")

# Separate policies allow different tuning per mode
REALTIME_POLICY = SafetyPolicy()
FORECAST_POLICY = SafetyPolicy()  # you can tweak thresholds separately if desired

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternative React port
        "*"  # Allow all origins (for development only)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/realtime", response_model=RealtimeResponse)
def realtime(req: RealtimeRequest):
    client = DiveWiseWeather(timeout=req.timeout, include_marine=req.include_marine)
    try:
        data = client.get_realtime(req.lat, req.lon)
    except OpenMeteoError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    safety = RiskEngine.assess_realtime(data, REALTIME_POLICY, fog_policy=req.fog_policy)
    data_out = dict(data)
    data_out["safety"] = safety
    return data_out


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    client = DiveWiseWeather(timeout=req.timeout, include_marine=req.include_marine)
    try:
        data = client.get_forecast_for_date(req.lat, req.lon, req.date)
    except OpenMeteoError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    safety = RiskEngine.assess_forecast(data, FORECAST_POLICY, fog_policy=req.fog_policy)
    data_out = dict(data)
    data_out["safety"] = safety
    return data_out
