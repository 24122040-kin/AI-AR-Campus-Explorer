"""
core/environmental_analyzer.py - Offline-friendly environmental heuristics.
Combines local crowd/weather observations with sensible defaults so routing can
stay useful even when no remote traffic/weather feeds are available.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from core.database import db


@dataclass
class EnvironmentalCell:
    crowd: list[float] = field(default_factory=list)
    weather: list[float] = field(default_factory=list)


class EnvironmentalAnalyzer:
    def __init__(self, cell_size_deg: float = 0.003):
        self.cell_size_deg = cell_size_deg
        self._grid: dict[tuple[int, int], EnvironmentalCell] = defaultdict(EnvironmentalCell)
        self._hourly_crowd: dict[tuple[int, int], float] = {}
        self._hourly_weather: dict[tuple[int, int], float] = {}
        self._last_refresh = datetime.min

    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        return (round(lat / self.cell_size_deg), round(lon / self.cell_size_deg))

    async def refresh(self, force: bool = False) -> None:
        age = (datetime.now() - self._last_refresh).total_seconds()
        if not force and age < 300:
            return

        rows = await db.fetchall(
            """
            SELECT hour, weekday,
                   AVG(COALESCE(crowd_level, 0)) AS avg_crowd,
                   AVG(COALESCE(weather_severity, 0)) AS avg_weather
            FROM environmental_observations
            GROUP BY hour, weekday
            """
        )
        self._hourly_crowd.clear()
        self._hourly_weather.clear()
        for row in rows:
            key = (row["hour"], row["weekday"])
            self._hourly_crowd[key] = float(row["avg_crowd"] or 0.0)
            self._hourly_weather[key] = float(row["avg_weather"] or 0.0)

        self._grid = defaultdict(EnvironmentalCell)
        spatial = await db.fetchall(
            """
            SELECT lat, lon, crowd_level, weather_severity
            FROM environmental_observations
            """
        )
        for row in spatial:
            cell = self._grid[self._key(row["lat"], row["lon"])]
            if row["crowd_level"] is not None:
                cell.crowd.append(float(row["crowd_level"]))
            if row["weather_severity"] is not None:
                cell.weather.append(float(row["weather_severity"]))

        self._last_refresh = datetime.now()

    @staticmethod
    def _default_crowd(hour: int, weekday: int) -> float:
        rush = 0.7 if hour in {7, 8, 17, 18} else 0.35
        weekend = 0.8 if weekday >= 5 else 1.0
        lunch = 0.15 if hour in {11, 12, 13} else 0.0
        return min(1.0, rush * weekend + lunch)

    @staticmethod
    def _default_weather(hour: int) -> float:
        if 14 <= hour <= 18:
            return 0.25
        if 19 <= hour <= 21:
            return 0.15
        return 0.05

    def crowd_level(self, lat: float | None, lon: float | None, hour: int, weekday: int) -> float:
        val = self._hourly_crowd.get((hour, weekday), self._default_crowd(hour, weekday))
        if lat is not None and lon is not None:
            cell = self._grid.get(self._key(lat, lon))
            if cell and cell.crowd:
                val = 0.5 * val + 0.5 * (sum(cell.crowd) / len(cell.crowd))
        return min(1.0, max(0.0, val))

    def weather_severity(self, lat: float | None, lon: float | None, hour: int, weekday: int) -> float:
        val = self._hourly_weather.get((hour, weekday), self._default_weather(hour))
        if lat is not None and lon is not None:
            cell = self._grid.get(self._key(lat, lon))
            if cell and cell.weather:
                val = 0.5 * val + 0.5 * (sum(cell.weather) / len(cell.weather))
        return min(1.0, max(0.0, val))

    def environmental_penalty(self, lat: float | None, lon: float | None, depart_time: datetime) -> tuple[float, dict]:
        hour = depart_time.hour
        weekday = depart_time.weekday()
        crowd = self.crowd_level(lat, lon, hour, weekday)
        weather = self.weather_severity(lat, lon, hour, weekday)
        penalty = 1.0 + crowd + weather
        return penalty, {
            "crowd_level": round(crowd, 3),
            "weather_severity": round(weather, 3),
            "hour": hour,
            "weekday": weekday,
        }


environmental_analyzer = EnvironmentalAnalyzer()
