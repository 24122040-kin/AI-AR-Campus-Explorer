"""
core/database.py — SQLite database with spatial support via SpatiaLite or raw coords
"""
from __future__ import annotations
import json
import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import aiosqlite
from loguru import logger

from config.settings import settings


# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Locations: every geo-tagged position the user has photographed
CREATE TABLE IF NOT EXISTS locations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    description TEXT,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    altitude    REAL,
    importance  INTEGER DEFAULT 1 CHECK(importance BETWEEN 1 AND 5),
    category    TEXT DEFAULT 'general',   -- cafe, landmark, alley, shop, ...
    tags        TEXT DEFAULT '[]',        -- JSON array of string tags
    osm_node_id INTEGER,                  -- link to OSM node if available
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Images: 2-4 photos per location
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id     INTEGER NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL UNIQUE,
    filepath        TEXT NOT NULL,
    caption         TEXT,
    bearing         REAL,                  -- compass direction of camera (0-360)
    taken_at        TEXT,
    width           INTEGER,
    height          INTEGER,
    file_size_kb    INTEGER,
    faiss_index_id  INTEGER,               -- row in FAISS index (-1 = not indexed)
    created_at      TEXT NOT NULL
);

-- Custom POI (places not on global map)
CREATE TABLE IF NOT EXISTS pois (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id INTEGER REFERENCES locations(id) ON DELETE SET NULL,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,             -- restaurant, shortcut, alley, ...
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    address     TEXT,
    phone       TEXT,
    hours       TEXT,                      -- opening hours, free text
    notes       TEXT,
    is_active   INTEGER DEFAULT 1,
    created_at  TEXT NOT NULL
);

-- Custom edges (shortcuts, alleyways not in OSM)
CREATE TABLE IF NOT EXISTS custom_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT,
    from_lat        REAL NOT NULL,
    from_lon        REAL NOT NULL,
    to_lat          REAL NOT NULL,
    to_lon          REAL NOT NULL,
    distance_m      REAL,
    travel_time_s   REAL,
    road_type       TEXT DEFAULT 'alley',  -- alley | shortcut | path
    is_bidirectional INTEGER DEFAULT 1,
    notes           TEXT,
    created_at      TEXT NOT NULL
);

-- Traffic observations (crowdsourced, feed into heuristic)
CREATE TABLE IF NOT EXISTS traffic_observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    hour        INTEGER NOT NULL CHECK(hour BETWEEN 0 AND 23),
    weekday     INTEGER NOT NULL CHECK(weekday BETWEEN 0 AND 6),
    speed_kmh   REAL,
    congestion  REAL CHECK(congestion BETWEEN 0.0 AND 1.0),
    observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS environmental_observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    lat              REAL NOT NULL,
    lon              REAL NOT NULL,
    hour             INTEGER NOT NULL CHECK(hour BETWEEN 0 AND 23),
    weekday          INTEGER NOT NULL CHECK(weekday BETWEEN 0 AND 6),
    crowd_level      REAL CHECK(crowd_level BETWEEN 0.0 AND 1.0),
    weather_severity REAL CHECK(weather_severity BETWEEN 0.0 AND 1.0),
    notes            TEXT,
    observed_at      TEXT NOT NULL
);

-- Navigation sessions (for analytics / re-routing)
CREATE TABLE IF NOT EXISTS nav_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    origin_lat      REAL NOT NULL,
    origin_lon      REAL NOT NULL,
    dest_lat        REAL NOT NULL,
    dest_lon        REAL NOT NULL,
    depart_at       TEXT,
    route_json      TEXT,
    total_distance_m REAL,
    total_time_s    REAL,
    created_at      TEXT NOT NULL
);

-- ── Indoor mapping ──────────────────────────────────────────────────────────

-- One row per floor-plan GeoJSON upload
CREATE TABLE IF NOT EXISTS floor_maps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    building_id     TEXT NOT NULL,
    floor           INTEGER NOT NULL,
    name            TEXT NOT NULL,
    geojson         TEXT NOT NULL,          -- full GeoJSON FeatureCollection
    lat_center      REAL,                   -- bounding-box centre for proximity lookup
    lon_center      REAL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- Denormalised node table for fast spatial queries without parsing GeoJSON
CREATE TABLE IF NOT EXISTS floor_nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    building_id     TEXT NOT NULL,
    floor           INTEGER NOT NULL,
    node_id         TEXT NOT NULL,          -- matches GeoJSON feature id
    name            TEXT NOT NULL,
    node_type       TEXT NOT NULL,          -- room|corridor|stairs|elevator|entrance|exit
    lat             REAL NOT NULL,
    lon             REAL NOT NULL,
    accessible      INTEGER DEFAULT 1,
    properties      TEXT DEFAULT '{}',      -- JSON blob for extra props
    created_at      TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_floor_maps_building_floor ON floor_maps(building_id, floor);
CREATE INDEX IF NOT EXISTS idx_floor_nodes_building ON floor_nodes(building_id);
CREATE INDEX IF NOT EXISTS idx_floor_nodes_building_floor ON floor_nodes(building_id, floor);
CREATE INDEX IF NOT EXISTS idx_floor_nodes_lat_lon ON floor_nodes(lat, lon);

-- ── Migration: add floor to locations (safe: IF NOT EXISTS via trigger) ──────
-- SQLite does not support IF NOT EXISTS on ALTER TABLE, so we use a workaround:
-- The column is added only if it doesn't exist yet (handled in Database.init).

-- ── Migration: add floor + geometry to custom_edges ──────────────────────────
-- Same pattern — handled in Database.init via ALTER TABLE with error suppression.

-- Indexes for spatial proximity queries
CREATE INDEX IF NOT EXISTS idx_locations_lat_lon ON locations(lat, lon);
CREATE INDEX IF NOT EXISTS idx_pois_lat_lon ON pois(lat, lon);
CREATE INDEX IF NOT EXISTS idx_images_location ON images(location_id);
CREATE INDEX IF NOT EXISTS idx_traffic_hour_weekday ON traffic_observations(hour, weekday);
CREATE INDEX IF NOT EXISTS idx_environment_hour_weekday ON environmental_observations(hour, weekday);
"""


class Database:
    """Async SQLite wrapper with convenience methods."""

    def __init__(self, path: Path = settings.db_path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()
        # ── Safe migrations (ALTER TABLE IF column not exists) ────────────────
        await self._migrate()
        logger.info(f"Database initialised at {self.path}")

    async def _migrate(self) -> None:
        """Apply additive schema migrations safely — never drops data."""
        migrations = [
            # locations: add floor (1 = ground floor default)
            "ALTER TABLE locations ADD COLUMN floor INTEGER DEFAULT 1",
            # locations: primary image for display in search results
            "ALTER TABLE locations ADD COLUMN primary_image_id INTEGER DEFAULT NULL",
            # custom_edges: add floor info for multi-floor paths
            "ALTER TABLE custom_edges ADD COLUMN from_floor INTEGER DEFAULT 1",
            "ALTER TABLE custom_edges ADD COLUMN to_floor INTEGER DEFAULT 1",
            # custom_edges: add geometry for walk-tracked curved paths (JSON array)
            "ALTER TABLE custom_edges ADD COLUMN geometry TEXT DEFAULT NULL",
            # custom_edges: physical properties
            "ALTER TABLE custom_edges ADD COLUMN is_covered INTEGER DEFAULT 0",
            "ALTER TABLE custom_edges ADD COLUMN width_m REAL DEFAULT NULL",
            "ALTER TABLE custom_edges ADD COLUMN surface TEXT DEFAULT 'concrete'",
            "ALTER TABLE custom_edges ADD COLUMN has_lighting INTEGER DEFAULT 1",
            "ALTER TABLE custom_edges ADD COLUMN slope_deg REAL DEFAULT 0",
            # images: mark as primary for a location
            "ALTER TABLE images ADD COLUMN is_primary INTEGER DEFAULT 0",
        ]
        async with aiosqlite.connect(self.path) as db:
            for sql in migrations:
                try:
                    await db.execute(sql)
                except Exception:
                    pass  # column already exists — safe to ignore
            await db.commit()

    # ── helpers ──────────────────────────────────────────────────────

    async def execute(self, sql: str, params: tuple = ()) -> int:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(sql, params)
            await db.commit()
            return cur.lastrowid  # type: ignore

    async def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(sql, params)
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(sql, params)
            row = await cur.fetchone()
            return dict(row) if row else None

    # ── Location CRUD ─────────────────────────────────────────────────

    async def add_location(
        self,
        name: str,
        lat: float,
        lon: float,
        description: str = "",
        category: str = "general",
        importance: int = 1,
        tags: list[str] | None = None,
        osm_node_id: int | None = None,
        floor: int = 1,
    ) -> int:
        now = datetime.utcnow().isoformat()
        return await self.execute(
            """INSERT INTO locations
               (name, description, lat, lon, floor, importance, category, tags, osm_node_id, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (name, description, lat, lon, floor, importance, category,
             json.dumps(tags or []), osm_node_id, now, now),
        )

    async def get_location(self, loc_id: int) -> Optional[dict]:
        return await self.fetchone("SELECT * FROM locations WHERE id=?", (loc_id,))

    async def nearby_locations(
        self, lat: float, lon: float, radius_deg: float = 0.01
    ) -> list[dict]:
        """Bounding-box proximity query (fast without PostGIS)."""
        return await self.fetchall(
            """SELECT *, (lat-?)*111000 AS dy, (lon-?)*111000*COS(lat*3.14159/180) AS dx
               FROM locations
               WHERE lat BETWEEN ?-? AND ?+?
                 AND lon BETWEEN ?-? AND ?+?
               ORDER BY (dy*dy + dx*dx)
               LIMIT 50""",
            (lat, lon, lat, radius_deg, lat, radius_deg, lon, radius_deg, lon, radius_deg),
        )

    async def search_locations(self, query: str) -> list[dict]:
        q = f"%{query}%"
        return await self.fetchall(
            "SELECT * FROM locations WHERE name LIKE ? OR description LIKE ? OR tags LIKE ?",
            (q, q, q),
        )

    # ── Image CRUD ────────────────────────────────────────────────────

    async def add_image(
        self,
        location_id: int,
        filename: str,
        filepath: str,
        caption: str = "",
        bearing: float | None = None,
        faiss_index_id: int = -1,
    ) -> int:
        now = datetime.utcnow().isoformat()
        return await self.execute(
            """INSERT INTO images (location_id, filename, filepath, caption, bearing, faiss_index_id, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (location_id, filename, filepath, caption, bearing, faiss_index_id, now),
        )

    async def set_primary_image(self, location_id: int, image_id: int) -> None:
        """Set the primary display image for a location."""
        await self.execute(
            "UPDATE images SET is_primary=0 WHERE location_id=?", (location_id,)
        )
        await self.execute(
            "UPDATE images SET is_primary=1 WHERE id=?", (image_id,)
        )
        await self.execute(
            "UPDATE locations SET primary_image_id=? WHERE id=?", (image_id, location_id)
        )

    async def get_primary_image(self, location_id: int) -> Optional[dict]:
        return await self.fetchone(
            "SELECT * FROM images WHERE location_id=? AND is_primary=1 LIMIT 1",
            (location_id,),
        )

    async def search_locations_ranked(self, query: str, limit: int = 10) -> list[dict]:
        """
        Semantic-ish search: exact name match first, then partial matches,
        then description/tags. Returns up to `limit` results with primary image.
        """
        q = query.strip().lower()
        q_like = f"%{q}%"
        rows = await self.fetchall(
            """
            SELECT l.*,
                   COUNT(i.id) AS image_count,
                   i2.filepath AS primary_image_path,
                   i2.id AS primary_image_id_img,
                   -- Relevance score: exact name=100, starts-with=80, contains=60, desc/tags=30
                   CASE
                     WHEN LOWER(l.name) = ?                    THEN 100
                     WHEN LOWER(l.name) LIKE ? || '%'          THEN 80
                     WHEN LOWER(l.name) LIKE '%' || ? || '%'   THEN 60
                     WHEN LOWER(l.description) LIKE '%'||?||'%'
                          OR LOWER(l.tags) LIKE '%'||?||'%'    THEN 30
                     ELSE 10
                   END AS relevance
            FROM locations l
            LEFT JOIN images i ON i.location_id = l.id
            LEFT JOIN images i2 ON i2.location_id = l.id AND i2.is_primary = 1
            WHERE LOWER(l.name) LIKE ?
               OR LOWER(l.description) LIKE ?
               OR LOWER(l.tags) LIKE ?
            GROUP BY l.id
            ORDER BY relevance DESC, l.importance DESC
            LIMIT ?
            """,
            (q, q, q, q, q, q_like, q_like, q_like, limit),
        )
        return rows
        return await self.fetchall(
            "SELECT * FROM images WHERE location_id=? ORDER BY bearing",
            (location_id,),
        )

    async def update_faiss_id(self, image_id: int, faiss_id: int) -> None:
        await self.execute(
            "UPDATE images SET faiss_index_id=? WHERE id=?", (faiss_id, image_id)
        )

    # ── POI CRUD ──────────────────────────────────────────────────────

    async def add_poi(
        self,
        name: str,
        poi_type: str,
        lat: float,
        lon: float,
        address: str = "",
        notes: str = "",
        location_id: int | None = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        return await self.execute(
            """INSERT INTO pois (name, type, lat, lon, address, notes, location_id, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (name, poi_type, lat, lon, address, notes, location_id, now),
        )

    async def search_pois(self, query: str) -> list[dict]:
        q = f"%{query}%"
        return await self.fetchall(
            "SELECT * FROM pois WHERE is_active=1 AND (name LIKE ? OR type LIKE ? OR address LIKE ?)",
            (q, q, q),
        )

    async def nearby_pois(self, lat: float, lon: float, radius_deg: float = 0.01) -> list[dict]:
        return await self.fetchall(
            """SELECT * FROM pois
               WHERE is_active=1
                 AND lat BETWEEN ?-? AND ?+?
                 AND lon BETWEEN ?-? AND ?+?""",
            (lat, radius_deg, lat, radius_deg, lon, radius_deg, lon, radius_deg),
        )

    # ── Custom edge CRUD ──────────────────────────────────────────────

    async def add_custom_edge(
        self,
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        name: str = "",
        road_type: str = "alley",
        bidirectional: bool = True,
        notes: str = "",
        from_floor: int = 1,
        to_floor: int = 1,
        geometry: list[tuple[float, float]] | None = None,
    ) -> tuple[int, float]:
        import math
        if geometry and len(geometry) >= 2:
            # compute distance along the actual path
            dist = sum(
                math.sqrt(
                    ((geometry[i+1][0] - geometry[i][0]) * 111000) ** 2 +
                    ((geometry[i+1][1] - geometry[i][1]) * 111000 *
                     math.cos(math.radians(geometry[i][0]))) ** 2
                )
                for i in range(len(geometry) - 1)
            )
        else:
            dlat = (to_lat - from_lat) * 111000
            dlon = (to_lon - from_lon) * 111000 * math.cos(math.radians(from_lat))
            dist = math.sqrt(dlat**2 + dlon**2)
        speed = 5 if road_type in ("alley", "path") else 20
        t = dist / (speed / 3.6)
        geom_json = json.dumps(geometry) if geometry else None
        now = datetime.utcnow().isoformat()
        eid = await self.execute(
            """INSERT INTO custom_edges
               (name, from_lat, from_lon, to_lat, to_lon, distance_m, travel_time_s,
                road_type, is_bidirectional, notes, from_floor, to_floor, geometry, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (name, from_lat, from_lon, to_lat, to_lon, dist, t,
             road_type, int(bidirectional), notes, from_floor, to_floor, geom_json, now),
        )
        return eid, float(dist)

    async def get_all_custom_edges(self) -> list[dict]:
        return await self.fetchall("SELECT * FROM custom_edges")

    async def find_edge_between_points(
        self,
        from_lat: float, from_lon: float,
        to_lat: float, to_lon: float,
        tolerance: float = 0.00001,  # ~1m tolerance
    ) -> list[dict]:
        """Find edges between two points (both directions)."""
        return await self.fetchall(
            """SELECT * FROM custom_edges
               WHERE (ABS(from_lat - ?) < ? AND ABS(from_lon - ?) < ?
                      AND ABS(to_lat - ?) < ? AND ABS(to_lon - ?) < ?)
                  OR (ABS(from_lat - ?) < ? AND ABS(from_lon - ?) < ?
                      AND ABS(to_lat - ?) < ? AND ABS(to_lon - ?) < ?)""",
            (from_lat, tolerance, from_lon, tolerance, to_lat, tolerance, to_lon, tolerance,
             to_lat, tolerance, to_lon, tolerance, from_lat, tolerance, from_lon, tolerance),
        )

    async def delete_edge(self, edge_id: int) -> bool:
        """Delete a custom edge by ID."""
        await self.execute("DELETE FROM custom_edges WHERE id=?", (edge_id,))
        return True

    # ── Traffic ───────────────────────────────────────────────────────

    async def add_traffic_obs(
        self, lat: float, lon: float, hour: int, weekday: int,
        speed_kmh: float | None, congestion: float | None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        return await self.execute(
            """INSERT INTO traffic_observations (lat, lon, hour, weekday, speed_kmh, congestion, observed_at)
               VALUES (?,?,?,?,?,?,?)""",
            (lat, lon, hour, weekday, speed_kmh, congestion, now),
        )

    async def avg_congestion(self, hour: int, weekday: int | None = None) -> float:
        """Return average congestion factor for a given hour (0=free, 1=jam)."""
        if weekday is not None:
            row = await self.fetchone(
                "SELECT AVG(congestion) AS c FROM traffic_observations WHERE hour=? AND weekday=?",
                (hour, weekday),
            )
        else:
            row = await self.fetchone(
                "SELECT AVG(congestion) AS c FROM traffic_observations WHERE hour=?",
                (hour,),
            )
        return row["c"] if row and row["c"] is not None else 0.0

    async def add_environment_obs(
        self,
        lat: float,
        lon: float,
        hour: int,
        weekday: int,
        crowd_level: float | None,
        weather_severity: float | None,
        notes: str = "",
    ) -> int:
        now = datetime.utcnow().isoformat()
        return await self.execute(
            """
            INSERT INTO environmental_observations
            (lat, lon, hour, weekday, crowd_level, weather_severity, notes, observed_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (lat, lon, hour, weekday, crowd_level, weather_severity, notes, now),
        )

    # ── Indoor floor maps ─────────────────────────────────────────────────────

    async def upsert_floor_map(
        self,
        building_id: str,
        floor: int,
        name: str,
        geojson: dict,
        lat_center: float | None = None,
        lon_center: float | None = None,
    ) -> int:
        """Insert or replace a floor-plan GeoJSON. Returns the row id."""
        import json as _json
        now = datetime.utcnow().isoformat()
        geojson_str = _json.dumps(geojson, ensure_ascii=False)
        existing = await self.fetchone(
            "SELECT id FROM floor_maps WHERE building_id=? AND floor=?",
            (building_id, floor),
        )
        if existing:
            await self.execute(
                """UPDATE floor_maps
                   SET name=?, geojson=?, lat_center=?, lon_center=?, updated_at=?
                   WHERE building_id=? AND floor=?""",
                (name, geojson_str, lat_center, lon_center, now, building_id, floor),
            )
            return existing["id"]
        return await self.execute(
            """INSERT INTO floor_maps (building_id, floor, name, geojson, lat_center, lon_center, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (building_id, floor, name, geojson_str, lat_center, lon_center, now, now),
        )

    async def get_floor_map(self, building_id: str, floor: int) -> dict | None:
        return await self.fetchone(
            "SELECT * FROM floor_maps WHERE building_id=? AND floor=?",
            (building_id, floor),
        )

    async def list_floor_maps(self, building_id: str) -> list[dict]:
        return await self.fetchall(
            "SELECT id, building_id, floor, name, lat_center, lon_center, created_at, updated_at "
            "FROM floor_maps WHERE building_id=? ORDER BY floor",
            (building_id,),
        )

    async def list_buildings(self) -> list[dict]:
        return await self.fetchall(
            """SELECT building_id,
                      COUNT(DISTINCT floor) AS floor_count,
                      MIN(floor) AS min_floor,
                      MAX(floor) AS max_floor,
                      MIN(lat_center) AS lat,
                      MIN(lon_center) AS lon
               FROM floor_maps
               GROUP BY building_id
               ORDER BY building_id"""
        )

    async def upsert_floor_nodes(self, building_id: str, floor: int, nodes: list[dict]) -> int:
        """Bulk-replace all nodes for a building+floor."""
        import json as _json
        now = datetime.utcnow().isoformat()
        await self.execute(
            "DELETE FROM floor_nodes WHERE building_id=? AND floor=?",
            (building_id, floor),
        )
        count = 0
        for n in nodes:
            await self.execute(
                """INSERT INTO floor_nodes
                   (building_id, floor, node_id, name, node_type, lat, lon, accessible, properties, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    building_id, floor, n["node_id"], n["name"], n["node_type"],
                    n["lat"], n["lon"], int(n.get("accessible", True)),
                    _json.dumps(n.get("properties", {})), now,
                ),
            )
            count += 1
        return count

    async def nearby_floor_nodes(
        self,
        lat: float,
        lon: float,
        radius_deg: float = 0.001,
        floor: int | None = None,
    ) -> list[dict]:
        if floor is not None:
            return await self.fetchall(
                """SELECT * FROM floor_nodes
                   WHERE floor=?
                     AND lat BETWEEN ?-? AND ?+?
                     AND lon BETWEEN ?-? AND ?+?
                   ORDER BY (lat-?)*(lat-?) + (lon-?)*(lon-?)
                   LIMIT 20""",
                (floor, lat, radius_deg, lat, radius_deg, lon, radius_deg, lon, radius_deg,
                 lat, lat, lon, lon),
            )
        return await self.fetchall(
            """SELECT * FROM floor_nodes
               WHERE lat BETWEEN ?-? AND ?+?
                 AND lon BETWEEN ?-? AND ?+?
               ORDER BY (lat-?)*(lat-?) + (lon-?)*(lon-?)
               LIMIT 20""",
            (lat, radius_deg, lat, radius_deg, lon, radius_deg, lon, radius_deg,
             lat, lat, lon, lon),
        )

    async def find_location_by_coords(
        self,
        lat: float,
        lon: float,
        tolerance: float = 0.0001,  # ~10m
    ) -> dict | None:
        """Find a location within tolerance of given coordinates."""
        return await self.fetchone(
            """SELECT * FROM locations
               WHERE ABS(lat - ?) < ? AND ABS(lon - ?) < ?
               ORDER BY (lat-?)*(lat-?) + (lon-?)*(lon-?)
               LIMIT 1""",
            (lat, tolerance, lon, tolerance, lat, lat, lon, lon),
        )


# Singleton
db = Database()
