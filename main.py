# VB_v104 — Vilyra Legacy Bot (Railway + Postgres) — FULL REPLACEMENT
# Goals (v104):
# - All commands ephemeral
# - Log ALL commands to #Legacy-Commands-Log (except /char_card)
# - Stable command sync (no duplicates / no CommandNotFound)
# - Character selection via autocomplete using numeric character_id token (still searchable by name)
# - Additive-only DB migrations (no drops/renames)
# - Avoid startup PATCH storms (no automatic dashboard refresh unless enabled)
#
# Required env:
# - DISCORD_TOKEN
# - DATABASE_URL
# Optional env:
# - GUILD_ID (for guild-scoped sync; recommended)
# - ALLOW_COMMAND_RESET=true + ALLOWED_GUILD_IDS=... (guarded hard reset)
# - DASHBOARD_CHANNEL_ID (defaults to constant)
# - COMMAND_LOG_CHANNEL_ID (fallback if channel name not found)
# - STARTUP_REFRESH=true (default false)
# - DASHBOARD_EDIT_MIN_INTERVAL (seconds; default 1.8)
# - STAFF_USER_IDS (comma-separated allowlist; optional)

from __future__ import annotations

import os
import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import discord
from discord import app_commands

import psycopg
from psycopg.rows import dict_row


# -----------------------------
# Logging
# -----------------------------

LOG = logging.getLogger("VilyraBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VilyraBot: %(message)s")


# -----------------------------
# Constants / Defaults
# -----------------------------

DEFAULT_DASHBOARD_CHANNEL_ID = 1469879866655768738
DEFAULT_COMMAND_LOG_CHANNEL_ID = 1469879960729817098
COMMAND_LOG_CHANNEL_NAME = "Legacy-Commands-Log"  # primary by name

MAX_ABILITY_STARS = 5
MAX_INFL_STARS_TOTAL = 5

STAR_COST = 10
MINOR_UPGRADE_COST = 5

REP_MIN = -100
REP_MAX = 100

DASHBOARD_TEMPLATE_VERSION = 1
DASHBOARD_EDIT_MIN_INTERVAL = float(os.getenv("DASHBOARD_EDIT_MIN_INTERVAL", "1.8"))
PLAYER_POST_SOFT_LIMIT = 1900

SERVER_RANKS = [
    "Newcomer",
    "Apprentice",
    "Adventurer",
    "Sentinel",
    "Champion",
    "Legend",
    "Sovereign",
    "Warden",
    "Guardian",
]

RANK_CHOICES: List[app_commands.Choice[str]] = [app_commands.Choice(name=r, value=r) for r in SERVER_RANKS]

KINGDOMS: List[str] = ["Velarith", "Lyvik", "Baelon", "Sethrathiel", "Avalea"]
KINGDOM_CHOICES: List[app_commands.Choice[str]] = [app_commands.Choice(name=k, value=k) for k in KINGDOMS]

BORDER_LEN = 20
PLAYER_BORDER = "═" * BORDER_LEN
CHAR_SEPARATOR = "-" * BORDER_LEN
CHAR_HEADER_LEFT = "꧁•⊹٭ "
CHAR_HEADER_RIGHT = " ٭⊹•꧂"


# -----------------------------
# Helpers
# -----------------------------

def env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v

def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def db_timeout() -> float:
    return float(os.getenv("DB_TIMEOUT", "8.0"))

async def run_db(coro, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=db_timeout())
    except asyncio.TimeoutError as e:
        raise RuntimeError(f"Database operation timed out ({label}).") from e

async def safe_reply(interaction: discord.Interaction, content: str, *, embed: discord.Embed | None = None) -> None:
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=True, embed=embed)
        else:
            await interaction.response.send_message(content, ephemeral=True, embed=embed)
    except Exception:
        LOG.exception("Failed to send response/followup")

async def defer_ephemeral(interaction: discord.Interaction) -> None:
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True, thinking=True)
    except Exception:
        pass

async def send_error(interaction: discord.Interaction, error: Exception | str) -> None:
    msg = str(error)
    await safe_reply(interaction, f"❌ {msg}")

def normalize_channel_name(name: str) -> str:
    return (name or "").strip().lower()


# -----------------------------
# Rate limiter for dashboard edits
# -----------------------------

class SimpleRateLimiter:
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait_for = (self._last + self.min_interval) - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last = asyncio.get_running_loop().time()


# -----------------------------
# UI renderers
# -----------------------------

def render_ability_star_bar(n: int) -> str:
    n = clamp(int(n), 0, MAX_ABILITY_STARS)
    return "★" * n + "☆" * (MAX_ABILITY_STARS - n)

def render_influence_star_bar(neg: int, pos: int) -> str:
    neg = clamp(int(neg), 0, MAX_INFL_STARS_TOTAL)
    pos = clamp(int(pos), 0, MAX_INFL_STARS_TOTAL)
    neg_slots = ["☆"] * MAX_INFL_STARS_TOTAL
    for i in range(neg):
        neg_slots[MAX_INFL_STARS_TOTAL - 1 - i] = "★"
    pos_slots = ["☆"] * MAX_INFL_STARS_TOTAL
    for i in range(pos):
        pos_slots[i] = "★"
    return "- " + "".join(neg_slots) + " | " + "".join(pos_slots) + " +"

def render_reputation_block(net_lifetime: int) -> str:
    net = clamp(int(net_lifetime), REP_MIN, REP_MAX)
    left_len = 20
    right_len = 20
    total = left_len + right_len
    pos = int(round((net - REP_MIN) / (REP_MAX - REP_MIN) * total))
    pos = clamp(pos, 0, total)
    bar = ["-"] * (total + 1)
    center_idx = left_len
    bar[center_idx] = "┃"
    bar[pos] = "▲"
    bar_line = "[" + "".join(bar) + "]"
    left_text = "MALEVOLENT ←"
    right_text = "→ BENEVOLENT"
    spaces = max(1, len(bar_line) - len(left_text) - len(right_text))
    explainer = left_text + (" " * spaces) + right_text
    return explainer + "\n" + bar_line


# -----------------------------
# Database layer
# -----------------------------

class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg.AsyncConnection] = None

        self.characters_cols: set[str] = set()
        self.abilities_cols: set[str] = set()
        self.abilities_level_col: str = "upgrade_level"
        self.abilities_char_col: str = "character_name"

    async def connect(self) -> None:
        LOG.info("Connecting to PostgreSQL...")
        self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True, row_factory=dict_row)
        LOG.info("PostgreSQL async connection established (autocommit=True)")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    def _require_conn(self) -> psycopg.AsyncConnection:
        if not self._conn:
            raise RuntimeError("Database not connected")
        return self._conn

    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        conn = self._require_conn()
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return int(cur.rowcount or 0)

    async def _fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        conn = self._require_conn()
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()

    async def _fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        conn = self._require_conn()
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            return list(rows or [])

    async def _load_table_columns(self, table: str) -> set[str]:
        rows = await self._fetchall(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s;
            """,
            (table,),
        )
        return {str(r["column_name"]) for r in rows if r and r.get("column_name")}

    async def detect_schema(self) -> None:
        self.characters_cols = await self._load_table_columns("characters")
        self.abilities_cols = await self._load_table_columns("abilities")

        LOG.info("Detected characters columns: %s", ", ".join(sorted(self.characters_cols)) if self.characters_cols else "(none)")
        LOG.info("Detected abilities columns: %s", ", ".join(sorted(self.abilities_cols)) if self.abilities_cols else "(none)")

        if "upgrade_level" in self.abilities_cols:
            self.abilities_level_col = "upgrade_level"
        elif "level" in self.abilities_cols:
            self.abilities_level_col = "level"
        else:
            self.abilities_level_col = "upgrade_level"

        if "character_name" in self.abilities_cols:
            self.abilities_char_col = "character_name"
        elif "name" in self.abilities_cols:
            self.abilities_char_col = "name"
        else:
            self.abilities_char_col = "character_name"

        LOG.info("Schema choices: abilities.%s as level, abilities.%s as character key",
                 self.abilities_level_col, self.abilities_char_col)

    async def init_schema(self) -> None:
        # ----- characters -----
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS archived BOOLEAN NOT NULL DEFAULT FALSE;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS legacy_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS legacy_minus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS lifetime_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS lifetime_minus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ability_stars INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS influence_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS influence_minus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS kingdom TEXT;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        # New: stable numeric id per character (additive; no renames)
        await self._execute("CREATE SEQUENCE IF NOT EXISTS vilyra_character_id_seq;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS character_id BIGINT;")
        await self._execute("ALTER TABLE characters ALTER COLUMN character_id SET DEFAULT nextval('vilyra_character_id_seq');")
        await self._execute("UPDATE characters SET character_id=nextval('vilyra_character_id_seq') WHERE character_id IS NULL;")

        # Indexes (best-effort)
        try:
            await self._execute("CREATE UNIQUE INDEX IF NOT EXISTS characters_unique_name ON characters (guild_id, user_id, name);")
        except Exception:
            LOG.exception("Could not create characters_unique_name; continuing")
        try:
            await self._execute("CREATE UNIQUE INDEX IF NOT EXISTS characters_unique_id ON characters (guild_id, character_id);")
        except Exception:
            LOG.exception("Could not create characters_unique_id; continuing")
        try:
            await self._execute("CREATE INDEX IF NOT EXISTS characters_name_lookup ON characters (guild_id, lower(name));")
        except Exception:
            LOG.exception("Could not create characters_name_lookup; continuing")

        # ----- abilities -----
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                ability_name   TEXT NOT NULL,
                upgrades       INT  NOT NULL DEFAULT 0,
                upgrade_level  INT  NOT NULL DEFAULT 0,
                level          INT  NULL,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS character_name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS ability_name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrades INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrade_level INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS level INT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        # New: numeric character_id link (additive)
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS character_id BIGINT;")
        await self._execute(
            """
            UPDATE abilities a
               SET character_id = c.character_id
              FROM characters c
             WHERE a.character_id IS NULL
               AND a.guild_id = c.guild_id
               AND a.user_id = c.user_id
               AND (
                    (a.character_name IS NOT NULL AND a.character_name = c.name)
                    OR
                    (a.name IS NOT NULL AND a.name = c.name)
               );
            """
        )

        try:
            await self._execute("CREATE INDEX IF NOT EXISTS abilities_lookup ON abilities (guild_id, user_id, character_id, ability_name);")
        except Exception:
            LOG.exception("Could not create abilities_lookup; continuing")

        # ----- dashboard tracking -----
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_messages (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                channel_id    BIGINT NOT NULL,
                message_ids   TEXT,
                content_hash  TEXT,
                template_version INT NOT NULL DEFAULT 0,
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id)
            );
            """
        )
        await self._execute("ALTER TABLE dashboard_messages ADD COLUMN IF NOT EXISTS content_hash TEXT;")
        await self._execute("ALTER TABLE dashboard_messages ADD COLUMN IF NOT EXISTS template_version INT NOT NULL DEFAULT 0;")

        # ----- players -----
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS players (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                server_rank   TEXT   NOT NULL DEFAULT 'Newcomer',
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id)
            );
            """
        )

        await self.detect_schema()
        LOG.info("Database schema initialized / updated")

    # -------- Characters --------

    async def create_character(self, guild_id: int, user_id: int, name: str, kingdom: Optional[str]) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("Character name cannot be empty.")
        k = (kingdom or "").strip() or None

        row = await self._fetchone(
            """
            INSERT INTO characters (guild_id, user_id, name, kingdom, archived, updated_at)
            VALUES (%s, %s, %s, %s, FALSE, NOW())
            ON CONFLICT (guild_id, user_id, name)
            DO UPDATE SET archived=FALSE,
                          kingdom=COALESCE(EXCLUDED.kingdom, characters.kingdom),
                          updated_at=NOW()
            RETURNING character_id;
            """,
            (guild_id, user_id, name, k),
        )
        if not row or row.get("character_id") is None:
            row2 = await self._fetchone(
                "SELECT character_id FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s LIMIT 1;",
                (guild_id, user_id, name),
            )
            if not row2 or row2.get("character_id") is None:
                raise RuntimeError("Failed to create character.")
            cid = int(row2["character_id"])
        else:
            cid = int(row["character_id"])

        await self._execute("INSERT INTO players (guild_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (guild_id, user_id))
        return cid

    async def get_character_by_id(self, guild_id: int, character_id: int, include_archived: bool = True) -> Optional[Dict[str, Any]]:
        where = "guild_id=%s AND character_id=%s"
        params: List[Any] = [guild_id, int(character_id)]
        if not include_archived:
            where += " AND COALESCE(archived, FALSE)=FALSE"
        return await self._fetchone(f"SELECT * FROM characters WHERE {where} LIMIT 1;", tuple(params))

    async def get_character_by_name(self, guild_id: int, name: str, include_archived: bool = True) -> Optional[Dict[str, Any]]:
        name = (name or "").strip()
        if not name:
            return None
        where = "guild_id=%s AND lower(name)=lower(%s)"
        params: List[Any] = [guild_id, name]
        if not include_archived:
            where += " AND COALESCE(archived, FALSE)=FALSE"
        return await self._fetchone(f"SELECT * FROM characters WHERE {where} LIMIT 1;", tuple(params))

    async def delete_character(self, guild_id: int, character_id: int) -> bool:
        await self._execute("DELETE FROM abilities WHERE guild_id=%s AND character_id=%s;", (guild_id, int(character_id)))
        rowcount = await self._execute("DELETE FROM characters WHERE guild_id=%s AND character_id=%s;", (guild_id, int(character_id)))
        return rowcount > 0

    async def rename_character(self, guild_id: int, character_id: int, new_name: str) -> bool:
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("New name cannot be empty.")

        collision = await self._fetchone(
            "SELECT 1 FROM characters WHERE guild_id=%s AND lower(name)=lower(%s) AND character_id<>%s LIMIT 1;",
            (guild_id, new_name, int(character_id)),
        )
        if collision:
            raise ValueError("A character with that name already exists in this guild.")

        row = await self.get_character_by_id(guild_id, character_id, include_archived=True)
        if not row:
            return False
        old_name = str(row.get("name") or "")

        await self._execute(
            "UPDATE characters SET name=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
            (new_name, guild_id, int(character_id)),
        )

        char_col = self.abilities_char_col
        try:
            await self._execute(
                f"UPDATE abilities SET {char_col}=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                (new_name, guild_id, int(character_id)),
            )
        except Exception:
            LOG.exception("Rename cascade by character_id failed; trying fallback by old name")
            await self._execute(
                f"UPDATE abilities SET {char_col}=%s, updated_at=NOW() WHERE guild_id=%s AND {char_col}=%s;",
                (new_name, guild_id, old_name),
            )
        return True

    async def get_character_state(self, guild_id: int, character_id: int) -> Dict[str, Any]:
        row = await self._fetchone(
            """
            SELECT name, user_id, character_id,
                   legacy_plus, legacy_minus, lifetime_plus, lifetime_minus,
                   influence_plus, influence_minus, ability_stars, kingdom,
                   COALESCE(archived, FALSE) AS archived
            FROM characters
            WHERE guild_id=%s AND character_id=%s
            LIMIT 1;
            """,
            (guild_id, int(character_id)),
        )
        if not row:
            raise ValueError("Character not found.")
        return {
            "name": str(row.get("name") or ""),
            "user_id": int(row.get("user_id") or 0),
            "character_id": int(row.get("character_id") or 0),
            "legacy_plus": safe_int(row.get("legacy_plus"), 0),
            "legacy_minus": safe_int(row.get("legacy_minus"), 0),
            "lifetime_plus": safe_int(row.get("lifetime_plus"), 0),
            "lifetime_minus": safe_int(row.get("lifetime_minus"), 0),
            "influence_plus": safe_int(row.get("influence_plus"), 0),
            "influence_minus": safe_int(row.get("influence_minus"), 0),
            "ability_stars": safe_int(row.get("ability_stars"), 0),
            "kingdom": (row.get("kingdom") or ""),
            "archived": bool(row.get("archived") or False),
        }

    async def list_characters_for_user(self, guild_id: int, user_id: int, include_archived: bool = False) -> List[Dict[str, Any]]:
        where = ["guild_id=%s", "user_id=%s"]
        params: List[Any] = [guild_id, user_id]
        if not include_archived:
            where.append("COALESCE(archived, FALSE)=FALSE")
        sql = f"""
            SELECT character_id, user_id, name, COALESCE(archived, FALSE) AS archived
            FROM characters
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(archived, FALSE) ASC, name ASC, character_id ASC;
        """
        return await self._fetchall(sql, tuple(params))

    async def list_all_characters_for_guild(self, guild_id: int, include_archived: bool, name_filter: str, limit: int = 25) -> List[Dict[str, Any]]:
        name_filter = (name_filter or "").strip()
        lim = max(1, min(int(limit or 25), 100))
        where = ["guild_id=%s"]
        params: List[Any] = [guild_id]
        if not include_archived:
            where.append("COALESCE(archived, FALSE)=FALSE")
        if name_filter:
            where.append("name ILIKE %s")
            params.append(f"%{name_filter}%")
        sql = f"""
            SELECT character_id, user_id, name, COALESCE(archived, FALSE) AS archived
            FROM characters
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(archived, FALSE) ASC, name ASC, character_id ASC
            LIMIT {lim};
        """
        return await self._fetchall(sql, tuple(params))

    async def list_player_ids(self, guild_id: int) -> List[int]:
        rows = await self._fetchall("SELECT user_id FROM players WHERE guild_id=%s ORDER BY user_id ASC;", (guild_id,))
        ids = [int(r["user_id"]) for r in rows if r and r.get("user_id") is not None]
        if ids:
            return ids
        rows2 = await self._fetchall("SELECT DISTINCT user_id FROM characters WHERE guild_id=%s ORDER BY user_id ASC;", (guild_id,))
        return [int(r["user_id"]) for r in rows2 if r and r.get("user_id") is not None]

    # -------- Players --------

    async def get_player_rank(self, guild_id: int, user_id: int) -> str:
        row = await self._fetchone("SELECT server_rank FROM players WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))
        return str(row["server_rank"]) if row and row.get("server_rank") else "Newcomer"


async def set_player_rank(self, guild_id: int, user_id: int, rank: str) -> None:
    """Set a player's server rank (validated)."""
    rank = (rank or "").strip()
    if rank not in SERVER_RANKS:
        raise ValueError("Invalid rank.")
    await self._execute(
        """
        INSERT INTO players (guild_id, user_id, server_rank)
        VALUES (%s, %s, %s)
        ON CONFLICT (guild_id, user_id)
        DO UPDATE SET server_rank=EXCLUDED.server_rank, updated_at=NOW();
        """,
        (guild_id, user_id, rank),
    )
# -------- Legacy points --------

    async def award_legacy(self, guild_id: int, character_id: int, pos: int, neg: int) -> None:
        pos = max(0, int(pos))
        neg = max(0, int(neg))
        await self._execute(
            """
            UPDATE characters
               SET legacy_plus = legacy_plus + %s,
                   legacy_minus = legacy_minus + %s,
                   lifetime_plus = lifetime_plus + %s,
                   lifetime_minus = lifetime_minus + %s,
                   updated_at = NOW()
             WHERE guild_id=%s AND character_id=%s;
            """,
            (pos, neg, pos, neg, guild_id, int(character_id)),
        )

    async def set_available_legacy(self, guild_id: int, character_id: int, pos: int, neg: int) -> None:
        pos = max(0, int(pos))
        neg = max(0, int(neg))
        await self._execute(
            """
            UPDATE characters
               SET legacy_plus=%s,
                   legacy_minus=%s,
                   updated_at=NOW()
             WHERE guild_id=%s AND character_id=%s;
            """,
            (pos, neg, guild_id, int(character_id)),
        )

    async def set_lifetime_legacy(self, guild_id: int, character_id: int, pos: int, neg: int) -> None:
        pos = max(0, int(pos))
        neg = max(0, int(neg))
        await self._execute(
            """
            UPDATE characters
               SET lifetime_plus=%s,
                   lifetime_minus=%s,
                   updated_at=NOW()
             WHERE guild_id=%s AND character_id=%s;
            """,
            (pos, neg, guild_id, int(character_id)),
        )

    async def convert_stars(self, guild_id: int, character_id: int, star_type: str, spend_plus: int, spend_minus: int) -> None:
        star_type = (star_type or "").strip().lower()
        spend_plus = max(0, int(spend_plus))
        spend_minus = max(0, int(spend_minus))
        if spend_plus + spend_minus != STAR_COST:
            raise ValueError(f"Star conversion cost is exactly {STAR_COST} total points (+ and - may split).")

        st = await self.get_character_state(guild_id, character_id)
        if st["legacy_plus"] < spend_plus:
            raise ValueError(f"Not enough available positive points (need {spend_plus}, have {st['legacy_plus']}).")
        if st["legacy_minus"] < spend_minus:
            raise ValueError(f"Not enough available negative points (need {spend_minus}, have {st['legacy_minus']}).")

        await self._execute(
            """
            UPDATE characters
               SET legacy_plus=legacy_plus-%s,
                   legacy_minus=legacy_minus-%s,
                   updated_at=NOW()
             WHERE guild_id=%s AND character_id=%s;
            """,
            (spend_plus, spend_minus, guild_id, int(character_id)),
        )

        if star_type == "ability":
            if st["ability_stars"] >= MAX_ABILITY_STARS:
                raise ValueError("Ability stars already at max (5).")
            await self._execute(
                "UPDATE characters SET ability_stars=ability_stars+1, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                (guild_id, int(character_id)),
            )
        elif star_type == "influence_positive":
            if st["influence_plus"] + st["influence_minus"] >= MAX_INFL_STARS_TOTAL:
                raise ValueError("Total influence stars (pos+neg) cannot exceed 5.")
            await self._execute(
                "UPDATE characters SET influence_plus=influence_plus+1, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                (guild_id, int(character_id)),
            )
        elif star_type == "influence_negative":
            if st["influence_plus"] + st["influence_minus"] >= MAX_INFL_STARS_TOTAL:
                raise ValueError("Total influence stars (pos+neg) cannot exceed 5.")
            await self._execute(
                "UPDATE characters SET influence_minus=influence_minus+1, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                (guild_id, int(character_id)),
            )
        else:
            raise ValueError("star_type must be: ability, influence_positive, influence_negative")

    # -------- Abilities --------

    def _ability_level_expr(self) -> str:
        if self.abilities_level_col == "upgrade_level":
            return "COALESCE(upgrade_level, 0) AS upgrade_level"
        return "COALESCE(level, 0) AS upgrade_level"

    async def list_abilities_for_character(self, guild_id: int, character_id: int) -> List[Tuple[str, int]]:
        rows = await self._fetchall(
            f"""
            SELECT ability_name, {self._ability_level_expr()}
            FROM abilities
            WHERE guild_id=%s AND character_id=%s
            ORDER BY created_at ASC, ability_name ASC;
            """,
            (guild_id, int(character_id)),
        )
        out: List[Tuple[str, int]] = []
        for r in rows:
            nm = str(r.get("ability_name") or "").strip()
            if not nm:
                continue
            out.append((nm, safe_int(r.get("upgrade_level"), 0)))
        return out

    async def add_ability(self, guild_id: int, character_id: int, ability_name: str) -> None:
        ability_name = (ability_name or "").strip()
        if not ability_name:
            raise ValueError("Ability name cannot be empty.")

        st = await self.get_character_state(guild_id, character_id)
        cap = 2 + clamp(st["ability_stars"], 0, MAX_ABILITY_STARS)
        current = await self.list_abilities_for_character(guild_id, character_id)
        if len(current) >= cap:
            raise ValueError(f"Ability capacity reached ({len(current)}/{cap}). Earn more Ability Stars to add abilities.")

        for nm, _ in current:
            if nm.lower() == ability_name.lower():
                raise ValueError("That ability already exists for this character.")

        await self._execute(
            """
            INSERT INTO abilities (guild_id, user_id, character_id, character_name, ability_name, upgrade_level, level, upgrades, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, 0, 0, 0, NOW(), NOW());
            """,
            (guild_id, st["user_id"], int(character_id), st["name"], ability_name),
        )

    async def rename_ability(self, guild_id: int, character_id: int, old_ability: str, new_ability: str) -> bool:
        old_ability = (old_ability or "").strip()
        new_ability = (new_ability or "").strip()
        if not old_ability or not new_ability:
            raise ValueError("Ability names cannot be empty.")

        current = await self.list_abilities_for_character(guild_id, character_id)
        for nm, _ in current:
            if nm.lower() == new_ability.lower():
                raise ValueError("That new ability name already exists for this character.")

        rowcount = await self._execute(
            """
            UPDATE abilities
               SET ability_name=%s, updated_at=NOW()
             WHERE guild_id=%s AND character_id=%s AND ability_name=%s;
            """,
            (new_ability, guild_id, int(character_id), old_ability),
        )
        return rowcount > 0

    async def upgrade_ability(self, guild_id: int, character_id: int, ability_name: str, pay_positive: int, pay_negative: int) -> Tuple[int, int]:
        ability_name = (ability_name or "").strip()
        pay_positive = max(0, int(pay_positive))
        pay_negative = max(0, int(pay_negative))
        total_cost = MINOR_UPGRADE_COST
        if pay_positive + pay_negative != total_cost:
            raise ValueError(f"Each upgrade costs exactly {total_cost} total points (+ and - may split).")

        st = await self.get_character_state(guild_id, character_id)
        if st["legacy_plus"] < pay_positive:
            raise ValueError(f"Not enough available positive points (need {pay_positive}, have {st['legacy_plus']}).")
        if st["legacy_minus"] < pay_negative:
            raise ValueError(f"Not enough available negative points (need {pay_negative}, have {st['legacy_minus']}).")

        level_col = self.abilities_level_col
        row = await self._fetchone(
            f"""
            SELECT COALESCE({level_col}, 0) AS cur_level
            FROM abilities
            WHERE guild_id=%s AND character_id=%s AND ability_name=%s
            ORDER BY created_at ASC
            LIMIT 1;
            """,
            (guild_id, int(character_id), ability_name),
        )
        if not row:
            raise ValueError("Ability not found. Add it first with /ability_add.")
        cur_level = safe_int(row.get("cur_level"), 0)
        max_level = 5
        if cur_level >= max_level:
            raise ValueError("Upgrade limit reached (5/5).")

        await self._execute(
            "UPDATE characters SET legacy_plus=legacy_plus-%s, legacy_minus=legacy_minus-%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
            (pay_positive, pay_negative, guild_id, int(character_id)),
        )

        new_level = cur_level + 1
        sets: List[str] = [f"{level_col}=%s", "upgrades=COALESCE(upgrades,0)+1", "updated_at=NOW()"]
        params: List[Any] = [new_level]

        if level_col != "upgrade_level" and "upgrade_level" in self.abilities_cols:
            sets.append("upgrade_level=%s")
            params.append(new_level)
        if level_col != "level" and "level" in self.abilities_cols:
            sets.append("level=%s")
            params.append(new_level)

        params.extend([guild_id, int(character_id), ability_name])
        await self._execute(
            f"UPDATE abilities SET {', '.join(sets)} WHERE guild_id=%s AND character_id=%s AND ability_name=%s;",
            tuple(params),
        )
        return new_level, max_level

    # -------- Dashboard tracking --------

    def _parse_ids(self, s: Optional[str]) -> List[int]:
        if not s:
            return []
        out: List[int] = []
        for part in str(s).split(","):
            part = part.strip()
            if part.isdigit():
                out.append(int(part))
        return out

    def _fmt_ids(self, ids: List[int]) -> str:
        return ",".join(str(i) for i in ids)

    async def get_dashboard_entry(self, guild_id: int, user_id: int) -> Tuple[List[int], Optional[str], Optional[Any], int]:
        row = await self._fetchone(
            "SELECT message_ids, content_hash, updated_at, COALESCE(template_version, 0) AS template_version FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )
        ids = self._parse_ids(row["message_ids"]) if row and row.get("message_ids") else []
        h = str(row["content_hash"]) if row and row.get("content_hash") else None
        ts = row["updated_at"] if row and row.get("updated_at") else None
        tv = int(row["template_version"]) if row and row.get("template_version") is not None else 0
        return ids, h, ts, tv

    async def set_dashboard_entry(self, guild_id: int, user_id: int, channel_id: int, msg_ids: List[int], h: Optional[str]) -> None:
        await self._execute(
            """
            INSERT INTO dashboard_messages (guild_id, user_id, channel_id, message_ids, content_hash, template_version, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (guild_id, user_id)
            DO UPDATE SET channel_id=EXCLUDED.channel_id,
                          message_ids=EXCLUDED.message_ids,
                          content_hash=EXCLUDED.content_hash,
                          template_version=EXCLUDED.template_version,
                          updated_at=NOW();
            """,
            (guild_id, user_id, channel_id, self._fmt_ids(msg_ids) if msg_ids else None, h, DASHBOARD_TEMPLATE_VERSION),
        )

    async def clear_dashboard_entry(self, guild_id: int, user_id: int) -> None:
        await self._execute("DELETE FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))


# -----------------------------
# Character dashboard rendering
# -----------------------------

@dataclass
class CharacterCard:
    name: str
    character_id: int
    user_id: int
    kingdom: str
    legacy_plus: int
    legacy_minus: int
    lifetime_plus: int
    lifetime_minus: int
    ability_stars: int
    infl_plus: int
    infl_minus: int
    abilities: List[Tuple[str, int]]

async def build_character_card(db: Database, guild_id: int, character_id: int) -> CharacterCard:
    st = await db.get_character_state(guild_id, character_id)
    abilities = await db.list_abilities_for_character(guild_id, character_id)
    return CharacterCard(
        name=st["name"],
        character_id=st["character_id"],
        user_id=st["user_id"],
        kingdom=st.get("kingdom", ""),
        legacy_plus=st["legacy_plus"],
        legacy_minus=st["legacy_minus"],
        lifetime_plus=st["lifetime_plus"],
        lifetime_minus=st["lifetime_minus"],
        ability_stars=st["ability_stars"],
        infl_plus=st["influence_plus"],
        infl_minus=st["influence_minus"],
        abilities=abilities,
    )

def render_character_block(card: CharacterCard) -> str:
    net_lifetime = card.lifetime_plus - card.lifetime_minus
    lines: List[str] = []
    lines.append(f"{CHAR_HEADER_LEFT}**{card.name}** {CHAR_HEADER_RIGHT}  `#{card.character_id}`")
    k = (card.kingdom or "").strip()
    lines.append(f"Kingdom: {k}" if k else "Kingdom:")
    lines.append("")
    lines.append(f"Legacy Points: +{card.legacy_plus}/-{card.legacy_minus} | Lifetime: +{card.lifetime_plus}/-{card.lifetime_minus}")
    lines.append("Ability Stars: " + render_ability_star_bar(card.ability_stars))
    lines.append("Influence Stars: " + render_influence_star_bar(card.infl_minus, card.infl_plus))
    lines.append(render_reputation_block(net_lifetime))
    if card.abilities:
        parts = [f"{nm} ({lvl})" for nm, lvl in card.abilities]
        lines.append("Abilities: " + " | ".join(parts))
    else:
        lines.append("Abilities: _none set_")
    return "\n".join(lines).strip()

async def render_player_post(db: Database, guild: discord.Guild, user_id: int) -> str:
    member = guild.get_member(user_id)
    display = member.display_name if member else f"User {user_id}"
    rank = await db.get_player_rank(guild.id, user_id)

    chars = await db.list_characters_for_user(guild.id, user_id, include_archived=False)
    if not chars:
        return ""

    lines: List[str] = []
    lines.append(PLAYER_BORDER)
    lines.append(f"__***{display}***__")
    lines.append(f"__***Server Rank: {rank}***__")
    lines.append("")

    for i, r in enumerate(chars):
        cid = int(r["character_id"])
        card = await build_character_card(db, guild.id, cid)
        lines.append(render_character_block(card))
        if i != len(chars) - 1:
            lines.append("")
            lines.append(CHAR_SEPARATOR)
            lines.append("")

    lines.append(PLAYER_BORDER)
    content = "\n".join(lines).rstrip()
    if len(content) > PLAYER_POST_SOFT_LIMIT:
        truncated = content[:PLAYER_POST_SOFT_LIMIT - 60]
        cut = truncated.rfind("\n")
        if cut > 0:
            truncated = truncated[:cut]
        content = truncated.rstrip() + "\n\n…(truncated: too many characters to fit in one post)"
    return content

async def get_dashboard_channel(guild: discord.Guild) -> Optional[discord.TextChannel]:
    ch_id = safe_int(os.getenv("DASHBOARD_CHANNEL_ID"), DEFAULT_DASHBOARD_CHANNEL_ID)
    ch = guild.get_channel(ch_id)
    if ch is None:
        try:
            ch = await guild.fetch_channel(ch_id)
        except Exception:
            ch = None
    return ch if isinstance(ch, discord.TextChannel) else None


# -----------------------------
# Command logging channel
# -----------------------------

async def get_command_log_channel(guild: discord.Guild) -> Optional[discord.TextChannel]:
    target = normalize_channel_name(COMMAND_LOG_CHANNEL_NAME)
    for ch in guild.text_channels:
        if normalize_channel_name(ch.name) == target:
            return ch

    ch_id = safe_int(os.getenv("COMMAND_LOG_CHANNEL_ID"), DEFAULT_COMMAND_LOG_CHANNEL_ID)
    ch = guild.get_channel(ch_id)
    if ch is None:
        try:
            ch = await guild.fetch_channel(ch_id)
        except Exception:
            ch = None
    return ch if isinstance(ch, discord.TextChannel) else None

async def log_command(interaction: discord.Interaction, text: str) -> None:
    guild = interaction.guild
    if not guild:
        return
    try:
        ch = await get_command_log_channel(guild)
        if not ch:
            return
        await ch.send(text, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        LOG.exception("Failed to write to command log channel")


# -----------------------------
# Autocomplete + resolvers
# -----------------------------

async def autocomplete_character(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    try:
        guild = interaction.guild
        if guild is None:
            return []
        q = (current or "").strip()
        rows = await interaction.client.db.list_all_characters_for_guild(
            guild_id=guild.id,
            include_archived=True,
            name_filter=q,
            limit=25,
        )
        out: List[app_commands.Choice[str]] = []
        for r in rows:
            cid = int(r.get("character_id") or 0)
            nm = str(r.get("name") or "").strip()
            if cid <= 0 or not nm:
                continue
            label = f"{nm}  (#{cid})"
            if len(label) > 100:
                label = label[:97] + "..."
            out.append(app_commands.Choice(name=label, value=str(cid)))
        return out
    except Exception:
        LOG.exception("Character autocomplete failed")
        return []

async def resolve_character(interaction: discord.Interaction, token: str) -> Tuple[int, Dict[str, Any]]:
    guild = interaction.guild
    if guild is None:
        raise ValueError("This command must be used in a server.")
    t = (token or "").strip()
    if not t:
        raise ValueError("Character is required.")
    if t.isdigit():
        cid = int(t)
        row = await interaction.client.db.get_character_by_id(guild.id, cid, include_archived=True)
        if not row:
            raise ValueError("Character not found.")
        return cid, row
    row2 = await interaction.client.db.get_character_by_name(guild.id, t, include_archived=True)
    if not row2:
        raise ValueError("Character not found.")
    return int(row2["character_id"]), row2

async def autocomplete_ability(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    try:
        guild = interaction.guild
        if guild is None:
            return []
        q = (current or "").strip().lower()
        char_token = getattr(interaction.namespace, "character", None) or getattr(interaction.namespace, "character_name", None)
        if not char_token:
            return []
        cid, _ = await resolve_character(interaction, str(char_token))
        abilities = await interaction.client.db.list_abilities_for_character(guild.id, cid)
        out: List[app_commands.Choice[str]] = []
        for nm, lvl in abilities:
            if q and q not in nm.lower():
                continue
            label = f"{nm} ({lvl})"
            if len(label) > 100:
                label = label[:97] + "..."
            out.append(app_commands.Choice(name=label, value=nm))
            if len(out) >= 25:
                break
        return out
    except Exception:
        LOG.exception("Ability autocomplete failed")
        return []


# -----------------------------
# Command guards
# -----------------------------

def in_guild_only(func=None):
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False
        return True
    decorator = app_commands.check(predicate)
    if callable(func):
        return decorator(func)
    def wrapper(f):
        return decorator(f)
    return wrapper

def staff_only(func=None):
    staff_user_ids: set[int] = set()
    raw = os.getenv("STAFF_USER_IDS", "") or ""
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            staff_user_ids.add(int(part))

    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False

        member = interaction.user
        if not isinstance(member, discord.Member):
            member = interaction.guild.get_member(interaction.user.id)  # type: ignore

        if interaction.user.id in staff_user_ids:
            return True
        if interaction.guild.owner_id == interaction.user.id:
            return True
        if isinstance(member, discord.Member):
            perms = member.guild_permissions
            if perms.administrator or perms.manage_guild:
                return True
        await safe_reply(interaction, "You don't have permission to use this command.")
        return False

    decorator = app_commands.check(predicate)
    if callable(func):
        return decorator(func)
    def wrapper(f):
        return decorator(f)
    return wrapper


# -----------------------------
# Dashboards
# -----------------------------

async def refresh_player_dashboard(client: "VilyraBotClient", guild: discord.Guild, user_id: int) -> str:
    db = client.db
    channel = await get_dashboard_channel(guild)
    if not channel:
        return "Dashboard channel not found."

    me = guild.me or (guild.get_member(client.user.id) if client.user else None)
    if me:
        perms = channel.permissions_for(me)
        if not (perms.view_channel and perms.send_messages):
            return f"Missing permissions in <#{channel.id}> (View Channel + Send Messages)."

    stored_ids, stored_hash, _, stored_tv = await db.get_dashboard_entry(guild.id, user_id)

    content = await render_player_post(db, guild, user_id)
    if not content:
        for mid in stored_ids:
            try:
                m = await channel.fetch_message(mid)
                await m.delete()
            except Exception:
                pass
        await db.clear_dashboard_entry(guild.id, user_id)
        return f"No characters for user_id={user_id}; dashboard cleared."

    new_hash = content_hash(content)
    if stored_hash == new_hash and stored_tv == DASHBOARD_TEMPLATE_VERSION:
        return "skipped"

    msg: Optional[discord.Message] = None
    if stored_ids:
        try:
            msg = await channel.fetch_message(stored_ids[0])
        except Exception:
            msg = None

    if msg is None:
        await client.dashboard_limiter.wait()
        msg = await channel.send(content)
        await db.set_dashboard_entry(guild.id, user_id, channel.id, [msg.id], new_hash)
        return "created"
    else:
        await client.dashboard_limiter.wait()
        await msg.edit(content=content)
        if len(stored_ids) > 1:
            for extra in stored_ids[1:]:
                try:
                    m2 = await channel.fetch_message(extra)
                    await m2.delete()
                except Exception:
                    pass
        await db.set_dashboard_entry(guild.id, user_id, channel.id, [msg.id], new_hash)
        return "updated"

async def refresh_all_dashboards(client: "VilyraBotClient", guild: discord.Guild) -> str:
    user_ids = await client.db.list_player_ids(guild.id)
    if not user_ids:
        return "No players with characters yet."

    ok = 0
    for uid in user_ids:
        try:
            await refresh_player_dashboard(client, guild, uid)
            ok += 1
            await asyncio.sleep(0.25)
        except Exception:
            LOG.exception("refresh_player_dashboard failed for user_id=%s", uid)
            await asyncio.sleep(0.25)
    return f"Refreshed dashboards for {ok} player(s)."


# -----------------------------
# Slash commands (names exactly as requested)
# -----------------------------

# -----------------------------
# Staff: Server Rank
# -----------------------------

@app_commands.command(name="set_server_rank", description="(Staff) Set a player's server rank.")
@in_guild_only()
@staff_only
@app_commands.describe(user="Player to update", rank="New server rank")
@app_commands.choices(rank=RANK_CHOICES)
async def set_server_rank(
    interaction: discord.Interaction,
    user: discord.Member,
    rank: app_commands.Choice[str],
) -> None:
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        new_rank = str(rank.value)
        await interaction.client.db.set_player_rank(interaction.guild.id, user.id, new_rank)
        await log_command(interaction, f"🏷️ {interaction.user.mention} set server rank for {user.mention} → **{new_rank}**")
        await refresh_all_dashboards(interaction.client, interaction.guild)
        await safe_reply(interaction, f"✅ Server rank for {user.mention} set to **{new_rank}**.")
    except Exception as e:
        LOG.exception("set_server_rank failed")
        await send_error(interaction, e)

@app_commands.command(name="staff_commands", description="(Staff) Show a list of staff commands.")
@in_guild_only()
@staff_only
async def staff_commands(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        lines = [
            "**Staff Commands**",
            "",
            "• **/character_create** — Create a new character for a user (with kingdom).",
            "• **/character_delete** — Delete a character (cannot be undone).",
            "• **/character_rename** — Rename a character (keeps character ID).",
            "• **/award_legacy** — Award legacy points (+/-) and lifetime totals.",
            "• **/convert_stars** — Spend 10 legacy points to add a star (ability/pos influence/neg influence).",
            "• **/ability_add** — Add an ability (capacity = 2 + ability stars).",
            "• **/ability_rename** — Rename an ability (keeps upgrades).",
            "• **/ability_upgrade** — Upgrade an ability (costs 5 legacy points; split +/−).",
            "• **/set_available_legacy** — Set available legacy points (does not change lifetime).",
            "• **/set_lifetime_legacy** — Set lifetime legacy points (does not change available).",
            "• **/refresh_dashboard** — Refresh all dashboard posts.",
            "",
            "Note: **/char_card** is public and not logged.",
        ]
        await safe_reply(interaction, "\n".join(lines))
        await log_command(interaction, f"📜 {interaction.user.mention} used /staff_commands")
    except Exception as e:
        LOG.exception("staff_commands failed")
        await send_error(interaction, e)


@app_commands.command(name="character_create", description="(Staff) Create a new character for a user.")
@in_guild_only()
@staff_only
@app_commands.describe(user="Player", character_name="Character name", kingdom="Home kingdom")
@app_commands.choices(kingdom=KINGDOM_CHOICES)
async def character_create(interaction: discord.Interaction, user: discord.Member, character_name: str, kingdom: app_commands.Choice[str]):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid = await run_db(interaction.client.db.create_character(interaction.guild.id, user.id, character_name, kingdom.value), "create_character")
        await log_command(interaction, f"🆕 {interaction.user.mention} created character **{character_name}** `#{cid}` for {user.mention} (kingdom={kingdom.value})")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, user.id), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Created **{character_name}** with id `#{cid}` for {user.mention}.")
    except Exception as e:
        LOG.exception("character_create failed")
        await send_error(interaction, e)


@app_commands.command(name="character_delete", description="(Staff) Delete a character (cannot be undone).")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)")
@app_commands.autocomplete(character=autocomplete_character)
async def character_delete(interaction: discord.Interaction, character: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, row = await resolve_character(interaction, character)
        user_id = int(row["user_id"])
        name = str(row["name"])
        ok = await run_db(interaction.client.db.delete_character(interaction.guild.id, cid), "delete_character")
        if not ok:
            raise RuntimeError("Character not found.")
        await log_command(interaction, f"🗑️ {interaction.user.mention} deleted **{name}** `#{cid}` (owner=<@{user_id}>)")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, user_id), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Deleted **{name}** `#{cid}`.")
    except Exception as e:
        LOG.exception("character_delete failed")
        await send_error(interaction, e)


@app_commands.command(name="character_rename", description="(Staff) Rename a character (keeps character ID).")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", new_name="New name")
@app_commands.autocomplete(character=autocomplete_character)
async def character_rename(interaction: discord.Interaction, character: str, new_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, row = await resolve_character(interaction, character)
        old = str(row["name"])
        user_id = int(row["user_id"])
        ok = await run_db(interaction.client.db.rename_character(interaction.guild.id, cid, new_name), "rename_character")
        if not ok:
            raise RuntimeError("Character not found.")
        await log_command(interaction, f"✏️ {interaction.user.mention} renamed **{old}** `#{cid}` → **{new_name.strip()}** (owner=<@{user_id}>)")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, user_id), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Renamed **{old}** → **{new_name.strip()}** (id stays `#{cid}`).")
    except Exception as e:
        LOG.exception("character_rename failed")
        await send_error(interaction, e)


@app_commands.command(name="award_legacy", description="(Staff) Award legacy points (+/-) and lifetime totals.")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", positive="Positive points to add", negative="Negative points to add")
@app_commands.autocomplete(character=autocomplete_character)
async def award_legacy(interaction: discord.Interaction, character: str, positive: int = 0, negative: int = 0):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, _ = await resolve_character(interaction, character)
        pos = max(0, int(positive))
        neg = max(0, int(negative))
        if pos == 0 and neg == 0:
            raise ValueError("Provide positive and/or negative points.")
        await run_db(interaction.client.db.award_legacy(interaction.guild.id, cid, pos, neg), "award_legacy")
        st = await run_db(interaction.client.db.get_character_state(interaction.guild.id, cid), "get_character_state")
        await log_command(interaction, f"🏅 {interaction.user.mention} awarded legacy to **{st['name']}** `#{cid}`: +{pos}/-{neg}")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, st["user_id"]), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Awarded **{st['name']}** `#{cid}`: +{pos}/-{neg}.")
    except Exception as e:
        LOG.exception("award_legacy failed")
        await send_error(interaction, e)


@app_commands.command(name="convert_stars", description="(Staff) Spend 10 legacy points (+/- split allowed) to add a star.")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", star_type="Star type", spend_plus="Spend + points", spend_minus="Spend - points")
@app_commands.choices(star_type=[
    app_commands.Choice(name="Ability Star", value="ability"),
    app_commands.Choice(name="Positive Influence Star", value="influence_positive"),
    app_commands.Choice(name="Negative Influence Star", value="influence_negative"),
])
@app_commands.autocomplete(character=autocomplete_character)
async def convert_stars(
    interaction: discord.Interaction,
    character: str,
    star_type: app_commands.Choice[str],
    spend_plus: int,
    spend_minus: int,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, _ = await resolve_character(interaction, character)
        await run_db(interaction.client.db.convert_stars(interaction.guild.id, cid, star_type.value, spend_plus, spend_minus), "convert_stars")
        st = await run_db(interaction.client.db.get_character_state(interaction.guild.id, cid), "get_character_state")
        await log_command(interaction, f"⭐ {interaction.user.mention} converted legacy to {star_type.name} for **{st['name']}** `#{cid}` (spent +{spend_plus}/-{spend_minus})")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, st["user_id"]), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Added **{star_type.name}** to **{st['name']}** `#{cid}` (spent +{spend_plus}/-{spend_minus}).")
    except Exception as e:
        LOG.exception("convert_stars failed")
        await send_error(interaction, e)


@app_commands.command(name="ability_add", description="(Staff) Add an ability (capacity = 2 + ability stars).")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", ability_name="Ability name")
@app_commands.autocomplete(character=autocomplete_character)
async def ability_add(interaction: discord.Interaction, character: str, ability_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, _ = await resolve_character(interaction, character)
        await run_db(interaction.client.db.add_ability(interaction.guild.id, cid, ability_name), "add_ability")
        st = await run_db(interaction.client.db.get_character_state(interaction.guild.id, cid), "get_character_state")
        await log_command(interaction, f"➕ {interaction.user.mention} added ability **{ability_name.strip()}** to **{st['name']}** `#{cid}`")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, st["user_id"]), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Added ability **{ability_name.strip()}** to **{st['name']}** `#{cid}`.")
    except Exception as e:
        LOG.exception("ability_add failed")
        await send_error(interaction, e)


@app_commands.command(name="ability_rename", description="(Staff) Rename an ability (keeps upgrades).")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", ability="Ability (autocomplete)", new_ability_name="New ability name")
@app_commands.autocomplete(character=autocomplete_character, ability=autocomplete_ability)
async def ability_rename(interaction: discord.Interaction, character: str, ability: str, new_ability_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, _ = await resolve_character(interaction, character)
        ok = await run_db(interaction.client.db.rename_ability(interaction.guild.id, cid, ability, new_ability_name), "rename_ability")
        if not ok:
            raise RuntimeError("Ability not found.")
        st = await run_db(interaction.client.db.get_character_state(interaction.guild.id, cid), "get_character_state")
        await log_command(interaction, f"✏️ {interaction.user.mention} renamed ability on **{st['name']}** `#{cid}`: **{ability}** → **{new_ability_name.strip()}**")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, st["user_id"]), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Renamed ability **{ability}** → **{new_ability_name.strip()}** for **{st['name']}** `#{cid}`.")
    except Exception as e:
        LOG.exception("ability_rename failed")
        await send_error(interaction, e)


@app_commands.command(name="ability_upgrade", description="(Staff) Upgrade an ability (cost 5 legacy points; split +/−).")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", ability="Ability (autocomplete)", positive_cost="Spend + points", negative_cost="Spend - points")
@app_commands.autocomplete(character=autocomplete_character, ability=autocomplete_ability)
async def ability_upgrade(interaction: discord.Interaction, character: str, ability: str, positive_cost: int, negative_cost: int):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, _ = await resolve_character(interaction, character)
        new_level, max_level = await run_db(interaction.client.db.upgrade_ability(interaction.guild.id, cid, ability, positive_cost, negative_cost), "upgrade_ability")
        st = await run_db(interaction.client.db.get_character_state(interaction.guild.id, cid), "get_character_state")
        await log_command(interaction, f"⬆️ {interaction.user.mention} upgraded **{ability}** for **{st['name']}** `#{cid}` to {new_level}/{max_level} (spent +{positive_cost}/-{negative_cost})")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, st["user_id"]), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Upgraded **{ability}** for **{st['name']}** `#{cid}` to {new_level}/{max_level}.")
    except Exception as e:
        LOG.exception("ability_upgrade failed")
        await send_error(interaction, e)


@app_commands.command(name="set_available_legacy", description="(Staff) Set available legacy points (does not change lifetime).")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", positive="Set available + points", negative="Set available - points")
@app_commands.autocomplete(character=autocomplete_character)
async def set_available_legacy(interaction: discord.Interaction, character: str, positive: int, negative: int):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, _ = await resolve_character(interaction, character)
        await run_db(interaction.client.db.set_available_legacy(interaction.guild.id, cid, positive, negative), "set_available_legacy")
        st = await run_db(interaction.client.db.get_character_state(interaction.guild.id, cid), "get_character_state")
        await log_command(interaction, f"🧮 {interaction.user.mention} set AVAILABLE legacy for **{st['name']}** `#{cid}` to +{positive}/-{negative}")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, st["user_id"]), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Set available legacy for **{st['name']}** `#{cid}` to +{positive}/-{negative}.")
    except Exception as e:
        LOG.exception("set_available_legacy failed")
        await send_error(interaction, e)


@app_commands.command(name="set_lifetime_legacy", description="(Staff) Set lifetime legacy totals (does not change available).")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", positive="Set lifetime + points", negative="Set lifetime - points")
@app_commands.autocomplete(character=autocomplete_character)
async def set_lifetime_legacy(interaction: discord.Interaction, character: str, positive: int, negative: int):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, _ = await resolve_character(interaction, character)
        await run_db(interaction.client.db.set_lifetime_legacy(interaction.guild.id, cid, positive, negative), "set_lifetime_legacy")
        st = await run_db(interaction.client.db.get_character_state(interaction.guild.id, cid), "get_character_state")
        await log_command(interaction, f"📈 {interaction.user.mention} set LIFETIME legacy for **{st['name']}** `#{cid}` to +{positive}/-{negative}")
        await run_db(refresh_player_dashboard(interaction.client, interaction.guild, st["user_id"]), "refresh_player_dashboard")
        await safe_reply(interaction, f"✅ Set lifetime legacy for **{st['name']}** `#{cid}` to +{positive}/-{negative}.")
    except Exception as e:
        LOG.exception("set_lifetime_legacy failed")
        await send_error(interaction, e)


@app_commands.command(name="refresh_dashboard", description="(Staff) Refresh all dashboard posts.")
@in_guild_only()
@staff_only
async def refresh_dashboard(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        status = await refresh_all_dashboards(interaction.client, interaction.guild)
        await log_command(interaction, f"🔄 {interaction.user.mention} ran /refresh_dashboard ({status})")
        await safe_reply(interaction, status)
    except Exception as e:
        LOG.exception("refresh_dashboard failed")
        await send_error(interaction, e)


@app_commands.command(name="char_card", description="Show a character card (public).")
@in_guild_only()
@app_commands.describe(character="Character (autocomplete)")
@app_commands.autocomplete(character=autocomplete_character)
async def char_card(interaction: discord.Interaction, character: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, row = await resolve_character(interaction, character)
        card = await run_db(build_character_card(interaction.client.db, interaction.guild.id, cid), "build_character_card")
        owner_id = int(row["user_id"])
        owner = interaction.guild.get_member(owner_id)
        owner_mention = owner.mention if owner else f"<@{owner_id}>"
        text = f"{owner_mention}\n\n" + render_character_block(card)
        await safe_reply(interaction, text)
    except Exception as e:
        LOG.exception("char_card failed")
        await send_error(interaction, e)


# -----------------------------
# Client
# -----------------------------

class VilyraBotClient(discord.Client):
    def __init__(self, db: Database) -> None:
        intents = discord.Intents.default()
        intents.members = True
        super().__init__(intents=intents)
        self.db = db
        self.tree = app_commands.CommandTree(self)
        self.dashboard_limiter = SimpleRateLimiter(DASHBOARD_EDIT_MIN_INTERVAL)
        self._did_sync = False

    async def setup_hook(self) -> None:
        # Register all commands exactly once
        self.tree.add_command(staff_commands)
        self.tree.add_command(set_server_rank)
        self.tree.add_command(character_create)
        self.tree.add_command(character_delete)
        self.tree.add_command(character_rename)
        self.tree.add_command(award_legacy)
        self.tree.add_command(convert_stars)
        self.tree.add_command(ability_add)
        self.tree.add_command(ability_rename)
        self.tree.add_command(ability_upgrade)
        self.tree.add_command(set_available_legacy)
        self.tree.add_command(set_lifetime_legacy)
        self.tree.add_command(refresh_dashboard)
        self.tree.add_command(char_card)

        # Validate duplicates in local tree
        names = [c.name for c in self.tree.get_commands()]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise RuntimeError(f"Duplicate command name(s) detected: {dupes}")

        LOG.info("Command tree prepared: %s command(s); GUILD_ID=%s", len(names), safe_int(os.getenv("GUILD_ID"), 0))
        await self._sync_commands()

    async def _sync_commands(self) -> None:
        if self._did_sync:
            return
        self._did_sync = True
        try:
            gid = safe_int(os.getenv("GUILD_ID"), 0)
            if not gid:
                synced = await self.tree.sync()
                LOG.info("Global command sync complete: %s commands", len(synced))
                return

            raw_allow = (os.getenv("ALLOWED_GUILD_IDS") or "").strip()
            allowed: Optional[set[int]] = None
            if raw_allow:
                try:
                    allowed = {int(x.strip()) for x in raw_allow.split(",") if x.strip()}
                except Exception:
                    allowed = None

            if allowed and gid not in allowed:
                LOG.error("GUILD_ID %s not in ALLOWED_GUILD_IDS; skipping sync/reset.", gid)
                return

            allow_reset = (os.getenv("ALLOW_COMMAND_RESET") or "").strip().lower() in ("1", "true", "yes", "y", "on")
            guild_obj = discord.Object(id=gid)
            self.tree.copy_global_to(guild=guild_obj)
            if allow_reset and self.application_id:
                await self.http.bulk_upsert_guild_commands(self.application_id, gid, [])
                LOG.warning("Performed hard guild command reset (ALLOW_COMMAND_RESET=true) for guild %s", gid)
            synced = await self.tree.sync(guild=guild_obj)
            LOG.info("Guild command sync complete: %s commands (hard_reset=%s)", len(synced), allow_reset)
        except Exception:
            LOG.exception("Command sync failed")

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")

        do_refresh = (os.getenv("STARTUP_REFRESH") or "").strip().lower() in ("1", "true", "yes", "y", "on")
        if not do_refresh:
            LOG.info("Startup refresh disabled (set STARTUP_REFRESH=true to enable).")
            return

        LOG.info("Startup dashboard refresh: beginning for %d guild(s)...", len(list(self.guilds)))
        for g in list(self.guilds):
            try:
                status = await refresh_all_dashboards(self, g)
                LOG.info("Startup dashboard refresh: %s", status)
            except Exception:
                LOG.exception("Startup dashboard refresh failed")


# -----------------------------
# Entrypoint
# -----------------------------

async def main_async() -> None:
    token = env("DISCORD_TOKEN")
    dsn = env("DATABASE_URL")

    db = Database(dsn)
    await db.connect()
    await db.init_schema()

    client = VilyraBotClient(db=db)
    try:
        await client.start(token)
    finally:
        await db.close()

def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
