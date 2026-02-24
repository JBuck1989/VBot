# VB_v51 — Vilyra Legacy Bot (Railway + Postgres) — FULL REPLACEMENT (self-check fixed to actual DB API; stable; no destructive DB ops)
# (self-check added; no destructive DB ops)

from __future__ import annotations

# Bump this whenever you change how dashboards/cards are rendered.
# It forces a refresh even if player data hasn't changed (prevents "skip" from hiding template updates).
DASHBOARD_TEMPLATE_VERSION = 3

import os
import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import discord
from discord import app_commands

import psycopg
from psycopg.rows import dict_row


# -----------------------------
# Config
# -----------------------------

DEFAULT_DASHBOARD_CHANNEL_ID = 1469879866655768738
DEFAULT_COMMAND_LOG_CHANNEL_ID = 1469879960729817098

GUARDIAN_ROLE_NAME = "Guardian"
WARDEN_ROLE_NAME = "Warden"

MAX_ABILITY_STARS = 5
MAX_INFL_STARS_TOTAL = 5

STAR_COST = 10
MINOR_UPGRADE_COST = 5

REP_MIN = -100
REP_MAX = 100


DASHBOARD_EDIT_MIN_INTERVAL = float(os.getenv("DASHBOARD_EDIT_MIN_INTERVAL", "1.2"))
PLAYER_POST_SOFT_LIMIT = 1900

SERVER_RANKS = [
    "Guardian",
    "Warden",
    "Newcomer",
    "Apprentice",
    "Adventurer",
    "Sentinel",
    "Champion",
    "Legend",
    "Sovereign",
]

KINGDOMS = ["Sethrathiel", "Velarith", "Lyvik", "Baelon", "Avalea"]


BORDER_LEN = 20
PLAYER_BORDER = "═" * BORDER_LEN
CHAR_SEPARATOR = "-" * BORDER_LEN
CHAR_HEADER_LEFT = "꧁•⊹٭ "
CHAR_HEADER_RIGHT = " ٭⊹•꧂"


LOG = logging.getLogger("VilyraBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VilyraBot: %(message)s")


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


def db_timeout() -> int:
    return max(3, safe_int(os.getenv("DB_TIMEOUT_SECONDS"), 12))


def is_staff(member: discord.abc.User | discord.Member) -> bool:
    roles = getattr(member, "roles", None) or []
    for r in roles:
        if getattr(r, "name", "") in (GUARDIAN_ROLE_NAME, WARDEN_ROLE_NAME):
            return True
    return False


async def defer_ephemeral(interaction: discord.Interaction) -> None:
    """Defer an interaction ephemerally (safe no-op if already responded)."""
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True, thinking=True)
    except Exception:
        pass

async def safe_reply(interaction: discord.Interaction, content: str, *, embed: discord.Embed | None = None) -> None:
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=True, embed=embed)
        else:
            await interaction.response.send_message(content, ephemeral=True, embed=embed)
    except Exception:
        LOG.exception("Failed to send response/followup")


async def run_db(coro, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=db_timeout())
    except asyncio.TimeoutError as e:
        raise RuntimeError(f"Database operation timed out ({label}).") from e


def fmt_ids(ids: List[int]) -> str:
    return ",".join(str(i) for i in ids)


def parse_ids(s: Optional[str]) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out


async def log_to_channel(guild: Optional[discord.Guild], text: str) -> None:
    if not guild:
        return
    ch_id = safe_int(os.getenv("COMMAND_LOG_CHANNEL_ID"), DEFAULT_COMMAND_LOG_CHANNEL_ID)
    try:
        ch = guild.get_channel(ch_id) or await guild.fetch_channel(ch_id)
        if isinstance(ch, discord.TextChannel):
            await ch.send(text, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        LOG.exception("Failed to write to command log channel")

class SimpleRateLimiter:
    """Serialize dashboard message edits/creates to reduce 429s.
    discord.py will still handle rate limits, but this prevents burst PATCH spam at startup."""
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


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# -----------------------------
# UI Renderers
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
    # 20/20 line with distinct center marker ┃ and ▲ indicator integrated in-line.
    # IMPORTANT: The explainer is now end-aligned (no centerline), so it won't "shift" visually between desktop/mobile.
    net = clamp(int(net_lifetime), REP_MIN, REP_MAX)

    left_len = 20
    right_len = 20
    total = left_len + right_len

    # Map REP_MIN..REP_MAX onto 0..total (inclusive). net=0 should land exactly at center (left_len).
    pos = int(round((net - REP_MIN) / (REP_MAX - REP_MIN) * total))
    pos = clamp(pos, 0, total)

    bar = ["-"] * (total + 1)
    center_idx = left_len
    bar[center_idx] = "┃"
    bar[pos] = "▲"

    bar_line = "[" + "".join(bar) + "]"

    left_text = "MALEVOLENT ←"
    right_text = "→ BENEVOLENT"

    # End-align the explainer to the same visual width as the bar line.
    spaces = max(1, len(bar_line) - len(left_text) - len(right_text))
    explainer = left_text + (" " * spaces) + right_text

    return explainer + "\n" + bar_line



# -----------------------------
# Database Layer (schema autodetect)
# -----------------------------

class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg.AsyncConnection] = None

        # Detected columns
        self.characters_cols: set[str] = set()
        self.abilities_cols: set[str] = set()

        # Detected "level" column in abilities (upgrade_level vs level)
        self.abilities_level_col: str = "upgrade_level"
        # Detected "character name" column in abilities (character_name vs name)
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

        # Abilities: level column
        if "upgrade_level" in self.abilities_cols:
            self.abilities_level_col = "upgrade_level"
        elif "level" in self.abilities_cols:
            self.abilities_level_col = "level"
        else:
            self.abilities_level_col = "upgrade_level"  # will be added

        # Abilities: character column
        if "character_name" in self.abilities_cols:
            self.abilities_char_col = "character_name"
        elif "name" in self.abilities_cols:
            self.abilities_char_col = "name"
        else:
            self.abilities_char_col = "character_name"  # will be added

        LOG.info("Schema choices: abilities.%s as level, abilities.%s as character key",
                 self.abilities_level_col, self.abilities_char_col)

    async def init_schema(self) -> None:
        # Characters table exists already in your DB. We never drop it.
        # Add the columns our bot needs, in case they were missing.
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ability_stars INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS influence_minus INT NOT NULL DEFAULT 0;")
        # Kingdom is optional; allow NULLs so you can backfill later.
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS kingdom TEXT;")
        # If the column existed from a prior version, ensure it's nullable and has no forced default.
        await self._execute("ALTER TABLE characters ALTER COLUMN kingdom DROP DEFAULT;")
        await self._execute("ALTER TABLE characters ALTER COLUMN kingdom DROP NOT NULL;")

        # Ensure these exist too (your DB already has most of them, but safe):
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS archived BOOLEAN NOT NULL DEFAULT FALSE;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS legacy_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS legacy_minus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS lifetime_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS lifetime_minus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS influence_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        # Unique index (needed for ON CONFLICT)
        try:
            await self._execute("CREATE UNIQUE INDEX IF NOT EXISTS characters_unique ON characters (guild_id, user_id, name);")
        except Exception:
            LOG.exception("Could not create unique index on characters; continuing")

        # Players table (server rank)
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

        # Abilities table — create if missing
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                ability_name   TEXT NOT NULL,
                upgrade_level  INT  NOT NULL DEFAULT 0,
                level          INT  NULL,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        # Add missing columns to existing abilities table
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS character_name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS ability_name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrade_level INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS level INT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        # If we have an older table that used (guild_id,user_id,name,ability_name) as keys,
        # we can't safely add a PRIMARY KEY without knowing duplicates; so we do not force it.
        # We *do* add an index that helps our lookups:
        try:
            await self._execute("CREATE INDEX IF NOT EXISTS abilities_lookup ON abilities (guild_id, user_id, character_name, ability_name);")
        except Exception:
            LOG.exception("Could not create abilities index; continuing")

        # Dashboard tracking
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
        await self.detect_schema()
        LOG.info("Database schema initialized / updated")

    # -------- Players --------

    async def get_player_rank(self, guild_id: int, user_id: int) -> str:
        row = await self._fetchone("SELECT server_rank FROM players WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))
        return str(row["server_rank"]) if row and row.get("server_rank") else "Newcomer"

    async def set_player_rank(self, guild_id: int, user_id: int, rank: str) -> None:
        if rank not in SERVER_RANKS:
            raise ValueError("Invalid rank")
        await self._execute(
            """
            INSERT INTO players (guild_id, user_id, server_rank)
            VALUES (%s, %s, %s)
            ON CONFLICT (guild_id, user_id)
            DO UPDATE SET server_rank=EXCLUDED.server_rank, updated_at=NOW();
            """,
            (guild_id, user_id, rank),
        )

    # -------- Characters --------

    async def add_character(self, guild_id: int, user_id: int, name: str, kingdom: str | None = None) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Character name cannot be empty.")

        kingdom = kingdom.strip() if isinstance(kingdom, str) else None
        if kingdom == "":
            kingdom = None

        await self._execute(
            """
            INSERT INTO characters (guild_id, user_id, name, kingdom, archived, legacy_plus, legacy_minus, lifetime_plus, lifetime_minus, influence_plus, influence_minus, ability_stars, updated_at)
            VALUES (%s, %s, %s, %s, FALSE, 0, 0, 0, 0, 0, 0, 0, NOW())
            ON CONFLICT (guild_id, user_id, name)
            DO UPDATE SET archived=FALSE,
                          kingdom=COALESCE(EXCLUDED.kingdom, characters.kingdom),
                          updated_at=NOW();
            """,
            (guild_id, user_id, name, kingdom),
        )

        await self._execute(
            "INSERT INTO players (guild_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (guild_id, user_id),
        )

    async def set_character_archived(self, guild_id: int, user_id: int, character_name: str, archived: bool) -> bool:
        """Archive/unarchive a character. Returns True if a row was updated."""
        sql = """
            UPDATE characters
               SET archived=%s,
                   updated_at=NOW()
             WHERE guild_id=%s
               AND user_id=%s
               AND name=%s;
        """
        rowcount = await self._execute(sql, (archived, guild_id, user_id, character_name))
        return bool(rowcount and rowcount > 0)

    async def delete_character(self, guild_id: int, user_id: int, character_name: str) -> bool:
        """Hard-delete a character and its abilities. Returns True if a character row was deleted."""
        # Delete abilities first (if any)
        await self._execute(
            "DELETE FROM abilities WHERE guild_id=%s AND user_id=%s AND character_name=%s;",
            (guild_id, user_id, character_name),
        )
        rowcount = await self._execute(
            "DELETE FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (guild_id, user_id, character_name),
        )
        return bool(rowcount and rowcount > 0)


    async def set_character_kingdom(self, guild_id: int, user_id: int, character_name: str, kingdom: str) -> bool:
        """Set a character's home kingdom. Returns True if updated."""
        sql = """
            UPDATE characters
               SET kingdom=%s,
                   updated_at=NOW()
             WHERE guild_id=%s
               AND user_id=%s
               AND name=%s;
        """
        rowcount = await self._execute(sql, (kingdom, guild_id, user_id, character_name))
        return rowcount > 0


    async def character_exists(self, guild_id: int, user_id: int, name: str) -> bool:
        row = await self._fetchone(
            "SELECT 1 FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s AND COALESCE(archived, FALSE)=FALSE LIMIT 1;",
            (guild_id, user_id, name.strip()),
        )
        return bool(row)

    async def list_characters(self, guild_id: int, user_id: int) -> List[str]:
        rows = await self._fetchall(
            """
            SELECT name
            FROM characters
            WHERE guild_id=%s AND user_id=%s AND COALESCE(archived, FALSE)=FALSE
            ORDER BY created_at ASC, name ASC;
            """,
            (guild_id, user_id),
        )
        return [str(r["name"]) for r in rows if r and r.get("name")]

    async def list_player_ids(self, guild_id: int) -> List[int]:
        rows = await self._fetchall(
            """
            SELECT DISTINCT user_id
            FROM characters
            WHERE guild_id=%s AND COALESCE(archived, FALSE)=FALSE
            ORDER BY user_id ASC;
            """,
            (guild_id,),
        )
        return [int(r["user_id"]) for r in rows if r and r.get("user_id") is not None]

    async def get_character_state(self, guild_id: int, user_id: int, name: str) -> Dict[str, Any]:
        row = await self._fetchone(
            """
            SELECT legacy_plus, legacy_minus, lifetime_plus, lifetime_minus,
                   influence_plus, influence_minus, ability_stars, kingdom
            FROM characters
            WHERE guild_id=%s AND user_id=%s AND name=%s AND COALESCE(archived, FALSE)=FALSE
            LIMIT 1;
            """,
            (guild_id, user_id, name.strip()),
        )
        if not row:
            raise ValueError("Character not found.")
        return {
            "legacy_plus": safe_int(row.get("legacy_plus"), 0),
            "legacy_minus": safe_int(row.get("legacy_minus"), 0),
            "lifetime_plus": safe_int(row.get("lifetime_plus"), 0),
            "lifetime_minus": safe_int(row.get("lifetime_minus"), 0),
            "influence_plus": safe_int(row.get("influence_plus"), 0),
            "influence_minus": safe_int(row.get("influence_minus"), 0),
            "ability_stars": safe_int(row.get("ability_stars"), 0),
            "kingdom": (row.get("kingdom") or ""),
        }

    async def award_legacy(self, guild_id: int, user_id: int, name: str, pos: int = 0, neg: int = 0) -> None:
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
            WHERE guild_id=%s AND user_id=%s AND name=%s AND COALESCE(archived, FALSE)=FALSE;
            """,
            (pos, neg, pos, neg, guild_id, user_id, name.strip()),
        )

    async def spend_legacy(self, guild_id: int, user_id: int, name: str, pool: str, amount: int) -> None:
        amount = max(0, int(amount))
        pool = pool.strip().lower()
        st = await self.get_character_state(guild_id, user_id, name)
        if pool == "positive":
            if st["legacy_plus"] < amount:
                raise ValueError(f"Not enough available positive points (need {amount}, have {st['legacy_plus']}).")
            await self._execute(
                "UPDATE characters SET legacy_plus=legacy_plus-%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (amount, guild_id, user_id, name.strip()),
            )
        elif pool == "negative":
            if st["legacy_minus"] < amount:
                raise ValueError(f"Not enough available negative points (need {amount}, have {st['legacy_minus']}).")
            await self._execute(
                "UPDATE characters SET legacy_minus=legacy_minus-%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (amount, guild_id, user_id, name.strip()),
            )
        else:
            raise ValueError("pool must be positive or negative")

    async def convert_star(
        self,
        guild_id: int,
        user_id: int,
        name: str,
        star_type: Literal['ability','influence_positive','influence_negative'],
        stars: int,
        spend_plus: int,
        spend_minus: int,
    ) -> None:
        """Convert AVAILABLE legacy points into stars (stars are assigned, never consumed).

        Costs:
          - ability: 10 total points per star (can split + and -)
          - influence_positive: 10 positive points per star (no split)
          - influence_negative: 10 negative points per star (no split)

        Caps:
          - ability_stars max 5
          - total influence stars (influence_plus + influence_minus) max 5
        """
        star_type = star_type.strip().lower()
        name = name.strip()
        stars = int(stars)
        spend_plus = int(spend_plus)
        spend_minus = int(spend_minus)

        if stars < 1:
            raise ValueError("stars must be >= 1")
        if spend_plus < 0 or spend_minus < 0:
            raise ValueError("Spend amounts must be >= 0")

        st = await self.get_character_state(guild_id, user_id, name)
        infl_total = st["influence_plus"] + st["influence_minus"]
        total_cost = STAR_COST * stars

        if star_type == "ability":
            if st["ability_stars"] + stars > MAX_ABILITY_STARS:
                raise ValueError("Ability stars already at max (5).")
            if spend_plus + spend_minus != total_cost:
                raise ValueError(f"Ability stars cost {total_cost} total points. Provide spend_plus + spend_minus = {total_cost}.")
        elif star_type == "influence_positive":
            if infl_total + stars > MAX_INFL_STARS_TOTAL:
                raise ValueError("Total influence stars (pos+neg) cannot exceed 5.")
            if spend_plus != total_cost or spend_minus != 0:
                raise ValueError(f"Positive influence stars cost {total_cost} POSITIVE points. Provide spend_plus={total_cost}, spend_minus=0.")
        elif star_type == "influence_negative":
            if infl_total + stars > MAX_INFL_STARS_TOTAL:
                raise ValueError("Total influence stars (pos+neg) cannot exceed 5.")
            if spend_minus != total_cost or spend_plus != 0:
                raise ValueError(f"Negative influence stars cost {total_cost} NEGATIVE points. Provide spend_plus=0, spend_minus={total_cost}.")
        else:
            raise ValueError("star_type must be ability, influence_positive, or influence_negative")

        if spend_plus > st["legacy_plus"]:
            raise ValueError(f"Not enough available positive points (need {spend_plus}, have {st['legacy_plus']}).")
        if spend_minus > st["legacy_minus"]:
            raise ValueError(f"Not enough available negative points (need {spend_minus}, have {st['legacy_minus']}).")

        # Spend ONLY from AVAILABLE pools
        await self._execute(
            "UPDATE characters SET legacy_plus=legacy_plus-%s, legacy_minus=legacy_minus-%s, updated_at=NOW() "
            "WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (spend_plus, spend_minus, guild_id, user_id, name),
        )

        # Apply stars
        if star_type == "ability":
            await self._execute(
                "UPDATE characters SET ability_stars=ability_stars+%s, updated_at=NOW() "
                "WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (stars, guild_id, user_id, name),
            )
        elif star_type == "influence_positive":
            await self._execute(
                "UPDATE characters SET influence_plus=influence_plus+%s, updated_at=NOW() "
                "WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (stars, guild_id, user_id, name),
            )
        else:  # influence_negative
            await self._execute(
                "UPDATE characters SET influence_minus=influence_minus+%s, updated_at=NOW() "
                "WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (stars, guild_id, user_id, name),
            )

    async def reset_points(self, guild_id: int, user_id: int, name: str,
                           legacy_plus: Optional[int], legacy_minus: Optional[int],
                           lifetime_plus: Optional[int], lifetime_minus: Optional[int]) -> None:
        st = await self.get_character_state(guild_id, user_id, name)
        lp = st["legacy_plus"] if legacy_plus is None else max(0, int(legacy_plus))
        lm = st["legacy_minus"] if legacy_minus is None else max(0, int(legacy_minus))
        ltp = st["lifetime_plus"] if lifetime_plus is None else max(0, int(lifetime_plus))
        ltm = st["lifetime_minus"] if lifetime_minus is None else max(0, int(lifetime_minus))
        await self._execute(
            """
            UPDATE characters
            SET legacy_plus=%s, legacy_minus=%s, lifetime_plus=%s, lifetime_minus=%s, updated_at=NOW()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (lp, lm, ltp, ltm, guild_id, user_id, name.strip()),
        )

    async def reset_stars(self, guild_id: int, user_id: int, name: str,
                          ability_stars: Optional[int], infl_plus: Optional[int], infl_minus: Optional[int]) -> None:
        st = await self.get_character_state(guild_id, user_id, name)
        a = st["ability_stars"] if ability_stars is None else clamp(int(ability_stars), 0, MAX_ABILITY_STARS)
        ip = st["influence_plus"] if infl_plus is None else clamp(int(infl_plus), 0, MAX_INFL_STARS_TOTAL)
        im = st["influence_minus"] if infl_minus is None else clamp(int(infl_minus), 0, MAX_INFL_STARS_TOTAL)
        if ip + im > MAX_INFL_STARS_TOTAL:
            raise ValueError("Total influence stars (pos+neg) cannot exceed 5.")
        await self._execute(
            """
            UPDATE characters
            SET ability_stars=%s, influence_plus=%s, influence_minus=%s, updated_at=NOW()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (a, ip, im, guild_id, user_id, name.strip()),
        )

    # -------- Abilities (schema-flex) --------

    def _ability_level_expr(self) -> str:
        # We always SELECT as "upgrade_level" to keep renderer stable
        if self.abilities_level_col == "upgrade_level":
            return "COALESCE(upgrade_level, 0) AS upgrade_level"
        return "COALESCE(level, 0) AS upgrade_level"

    def _ability_where_char(self) -> str:
        return self.abilities_char_col

    async def list_abilities(self, guild_id: int, user_id: int, name: str) -> List[Tuple[str, int]]:
        # If schema changed since startup, re-detect once
        try:
            rows = await self._fetchall(
                f"""
                SELECT ability_name, {self._ability_level_expr()}
                FROM abilities
                WHERE guild_id=%s AND user_id=%s AND {self._ability_where_char()}=%s
                ORDER BY created_at ASC, ability_name ASC;
                """,
                (guild_id, user_id, name.strip()),
            )
        except Exception:
            # Last-resort: refresh detection and retry once
            await self.detect_schema()
            rows = await self._fetchall(
                f"""
                SELECT ability_name, {self._ability_level_expr()}
                FROM abilities
                WHERE guild_id=%s AND user_id=%s AND {self._ability_where_char()}=%s
                ORDER BY created_at ASC, ability_name ASC;
                """,
                (guild_id, user_id, name.strip()),
            )

        out: List[Tuple[str, int]] = []
        for r in rows:
            if r and r.get("ability_name"):
                out.append((str(r["ability_name"]), safe_int(r.get("upgrade_level"), 0)))
        return out

    async def add_ability(self, guild_id: int, user_id: int, name: str, ability_name: str) -> None:
        ability_name = ability_name.strip()
        if not ability_name:
            raise ValueError("Ability name cannot be empty.")
        st = await self.get_character_state(guild_id, user_id, name)
        current = await self.list_abilities(guild_id, user_id, name)
        cap = 2 + clamp(st["ability_stars"], 0, MAX_ABILITY_STARS)
        if len(current) >= cap:
            raise ValueError(f"Ability capacity reached ({len(current)}/{cap}). Earn more Ability Stars to add abilities.")

        # Insert using whatever character column exists; also initialize both level columns if present.
        char_col = self._ability_where_char()
        cols = ["guild_id", "user_id", char_col, "ability_name", "created_at"]
        vals = ["%s", "%s", "%s", "%s", "NOW()"]
        params: List[Any] = [guild_id, user_id, name.strip(), ability_name]

        if "upgrade_level" in self.abilities_cols:
            cols.append("upgrade_level")
            vals.append("0")
        if "level" in self.abilities_cols:
            cols.append("level")
            vals.append("0")

        sql = "INSERT INTO abilities (" + ", ".join(cols) + ") VALUES (" + ", ".join(vals) + ");"
        await self._execute(sql, params)


    async def upgrade_ability(
        self,
        guild_id: int,
        user_id: int,
        name: str,
        ability_name: str,
        upgrades: int,
        pay_positive: int,
        pay_negative: int,
    ) -> Tuple[int, int]:
        """Apply ability upgrades (max 5 per ability). Each upgrade costs 5 legacy points.
        Points may be paid using any mix of positive/negative AVAILABLE legacy points, but the caller must specify the split.
        This method NEVER touches lifetime totals.
        """
        ability_name = ability_name.strip()
        upgrades = max(1, int(upgrades))
        pay_positive = max(0, int(pay_positive))
        pay_negative = max(0, int(pay_negative))

        if upgrades < 1:
            raise ValueError("upgrades must be >= 1")

        total_cost = upgrades * MINOR_UPGRADE_COST
        if pay_positive + pay_negative != total_cost:
            raise ValueError(f"Payment must equal {total_cost} points total (5 per upgrade).")

        char_col = self._ability_where_char()
        level_col = self.abilities_level_col

        row = await self._fetchone(
            f"""
            SELECT COALESCE({level_col}, 0) AS cur_level
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND ability_name=%s
            ORDER BY created_at ASC
            LIMIT 1;
            """,
            (guild_id, user_id, name.strip(), ability_name),
        )
        if not row:
            raise ValueError("Ability not found. Add it first with /add_ability.")
        cur_level = safe_int(row.get("cur_level"), 0)

        max_level = 5
        if cur_level >= max_level:
            raise ValueError(f"Upgrade limit reached ({cur_level}/{max_level}).")

        # Clamp requested upgrades to remaining cap
        remaining = max_level - cur_level
        if upgrades > remaining:
            raise ValueError(f"Only {remaining} upgrade(s) remaining for this ability (max {max_level}).")

        # Validate available pools and deduct ONLY from available points
        st = await self.get_character_state(guild_id, user_id, name)
        if st["legacy_plus"] < pay_positive:
            raise ValueError(f"Not enough available positive points (need {pay_positive}, have {st['legacy_plus']}).")
        if st["legacy_minus"] < pay_negative:
            raise ValueError(f"Not enough available negative points (need {pay_negative}, have {st['legacy_minus']}).")

        if pay_positive:
            await self._execute(
                "UPDATE characters SET legacy_plus=legacy_plus-%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (pay_positive, guild_id, user_id, name.strip()),
            )
        if pay_negative:
            await self._execute(
                "UPDATE characters SET legacy_minus=legacy_minus-%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (pay_negative, guild_id, user_id, name.strip()),
            )

        new_level = cur_level + upgrades

        # Update the detected column, and also keep the other column in sync if present
        sets: List[str] = [f"{level_col}=%s"]
        params: List[Any] = [new_level]

        if level_col != "upgrade_level" and "upgrade_level" in self.abilities_cols:
            sets.append("upgrade_level=%s")
            params.append(new_level)
        if level_col != "level" and "level" in self.abilities_cols:
            sets.append("level=%s")
            params.append(new_level)

        params.extend([guild_id, user_id, name.strip(), ability_name])
        sql = f"UPDATE abilities SET {', '.join(sets)} WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND ability_name=%s;"
        await self._execute(sql, params)
        return new_level, max_level

    # -------- Dashboard message tracking --------

    async def get_dashboard_entry(self, guild_id: int, user_id: int) -> Tuple[List[int], Optional[str], Optional[Any], int]:
        """Return (message_ids, content_hash, updated_at, template_version)."""
        try:
            row = await self._fetchone(
                "SELECT message_ids, content_hash, updated_at, COALESCE(template_version, 0) AS template_version FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
                (guild_id, user_id),
            )
            ids = parse_ids(row["message_ids"]) if row and row.get("message_ids") else []
            h = str(row["content_hash"]) if row and row.get("content_hash") else None
            ts = row["updated_at"] if row and row.get("updated_at") else None
            tv = int(row["template_version"]) if row and row.get("template_version") is not None else 0
            return ids, h, ts, tv
        except psycopg.errors.UndefinedColumn:
            row = await self._fetchone(
                "SELECT message_ids, updated_at FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
                (guild_id, user_id),
            )
            ids = parse_ids(row["message_ids"]) if row and row.get("message_ids") else []
            ts = row["updated_at"] if row and row.get("updated_at") else None
            return ids, None, ts, 0

    async def get_dashboard_message_ids(self, guild_id: int, user_id: int) -> List[int]:
        ids, _, _, _ = await self.get_dashboard_entry(guild_id, user_id)
        return ids

    async def set_dashboard_message_ids(
        self,
        guild_id: int,
        user_id: int,
        channel_id: int,
        ids: List[int],
        h: Optional[str] = None,
    ) -> None:
        """Persist dashboard message IDs (and optional content hash) for a player."""
        try:
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
                (guild_id, user_id, channel_id, fmt_ids(ids) if ids else None, h, DASHBOARD_TEMPLATE_VERSION),
            )
        except psycopg.errors.UndefinedColumn:
            await self._execute(
                """
                INSERT INTO dashboard_messages (guild_id, user_id, channel_id, message_ids, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (guild_id, user_id)
                DO UPDATE SET channel_id=EXCLUDED.channel_id,
                              message_ids=EXCLUDED.message_ids,
                              updated_at=NOW();
                """,
                (guild_id, user_id, channel_id, fmt_ids(ids) if ids else None),
            )

    async def clear_dashboard_message_ids(self, guild_id: int, user_id: int) -> None:
        await self._execute(
            "DELETE FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )

    async def get_latest_player_data_updated_at(self, guild_id: int, user_id: int) -> Optional[Any]:
        """Max updated_at across characters + abilities for this player."""
        row = await self._fetchone(
            """
            SELECT GREATEST(
                COALESCE((SELECT MAX(updated_at) FROM characters WHERE guild_id=%s AND user_id=%s), to_timestamp(0)),
                COALESCE((SELECT MAX(updated_at) FROM abilities WHERE guild_id=%s AND user_id=%s), to_timestamp(0)),
                COALESCE((SELECT MAX(updated_at) FROM players WHERE guild_id=%s AND user_id=%s), to_timestamp(0))
            ) AS ts
            """,
            (guild_id, user_id, guild_id, user_id, guild_id, user_id),
        )
        return row["ts"] if row else None


    # -------- Dashboard message tracking --------


@dataclass
class CharacterCard:
    name: str
    kingdom: str
    legacy_plus: int
    legacy_minus: int
    lifetime_plus: int
    lifetime_minus: int
    ability_stars: int
    infl_plus: int
    infl_minus: int
    abilities: List[Tuple[str, int]]  # (ability_name, upgrades/level)


async def build_character_card(db: Database, guild_id: int, user_id: int, name: str) -> CharacterCard:
    st = await db.get_character_state(guild_id, user_id, name)
    abilities = await db.list_abilities(guild_id, user_id, name)
    return CharacterCard(
        name=name,
        kingdom=st.get("kingdom", "Unassigned"),
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
    # Keep the decorative header but bold the name and add spacing so it doesn't wrap awkwardly on mobile
    lines.append(f"{CHAR_HEADER_LEFT}**{card.name}** {CHAR_HEADER_RIGHT}")
    # Kingdom directly under the character name.
    # Always show the line; if NULL/empty/unassigned, show a blank value (per spec).
    k = (card.kingdom or "").strip()
    if (not k) or (k.lower() == "unassigned"):
        lines.append("Kingdom:")
    else:
        lines.append(f"Kingdom: {k}")
    lines.append("")  # spacer line between header and stats
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
    nickname = member.display_name if member else f"User {user_id}"
    rank = await db.get_player_rank(guild.id, user_id)

    chars = await db.list_characters(guild.id, user_id)
    if not chars:
        return ""

    lines: List[str] = []
    lines.append(PLAYER_BORDER)
    lines.append(f"__***{nickname}***__")
    lines.append(f"__***Server Rank: {rank}***__")
    lines.append("")

    for i, cname in enumerate(chars):
        card = await build_character_card(db, guild.id, user_id, cname)
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


async def refresh_player_dashboard(client: "VilyraBotClient", guild: discord.Guild, user_id: int) -> str:
    db = client.db
    channel = await get_dashboard_channel(guild)
    if not channel:
        return "Dashboard channel not found or not a text channel."

    me = guild.me or (guild.get_member(client.user.id) if client.user else None)
    if me:
        perms = channel.permissions_for(me)
        if not (perms.view_channel and perms.send_messages):
            return f"Missing permissions in <#{channel.id}>: need View Channel + Send Messages."

    chars = await db.list_characters(guild.id, user_id)
    stored_ids, stored_hash, dash_ts, stored_tv = await db.get_dashboard_entry(guild.id, user_id)

    # Skip startup refresh for this player if nothing changed since last dashboard update.
    try:
        latest_ts = await db.get_latest_player_data_updated_at(guild.id, user_id)
        # Only skip if this dashboard was rendered with the CURRENT template.
        if stored_tv == DASHBOARD_TEMPLATE_VERSION and dash_ts and latest_ts and latest_ts <= dash_ts:
            LOG.info(
                "Dashboard up-to-date for user_id=%s (latest_ts=%s <= dash_ts=%s); skipping.",
                user_id,
                latest_ts,
                dash_ts,
            )
            return "skipped"
    except Exception as ex:
        LOG.warning("Could not compute latest player data timestamp for user_id=%s: %s", user_id, ex)

    if not chars:
        for mid in stored_ids:
            try:
                m = await channel.fetch_message(mid)
                await m.delete()
            except Exception:
                pass
        await db.clear_dashboard_message_ids(guild.id, user_id)
        return f"No characters for user_id={user_id}; dashboard entry cleared."

    content = await render_player_post(db, guild, user_id)
    if not content:
        return f"No content rendered for user_id={user_id}."

    new_hash = content_hash(content)

    msg: Optional[discord.Message] = None
    if stored_ids:
        try:
            msg = await channel.fetch_message(stored_ids[0])
        except Exception:
            msg = None

    if msg is None:
        await client.dashboard_limiter.wait()
        msg = await channel.send(content)
        await db.set_dashboard_message_ids(guild.id, user_id, channel.id, [msg.id], new_hash)
        return f"Dashboard created for user_id={user_id}."
    else:
        await client.dashboard_limiter.wait()
        await msg.edit(content=content)
        if len(stored_ids) > 1:
            for extra_id in stored_ids[1:]:
                try:
                    extra_msg = await channel.fetch_message(extra_id)
                    await extra_msg.delete()
                except Exception:
                    pass
            await db.set_dashboard_message_ids(guild.id, user_id, channel.id, [msg.id], new_hash)
        await db.set_dashboard_message_ids(guild.id, user_id, channel.id, [msg.id], new_hash)
        return f"Dashboard updated for user_id={user_id}."


async def refresh_all_dashboards(client: "VilyraBotClient", guild: discord.Guild) -> str:
    user_ids = await client.db.list_player_ids(guild.id)
    if not user_ids:
        return "No players with characters yet."
    ok = 0
    for uid in user_ids:
        await refresh_player_dashboard(client, guild, uid)
        ok += 1
        # gentle spacing between players (prevents burst edits on startup)
        await asyncio.sleep(0.2)
    return f"Refreshed dashboards for {ok} player(s)."


# -----------------------------
# Command guards
# -----------------------------

def staff_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild:
            return False
        if isinstance(interaction.user, discord.Member) and is_staff(interaction.user):
            return True
        await safe_reply(interaction, "Staff only (Guardian/Warden).")
        return False
    return app_commands.check(predicate)


def in_guild_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False
        return True
    return app_commands.check(predicate)


async def require_character(db: Database, guild_id: int, user_id: int, name: str) -> None:
    if not await db.character_exists(guild_id, user_id, name):
        raise ValueError("Character not found for that user.")


# -----------------------------
# Slash commands
# -----------------------------

@app_commands.command(name="set_server_rank", description="(Staff) Set a player's server rank.")
@in_guild_only()
@staff_only()
async def set_server_rank(interaction: discord.Interaction, user: discord.Member, rank: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        rank = rank.strip()
        if rank not in SERVER_RANKS:
            await safe_reply(interaction, "Invalid rank. Options: " + ", ".join(SERVER_RANKS))
            return
        await run_db(interaction.client.db.set_player_rank(interaction.guild.id, user.id, rank), "set_server_rank")
        await log_to_channel(interaction.guild, f"🏷️ {interaction.user.mention} set server rank for {user.mention} to **{rank}**")
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Rank set. " + status)
    except Exception as e:
        LOG.exception("set_server_rank failed")
        await safe_reply(interaction, f"Set rank failed: {e}")


@app_commands.command(name="set_char_kingdom", description="(Staff) Set a character's home kingdom.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player who owns the character", character_name="Exact character name", kingdom="New home kingdom")
@app_commands.choices(kingdom=[app_commands.Choice(name=k, value=k) for k in KINGDOMS])
async def set_char_kingdom(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    kingdom: app_commands.Choice[str],
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        char_name = character_name.strip()
        if not char_name:
            await safe_reply(interaction, "Character name is required.")
            return
        # Ensure the character exists for this user
        exists = await run_db(
            interaction.client.db.character_exists(interaction.guild.id, user.id, char_name),
            "character_exists"
        )
        if not exists:
            await safe_reply(interaction, f"Character not found for {user.mention}: **{char_name}**")
            return
        await run_db(
            interaction.client.db.set_character_kingdom(interaction.guild.id, user.id, char_name, kingdom.value),
            "set_character_kingdom"
        )
        await log_to_channel(
            interaction.guild,
            f"🏰 {interaction.user.mention} set kingdom for **{char_name}** ({user.mention}) to **{kingdom.value}**"
        )
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Kingdom updated. " + status)
    except Exception as e:
        LOG.exception("set_char_kingdom failed")
        await safe_reply(interaction, f"Set kingdom failed: {e}")


async def _add_character_impl(interaction: discord.Interaction, user: discord.Member, character_name: str, kingdom: Optional[str]) -> None:
    """Core implementation for /add_character."""
    await defer_ephemeral(interaction)

    try:
        assert_staff(interaction)
        # Normalize inputs
        cname = (character_name or "").strip()
        if not cname:
            raise ValueError("Character name is required.")

        k = (kingdom or "").strip() or None

        db: Database = interaction.client.db  # type: ignore[attr-defined]
        await run_db(db.add_character(interaction.guild.id, user.id, cname, k), "add_character")

        await interaction.followup.send(f"✅ Added **{cname}** for {user.mention}.", ephemeral=True)

        # Ensure dashboard reflects the new character immediately
        await refresh_player_dashboard(interaction.client, interaction.guild, user.id)

        await log_command(
            interaction.client,
            interaction.guild,
            f"➕ {interaction.user.mention} added character **{cname}** for {user.mention}" + (f" (kingdom: **{k}**)." if k else "."),
        )
    except Exception as e:
        await interaction.followup.send(f"❌ Add character failed: {e}", ephemeral=True)



    kingdom_value = (kingdom or None)

    try:
        await run_db(db.add_character(interaction.guild.id, user.id, character_name, kingdom_value), "add_character")
    except Exception as e:
        await interaction.followup.send(f"Add character failed: {e}", ephemeral=True)
        return

    await log_command(interaction, f"➕ {interaction.user.mention} added **{character_name}** for {user.mention} (kingdom: {kingdom_value or '—'}).")
    await refresh_player_dashboard(interaction.client, interaction.guild, user.id, force=True)

    await interaction.followup.send(f"Character **{character_name}** added for {user.mention}. Dashboard updated.", ephemeral=True)


@app_commands.command(name="add_character", description="Add a character for a player (staff only).")
@app_commands.describe(
    user="The player who will own this character",
    character_name="Character name (exact)",
    kingdom="Starting kingdom (optional)",
)
@app_commands.choices(
    kingdom=[
        app_commands.Choice(name="Velarith", value="Velarith"),
        app_commands.Choice(name="Lyvik", value="Lyvik"),
        app_commands.Choice(name="Baelon", value="Baelon"),
        app_commands.Choice(name="Sethrathiel", value="Sethrathiel"),
        app_commands.Choice(name="Avalea", value="Avalea"),
    ]
)
async def add_character(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    kingdom: Optional[app_commands.Choice[str]] = None,
) -> None:
    await _add_character_impl(interaction, user, character_name, kingdom.value if kingdom else None)





@app_commands.command(name="character_archive", description="(Staff) Archive or unarchive a character (hide/show on dashboard).")
@app_commands.guild_only()
@staff_only()
@app_commands.describe(user="Player who owns the character", character_name="Character to archive/unarchive")
@app_commands.choices(action=[
    app_commands.Choice(name="Archive", value="archive"),
    app_commands.Choice(name="Unarchive", value="unarchive"),
])
async def character_archive(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    action: app_commands.Choice[str],
):
    """Archive/unarchive a character. Archived characters are hidden from dashboard + cards."""
    await interaction.response.defer(ephemeral=True, thinking=True)
    do_archive = (action.value == "archive")

    ok = await run_db(
        interaction.client.db.set_character_archived(interaction.guild.id, user.id, character_name, do_archive),
        "set_character_archived",
            "set_character_kingdom",
    )
    if not ok:
        await interaction.followup.send(f"Character not found: **{character_name}**", ephemeral=True)
        return

    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    # Command log
    verb = "archived" if do_archive else "unarchived"
    await log_to_channel(
        interaction.guild,
        f"🗄 {interaction.user.mention} {verb} **{character_name}** for {user.mention}",
    )

    await interaction.followup.send(f"✅ {verb.title()} **{character_name}**. {status}", ephemeral=True)






@app_commands.command(name="character_delete", description="(Staff) Delete a character (cannot be undone).")
@in_guild_only()
@staff_only()
async def character_delete(interaction: discord.Interaction, user: discord.Member, character_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        ok = await run_db(
            interaction.client.db.delete_character(interaction.guild.id, user.id, character_name),
            "delete_character",
        )
        if not ok:
            await safe_reply(interaction, f"Character not found: **{character_name}**")
            return

        await log_to_channel(
            interaction.guild,
            f"🗑️ {interaction.user.mention} deleted character **{character_name.strip()}** for {user.mention}",
        )
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Character deleted. " + status)
    except Exception as e:
        LOG.exception("character_delete failed")
        await safe_reply(interaction, f"Delete character failed: {e}")


@app_commands.command(name="character_archive_by_id", description="(Staff) Archive/unarchive a character by user ID (use for players who left).")
@in_guild_only()
@staff_only()
@app_commands.describe(
    user_id="Discord user ID of the player (numbers only).",
    character_name="Exact character name to archive/unarchive.",
    action="Archive hides the character from the dashboard; Unarchive shows it again.",
)
@app_commands.choices(
    action=[
        app_commands.Choice(name="Archive", value="archive"),
        app_commands.Choice(name="Unarchive", value="unarchive"),
    ]
)
async def character_archive_by_id(
    interaction: discord.Interaction,
    user_id: str,
    character_name: str,
    action: app_commands.Choice[str],
):
    """
    Archive/unarchive a character when you can't pick the user via Discord's user picker (e.g., they left the server).
    """
    await defer_ephemeral(interaction)

    uid_str = (user_id or "").strip()
    if not uid_str.isdigit():
        await interaction.followup.send("❌ `user_id` must be numbers only (a Discord user ID).", ephemeral=True)
        return
    uid = int(uid_str)

    do_archive = (action.value == "archive")
    try:
        status = await run_db(
            interaction.client.db.set_character_archived(interaction.guild.id, uid, character_name, do_archive),
            "set_character_archived",
        )
    except Exception as e:
        await interaction.followup.send(f"❌ Archive failed: {e}", ephemeral=True)
        return

    # Update dashboard for that user id (even if the user isn't in guild anymore, the post is keyed by user_id)
    try:
        dash_status = await refresh_player_dashboard(interaction.client, interaction.guild, uid)
    except Exception as e:
        dash_status = f"(Dashboard refresh failed: {e})"

    verb = "archived" if do_archive else "unarchived"
    await log_to_channel(
        interaction.guild,
        f"🗄 {interaction.user.mention} {verb} **{character_name}** for user_id={uid}",
    )

    await interaction.followup.send(f"✅ {verb.title()} **{character_name}** for user_id={uid}. {dash_status}", ephemeral=True)

@app_commands.command(name="award_legacy_points", description="(Staff) Award positive and/or negative legacy points to a character.")
@in_guild_only()
@staff_only()
async def award_legacy_points(interaction: discord.Interaction, user: discord.Member, character_name: str, positive: int = 0, negative: int = 0):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        character_name = character_name.strip()
        await run_db(require_character(interaction.client.db, interaction.guild.id, user.id, character_name), "require_character")
        if positive < 0 or negative < 0:
            await safe_reply(interaction, "Points must be >= 0.")
            return
        if positive == 0 and negative == 0:
            await safe_reply(interaction, "Provide positive and/or negative points to award.")
            return
        await run_db(interaction.client.db.award_legacy(interaction.guild.id, user.id, character_name, pos=positive, neg=negative), "award_legacy")
        await log_to_channel(interaction.guild, f"🏅 {interaction.user.mention} awarded **{character_name}** ({user.mention}) legacy: +{positive} / -{negative}")
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Awarded. " + status)
    except Exception as e:
        LOG.exception("award_legacy_points failed")
        await safe_reply(interaction, f"Award failed: {e}")



async def _handle_convert_star_interaction(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    star_type: str,
    stars: int,
    spend_positive: int,
    spend_negative: int,
) -> None:
    """Shared implementation for /convert_star and /convert_points_to_stars."""
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        character_name = character_name.strip()
        star_type = str(star_type).strip().lower()

        await run_db(require_character(interaction.client.db, interaction.guild.id, user.id, character_name), "require_character")

        await run_db(
            interaction.client.db.convert_star(
                interaction.guild.id,
                user.id,
                character_name,
                star_type,
                stars,
                spend_positive,
                spend_negative,
            ),
            "convert_star",
        )

        await log_to_channel(
            interaction.guild,
            f"⭐ {interaction.user.mention} converted legacy -> **{star_type}** x{stars} for **{character_name}** ({user.mention}) (spent +{spend_positive}/-{spend_negative})",
        )
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Converted. " + status)
    except Exception as e:
        LOG.exception("convert_star failed")
        await safe_reply(interaction, f"Convert failed: {e}")


@app_commands.command(name="convert_star", description="(Staff) Convert available legacy points into stars (10 points per star).")
@app_commands.choices(star_type=[
    app_commands.Choice(name="Ability Star", value="ability"),
    app_commands.Choice(name="Positive Influence Star", value="influence_positive"),
    app_commands.Choice(name="Negative Influence Star", value="influence_negative"),
])
@in_guild_only()
@staff_only()
async def convert_star(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    star_type: Literal['ability','influence_positive','influence_negative'],
    stars: int,
    spend_positive: int,
    spend_negative: int,
):
    await _handle_convert_star_interaction(interaction, user, character_name, star_type, stars, spend_positive, spend_negative)

@app_commands.command(name="convert_points_to_stars", description="(Staff) Convert available legacy points into stars (10 points per star).")
@app_commands.choices(star_type=[
    app_commands.Choice(name="Ability Star", value="ability"),
    app_commands.Choice(name="Positive Influence Star", value="influence_positive"),
    app_commands.Choice(name="Negative Influence Star", value="influence_negative"),
])
@in_guild_only()
@staff_only()
async def convert_points_to_stars(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    star_type: Literal['ability','influence_positive','influence_negative'],
    stars: int,
    spend_positive: int,
    spend_negative: int,
):
    await _handle_convert_star_interaction(interaction, user, character_name, star_type, stars, spend_positive, spend_negative)

@app_commands.command(name="staff_commands", description="(Staff) Show a quick list of staff commands and what they do.")
@in_guild_only()
@staff_only()
async def staff_commands(interaction: discord.Interaction):
    """Ephemeral staff help: lists core staff/admin actions (no legacy/internal/debug commands)."""
    await defer_ephemeral(interaction)

    # Keep this list intentionally small and player-friendly (but staff-only).
    items: list[tuple[str, str]] = [
        ("/add_character", "Add a new character for a player."),
        ("/character_archive", "Archive or unarchive a character (hide/show on the dashboard)."),
        ("/character_archive_by_id", "Archive/unarchive by user ID (for players who left the server)."),
        ("/award_points", "Award legacy points to a character (positive or negative)."),
        ("/add_ability", "Add a new ability to a character (does not spend stars)."),
        ("/upgrade_ability", "Upgrade an existing ability (costs legacy points; max 5 upgrades)."),
        ("/convert_points_to_stars", "Convert available legacy points into stars (ability or influence)."),
        ("/refresh_dashboard", "Force-refresh a player’s dashboard post right now."),
        ("/char_card", "Show a character card (ephemeral) exactly like the dashboard view."),
    ]

    lines = ["**Staff Commands**", ""]
    for cmd, desc in items:
        lines.append(f"• **{cmd}** — {desc}")
    await safe_reply(interaction, "\n".join(lines))


@app_commands.command(name="reset_points", description="(Staff) Set legacy/lifetime totals for a character (use for corrections).")
@in_guild_only()
@staff_only()
async def reset_points(interaction: discord.Interaction, user: discord.Member, character_name: str,
                       legacy_plus: Optional[int] = None, legacy_minus: Optional[int] = None,
                       lifetime_plus: Optional[int] = None, lifetime_minus: Optional[int] = None):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        character_name = character_name.strip()
        await run_db(require_character(interaction.client.db, interaction.guild.id, user.id, character_name), "require_character")
        await run_db(interaction.client.db.reset_points(interaction.guild.id, user.id, character_name, legacy_plus, legacy_minus, lifetime_plus, lifetime_minus), "reset_points")
        await log_to_channel(interaction.guild, f"🧾 {interaction.user.mention} reset points for **{character_name}** ({user.mention})")
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Reset complete. " + status)
    except Exception as e:
        LOG.exception("reset_points failed")
        await safe_reply(interaction, f"Reset failed: {e}")


@app_commands.command(name="reset_stars", description="(Staff) Set ability stars and/or influence stars for a character.")
@in_guild_only()
@staff_only()
async def reset_stars(interaction: discord.Interaction, user: discord.Member, character_name: str,
                      ability_stars: Optional[int] = None, influence_plus: Optional[int] = None, influence_minus: Optional[int] = None):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        character_name = character_name.strip()
        await run_db(require_character(interaction.client.db, interaction.guild.id, user.id, character_name), "require_character")
        await run_db(interaction.client.db.reset_stars(interaction.guild.id, user.id, character_name, ability_stars, influence_plus, influence_minus), "reset_stars")
        await log_to_channel(interaction.guild, f"⚖️ {interaction.user.mention} reset stars for **{character_name}** ({user.mention})")
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Stars set. " + status)
    except Exception as e:
        LOG.exception("reset_stars failed")
        await safe_reply(interaction, f"Reset failed: {e}")


@app_commands.command(name="add_ability", description="(Staff) Add an ability to a character (capacity = 2 + ability stars).")
@in_guild_only()
@staff_only()
async def add_ability(interaction: discord.Interaction, user: discord.Member, character_name: str, ability_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        character_name = character_name.strip()
        ability_name = ability_name.strip()
        await run_db(require_character(interaction.client.db, interaction.guild.id, user.id, character_name), "require_character")
        await run_db(interaction.client.db.add_ability(interaction.guild.id, user.id, character_name, ability_name), "add_ability")
        await log_to_channel(interaction.guild, f"🧩 {interaction.user.mention} added ability **{ability_name}** to **{character_name}** ({user.mention})")
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Ability added. " + status)
    except Exception as e:
        LOG.exception("add_ability failed")
        await safe_reply(interaction, f"Add ability failed: {e}")



@app_commands.command(name="upgrade_ability", description="(Staff) Spend 5 legacy points per upgrade (max 5 upgrades per ability). Requires explicit +/− split.")
@in_guild_only()
@staff_only()
async def upgrade_ability(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    ability_name: str,
    upgrades: int,
    pay_positive: int,
    pay_negative: int,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        character_name = character_name.strip()
        ability_name = ability_name.strip()

        if upgrades < 1:
            await safe_reply(interaction, "Upgrades must be >= 1.")
            return
        if pay_positive < 0 or pay_negative < 0:
            await safe_reply(interaction, "Payment values must be >= 0.")
            return

        await run_db(require_character(interaction.client.db, interaction.guild.id, user.id, character_name), "require_character")
        new_level, max_level = await run_db(
            interaction.client.db.upgrade_ability(
                interaction.guild.id,
                user.id,
                character_name,
                ability_name,
                upgrades,
                pay_positive,
                pay_negative,
            ),
            "upgrade_ability",
        )
        await log_to_channel(
            interaction.guild,
            f"🔧 {interaction.user.mention} upgraded **{ability_name}** on **{character_name}** ({user.mention}) -> {new_level}/{max_level} (paid +{pay_positive}/-{pay_negative})",
        )
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, f"Upgraded to level {new_level}/{max_level}. {status}")
    except Exception as e:
        LOG.exception("upgrade_ability failed")
        await safe_reply(interaction, f"Upgrade failed: {e}")



@app_commands.command(name="refresh_dashboard", description="(Staff) Force refresh the whole dashboard.")
@in_guild_only()
@staff_only()
async def refresh_dashboard(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        status = await refresh_all_dashboards(interaction.client, interaction.guild)
        await log_to_channel(interaction.guild, f"🔄 {interaction.user.mention} refreshed the dashboard")
        await safe_reply(interaction, status)
    except Exception as e:
        LOG.exception("refresh_dashboard failed")
        await safe_reply(interaction, f"Refresh failed: {e}")


@app_commands.command(name="char_card", description="Show a character card ephemerally.")
@in_guild_only()
async def char_card(interaction: discord.Interaction, character_name: str, user: Optional[discord.Member] = None):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        character_name = character_name.strip()
        target = user or interaction.user

        if user is not None:
            if not (isinstance(interaction.user, discord.Member) and is_staff(interaction.user)):
                embed = discord.Embed()
                embed.set_image(url='https://media.discordapp.net/attachments/1324994929176612936/1473872568191553568/1631280-doc_brown_full.jpg?ex=6997ca4b&is=699678cb&hm=fdc25510e3a9575ccf7f1cad504c577ac1d2a6b494e2810b0425c3b9211c8e7b&=&format=webp&width=869&height=856')
                await safe_reply(interaction, "You can only look up your own characters.", embed=embed)
                return

        await run_db(require_character(interaction.client.db, interaction.guild.id, target.id, character_name), "require_character")
        card = await run_db(build_character_card(interaction.client.db, interaction.guild.id, target.id, character_name), "build_character_card")
        embed = discord.Embed()
        embed.set_image(url='https://media.discordapp.net/attachments/1324994929176612936/1473872568191553568/1631280-doc_brown_full.jpg?ex=6997ca4b&is=699678cb&hm=fdc25510e3a9575ccf7f1cad504c577ac1d2a6b494e2810b0425c3b9211c8e7b&=&format=webp&width=869&height=856')
        await safe_reply(interaction, render_character_block(card), embed=embed)
    except Exception as e:
        LOG.exception("char_card failed")
        await safe_reply(interaction, f"Lookup failed: {e}")


# -----------------------------
# Bot client
# -----------------------------

class VilyraBotClient(discord.Client):
    def __init__(self, db: Database):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.db = db
        self.dashboard_limiter = SimpleRateLimiter(DASHBOARD_EDIT_MIN_INTERVAL)
    def _selfcheck(self) -> None:
        """Lightweight startup audit to prevent 'orphaned' functions after edits."""
        required_db_methods = [
            "connect", "close", "init_schema",
            "detect_schema",
            "add_character", "add_ability",
            "award_legacy", "convert_star", "spend_legacy",
            "upgrade_ability",
            "list_characters", "list_abilities", "list_player_ids",
            "get_player_rank", "set_player_rank",
            "reset_points", "reset_stars",
            "get_dashboard_entry", "get_dashboard_message_ids",
            "set_dashboard_message_ids", "clear_dashboard_message_ids",
            "get_latest_player_data_updated_at",
            "set_character_archived",
        ]
        missing = [m for m in required_db_methods if not hasattr(self.db, m)]
        if missing:
            LOG.error("SELF-CHECK FAILED: Database missing methods: %s", ", ".join(missing))
        else:
            LOG.info("Self-check: Database methods OK (%d checked).", len(required_db_methods))

        required_commands = [
            "set_server_rank", "add_character", "award_legacy_points",
            "convert_star", "reset_points", "reset_stars", "add_ability",
            "upgrade_ability", "refresh_dashboard", "char_card",
            "convert_points_to_stars",
            "staff_commands",
            "character_archive",
            "character_archive_by_id",
            "set_char_kingdom",
        ]
        present = {c.name for c in self.tree.get_commands()}
        missing_cmds = [c for c in required_commands if c not in present]
        if missing_cmds:
            LOG.error("SELF-CHECK FAILED: Command(s) not registered in tree: %s", ", ".join(missing_cmds))
        else:
            LOG.info("Self-check: Command tree OK (%d commands).", len(present))



    async def setup_hook(self) -> None:
        # Register commands
        self.tree.add_command(set_server_rank)
        self.tree.add_command(set_char_kingdom)
        self.tree.add_command(add_character)
        self.tree.add_command(character_archive)
        self.tree.add_command(character_archive_by_id)
        self.tree.add_command(character_delete)
        self.tree.add_command(award_legacy_points)
        self.tree.add_command(convert_star)
        self.tree.add_command(convert_points_to_stars)
        self.tree.add_command(staff_commands)
        self.tree.add_command(reset_points)
        self.tree.add_command(reset_stars)
        self.tree.add_command(add_ability)
        self.tree.add_command(upgrade_ability)
        self.tree.add_command(refresh_dashboard)
        self.tree.add_command(char_card)

        LOG.info("Command tree prepared: %s command(s); GUILD_ID=%s", len(self.tree.get_commands()), os.getenv("GUILD_ID"))

        self._selfcheck()

        # Sync commands
        try:
            gid = safe_int(os.getenv("GUILD_ID"), 0)
            if gid:
                guild_obj = discord.Object(id=gid)

                # IMPORTANT: Clear + resync to prevent Discord-side signature mismatches from older deployments.
                # This is safe because we always re-upload the current command set immediately after clearing.
                self.tree.clear_commands(guild=guild_obj)
                await self.tree.sync(guild=guild_obj)

                self.tree.copy_global_to(guild=guild_obj)
                synced = await self.tree.sync(guild=guild_obj)
                LOG.info("Guild sync succeeded: %s commands", len(synced))

                if len(synced)==0:
                    LOG.warning("Guild sync returned 0 commands; attempting global sync fallback...")
                    try:
                        synced2 = await self.tree.sync()
                        LOG.info("Global sync fallback succeeded: %s commands", len(synced2))
                        try:
                            # As a last resort, clear guild commands and re-sync to resolve "outdated" command schemas.
                            self.tree.clear_commands(guild=discord.Object(id=gid))
                            self.tree.copy_global_to(guild=discord.Object(id=gid))
                            synced3 = await self.tree.sync(guild=discord.Object(id=gid))
                            LOG.info("Guild re-sync after clear succeeded: %s commands", len(synced3))
                        except Exception as ex3:
                            LOG.warning("Guild re-sync after clear failed: %s", ex3)
                    except Exception as ex2:
                        LOG.exception("Global sync fallback failed: %s", ex2)
                if len(synced)==0:
                    LOG.warning("Guild sync returned 0 commands. Check GUILD_ID, bot invite scopes (applications.commands), and that commands are not pending global propagation.")
            else:
                synced = await self.tree.sync()
                LOG.info("Global sync succeeded: %s commands", len(synced))
                if len(synced)==0:
                    LOG.warning("Global sync returned 0 commands. This can be normal if no changes were detected, but if commands are missing, check bot invite scopes and app command permissions.")
        except Exception:
            LOG.exception("Command sync failed")

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")
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
