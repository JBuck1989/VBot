# VB_v101 — Vilyra Legacy Bot (Railway + Postgres) — FULL REPLACEMENT
# Goals:
# - Clean, import-safe file (no stray decorators / missing symbols / bad indentation)
# - Additive-only DB changes (no drops/renames)
# - Stable, guild-wide character autocomplete that actually returns options
# - Commands wired correctly to DB methods (no signature mismatches)
# - Startup dashboard refresh that will not crash the bot on missing data
#
# Tables expected (existing):
#   characters(guild_id, user_id, name, archived, legacy_plus, legacy_minus, lifetime_plus, lifetime_minus,
#              ability_stars, influence_minus, influence_plus, created_at, updated_at, kingdom)
#   abilities(guild_id, user_id, character_name, ability_name, upgrades, created_at, updated_at, name, upgrade_level, level)
#
# NOTE: character names are unique across the guild (per user). We therefore use character name as the autocomplete value
# and resolve owner user_id from characters table.

from __future__ import annotations

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
# Logging
# -----------------------------

LOG = logging.getLogger("VilyraBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VilyraBot: %(message)s")


# -----------------------------
# Config
# -----------------------------

DEFAULT_DASHBOARD_CHANNEL_ID = 1469879866655768738
DEFAULT_COMMAND_LOG_CHANNEL_ID = 1469879960729817098

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

KINGDOMS: List[str] = ["Velarith", "Lyvik", "Baelon", "Sethrathiel", "Avalea"]
KINGDOM_CHOICES: List[app_commands.Choice[str]] = [app_commands.Choice(name=k, value=k) for k in KINGDOMS]

MAX_ABILITY_STARS = 5
MAX_INFL_STARS_TOTAL = 5

STAR_COST = 10
UPGRADE_COST = 5

REP_MIN = -100
REP_MAX = 100

# Dashboard rendering version bump forces refresh even if data unchanged.
DASHBOARD_TEMPLATE_VERSION = 1

# Throttle dashboard edits
DASHBOARD_EDIT_MIN_INTERVAL = float(os.getenv("DASHBOARD_EDIT_MIN_INTERVAL", "1.2"))
PLAYER_POST_SOFT_LIMIT = 1900

BORDER_LEN = 20
PLAYER_BORDER = "═" * BORDER_LEN
CHAR_SEPARATOR = "-" * BORDER_LEN
CHAR_HEADER_LEFT = "꧁•⊹٭ "
CHAR_HEADER_RIGHT = " ٭⊹•꧂"


# -----------------------------
# Small helpers
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
    return lo if n < lo else hi if n > hi else n

def db_timeout() -> float:
    return float(os.getenv("DB_TIMEOUT_SECONDS", "8.0"))

async def defer_ephemeral(interaction: discord.Interaction) -> None:
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True, thinking=False)
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

async def send_error(interaction: discord.Interaction, error: Exception | str) -> None:
    await safe_reply(interaction, f"❌ {str(error)}")

async def run_db(coro, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=db_timeout())
    except asyncio.TimeoutError as e:
        raise RuntimeError(f"Database operation timed out ({label}).") from e

def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

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
# Database Layer
# -----------------------------

class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg.AsyncConnection] = None

        self.abilities_cols: set[str] = set()
        self.abilities_level_col: str = "upgrade_level"
        self.abilities_char_col: str = "character_name"
        self.abilities_updated_at_col: Optional[str] = None

        self.characters_cols: set[str] = set()

    async def connect(self) -> None:
        LOG.info("Connecting to PostgreSQL...")
        self._conn = await psycopg.AsyncConnection.connect(
            self._dsn, autocommit=True, row_factory=dict_row
        )
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

        LOG.info("Detected characters columns: %s",
                 ", ".join(sorted(self.characters_cols)) if self.characters_cols else "(none)")
        LOG.info("Detected abilities columns: %s",
                 ", ".join(sorted(self.abilities_cols)) if self.abilities_cols else "(none)")

        if "character_name" in self.abilities_cols:
            self.abilities_char_col = "character_name"
        elif "name" in self.abilities_cols:
            self.abilities_char_col = "name"
        else:
            self.abilities_char_col = "character_name"

        if "upgrade_level" in self.abilities_cols:
            self.abilities_level_col = "upgrade_level"
        elif "level" in self.abilities_cols:
            self.abilities_level_col = "level"
        else:
            self.abilities_level_col = "upgrade_level"

        self.abilities_updated_at_col = "updated_at" if "updated_at" in self.abilities_cols else None

        LOG.info("Schema choices: abilities.%s as level, abilities.%s as character key",
                 self.abilities_level_col, self.abilities_char_col)

    async def init_schema(self) -> None:
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

        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                ability_name   TEXT NOT NULL,
                upgrade_level  INT  NOT NULL DEFAULT 0,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS character_name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS ability_name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrades INT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrade_level INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS level INT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        try:
            await self._execute("CREATE UNIQUE INDEX IF NOT EXISTS characters_unique ON characters (guild_id, user_id, name);")
        except Exception:
            LOG.exception("Could not create unique index on characters; continuing")
        try:
            await self._execute("CREATE INDEX IF NOT EXISTS abilities_lookup ON abilities (guild_id, user_id, character_name, ability_name);")
        except Exception:
            LOG.exception("Could not create abilities index; continuing")

        await self.detect_schema()
        LOG.info("Database schema initialized / updated")

    async def get_player_rank(self, guild_id: int, user_id: int) -> str:
        row = await self._fetchone(
            "SELECT server_rank FROM players WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )
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

    async def list_player_ids(self, guild_id: int) -> List[int]:
        try:
            rows = await self._fetchall(
                "SELECT user_id FROM players WHERE guild_id=%s ORDER BY user_id ASC;",
                (guild_id,),
            )
            ids = [int(r["user_id"]) for r in rows if r and r.get("user_id") is not None]
            if ids:
                return ids
        except Exception:
            LOG.debug("players table missing/empty; falling back to characters", exc_info=True)

        rows2 = await self._fetchall(
            "SELECT DISTINCT user_id FROM characters WHERE guild_id=%s ORDER BY user_id ASC;",
            (guild_id,),
        )
        return [int(r["user_id"]) for r in rows2 if r and r.get("user_id") is not None]

    async def add_character(self, guild_id: int, user_id: int, name: str, kingdom: Optional[str] = None) -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("Character name cannot be empty.")
        kingdom = (kingdom or "").strip() or None

        await self._execute(
            """
            INSERT INTO characters (
                guild_id, user_id, name, kingdom, archived,
                legacy_plus, legacy_minus, lifetime_plus, lifetime_minus,
                influence_plus, influence_minus, ability_stars, updated_at
            )
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

    async def character_exists(self, guild_id: int, user_id: int, name: str, *, include_archived: bool = False) -> bool:
        name = (name or "").strip()
        if not name:
            return False
        if include_archived:
            row = await self._fetchone(
                "SELECT 1 FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s LIMIT 1;",
                (guild_id, user_id, name),
            )
        else:
            row = await self._fetchone(
                "SELECT 1 FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s AND COALESCE(archived,FALSE)=FALSE LIMIT 1;",
                (guild_id, user_id, name),
            )
        return bool(row)

    async def resolve_character_owner(self, guild_id: int, name: str) -> Tuple[int, str]:
        name = (name or "").strip()
        if not name:
            raise ValueError("Character name is required.")
        row = await self._fetchone(
            "SELECT user_id, name FROM characters WHERE guild_id=%s AND lower(name)=lower(%s) LIMIT 1;",
            (guild_id, name),
        )
        if not row:
            raise ValueError("No matching character found in this guild.")
        return int(row["user_id"]), str(row["name"])

    async def list_characters(self, guild_id: int, user_id: int) -> List[str]:
        rows = await self._fetchall(
            """
            SELECT name
            FROM characters
            WHERE guild_id=%s AND user_id=%s AND COALESCE(archived,FALSE)=FALSE
            ORDER BY created_at ASC, name ASC;
            """,
            (guild_id, user_id),
        )
        return [str(r["name"]) for r in rows if r and r.get("name")]

    async def list_all_characters_for_guild(
        self,
        guild_id: int,
        *,
        include_archived: bool = True,
        name_filter: str = "",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        name_filter = (name_filter or "").strip()
        lim = max(1, min(int(limit or 25), 200))

        where = ["guild_id=%s"]
        params: List[Any] = [guild_id]

        if not include_archived:
            where.append("COALESCE(archived,FALSE)=FALSE")
        if name_filter:
            where.append("name ILIKE %s")
            params.append(f"%{name_filter}%")

        sql = f"""
            SELECT user_id, name, COALESCE(archived,FALSE) AS archived
            FROM characters
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(archived,FALSE) ASC, name ASC
            LIMIT {lim};
        """
        return await self._fetchall(sql, tuple(params))

    async def set_character_archived(self, guild_id: int, user_id: int, name: str, archived: bool) -> bool:
        rc = await self._execute(
            "UPDATE characters SET archived=%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (archived, guild_id, user_id, name),
        )
        return rc > 0

    async def delete_character(self, guild_id: int, user_id: int, name: str) -> bool:
        char_col = self.abilities_char_col
        await self._execute(
            f"DELETE FROM abilities WHERE guild_id=%s AND user_id=%s AND {char_col}=%s;",
            (guild_id, user_id, name),
        )
        rc = await self._execute(
            "DELETE FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (guild_id, user_id, name),
        )
        return rc > 0

    async def set_character_kingdom(self, guild_id: int, user_id: int, name: str, kingdom: str) -> bool:
        rc = await self._execute(
            "UPDATE characters SET kingdom=%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (kingdom, guild_id, user_id, name),
        )
        return rc > 0

    async def rename_character(self, guild_id: int, user_id: int, old_name: str, new_name: str) -> bool:
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()
        if not old_name or not new_name:
            raise ValueError("Both old and new names are required.")
        if old_name.lower() == new_name.lower():
            return True

        conn = self._require_conn()
        async with conn.transaction():
            exists = await self._fetchone(
                "SELECT 1 FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s LIMIT 1;",
                (guild_id, user_id, old_name),
            )
            if not exists:
                return False

            collision = await self._fetchone(
                "SELECT 1 FROM characters WHERE guild_id=%s AND lower(name)=lower(%s) LIMIT 1;",
                (guild_id, new_name),
            )
            if collision:
                raise ValueError("A character with that name already exists in this guild.")

            rc = await self._execute(
                "UPDATE characters SET name=%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (new_name, guild_id, user_id, old_name),
            )
            if rc <= 0:
                return False

            char_col = self.abilities_char_col
            if self.abilities_updated_at_col:
                await self._execute(
                    f"UPDATE abilities SET {char_col}=%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND {char_col}=%s;",
                    (new_name, guild_id, user_id, old_name),
                )
            else:
                await self._execute(
                    f"UPDATE abilities SET {char_col}=%s WHERE guild_id=%s AND user_id=%s AND {char_col}=%s;",
                    (new_name, guild_id, user_id, old_name),
                )
            return True

    async def get_character_state(self, guild_id: int, user_id: int, name: str) -> Dict[str, Any]:
        row = await self._fetchone(
            """
            SELECT legacy_plus, legacy_minus, lifetime_plus, lifetime_minus,
                   influence_plus, influence_minus, ability_stars, kingdom
            FROM characters
            WHERE guild_id=%s AND user_id=%s AND name=%s AND COALESCE(archived,FALSE)=FALSE
            LIMIT 1;
            """,
            (guild_id, user_id, name),
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
            WHERE guild_id=%s AND user_id=%s AND name=%s AND COALESCE(archived,FALSE)=FALSE;
            """,
            (pos, neg, pos, neg, guild_id, user_id, name),
        )

    async def reset_points(
        self,
        guild_id: int,
        user_id: int,
        name: str,
        legacy_plus: int,
        legacy_minus: int,
        lifetime_plus: int,
        lifetime_minus: int,
    ) -> None:
        await self._execute(
            """
            UPDATE characters
            SET legacy_plus=%s, legacy_minus=%s, lifetime_plus=%s, lifetime_minus=%s, updated_at=NOW()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (max(0, int(legacy_plus)), max(0, int(legacy_minus)), max(0, int(lifetime_plus)), max(0, int(lifetime_minus)),
             guild_id, user_id, name),
        )

    async def reset_stars(
        self,
        guild_id: int,
        user_id: int,
        name: str,
        ability_stars: Optional[int],
        influence_plus: Optional[int],
        influence_minus: Optional[int],
    ) -> None:
        st = await self.get_character_state(guild_id, user_id, name)
        a = st["ability_stars"] if ability_stars is None else clamp(int(ability_stars), 0, MAX_ABILITY_STARS)
        ip = st["influence_plus"] if influence_plus is None else clamp(int(influence_plus), 0, MAX_INFL_STARS_TOTAL)
        im = st["influence_minus"] if influence_minus is None else clamp(int(influence_minus), 0, MAX_INFL_STARS_TOTAL)
        if ip + im > MAX_INFL_STARS_TOTAL:
            raise ValueError("Total influence stars (pos+neg) cannot exceed 5.")
        await self._execute(
            """
            UPDATE characters
            SET ability_stars=%s, influence_plus=%s, influence_minus=%s, updated_at=NOW()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (a, ip, im, guild_id, user_id, name),
        )

    async def convert_star(
        self,
        guild_id: int,
        user_id: int,
        name: str,
        star_type: Literal["ability", "influence_positive", "influence_negative"],
        stars: int,
        spend_plus: int,
        spend_minus: int,
    ) -> None:
        star_type = (star_type or "").strip().lower()
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

        await self._execute(
            "UPDATE characters SET legacy_plus=legacy_plus-%s, legacy_minus=legacy_minus-%s, updated_at=NOW() "
            "WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (spend_plus, spend_minus, guild_id, user_id, name),
        )

        if star_type == "ability":
            await self._execute(
                "UPDATE characters SET ability_stars=ability_stars+%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (stars, guild_id, user_id, name),
            )
        elif star_type == "influence_positive":
            await self._execute(
                "UPDATE characters SET influence_plus=influence_plus+%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (stars, guild_id, user_id, name),
            )
        else:
            await self._execute(
                "UPDATE characters SET influence_minus=influence_minus+%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                (stars, guild_id, user_id, name),
            )

    def _ability_level_expr(self) -> str:
        if self.abilities_level_col == "upgrade_level":
            return "COALESCE(upgrade_level, 0) AS upgrade_level"
        return "COALESCE(level, 0) AS upgrade_level"

    def _ability_char_col(self) -> str:
        return self.abilities_char_col

    async def list_abilities(self, guild_id: int, user_id: int, name: str) -> List[Tuple[str, int]]:
        char_col = self._ability_char_col()
        rows = await self._fetchall(
            f"""
            SELECT ability_name, {self._ability_level_expr()}
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND {char_col}=%s
            ORDER BY created_at ASC, ability_name ASC;
            """,
            (guild_id, user_id, name),
        )
        out: List[Tuple[str, int]] = []
        for r in rows:
            if r and r.get("ability_name"):
                out.append((str(r["ability_name"]), safe_int(r.get("upgrade_level"), 0)))
        return out

    async def add_ability(self, guild_id: int, user_id: int, name: str, ability_name: str) -> None:
        ability_name = (ability_name or "").strip()
        if not ability_name:
            raise ValueError("Ability name cannot be empty.")

        st = await self.get_character_state(guild_id, user_id, name)
        current = await self.list_abilities(guild_id, user_id, name)
        cap = 2 + clamp(st["ability_stars"], 0, MAX_ABILITY_STARS)
        if len(current) >= cap:
            raise ValueError(f"Ability capacity reached ({len(current)}/{cap}). Earn more Ability Stars to add abilities.")

        char_col = self._ability_char_col()
        cols = ["guild_id", "user_id", char_col, "ability_name", "created_at"]
        vals = ["%s", "%s", "%s", "%s", "NOW()"]
        params: List[Any] = [guild_id, user_id, name, ability_name]

        if "upgrade_level" in self.abilities_cols:
            cols.append("upgrade_level")
            vals.append("0")
        if "level" in self.abilities_cols:
            cols.append("level")
            vals.append("0")
        if "updated_at" in self.abilities_cols:
            cols.append("updated_at")
            vals.append("NOW()")

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
        ability_name = (ability_name or "").strip()
        upgrades = max(1, int(upgrades))
        pay_positive = max(0, int(pay_positive))
        pay_negative = max(0, int(pay_negative))

        total_cost = upgrades * UPGRADE_COST
        if pay_positive + pay_negative != total_cost:
            raise ValueError(f"Payment must equal {total_cost} points total (5 per upgrade).")

        char_col = self._ability_char_col()
        level_col = self.abilities_level_col

        row = await self._fetchone(
            f"""
            SELECT COALESCE({level_col}, 0) AS cur_level
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND ability_name=%s
            ORDER BY created_at ASC
            LIMIT 1;
            """,
            (guild_id, user_id, name, ability_name),
        )
        if not row:
            raise ValueError("Ability not found. Add it first with /add_ability.")
        cur_level = safe_int(row.get("cur_level"), 0)

        max_level = 5
        if cur_level >= max_level:
            raise ValueError(f"Upgrade limit reached ({cur_level}/{max_level}).")

        remaining = max_level - cur_level
        if upgrades > remaining:
            raise ValueError(f"Only {remaining} upgrade(s) remaining for this ability (max {max_level}).")

        st = await self.get_character_state(guild_id, user_id, name)
        if st["legacy_plus"] < pay_positive:
            raise ValueError(f"Not enough available positive points (need {pay_positive}, have {st['legacy_plus']}).")
        if st["legacy_minus"] < pay_negative:
            raise ValueError(f"Not enough available negative points (need {pay_negative}, have {st['legacy_minus']}).")

        await self._execute(
            "UPDATE characters SET legacy_plus=legacy_plus-%s, legacy_minus=legacy_minus-%s, updated_at=NOW() "
            "WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (pay_positive, pay_negative, guild_id, user_id, name),
        )

        new_level = cur_level + upgrades

        sets: List[str] = [f"{level_col}=%s"]
        params: List[Any] = [new_level]

        if level_col != "upgrade_level" and "upgrade_level" in self.abilities_cols:
            sets.append("upgrade_level=%s")
            params.append(new_level)
        if level_col != "level" and "level" in self.abilities_cols:
            sets.append("level=%s")
            params.append(new_level)
        if self.abilities_updated_at_col:
            sets.append("updated_at=NOW()")

        params.extend([guild_id, user_id, name, ability_name])

        sql = f"UPDATE abilities SET {', '.join(sets)} WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND ability_name=%s;"
        await self._execute(sql, params)
        return new_level, max_level

    async def get_dashboard_entry(self, guild_id: int, user_id: int) -> Tuple[List[int], Optional[str], Optional[Any], int]:
        row = await self._fetchone(
            """
            SELECT message_ids, content_hash, updated_at, COALESCE(template_version, 0) AS template_version
            FROM dashboard_messages
            WHERE guild_id=%s AND user_id=%s;
            """,
            (guild_id, user_id),
        )
        ids = parse_ids(row["message_ids"]) if row and row.get("message_ids") else []
        h = str(row["content_hash"]) if row and row.get("content_hash") else None
        ts = row["updated_at"] if row and row.get("updated_at") else None
        tv = int(row["template_version"]) if row and row.get("template_version") is not None else 0
        return ids, h, ts, tv

    async def set_dashboard_message_ids(
        self,
        guild_id: int,
        user_id: int,
        channel_id: int,
        ids: List[int],
        h: Optional[str],
    ) -> None:
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

    async def clear_dashboard_entry(self, guild_id: int, user_id: int) -> None:
        await self._execute(
            "DELETE FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )

    async def get_latest_player_data_updated_at(self, guild_id: int, user_id: int) -> Optional[Any]:
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


# -----------------------------
# Autocomplete + resolvers
# -----------------------------

async def autocomplete_character_guild(
    interaction: discord.Interaction,
    current: str,
) -> List[app_commands.Choice[str]]:
    try:
        guild = interaction.guild
        if guild is None:
            return []
        q = (current or "").strip()
        rows = await interaction.client.db.list_all_characters_for_guild(
            guild.id,
            include_archived=True,
            name_filter=q,
            limit=25,
        )
        choices: List[app_commands.Choice[str]] = []
        for r in rows:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            label = name
            if len(label) > 100:
                label = label[:97] + "..."
            choices.append(app_commands.Choice(name=label, value=name))
        return choices
    except Exception:
        LOG.exception("Character autocomplete failed")
        return []

async def resolve_character_input(interaction: discord.Interaction, character_name: str) -> Tuple[int, str]:
    guild = interaction.guild
    if guild is None:
        raise ValueError("This command must be used in a server.")
    return await interaction.client.db.resolve_character_owner(guild.id, character_name)


# -----------------------------
# Dashboard renderers
# -----------------------------

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
    abilities: List[Tuple[str, int]]

async def build_character_card(db: Database, guild_id: int, user_id: int, name: str) -> CharacterCard:
    st = await db.get_character_state(guild_id, user_id, name)
    abilities = await db.list_abilities(guild_id, user_id, name)
    return CharacterCard(
        name=name,
        kingdom=(st.get("kingdom") or ""),
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
    lines.append(f"{CHAR_HEADER_LEFT}**{card.name}**{CHAR_HEADER_RIGHT}")
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

    try:
        latest_ts = await db.get_latest_player_data_updated_at(guild.id, user_id)
        if stored_tv == DASHBOARD_TEMPLATE_VERSION and dash_ts and latest_ts and latest_ts <= dash_ts:
            return "skipped"
    except Exception as ex:
        LOG.warning("Could not compute latest player ts for user_id=%s: %s", user_id, ex)

    if not chars:
        for mid in stored_ids:
            try:
                m = await channel.fetch_message(mid)
                await m.delete()
            except Exception:
                pass
        await db.clear_dashboard_entry(guild.id, user_id)
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
        await db.set_dashboard_message_ids(guild.id, user_id, channel.id, [msg.id], new_hash)
        if len(stored_ids) > 1:
            for extra_id in stored_ids[1:]:
                try:
                    extra_msg = await channel.fetch_message(extra_id)
                    await extra_msg.delete()
                except Exception:
                    pass
        return f"Dashboard updated for user_id={user_id}."


async def refresh_all_dashboards(client: "VilyraBotClient", guild: discord.Guild) -> str:
    try:
        user_ids = await client.db.list_player_ids(guild.id)
    except Exception:
        LOG.exception("list_player_ids failed during refresh_all_dashboards")
        user_ids = []

    if not user_ids:
        return "No players with characters yet."

    ok = 0
    for uid in user_ids:
        try:
            await refresh_player_dashboard(client, guild, uid)
            ok += 1
            await asyncio.sleep(0.2)
        except Exception:
            LOG.exception("Failed refreshing dashboard for uid=%s", uid)
    return f"Refreshed dashboards for {ok} player(s)."


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

def _staff_allowlist() -> set[int]:
    ids: set[int] = set()
    raw = (os.getenv("STAFF_USER_IDS") or "").strip()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return ids

def is_staff(member: discord.Member) -> bool:
    if member.id in _staff_allowlist():
        return True
    if member.guild.owner_id == member.id:
        return True
    perms = member.guild_permissions
    return bool(perms.administrator or perms.manage_guild)

def staff_only(func=None):
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False
        m = interaction.user
        if not isinstance(m, discord.Member):
            m = interaction.guild.get_member(interaction.user.id)  # type: ignore
        if isinstance(m, discord.Member) and is_staff(m):
            return True
        await safe_reply(interaction, "You don't have permission to use this command.")
        return False
    decorator = app_commands.check(predicate)
    if callable(func):
        return decorator(func)
    def wrapper(f):
        return decorator(f)
    return wrapper

async def require_character(db: Database, guild_id: int, user_id: int, name: str) -> None:
    if not await db.character_exists(guild_id, user_id, name, include_archived=True):
        raise ValueError("Character not found.")


# -----------------------------
# Slash commands
# -----------------------------

@app_commands.command(name="set_server_rank", description="(Staff) Set a player's server rank.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", rank="Rank name")
async def set_server_rank(interaction: discord.Interaction, user: discord.Member, rank: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        rank = (rank or "").strip()
        if rank not in SERVER_RANKS:
            await safe_reply(interaction, "Invalid rank. Options: " + ", ".join(SERVER_RANKS))
            return
        await run_db(interaction.client.db.set_player_rank(interaction.guild.id, user.id, rank), "set_player_rank")
        await log_to_channel(interaction.guild, f"🏷️ {interaction.user.mention} set server rank for {user.mention} to **{rank}**")
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, "Rank set. " + status)
    except Exception as e:
        LOG.exception("set_server_rank failed")
        await send_error(interaction, e)


@app_commands.command(name="add_character", description="(Staff) Add a character for a user.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", character_name="Character name", kingdom="Optional kingdom")
@app_commands.choices(kingdom=KINGDOM_CHOICES)
async def add_character(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    kingdom: Optional[app_commands.Choice[str]] = None,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        k = kingdom.value if kingdom else None
        await run_db(interaction.client.db.add_character(interaction.guild.id, user.id, character_name, k), "add_character")
        await log_to_channel(interaction.guild, f"➕ {interaction.user.mention} added character **{character_name}** for {user.mention}")
        await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_reply(interaction, f"✅ Added **{character_name}** for {user.mention}.")
    except Exception as e:
        LOG.exception("add_character failed")
        await send_error(interaction, e)


@app_commands.command(name="set_char_kingdom", description="(Staff) Set a character's home kingdom.")
@in_guild_only()
@staff_only()
@app_commands.describe(character_name="Character (autocomplete)", kingdom="New kingdom")
@app_commands.autocomplete(character_name=autocomplete_character_guild)
@app_commands.choices(kingdom=KINGDOM_CHOICES)
async def set_char_kingdom(interaction: discord.Interaction, character_name: str, kingdom: app_commands.Choice[str]):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        ok = await run_db(interaction.client.db.set_character_kingdom(interaction.guild.id, user_id, cname, kingdom.value), "set_character_kingdom")
        if not ok:
            raise RuntimeError("Character not updated.")
        await refresh_all_dashboards(interaction.client, interaction.guild)
        embed = discord.Embed(title="Kingdom updated")
        embed.add_field(name="Character", value=cname, inline=False)
        embed.add_field(name="Kingdom", value=kingdom.value, inline=False)
        await safe_reply(interaction, "", embed=embed)
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="character_archive", description="(Staff) Archive/unarchive a character (hide/show on dashboard).")
@in_guild_only()
@staff_only()
@app_commands.describe(character_name="Character (autocomplete)", action="Archive or Unarchive")
@app_commands.choices(action=[
    app_commands.Choice(name="Archive", value="archive"),
    app_commands.Choice(name="Unarchive", value="unarchive"),
])
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def character_archive(interaction: discord.Interaction, character_name: str, action: app_commands.Choice[str]):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        archived = action.value == "archive"
        updated = await run_db(interaction.client.db.set_character_archived(interaction.guild.id, user_id, cname, archived), "set_character_archived")
        if not updated:
            raise RuntimeError("Character not found.")
        await refresh_all_dashboards(interaction.client, interaction.guild)
        embed = discord.Embed(title="Character updated")
        embed.add_field(name="Character", value=cname, inline=False)
        embed.add_field(name="Archived", value=str(archived), inline=False)
        await safe_reply(interaction, "", embed=embed)
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="character_delete", description="(Staff) Delete a character (cannot be undone).")
@in_guild_only()
@staff_only()
@app_commands.describe(character_name="Character (autocomplete)")
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def character_delete(interaction: discord.Interaction, character_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        deleted = await run_db(interaction.client.db.delete_character(interaction.guild.id, user_id, cname), "delete_character")
        if not deleted:
            raise RuntimeError("Character not found.")
        await refresh_all_dashboards(interaction.client, interaction.guild)
        embed = discord.Embed(title="Character deleted")
        embed.add_field(name="Character", value=cname, inline=False)
        await safe_reply(interaction, "", embed=embed)
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="character_rename", description="(Staff) Rename a character (preserves stats; updates abilities).")
@in_guild_only()
@staff_only()
@app_commands.describe(character_name="Character (autocomplete)", new_name="New character name")
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def character_rename(interaction: discord.Interaction, character_name: str, new_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, old_name = await resolve_character_input(interaction, character_name)
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("New name cannot be empty.")
        ok = await run_db(interaction.client.db.rename_character(interaction.guild.id, user_id, old_name, new_name), "rename_character")
        if not ok:
            raise RuntimeError("Character not found.")
        await refresh_all_dashboards(interaction.client, interaction.guild)
        await safe_reply(interaction, f"✅ Renamed **{old_name}** → **{new_name}**.")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="award_points", description="(Staff) Award legacy points to a character (+ and/or -).")
@in_guild_only()
@staff_only()
@app_commands.describe(character_name="Character (autocomplete)", positive="Positive points to add", negative="Negative points to add")
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def award_points(interaction: discord.Interaction, character_name: str, positive: int = 0, negative: int = 0):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        await run_db(interaction.client.db.award_legacy(interaction.guild.id, user_id, cname, positive, negative), "award_legacy")
        await refresh_all_dashboards(interaction.client, interaction.guild)
        await log_to_channel(interaction.guild, f"🎁 {interaction.user.mention} awarded **{cname}** points: +{positive} / -{negative}")
        await safe_reply(interaction, f"✅ Awarded **{cname}**: +{positive} / -{negative}.")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="convert_star", description="(Staff) Convert available legacy points into stars (10 points per star).")
@in_guild_only()
@staff_only()
@app_commands.choices(star_type=[
    app_commands.Choice(name="Ability Star", value="ability"),
    app_commands.Choice(name="Positive Influence Star", value="influence_positive"),
    app_commands.Choice(name="Negative Influence Star", value="influence_negative"),
])
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def convert_star(
    interaction: discord.Interaction,
    character_name: str,
    star_type: app_commands.Choice[str],
    stars: int,
    spend_plus: int,
    spend_minus: int,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        await run_db(
            interaction.client.db.convert_star(interaction.guild.id, user_id, cname, star_type.value, stars, spend_plus, spend_minus),
            "convert_star",
        )
        await refresh_all_dashboards(interaction.client, interaction.guild)
        embed = discord.Embed(title="Converted points to stars")
        embed.add_field(name="Character", value=cname, inline=False)
        embed.add_field(name="Star type", value=star_type.name, inline=True)
        embed.add_field(name="Stars", value=str(stars), inline=True)
        embed.add_field(name="Spent", value=f"+{spend_plus} / -{spend_minus}", inline=False)
        await safe_reply(interaction, "", embed=embed)
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="reset_points", description="(Staff) Set legacy/lifetime totals for a character (corrections).")
@in_guild_only()
@staff_only()
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def reset_points(
    interaction: discord.Interaction,
    character_name: str,
    legacy_plus: int = 0,
    legacy_minus: int = 0,
    lifetime_plus: int = 0,
    lifetime_minus: int = 0,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        await run_db(
            interaction.client.db.reset_points(interaction.guild.id, user_id, cname, legacy_plus, legacy_minus, lifetime_plus, lifetime_minus),
            "reset_points",
        )
        await refresh_all_dashboards(interaction.client, interaction.guild)
        await safe_reply(interaction, f"✅ Reset **{cname}** points.")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="reset_stars", description="(Staff) Set ability/influence stars for a character (corrections).")
@in_guild_only()
@staff_only()
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def reset_stars(
    interaction: discord.Interaction,
    character_name: str,
    ability_stars: Optional[int] = None,
    influence_plus: Optional[int] = None,
    influence_minus: Optional[int] = None,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        await run_db(
            interaction.client.db.reset_stars(interaction.guild.id, user_id, cname, ability_stars, influence_plus, influence_minus),
            "reset_stars",
        )
        await refresh_all_dashboards(interaction.client, interaction.guild)
        await safe_reply(interaction, f"✅ Reset **{cname}** stars.")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="add_ability", description="(Staff) Add an ability to a character (capacity = 2 + ability stars).")
@in_guild_only()
@staff_only()
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def add_ability(interaction: discord.Interaction, character_name: str, ability_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        await run_db(interaction.client.db.add_ability(interaction.guild.id, user_id, cname, ability_name), "add_ability")
        await refresh_all_dashboards(interaction.client, interaction.guild)
        await safe_reply(interaction, f"✅ Added ability **{ability_name.strip()}** to **{cname}**.")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="upgrade_ability", description="(Staff) Upgrade an ability (max 5). Costs 5 legacy points per upgrade.")
@in_guild_only()
@staff_only()
@app_commands.autocomplete(character_name=autocomplete_character_guild)
@app_commands.describe(upgrades="How many upgrades to apply", pay_positive="Spend from + pool", pay_negative="Spend from - pool")
async def upgrade_ability(
    interaction: discord.Interaction,
    character_name: str,
    ability_name: str,
    upgrades: int,
    pay_positive: int,
    pay_negative: int,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)
        await run_db(require_character(interaction.client.db, interaction.guild.id, user_id, cname), "require_character")
        new_level, max_level = await run_db(
            interaction.client.db.upgrade_ability(interaction.guild.id, user_id, cname, ability_name, upgrades, pay_positive, pay_negative),
            "upgrade_ability",
        )
        await refresh_all_dashboards(interaction.client, interaction.guild)
        await safe_reply(interaction, f"✅ Upgraded **{ability_name.strip()}** for **{cname}** to **{new_level}/{max_level}**.")
    except Exception as e:
        await send_error(interaction, e)


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
        await send_error(interaction, e)


@app_commands.command(name="debug_characters", description="(Staff) Debug: show character counts for this guild.")
@in_guild_only()
@staff_only()
async def debug_characters(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        rows = await interaction.client.db._fetchall(
            "SELECT COALESCE(archived,FALSE) AS archived, COUNT(*) AS n FROM characters WHERE guild_id=%s GROUP BY COALESCE(archived,FALSE) ORDER BY COALESCE(archived,FALSE);",
            (interaction.guild.id,),
        )
        total = 0
        parts = []
        for r in rows:
            n = int(r["n"])
            total += n
            parts.append(f"archived={bool(r['archived'])}: {n}")
        msg = " | ".join(parts) if parts else "no rows"
        await safe_reply(interaction, f"Guild {interaction.guild.id}: characters total={total} ({msg})")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="char_card", description="Show a character card ephemerally (same format as dashboard).")
@in_guild_only()
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def char_card(interaction: discord.Interaction, character_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        user_id, cname = await resolve_character_input(interaction, character_name)

        member = interaction.user if isinstance(interaction.user, discord.Member) else interaction.guild.get_member(interaction.user.id)
        if not isinstance(member, discord.Member):
            raise RuntimeError("Could not resolve member.")
        if user_id != member.id and not is_staff(member):
            await safe_reply(interaction, "You can only view your own characters.")
            return

        card = await build_character_card(interaction.client.db, interaction.guild.id, user_id, cname)
        text = render_character_block(card)
        await safe_reply(interaction, text)
    except Exception as e:
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
        self._did_hard_sync = False

    async def setup_hook(self) -> None:
        for cmd in [
            set_server_rank,
            add_character,
            set_char_kingdom,
            character_archive,
            character_delete,
            character_rename,
            award_points,
            convert_star,
            reset_points,
            reset_stars,
            add_ability,
            upgrade_ability,
            refresh_dashboard,
            debug_characters,
            char_card,
        ]:
            self.tree.add_command(cmd)

        names = [c.name for c in self.tree.get_commands()]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise RuntimeError(f"Duplicate command name(s) detected: {dupes}")

        LOG.info("Command tree prepared: %s command(s); GUILD_ID=%s", len(names), safe_int(os.getenv("GUILD_ID"), 0))
        await self._guild_sync_guarded()

    async def _guild_sync_guarded(self) -> None:
        try:
            gid = safe_int(os.getenv("GUILD_ID"), 0)
            if not gid:
                return

            raw_allow = (os.getenv("ALLOWED_GUILD_IDS") or "").strip()
            allowed: Optional[set[int]] = None
            if raw_allow:
                try:
                    allowed = {int(x.strip()) for x in raw_allow.split(",") if x.strip()}
                except Exception:
                    allowed = None

            if allowed and gid not in allowed:
                LOG.error("GUILD_ID %s not in ALLOWED_GUILD_IDS; skipping guild sync/reset.", gid)
                return

            allow_reset = (os.getenv("ALLOW_COMMAND_RESET") or "").strip().lower() in ("1", "true", "yes", "y", "on")
            guild_obj = discord.Object(id=gid)

            self.tree.copy_global_to(guild=guild_obj)

            if allow_reset:
                await self.http.bulk_upsert_guild_commands(self.application_id, gid, [])
                LOG.warning("Performed hard guild command reset (ALLOW_COMMAND_RESET=true) for guild %s", gid)

            synced = await self.tree.sync(guild=guild_obj)
            LOG.info("Guild command sync complete: %s commands (hard_reset=%s)", len(synced), allow_reset)
        except Exception:
            LOG.exception("Guild command sync failed")

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")
        if not self._did_hard_sync:
            await self._guild_sync_guarded()
            self._did_hard_sync = True

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
