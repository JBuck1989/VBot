# VB_v105 — Vilyra Legacy Bot (Railway + Postgres) — FULL REPLACEMENT
# Goals:
# - NEW command names ONLY (no old/duplicate commands registered)
# - Guild-only sync (no global sync), plus OPTIONAL one-time GLOBAL purge guard
# - All commands ephemeral
# - All staff commands logged to Legacy-Commands-Log (except /char_card)
# - Character autocomplete works everywhere; ability autocomplete works per-character
# - Fix/avoid rate-limit spam during startup dashboard refresh (hash + template skip + limiter)
# - Additive-only DB changes (no drops/renames). Adds characters.character_id safely.

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
# Config
# -----------------------------

LOG = logging.getLogger("VilyraBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VilyraBot: %(message)s")

DEFAULT_DASHBOARD_CHANNEL_ID = 1469879866655768738
DEFAULT_COMMAND_LOG_CHANNEL_ID = 1469879960729817098

# Dashboard rendering version (bump when formatting changes)
DASHBOARD_TEMPLATE_VERSION = 5

# Discord edit pacing (helps reduce 429 bursts)
DASHBOARD_EDIT_MIN_INTERVAL = float(os.getenv("DASHBOARD_EDIT_MIN_INTERVAL", "2.2"))
DASHBOARD_PLAYER_SPACING_SEC = float(os.getenv("DASHBOARD_PLAYER_SPACING_SEC", "0.35"))

# Limits / rules
MAX_ABILITY_STARS = 5
MAX_INFL_STARS_TOTAL = 5
STAR_COST = 10
UPGRADE_COST = 5
MAX_ABILITY_UPGRADES = 5

REP_MIN = -100
REP_MAX = 100

# Kingdom options (dropdown)
KINGDOMS: List[str] = ["Velarith", "Lyvik", "Baelon", "Sethrathiel", "Avalea"]
KINGDOM_CHOICES: List[app_commands.Choice[str]] = [app_commands.Choice(name=k, value=k) for k in KINGDOMS]

# Server ranks (dropdown)
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

# Dashboard text formatting
BORDER_LEN = 20
PLAYER_BORDER = "═" * BORDER_LEN
CHAR_SEPARATOR = "-" * BORDER_LEN
CHAR_HEADER_LEFT = "꧁•⊹٭ "
CHAR_HEADER_RIGHT = " ٭⊹•꧂"

PLAYER_POST_SOFT_LIMIT = 1900


# -----------------------------
# Small utilities
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
    return max(lo, min(hi, int(n)))


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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


def db_timeout() -> float:
    return float(os.getenv("DB_TIMEOUT", "10.0"))


async def run_db(coro, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=db_timeout())
    except asyncio.TimeoutError as e:
        raise RuntimeError(f"Database operation timed out ({label}).") from e


async def log_to_channel(guild: Optional[discord.Guild], text: str) -> None:
    """Best-effort staff command logging. Must never block/kill the command response."""
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
    """Serialize dashboard message edits/creates to reduce 429s."""
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
# Render helpers
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

        LOG.info(
            "Schema choices: abilities.%s as level, abilities.%s as character key",
            self.abilities_level_col,
            self.abilities_char_col,
        )

    async def init_schema(self) -> None:
        # Characters: additive columns + safe numeric id
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS archived BOOLEAN NOT NULL DEFAULT FALSE;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS legacy_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS legacy_minus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS lifetime_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS lifetime_minus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS ability_stars INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS influence_plus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS influence_minus INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS kingdom TEXT;")
        await self._execute("ALTER TABLE characters ALTER COLUMN kingdom DROP DEFAULT;")
        await self._execute("ALTER TABLE characters ALTER COLUMN kingdom DROP NOT NULL;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        # Safe character_id addition (no drops/renames)
        # Strategy:
        # - create sequence if missing
        # - add column if missing
        # - set DEFAULT nextval if missing
        # - backfill NULL ids
        await self._execute("CREATE SEQUENCE IF NOT EXISTS characters_character_id_seq;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS character_id BIGINT;")
        await self._execute(
            "ALTER TABLE characters ALTER COLUMN character_id SET DEFAULT nextval('characters_character_id_seq');"
        )
        await self._execute(
            "UPDATE characters SET character_id = nextval('characters_character_id_seq') WHERE character_id IS NULL;"
        )
        try:
            await self._execute("CREATE UNIQUE INDEX IF NOT EXISTS characters_character_id_unique ON characters (character_id);")
        except Exception:
            LOG.exception("Could not create unique index on characters.character_id; continuing")

        # Unique composite (needed for ON CONFLICT in some ops)
        try:
            await self._execute("CREATE UNIQUE INDEX IF NOT EXISTS characters_unique ON characters (guild_id, user_id, name);")
        except Exception:
            LOG.exception("Could not create unique index on characters; continuing")

        # Players
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

        # Abilities (create if missing + additive cols)
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                ability_name   TEXT NOT NULL,
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
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrade_level INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS level INT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        try:
            await self._execute("CREATE INDEX IF NOT EXISTS abilities_lookup ON abilities (guild_id, user_id, character_name, ability_name);")
        except Exception:
            LOG.exception("Could not create abilities index; continuing")

        # Dashboard tracking (hash + template version)
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

    # ---- players ----

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

    # ---- characters ----

    async def add_character(self, guild_id: int, user_id: int, name: str, kingdom: str | None) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("Character name cannot be empty.")

        k = (kingdom or "").strip()
        if not k:
            k = None

        # insert (id is auto-filled by DEFAULT, but if existing row, keep same)
        await self._execute(
            """
            INSERT INTO characters (guild_id, user_id, name, kingdom, archived, legacy_plus, legacy_minus, lifetime_plus, lifetime_minus,
                                   influence_plus, influence_minus, ability_stars, updated_at)
            VALUES (%s, %s, %s, %s, FALSE, 0, 0, 0, 0, 0, 0, 0, NOW())
            ON CONFLICT (guild_id, user_id, name)
            DO UPDATE SET archived=FALSE,
                          kingdom=COALESCE(EXCLUDED.kingdom, characters.kingdom),
                          updated_at=NOW();
            """,
            (guild_id, user_id, name, k),
        )
        await self._execute(
            "INSERT INTO players (guild_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (guild_id, user_id),
        )
        row = await self._fetchone(
            "SELECT character_id FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s LIMIT 1;",
            (guild_id, user_id, name),
        )
        return int(row["character_id"]) if row and row.get("character_id") is not None else 0

    async def get_character_by_id(self, guild_id: int, character_id: int) -> Optional[Dict[str, Any]]:
        row = await self._fetchone(
            "SELECT character_id, user_id, name, COALESCE(archived,FALSE) AS archived FROM characters WHERE guild_id=%s AND character_id=%s LIMIT 1;",
            (guild_id, character_id),
        )
        return dict(row) if row else None

    async def character_exists(self, guild_id: int, user_id: int, name: str) -> bool:
        row = await self._fetchone(
            "SELECT 1 FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s AND COALESCE(archived,FALSE)=FALSE LIMIT 1;",
            (guild_id, user_id, (name or "").strip()),
        )
        return bool(row)

    async def list_characters(self, guild_id: int, user_id: int) -> List[Dict[str, Any]]:
        rows = await self._fetchall(
            """
            SELECT character_id, name, COALESCE(archived,FALSE) AS archived
            FROM characters
            WHERE guild_id=%s AND user_id=%s AND COALESCE(archived,FALSE)=FALSE
            ORDER BY created_at ASC, name ASC;
            """,
            (guild_id, user_id),
        )
        return [dict(r) for r in rows]

    async def list_all_characters_for_guild(
        self,
        guild_id: int,
        include_archived: bool = True,
        name_filter: str = "",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        name_filter = (name_filter or "").strip()
        lim = max(1, min(int(limit or 25), 200))
        where = ["guild_id=%s"]
        params: List[Any] = [guild_id]

        if not include_archived:
            where.append("COALESCE(archived, FALSE)=FALSE")
        if name_filter:
            where.append("name ILIKE %s")
            params.append(f"%{name_filter}%")

        sql = f"""
            SELECT character_id, user_id, name, COALESCE(archived,FALSE) AS archived
            FROM characters
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(archived,FALSE) ASC, name ASC, user_id ASC
            LIMIT {lim};
        """
        return await self._fetchall(sql, tuple(params))

    async def delete_character_by_id(self, guild_id: int, character_id: int) -> bool:
        row = await self.get_character_by_id(guild_id, character_id)
        if not row:
            return False
        user_id = int(row["user_id"])
        name = str(row["name"])

        # delete abilities first
        await self._execute(
            "DELETE FROM abilities WHERE guild_id=%s AND user_id=%s AND character_name=%s;",
            (guild_id, user_id, name),
        )
        # delete character row
        rc = await self._execute(
            "DELETE FROM characters WHERE guild_id=%s AND character_id=%s;",
            (guild_id, character_id),
        )
        return rc > 0

    async def rename_character_by_id(self, guild_id: int, character_id: int, new_name: str) -> Tuple[str, str, int]:
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("New name cannot be empty.")

        row = await self.get_character_by_id(guild_id, character_id)
        if not row:
            raise ValueError("Character not found.")

        user_id = int(row["user_id"])
        old_name = str(row["name"])

        collision = await self._fetchone(
            "SELECT 1 FROM characters WHERE guild_id=%s AND lower(name)=lower(%s) LIMIT 1;",
            (guild_id, new_name),
        )
        if collision:
            raise ValueError("A character with that name already exists in this guild.")

        conn = self._require_conn()
        async with conn.transaction():
            await self._execute(
                "UPDATE characters SET name=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                (new_name, guild_id, character_id),
            )
            # cascade in abilities (both possible char cols)
            if self.abilities_cols:
                if "character_name" in self.abilities_cols:
                    await self._execute(
                        "UPDATE abilities SET character_name=%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND character_name=%s;",
                        (new_name, guild_id, user_id, old_name),
                    )
                if "name" in self.abilities_cols:
                    await self._execute(
                        "UPDATE abilities SET name=%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s AND name=%s;",
                        (new_name, guild_id, user_id, old_name),
                    )

        return old_name, new_name, user_id

    async def set_character_kingdom_by_id(self, guild_id: int, character_id: int, kingdom: str) -> bool:
        rc = await self._execute(
            "UPDATE characters SET kingdom=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
            ((kingdom or "").strip() or None, guild_id, character_id),
        )
        return rc > 0

    async def award_legacy_by_id(self, guild_id: int, character_id: int, pos: int, neg: int) -> Tuple[int, int, int]:
        pos = max(0, int(pos))
        neg = max(0, int(neg))
        row = await self.get_character_by_id(guild_id, character_id)
        if not row:
            raise ValueError("Character not found.")
        user_id = int(row["user_id"])
        name = str(row["name"])
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
            (pos, neg, pos, neg, guild_id, character_id),
        )
        return user_id, character_id, user_id  # (owner id, cid, owner id)

    async def set_available_legacy_by_id(self, guild_id: int, character_id: int, pos: int, neg: int) -> None:
        pos = max(0, int(pos))
        neg = max(0, int(neg))
        await self._execute(
            "UPDATE characters SET legacy_plus=%s, legacy_minus=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
            (pos, neg, guild_id, character_id),
        )

    async def set_lifetime_legacy_by_id(self, guild_id: int, character_id: int, pos: int, neg: int) -> None:
        pos = max(0, int(pos))
        neg = max(0, int(neg))
        await self._execute(
            "UPDATE characters SET lifetime_plus=%s, lifetime_minus=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
            (pos, neg, guild_id, character_id),
        )

    async def get_character_state_by_id(self, guild_id: int, character_id: int) -> Dict[str, Any]:
        row = await self._fetchone(
            """
            SELECT user_id, name, kingdom,
                   legacy_plus, legacy_minus,
                   lifetime_plus, lifetime_minus,
                   influence_plus, influence_minus,
                   ability_stars
            FROM characters
            WHERE guild_id=%s AND character_id=%s AND COALESCE(archived,FALSE)=FALSE
            LIMIT 1;
            """,
            (guild_id, character_id),
        )
        if not row:
            raise ValueError("Character not found.")
        return dict(row)

    async def set_stars_correction_by_id(self, guild_id: int, character_id: int, ability: int, infl_pos: int, infl_neg: int) -> None:
        ability = clamp(int(ability), 0, MAX_ABILITY_STARS)
        infl_pos = clamp(int(infl_pos), 0, MAX_INFL_STARS_TOTAL)
        infl_neg = clamp(int(infl_neg), 0, MAX_INFL_STARS_TOTAL)
        if infl_pos + infl_neg > MAX_INFL_STARS_TOTAL:
            raise ValueError("Total influence stars (pos + neg) cannot exceed 5.")
        await self._execute(
            """
            UPDATE characters
            SET ability_stars=%s,
                influence_plus=%s,
                influence_minus=%s,
                updated_at=NOW()
            WHERE guild_id=%s AND character_id=%s;
            """,
            (ability, infl_pos, infl_neg, guild_id, character_id),
        )

    # ---- abilities ----

    def _ability_level_expr(self) -> str:
        if self.abilities_level_col == "upgrade_level":
            return "COALESCE(upgrade_level, 0) AS lvl"
        return "COALESCE(level, 0) AS lvl"

    async def list_abilities_for_character(self, guild_id: int, user_id: int, character_name: str) -> List[Tuple[str, int]]:
        char_col = self.abilities_char_col
        rows = await self._fetchall(
            f"""
            SELECT ability_name, {self._ability_level_expr()}
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND {char_col}=%s
            ORDER BY created_at ASC, ability_name ASC;
            """,
            (guild_id, user_id, (character_name or "").strip()),
        )
        out: List[Tuple[str, int]] = []
        for r in rows:
            nm = (r.get("ability_name") or "").strip()
            if nm:
                out.append((nm, safe_int(r.get("lvl"), 0)))
        return out

    async def add_ability(self, guild_id: int, character_id: int, ability_name: str) -> Tuple[int, str]:
        ability_name = (ability_name or "").strip()
        if not ability_name:
            raise ValueError("Ability name cannot be empty.")

        st = await self.get_character_state_by_id(guild_id, character_id)
        user_id = int(st["user_id"])
        cname = str(st["name"])

        current = await self.list_abilities_for_character(guild_id, user_id, cname)
        cap = 2 + clamp(safe_int(st.get("ability_stars"), 0), 0, MAX_ABILITY_STARS)
        if len(current) >= cap:
            raise ValueError(f"Ability capacity reached ({len(current)}/{cap}). Earn more Ability Stars to add abilities.")

        char_col = self.abilities_char_col
        cols = ["guild_id", "user_id", char_col, "ability_name", "created_at", "updated_at"]
        vals = ["%s", "%s", "%s", "%s", "NOW()", "NOW()"]
        params: List[Any] = [guild_id, user_id, cname, ability_name]

        if "upgrade_level" in self.abilities_cols:
            cols.append("upgrade_level")
            vals.append("0")
        if "level" in self.abilities_cols:
            cols.append("level")
            vals.append("0")

        sql = "INSERT INTO abilities (" + ", ".join(cols) + ") VALUES (" + ", ".join(vals) + ");"
        await self._execute(sql, params)
        return user_id, cname

    async def rename_ability(self, guild_id: int, character_id: int, old_ability: str, new_ability: str) -> Tuple[int, str]:
        old_ability = (old_ability or "").strip()
        new_ability = (new_ability or "").strip()
        if not old_ability or not new_ability:
            raise ValueError("Ability names cannot be empty.")

        st = await self.get_character_state_by_id(guild_id, character_id)
        user_id = int(st["user_id"])
        cname = str(st["name"])
        char_col = self.abilities_char_col

        exists = await self._fetchone(
            f"SELECT 1 FROM abilities WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND lower(ability_name)=lower(%s) LIMIT 1;",
            (guild_id, user_id, cname, old_ability),
        )
        if not exists:
            raise ValueError("Ability not found for that character.")

        collision = await self._fetchone(
            f"SELECT 1 FROM abilities WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND lower(ability_name)=lower(%s) LIMIT 1;",
            (guild_id, user_id, cname, new_ability),
        )
        if collision:
            raise ValueError("That character already has an ability with the new name.")

        await self._execute(
            f"""
            UPDATE abilities
            SET ability_name=%s, updated_at=NOW()
            WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND ability_name=%s;
            """,
            (new_ability, guild_id, user_id, cname, old_ability),
        )
        return user_id, cname

    async def upgrade_ability(self, guild_id: int, character_id: int, ability_name: str, pos_cost: int, neg_cost: int, upgrades: int) -> Tuple[int, str, int]:
        ability_name = (ability_name or "").strip()
        if not ability_name:
            raise ValueError("Ability name cannot be empty.")

        upgrades = max(1, int(upgrades))
        pos_cost = max(0, int(pos_cost))
        neg_cost = max(0, int(neg_cost))

        total_cost = upgrades * UPGRADE_COST
        if pos_cost + neg_cost != total_cost:
            raise ValueError(f"Payment must equal {total_cost} total points (5 per upgrade).")

        st = await self.get_character_state_by_id(guild_id, character_id)
        user_id = int(st["user_id"])
        cname = str(st["name"])
        char_col = self.abilities_char_col
        level_col = self.abilities_level_col

        row = await self._fetchone(
            f"""
            SELECT COALESCE({level_col}, 0) AS cur_level
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND ability_name=%s
            ORDER BY created_at ASC
            LIMIT 1;
            """,
            (guild_id, user_id, cname, ability_name),
        )
        if not row:
            raise ValueError("Ability not found. Add it first with /ability_add.")

        cur_level = safe_int(row.get("cur_level"), 0)
        if cur_level >= MAX_ABILITY_UPGRADES:
            raise ValueError(f"Upgrade limit reached ({cur_level}/{MAX_ABILITY_UPGRADES}).")

        remaining = MAX_ABILITY_UPGRADES - cur_level
        if upgrades > remaining:
            raise ValueError(f"Only {remaining} upgrade(s) remaining for this ability (max {MAX_ABILITY_UPGRADES}).")

        # validate and deduct available legacy pools
        legacy_plus = safe_int(st.get("legacy_plus"), 0)
        legacy_minus = safe_int(st.get("legacy_minus"), 0)
        if legacy_plus < pos_cost:
            raise ValueError(f"Not enough available positive points (need {pos_cost}, have {legacy_plus}).")
        if legacy_minus < neg_cost:
            raise ValueError(f"Not enough available negative points (need {neg_cost}, have {legacy_minus}).")

        conn = self._require_conn()
        async with conn.transaction():
            if pos_cost:
                await self._execute(
                    "UPDATE characters SET legacy_plus=legacy_plus-%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                    (pos_cost, guild_id, character_id),
                )
            if neg_cost:
                await self._execute(
                    "UPDATE characters SET legacy_minus=legacy_minus-%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                    (neg_cost, guild_id, character_id),
                )

            new_level = cur_level + upgrades
            sets: List[str] = [f"{level_col}=%s", "updated_at=NOW()"]
            params: List[Any] = [new_level]

            # keep both columns in sync if both exist
            if level_col != "upgrade_level" and "upgrade_level" in self.abilities_cols:
                sets.append("upgrade_level=%s")
                params.append(new_level)
            if level_col != "level" and "level" in self.abilities_cols:
                sets.append("level=%s")
                params.append(new_level)

            params.extend([guild_id, user_id, cname, ability_name])
            await self._execute(
                f"UPDATE abilities SET {', '.join(sets)} WHERE guild_id=%s AND user_id=%s AND {char_col}=%s AND ability_name=%s;",
                params,
            )

        return user_id, cname, cur_level + upgrades

    # ---- dashboards ----

    async def get_dashboard_entry(self, guild_id: int, user_id: int) -> Tuple[List[int], Optional[str], Optional[Any], int]:
        row = await self._fetchone(
            "SELECT message_ids, content_hash, updated_at, COALESCE(template_version,0) AS template_version FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )
        ids: List[int] = []
        if row and row.get("message_ids"):
            for part in str(row["message_ids"]).split(","):
                part = part.strip()
                if part.isdigit():
                    ids.append(int(part))
        h = str(row["content_hash"]) if row and row.get("content_hash") else None
        ts = row["updated_at"] if row and row.get("updated_at") else None
        tv = int(row["template_version"]) if row and row.get("template_version") is not None else 0
        return ids, h, ts, tv

    async def set_dashboard_entry(self, guild_id: int, user_id: int, channel_id: int, message_id: int, h: str) -> None:
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
            (guild_id, user_id, channel_id, str(message_id), h, DASHBOARD_TEMPLATE_VERSION),
        )

    async def clear_dashboard_entry(self, guild_id: int, user_id: int) -> None:
        await self._execute("DELETE FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))

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

    async def list_player_ids(self, guild_id: int) -> List[int]:
        rows = await self._fetchall(
            "SELECT user_id FROM players WHERE guild_id=%s ORDER BY user_id ASC;",
            (guild_id,),
        )
        ids = [int(r["user_id"]) for r in rows if r and r.get("user_id") is not None]
        if ids:
            return ids
        rows2 = await self._fetchall(
            "SELECT DISTINCT user_id FROM characters WHERE guild_id=%s ORDER BY user_id ASC;",
            (guild_id,),
        )
        return [int(r["user_id"]) for r in rows2 if r and r.get("user_id") is not None]


# -----------------------------
# Autocomplete (character + ability)
# -----------------------------

async def resolve_character_token(interaction: discord.Interaction, token: str) -> Tuple[int, int, str]:
    """Returns (character_id, owner_user_id, character_name). Token may be digits (character_id) or raw name."""
    if interaction.guild is None:
        raise ValueError("This command must be used in a server.")
    db: Database = interaction.client.db  # type: ignore

    t = (token or "").strip()
    if not t:
        raise ValueError("Character is required.")

    if t.isdigit():
        cid = int(t)
        row = await db.get_character_by_id(interaction.guild.id, cid)
        if not row:
            raise ValueError("Character not found.")
        return int(row["character_id"]), int(row["user_id"]), str(row["name"])

    # fallback: unique names
    rows = await db.list_all_characters_for_guild(interaction.guild.id, include_archived=True, name_filter=t, limit=50)
    exact = [r for r in rows if str(r.get("name", "")).lower() == t.lower()]
    if len(exact) == 1:
        r = exact[0]
        return int(r["character_id"]), int(r["user_id"]), str(r["name"])
    if len(exact) > 1:
        raise ValueError("Multiple characters matched. Please select from the autocomplete list.")
    raise ValueError("No matching character found.")


async def autocomplete_character(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    try:
        guild = interaction.guild
        if guild is None:
            return []
        q = (current or "").strip()
        db: Database = interaction.client.db  # type: ignore
        rows = await db.list_all_characters_for_guild(guild.id, include_archived=True, name_filter=q, limit=25)
        out: List[app_commands.Choice[str]] = []
        for r in rows:
            cid = safe_int(r.get("character_id"), 0)
            name = str(r.get("name") or "").strip()
            archived = bool(r.get("archived", False))
            if cid <= 0 or not name:
                continue
            label = f"{name}" + (" [archived]" if archived else "")
            if len(label) > 100:
                label = label[:97] + "..."
            out.append(app_commands.Choice(name=label, value=str(cid)))
        return out
    except Exception:
        LOG.exception("Character autocomplete failed")
        return []


async def autocomplete_ability_for_character(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    """Ability autocomplete that depends on selected 'character' param."""
    try:
        guild = interaction.guild
        if guild is None:
            return []
        ns = getattr(interaction, "namespace", None)
        char_token = ""
        if ns is not None:
            # must match command parameter name "character"
            char_token = getattr(ns, "character", "") or ""
        if not char_token:
            return []
        cid, owner_id, cname = await resolve_character_token(interaction, str(char_token))
        db: Database = interaction.client.db  # type: ignore
        abilities = await db.list_abilities_for_character(guild.id, owner_id, cname)
        q = (current or "").strip().lower()
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

def in_guild_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False
        return True
    return app_commands.check(predicate)


def staff_only():
    raw = os.getenv("STAFF_USER_IDS", "") or ""
    staff_user_ids: set[int] = {int(x.strip()) for x in raw.split(",") if x.strip().isdigit()}

    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False

        # allowlist
        if interaction.user.id in staff_user_ids:
            return True

        # owner / perms
        member = interaction.user
        if not isinstance(member, discord.Member):
            member = interaction.guild.get_member(interaction.user.id)  # type: ignore

        if interaction.guild.owner_id == interaction.user.id:
            return True

        if isinstance(member, discord.Member):
            perms = member.guild_permissions
            if perms.administrator or perms.manage_guild:
                return True

        await safe_reply(interaction, "You don't have permission to use this command.")
        return False

    return app_commands.check(predicate)


# -----------------------------
# Dashboard rendering
# -----------------------------

@dataclass
class CharacterCard:
    character_id: int
    owner_id: int
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


async def build_character_card(db: Database, guild_id: int, character_id: int) -> CharacterCard:
    st = await db.get_character_state_by_id(guild_id, character_id)
    owner_id = int(st["user_id"])
    name = str(st["name"])
    abilities = await db.list_abilities_for_character(guild_id, owner_id, name)
    return CharacterCard(
        character_id=character_id,
        owner_id=owner_id,
        name=name,
        kingdom=str(st.get("kingdom") or ""),
        legacy_plus=safe_int(st.get("legacy_plus"), 0),
        legacy_minus=safe_int(st.get("legacy_minus"), 0),
        lifetime_plus=safe_int(st.get("lifetime_plus"), 0),
        lifetime_minus=safe_int(st.get("lifetime_minus"), 0),
        ability_stars=safe_int(st.get("ability_stars"), 0),
        infl_plus=safe_int(st.get("influence_plus"), 0),
        infl_minus=safe_int(st.get("influence_minus"), 0),
        abilities=abilities,
    )


def render_character_block(card: CharacterCard) -> str:
    net_lifetime = card.lifetime_plus - card.lifetime_minus
    lines: List[str] = []
    lines.append(f"{CHAR_HEADER_LEFT}**{card.name}** {CHAR_HEADER_RIGHT}")

    k = (card.kingdom or "").strip()
    lines.append("Kingdom:" if not k else f"Kingdom: {k}")
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

    for i, c in enumerate(chars):
        cid = safe_int(c.get("character_id"), 0)
        if cid <= 0:
            continue
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

    # Skip if data unchanged since last dashboard update AND template unchanged
    try:
        latest_ts = await db.get_latest_player_data_updated_at(guild.id, user_id)
        if stored_tv == DASHBOARD_TEMPLATE_VERSION and dash_ts and latest_ts and latest_ts <= dash_ts:
            return "skipped"
    except Exception as ex:
        LOG.warning("Could not compute latest player data ts for user_id=%s: %s", user_id, ex)

    if not chars:
        # if they have no characters, delete old dashboard if we can
        if stored_ids:
            for mid in stored_ids:
                try:
                    m = await channel.fetch_message(mid)
                    await m.delete()
                except Exception:
                    pass
        await db.clear_dashboard_entry(guild.id, user_id)
        return f"No characters for user_id={user_id}; dashboard cleared."

    content = await render_player_post(db, guild, user_id)
    if not content:
        return f"No content rendered for user_id={user_id}."

    new_hash = content_hash(content)

    # If hash is identical AND template is identical, skip edit entirely
    if stored_hash == new_hash and stored_tv == DASHBOARD_TEMPLATE_VERSION and stored_ids:
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
        await db.set_dashboard_entry(guild.id, user_id, channel.id, msg.id, new_hash)
        return f"Dashboard created for user_id={user_id}."
    else:
        await client.dashboard_limiter.wait()
        await msg.edit(content=content)
        await db.set_dashboard_entry(guild.id, user_id, channel.id, msg.id, new_hash)
        return f"Dashboard updated for user_id={user_id}."


async def refresh_all_dashboards(client: "VilyraBotClient", guild: discord.Guild) -> str:
    user_ids = await client.db.list_player_ids(guild.id)
    if not user_ids:
        return "No players with characters yet."

    ok = 0
    for uid in user_ids:
        try:
            await refresh_player_dashboard(client, guild, uid)
        except Exception:
            LOG.exception("refresh_player_dashboard failed for user_id=%s", uid)
        ok += 1
        await asyncio.sleep(DASHBOARD_PLAYER_SPACING_SEC)
    return f"Refreshed dashboards for {ok} player(s)."


# -----------------------------
# Slash commands (NEW NAMES ONLY)
# -----------------------------

@app_commands.command(name="staff_commands", description="(Staff) Show the staff command list.")
@in_guild_only()
@staff_only()
async def staff_commands(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        # Only the commands in THIS bot build (as requested)
        items: list[tuple[str, str]] = [
            ("/character_create", "Create a character for a player (user, name, kingdom)."),
            ("/character_delete", "Delete a character (select from autocomplete)."),
            ("/character_rename", "Rename a character (keeps character id; cascades to abilities)."),
            ("/set_char_kingdom", "Set a character’s kingdom."),
            ("/award_legacy", "Award legacy points (+/-) to a character (updates available + lifetime)."),
            ("/set_available_legacy", "Correction: set available legacy pools (does NOT change lifetime)."),
            ("/set_lifetime_legacy", "Correction: set lifetime legacy totals (does NOT change available)."),
            ("/convert_stars", "Convert 10 legacy points per star into ability/influence stars."),
            ("/set_stars_correction", "Rare correction: set Ability / Influence+ / Influence- stars."),
            ("/ability_add", "Add an ability to a character (capacity = 2 + ability stars)."),
            ("/ability_rename", "Rename an ability (keeps upgrades)."),
            ("/ability_upgrade", "Upgrade an ability (max 5; costs 5 legacy per upgrade)."),
            ("/set_server_rank", "Set a player’s server rank."),
            ("/refresh_dashboard", "Force-refresh all dashboard posts."),
        ]
        lines = ["**Staff Commands**", ""]
        for cmd, desc in items:
            lines.append(f"• **{cmd}** — {desc}")
        await safe_reply(interaction, "\n".join(lines))
        await log_to_channel(interaction.guild, f"📜 {interaction.user.mention} used **/staff_commands**")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="character_create", description="(Staff) Create a character for a player.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="Player", character_name="Character name", kingdom="Home kingdom")
@app_commands.choices(kingdom=KINGDOM_CHOICES)
async def character_create(interaction: discord.Interaction, user: discord.Member, character_name: str, kingdom: app_commands.Choice[str]):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid = await run_db(interaction.client.db.add_character(interaction.guild.id, user.id, character_name, kingdom.value), "add_character")  # type: ignore
        await refresh_player_dashboard(interaction.client, interaction.guild, user.id)  # type: ignore
        await safe_reply(interaction, f"✅ Created/updated **{character_name}** for {user.mention}. (ID: `{cid}`; Kingdom: **{kingdom.value}**)")
        await log_to_channel(interaction.guild, f"🆕 {interaction.user.mention} created **{character_name}** (id {cid}) for {user.mention} — kingdom **{kingdom.value}**")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="character_delete", description="(Staff) Delete a character (cannot be undone).")
@in_guild_only()
@staff_only()
@app_commands.describe(character="Character (select from autocomplete)")
@app_commands.autocomplete(character=autocomplete_character)
async def character_delete(interaction: discord.Interaction, character: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        ok = await run_db(interaction.client.db.delete_character_by_id(interaction.guild.id, cid), "delete_character")  # type: ignore
        if not ok:
            raise RuntimeError("Character not found.")
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Deleted **{cname}** (ID: `{cid}`).")
        await log_to_channel(interaction.guild, f"🗑️ {interaction.user.mention} deleted **{cname}** (id {cid}) owned by <@{owner_id}>")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="character_rename", description="(Staff) Rename a character (keeps ID; cascades to abilities).")
@in_guild_only()
@staff_only()
@app_commands.describe(character="Character (select from autocomplete)", new_name="New character name")
@app_commands.autocomplete(character=autocomplete_character)
async def character_rename(interaction: discord.Interaction, character: str, new_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, _ = await resolve_character_token(interaction, character)
        old_name, new_name2, owner_id2 = await run_db(
            interaction.client.db.rename_character_by_id(interaction.guild.id, cid, new_name),  # type: ignore
            "rename_character",
        )
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id2)  # type: ignore
        await safe_reply(interaction, f"✅ Renamed **{old_name}** → **{new_name2}** (ID: `{cid}`).")
        await log_to_channel(interaction.guild, f"✏️ {interaction.user.mention} renamed **{old_name}** → **{new_name2}** (id {cid}) owned by <@{owner_id}>")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="set_char_kingdom", description="(Staff) Set a character's kingdom.")
@in_guild_only()
@staff_only()
@app_commands.describe(character="Character (select from autocomplete)", kingdom="Home kingdom")
@app_commands.autocomplete(character=autocomplete_character)
@app_commands.choices(kingdom=KINGDOM_CHOICES)
async def set_char_kingdom(interaction: discord.Interaction, character: str, kingdom: app_commands.Choice[str]):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        ok = await run_db(interaction.client.db.set_character_kingdom_by_id(interaction.guild.id, cid, kingdom.value), "set_char_kingdom")  # type: ignore
        if not ok:
            raise RuntimeError("Character not found.")
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Kingdom set: **{cname}** → **{kingdom.value}**.")
        await log_to_channel(interaction.guild, f"🏰 {interaction.user.mention} set kingdom for **{cname}** (id {cid}) to **{kingdom.value}**")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="set_server_rank", description="(Staff) Set a player's server rank.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="Player", rank="Rank")
@app_commands.choices(rank=[app_commands.Choice(name=r, value=r) for r in SERVER_RANKS])
async def set_server_rank(interaction: discord.Interaction, user: discord.Member, rank: app_commands.Choice[str]):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        await run_db(interaction.client.db.set_player_rank(interaction.guild.id, user.id, rank.value), "set_player_rank")  # type: ignore
        await refresh_player_dashboard(interaction.client, interaction.guild, user.id)  # type: ignore
        await safe_reply(interaction, f"✅ Set rank for {user.mention} to **{rank.value}**.")
        await log_to_channel(interaction.guild, f"🏷️ {interaction.user.mention} set server rank for {user.mention} to **{rank.value}**")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="award_legacy", description="(Staff) Award legacy points (+/-) to a character (updates available + lifetime).")
@in_guild_only()
@staff_only()
@app_commands.describe(character="Character (select from autocomplete)", positive="Positive points to award", negative="Negative points to award")
@app_commands.autocomplete(character=autocomplete_character)
async def award_legacy(interaction: discord.Interaction, character: str, positive: int = 0, negative: int = 0):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        positive = max(0, int(positive))
        negative = max(0, int(negative))
        await run_db(interaction.client.db.award_legacy_by_id(interaction.guild.id, cid, positive, negative), "award_legacy")  # type: ignore
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Awarded **{cname}**: +{positive} / -{negative}.")
        await log_to_channel(interaction.guild, f"➕ {interaction.user.mention} awarded **{cname}** (id {cid}) +{positive}/-{negative}")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="set_available_legacy", description="(Staff) Correction: set available legacy pools (does NOT change lifetime totals).")
@in_guild_only()
@staff_only()
@app_commands.describe(character="Character (select from autocomplete)", positive="Set available positive legacy", negative="Set available negative legacy")
@app_commands.autocomplete(character=autocomplete_character)
async def set_available_legacy(interaction: discord.Interaction, character: str, positive: int, negative: int):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        await run_db(interaction.client.db.set_available_legacy_by_id(interaction.guild.id, cid, positive, negative), "set_available_legacy")  # type: ignore
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Set available legacy for **{cname}** to +{max(0,int(positive))} / -{max(0,int(negative))}. (Lifetime unchanged)")
        await log_to_channel(interaction.guild, f"🛠️ {interaction.user.mention} set AVAILABLE legacy for **{cname}** (id {cid}) to +{max(0,int(positive))}/-{max(0,int(negative))}")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="set_lifetime_legacy", description="(Staff) Correction: set lifetime legacy totals (does NOT change available pools).")
@in_guild_only()
@staff_only()
@app_commands.describe(character="Character (select from autocomplete)", positive="Set lifetime positive legacy", negative="Set lifetime negative legacy")
@app_commands.autocomplete(character=autocomplete_character)
async def set_lifetime_legacy(interaction: discord.Interaction, character: str, positive: int, negative: int):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        await run_db(interaction.client.db.set_lifetime_legacy_by_id(interaction.guild.id, cid, positive, negative), "set_lifetime_legacy")  # type: ignore
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Set lifetime legacy for **{cname}** to +{max(0,int(positive))} / -{max(0,int(negative))}. (Available unchanged)")
        await log_to_channel(interaction.guild, f"🛠️ {interaction.user.mention} set LIFETIME legacy for **{cname}** (id {cid}) to +{max(0,int(positive))}/-{max(0,int(negative))}")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="convert_stars", description="(Staff) Convert 10 legacy points per star into ability/influence stars.")
@in_guild_only()
@staff_only()
@app_commands.describe(
    character="Character (select from autocomplete)",
    star_type="Star type",
    stars="How many stars to convert (each costs 10 points)",
    spend_positive="Positive points spent",
    spend_negative="Negative points spent",
)
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
    stars: int,
    spend_positive: int,
    spend_negative: int,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)

        stars = max(1, int(stars))
        spend_positive = max(0, int(spend_positive))
        spend_negative = max(0, int(spend_negative))
        total_cost = STAR_COST * stars

        if spend_positive + spend_negative != total_cost:
            raise ValueError(f"Spend must total {total_cost} points (10 per star).")

        st = await interaction.client.db.get_character_state_by_id(interaction.guild.id, cid)  # type: ignore
        legacy_plus = safe_int(st.get("legacy_plus"), 0)
        legacy_minus = safe_int(st.get("legacy_minus"), 0)
        ability_stars = safe_int(st.get("ability_stars"), 0)
        infl_pos = safe_int(st.get("influence_plus"), 0)
        infl_neg = safe_int(st.get("influence_minus"), 0)

        infl_total = infl_pos + infl_neg
        t = star_type.value

        if t == "ability":
            if ability_stars + stars > MAX_ABILITY_STARS:
                raise ValueError("Ability stars already at max (5).")
        elif t == "influence_positive":
            if infl_total + stars > MAX_INFL_STARS_TOTAL:
                raise ValueError("Total influence stars (pos+neg) cannot exceed 5.")
            if spend_negative != 0:
                raise ValueError("Positive influence stars must be paid with positive points only (spend_negative=0).")
        elif t == "influence_negative":
            if infl_total + stars > MAX_INFL_STARS_TOTAL:
                raise ValueError("Total influence stars (pos+neg) cannot exceed 5.")
            if spend_positive != 0:
                raise ValueError("Negative influence stars must be paid with negative points only (spend_positive=0).")
        else:
            raise ValueError("Invalid star type.")

        if legacy_plus < spend_positive:
            raise ValueError(f"Not enough available positive points (need {spend_positive}, have {legacy_plus}).")
        if legacy_minus < spend_negative:
            raise ValueError(f"Not enough available negative points (need {spend_negative}, have {legacy_minus}).")

        # Deduct costs from available pools
        conn = interaction.client.db._require_conn()  # type: ignore
        async with conn.transaction():
            if spend_positive:
                await interaction.client.db._execute(  # type: ignore
                    "UPDATE characters SET legacy_plus=legacy_plus-%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                    (spend_positive, interaction.guild.id, cid),
                )
            if spend_negative:
                await interaction.client.db._execute(  # type: ignore
                    "UPDATE characters SET legacy_minus=legacy_minus-%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                    (spend_negative, interaction.guild.id, cid),
                )

            # Apply stars
            if t == "ability":
                await interaction.client.db._execute(  # type: ignore
                    "UPDATE characters SET ability_stars=ability_stars+%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                    (stars, interaction.guild.id, cid),
                )
            elif t == "influence_positive":
                await interaction.client.db._execute(  # type: ignore
                    "UPDATE characters SET influence_plus=influence_plus+%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                    (stars, interaction.guild.id, cid),
                )
            else:
                await interaction.client.db._execute(  # type: ignore
                    "UPDATE characters SET influence_minus=influence_minus+%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;",
                    (stars, interaction.guild.id, cid),
                )

        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Converted points to stars for **{cname}**: {star_type.name} x{stars} (spent +{spend_positive}/-{spend_negative}).")
        await log_to_channel(interaction.guild, f"⭐ {interaction.user.mention} converted stars for **{cname}** (id {cid}) — {star_type.value} x{stars} spent +{spend_positive}/-{spend_negative}")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="set_stars_correction", description="(Staff) Rare correction: set Ability / Influence+ / Influence- stars exactly.")
@in_guild_only()
@staff_only()
@app_commands.describe(
    character="Character (select from autocomplete)",
    ability_stars="Set ability stars (0-5)",
    influence_positive="Set positive influence stars (0-5)",
    influence_negative="Set negative influence stars (0-5)",
)
@app_commands.autocomplete(character=autocomplete_character)
async def set_stars_correction(
    interaction: discord.Interaction,
    character: str,
    ability_stars: int,
    influence_positive: int,
    influence_negative: int,
):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        await run_db(
            interaction.client.db.set_stars_correction_by_id(interaction.guild.id, cid, ability_stars, influence_positive, influence_negative),  # type: ignore
            "set_stars_correction",
        )
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Stars corrected for **{cname}**: Ability={clamp(ability_stars,0,5)}, +Inf={clamp(influence_positive,0,5)}, -Inf={clamp(influence_negative,0,5)}.")
        await log_to_channel(interaction.guild, f"🛠️ {interaction.user.mention} corrected stars for **{cname}** (id {cid}) — ability={clamp(ability_stars,0,5)} +inf={clamp(influence_positive,0,5)} -inf={clamp(influence_negative,0,5)}")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="ability_add", description="(Staff) Add an ability to a character (capacity = 2 + ability stars).")
@in_guild_only()
@staff_only()
@app_commands.describe(character="Character (select from autocomplete)", ability_name="Ability name")
@app_commands.autocomplete(character=autocomplete_character)
async def ability_add(interaction: discord.Interaction, character: str, ability_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        await run_db(interaction.client.db.add_ability(interaction.guild.id, cid, ability_name), "ability_add")  # type: ignore
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Added ability to **{cname}**: **{ability_name.strip()}**.")
        await log_to_channel(interaction.guild, f"🧩 {interaction.user.mention} added ability **{ability_name.strip()}** to **{cname}** (id {cid})")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="ability_rename", description="(Staff) Rename an ability (keeps upgrades).")
@in_guild_only()
@staff_only()
@app_commands.describe(character="Character (select from autocomplete)", ability="Ability (autocomplete)", new_ability_name="New ability name")
@app_commands.autocomplete(character=autocomplete_character, ability=autocomplete_ability_for_character)
async def ability_rename(interaction: discord.Interaction, character: str, ability: str, new_ability_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        await run_db(interaction.client.db.rename_ability(interaction.guild.id, cid, ability, new_ability_name), "ability_rename")  # type: ignore
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Renamed ability for **{cname}**: **{ability}** → **{new_ability_name.strip()}**.")
        await log_to_channel(interaction.guild, f"✏️ {interaction.user.mention} renamed ability for **{cname}** (id {cid}): **{ability}** → **{new_ability_name.strip()}**")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="ability_upgrade", description="(Staff) Upgrade an ability (max 5; costs 5 legacy per upgrade).")
@in_guild_only()
@staff_only()
@app_commands.describe(
    character="Character (select from autocomplete)",
    ability="Ability (autocomplete)",
    positive_cost="Positive points spent",
    negative_cost="Negative points spent",
    upgrades="How many upgrades to apply (each costs 5 points)",
)
@app_commands.autocomplete(character=autocomplete_character, ability=autocomplete_ability_for_character)
async def ability_upgrade(interaction: discord.Interaction, character: str, ability: str, positive_cost: int, negative_cost: int, upgrades: int = 1):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        _, _, new_lvl = await run_db(
            interaction.client.db.upgrade_ability(interaction.guild.id, cid, ability, positive_cost, negative_cost, upgrades),  # type: ignore
            "ability_upgrade",
        )
        await refresh_player_dashboard(interaction.client, interaction.guild, owner_id)  # type: ignore
        await safe_reply(interaction, f"✅ Upgraded **{ability}** for **{cname}** → level **{new_lvl}** (spent +{max(0,int(positive_cost))}/-{max(0,int(negative_cost))}).")
        await log_to_channel(interaction.guild, f"⬆️ {interaction.user.mention} upgraded **{ability}** for **{cname}** (id {cid}) to lvl {new_lvl} spent +{max(0,int(positive_cost))}/-{max(0,int(negative_cost))}")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="refresh_dashboard", description="(Staff) Force-refresh all dashboard posts.")
@in_guild_only()
@staff_only()
async def refresh_dashboard(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        status = await refresh_all_dashboards(interaction.client, interaction.guild)  # type: ignore
        await safe_reply(interaction, status)
        await log_to_channel(interaction.guild, f"🔄 {interaction.user.mention} ran **/refresh_dashboard** — {status}")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="debug_characters", description="(Staff) Debug: show character counts for this guild.")
@in_guild_only()
@staff_only()
async def debug_characters(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        rows = await interaction.client.db._fetchall(  # type: ignore
            "SELECT COALESCE(archived, FALSE) AS archived, COUNT(*) AS n FROM characters WHERE guild_id=%s GROUP BY COALESCE(archived, FALSE) ORDER BY COALESCE(archived, FALSE)",
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
        await log_to_channel(interaction.guild, f"🧪 {interaction.user.mention} ran **/debug_characters**")
    except Exception as e:
        await send_error(interaction, e)


@app_commands.command(name="char_card", description="Show a character card (any member can view).")
@in_guild_only()
@app_commands.describe(character="Character (select from autocomplete)")
@app_commands.autocomplete(character=autocomplete_character)
async def char_card(interaction: discord.Interaction, character: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, owner_id, cname = await resolve_character_token(interaction, character)
        card = await build_character_card(interaction.client.db, interaction.guild.id, cid)  # type: ignore

        owner_mention = f"<@{owner_id}>"
        header = f"__***{owner_mention}***__"
        body = render_character_block(card)
        await safe_reply(interaction, header + "\n\n" + body)
        # NOTE: per spec, /char_card is NOT logged anywhere
    except Exception as e:
        await send_error(interaction, e)


# -----------------------------
# Discord Client
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
        # Register ONLY the intended commands (new names only)
        commands = [
            staff_commands,
            character_create,
            character_delete,
            character_rename,
            set_char_kingdom,
            set_server_rank,
            award_legacy,
            set_available_legacy,
            set_lifetime_legacy,
            convert_stars,
            set_stars_correction,
            ability_add,
            ability_rename,
            ability_upgrade,
            refresh_dashboard,
            debug_characters,
            char_card,
        ]
        for c in commands:
            self.tree.add_command(c)

        # Self-check duplicates in code registration
        names = [c.name for c in self.tree.get_commands()]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise RuntimeError(f"Duplicate command name(s) registered in code: {dupes}")
        LOG.info("Command tree prepared: %s command(s); GUILD_ID=%s", len(names), safe_int(os.getenv("GUILD_ID"), 0))

        # ---- OPTIONAL: ONE-TIME GLOBAL PURGE (removes old global commands that cause duplicates) ----
        # Use ONLY for one deploy, then turn it off immediately.
        # Env:
        #   ALLOW_GLOBAL_COMMAND_RESET=true
        # Safety latch:
        #   ALLOWED_GUILD_IDS must include your GUILD_ID (we use it as a required "I know what I'm doing" latch).
        try:
            allow_global_reset = (os.getenv("ALLOW_GLOBAL_COMMAND_RESET") or "").strip().lower() in ("1", "true", "yes", "y", "on")
            gid = safe_int(os.getenv("GUILD_ID"), 0)
            raw_allow = (os.getenv("ALLOWED_GUILD_IDS") or "").strip()
            allowed = {int(x.strip()) for x in raw_allow.split(",") if x.strip().isdigit()} if raw_allow else set()

            if allow_global_reset:
                if not gid or gid not in allowed:
                    LOG.error("ALLOW_GLOBAL_COMMAND_RESET set but GUILD_ID not in ALLOWED_GUILD_IDS; refusing global purge.")
                else:
                    if getattr(self, "application_id", None):
                        await self.http.bulk_upsert_global_commands(self.application_id, [])
                        LOG.warning("Performed GLOBAL command purge (ALLOW_GLOBAL_COMMAND_RESET=true). Turn it OFF after this deploy.")
        except Exception:
            LOG.exception("Global command purge failed")

        # ---- Guild-only sync (no global sync) ----
        # Optional one-time GUILD wipe:
        #   ALLOW_COMMAND_RESET=true
        # plus safety:
        #   ALLOWED_GUILD_IDS includes GUILD_ID
        try:
            gid = safe_int(os.getenv("GUILD_ID"), 0)
            if not gid:
                LOG.warning("GUILD_ID not set; skipping guild sync. (This bot is intended to run guild-scoped.)")
                return

            raw_allow = (os.getenv("ALLOWED_GUILD_IDS") or "").strip()
            allowed = {int(x.strip()) for x in raw_allow.split(",") if x.strip().isdigit()} if raw_allow else set()

            if allowed and gid not in allowed:
                LOG.error("GUILD_ID %s not in ALLOWED_GUILD_IDS; skipping guild sync/reset.", gid)
                return

            allow_reset = (os.getenv("ALLOW_COMMAND_RESET") or "").strip().lower() in ("1", "true", "yes", "y", "on")
            guild_obj = discord.Object(id=gid)

            # copy globals into guild scope, then sync to guild
            self.tree.copy_global_to(guild=guild_obj)

            if allow_reset and getattr(self, "application_id", None):
                await self.http.bulk_upsert_guild_commands(self.application_id, gid, [])
                LOG.warning("Performed GUILD command reset (ALLOW_COMMAND_RESET=true) for guild %s", gid)

            synced = await self.tree.sync(guild=guild_obj)
            LOG.info("Guild command sync complete: %s commands (hard_reset=%s)", len(synced), allow_reset)
            self._did_sync = True
        except Exception:
            LOG.exception("Guild command sync failed")

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")

        # Startup dashboard refresh (best-effort, paced, and hash-skipping)
        try:
            guilds = list(self.guilds)
            LOG.info("Startup dashboard refresh: beginning for %d guild(s)...", len(guilds))
            for g in guilds:
                try:
                    status = await refresh_all_dashboards(self, g)
                    LOG.info("Startup dashboard refresh: %s", status)
                except Exception:
                    LOG.exception("Startup dashboard refresh failed for guild %s", getattr(g, "id", "unknown"))
        except Exception:
            LOG.exception("Startup refresh wrapper failed")


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
