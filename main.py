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

LOG = logging.getLogger("VilyraBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VilyraBot: %(message)s")

DEFAULT_DASHBOARD_CHANNEL_ID = 1469879866655768738
DEFAULT_COMMAND_LOG_CHANNEL_ID = 1469879960729817098
COMMAND_LOG_CHANNEL_NAME = "Legacy-Commands-Log"

MAX_ABILITY_STARS = 5
MAX_INFL_STARS_TOTAL = 5
STAR_COST = 10
MINOR_UPGRADE_COST = 5
REP_MIN = -100
REP_MAX = 100

DASHBOARD_TEMPLATE_VERSION = 8
DASHBOARD_EDIT_MIN_INTERVAL = float(os.getenv("DASHBOARD_EDIT_MIN_INTERVAL", "3.0"))
NAV_REBUILD_DEBOUNCE_SECONDS = float(os.getenv("NAV_REBUILD_DEBOUNCE_SECONDS", "12.0"))
BOOTSTRAP_PLAYER_PAUSE_SECONDS = float(os.getenv("BOOTSTRAP_PLAYER_PAUSE_SECONDS", "3.0"))
BOOTSTRAP_BATCH_SIZE = max(1, int(os.getenv("BOOTSTRAP_BATCH_SIZE", "10")))
BOOTSTRAP_BATCH_PAUSE_SECONDS = float(os.getenv("BOOTSTRAP_BATCH_PAUSE_SECONDS", "8.0"))
BOOTSTRAP_NAV_PAUSE_SECONDS = float(os.getenv("BOOTSTRAP_NAV_PAUSE_SECONDS", "10.0"))
MAINTENANCE_PLAYER_PAUSE_SECONDS = float(os.getenv("MAINTENANCE_PLAYER_PAUSE_SECONDS", "0.8"))
EMBED_COLOR = 0x7FA1B1

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

REFRESH_MODE_CHOICES: List[app_commands.Choice[str]] = [
    app_commands.Choice(name="Initialize (slow full rebuild)", value="initialize"),
    app_commands.Choice(name="Maintenance (lighter full upkeep)", value="maintenance"),
]

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
    await safe_reply(interaction, f"❌ {str(error)}")

def normalize_channel_name(name: str) -> str:
    return (name or "").strip().lower()

def embed_hash(embed: discord.Embed) -> str:
    parts: List[str] = [
        embed.title or "",
        embed.description or "",
        str(embed.color.value if embed.color else 0),
        embed.footer.text if embed.footer else "",
    ]
    for field in embed.fields:
        parts.extend([field.name or "", field.value or "", "1" if field.inline else "0"])
    return content_hash("\n".join(parts))

def build_message_link(guild_id: int, channel_id: int, message_id: int) -> str:
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"

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
    return left_text + (" " * spaces) + right_text + "\n" + bar_line

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
            "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s;",
            (table,),
        )
        return {str(r["column_name"]) for r in rows if r and r.get("column_name")}

    async def detect_schema(self) -> None:
        self.characters_cols = await self._load_table_columns("characters")
        self.abilities_cols = await self._load_table_columns("abilities")
        self.abilities_level_col = "upgrade_level" if "upgrade_level" in self.abilities_cols else ("level" if "level" in self.abilities_cols else "upgrade_level")
        self.abilities_char_col = "character_name" if "character_name" in self.abilities_cols else ("name" if "name" in self.abilities_cols else "character_name")

    async def init_schema(self) -> None:
        stmts = [
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS archived BOOLEAN NOT NULL DEFAULT FALSE;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS legacy_plus INT NOT NULL DEFAULT 0;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS legacy_minus INT NOT NULL DEFAULT 0;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS lifetime_plus INT NOT NULL DEFAULT 0;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS lifetime_minus INT NOT NULL DEFAULT 0;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS ability_stars INT NOT NULL DEFAULT 0;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS influence_plus INT NOT NULL DEFAULT 0;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS influence_minus INT NOT NULL DEFAULT 0;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS kingdom TEXT;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();",
            "CREATE SEQUENCE IF NOT EXISTS vilyra_character_id_seq;",
            "ALTER TABLE characters ADD COLUMN IF NOT EXISTS character_id BIGINT;",
            "ALTER TABLE characters ALTER COLUMN character_id SET DEFAULT nextval('vilyra_character_id_seq');",
            "UPDATE characters SET character_id=nextval('vilyra_character_id_seq') WHERE character_id IS NULL;",
        ]
        for s in stmts:
            await self._execute(s)
        for s in [
            "CREATE UNIQUE INDEX IF NOT EXISTS characters_unique_id ON characters (guild_id, character_id);",
            "CREATE INDEX IF NOT EXISTS characters_name_lookup ON characters (guild_id, lower(name));",
            "CREATE UNIQUE INDEX IF NOT EXISTS characters_unique_guild_lower_name ON characters (guild_id, lower(name));",
        ]:
            try:
                await self._execute(s)
            except Exception:
                LOG.exception("Index creation failed; continuing")
        await self._execute(
            '''
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
            '''
        )
        for s in [
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS character_name TEXT;",
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS name TEXT;",
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS ability_name TEXT;",
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrades INT NOT NULL DEFAULT 0;",
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrade_level INT NOT NULL DEFAULT 0;",
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS level INT;",
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();",
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();",
            "ALTER TABLE abilities ADD COLUMN IF NOT EXISTS character_id BIGINT;",
        ]:
            await self._execute(s)
        await self._execute(
            '''
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
            '''
        )
        try:
            await self._execute("CREATE INDEX IF NOT EXISTS abilities_lookup ON abilities (guild_id, user_id, character_id, ability_name);")
        except Exception:
            LOG.exception("Could not create abilities_lookup; continuing")

        await self._execute(
            '''
            CREATE TABLE IF NOT EXISTS dashboard_character_messages (
                guild_id          BIGINT NOT NULL,
                character_id      BIGINT NOT NULL,
                channel_id        BIGINT NOT NULL,
                message_id        BIGINT,
                content_hash      TEXT,
                template_version  INT NOT NULL DEFAULT 0,
                updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, character_id)
            );
            '''
        )
        await self._execute("ALTER TABLE dashboard_character_messages ADD COLUMN IF NOT EXISTS content_hash TEXT;")
        await self._execute("ALTER TABLE dashboard_character_messages ADD COLUMN IF NOT EXISTS template_version INT NOT NULL DEFAULT 0;")
        await self._execute(
            '''
            CREATE TABLE IF NOT EXISTS players (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                server_rank   TEXT   NOT NULL DEFAULT 'Newcomer',
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id)
            );
            '''
        )
        await self._execute(
            '''
            CREATE TABLE IF NOT EXISTS dashboard_channel_meta (
                guild_id                 BIGINT PRIMARY KEY,
                channel_id               BIGINT NOT NULL,
                leaderboard_message_id   BIGINT,
                quicklinks_message_id    BIGINT,
                leaderboard_hash         TEXT,
                quicklinks_hash          TEXT,
                updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            '''
        )
        for s in [
            "ALTER TABLE dashboard_channel_meta ADD COLUMN IF NOT EXISTS leaderboard_message_id BIGINT;",
            "ALTER TABLE dashboard_channel_meta ADD COLUMN IF NOT EXISTS quicklinks_message_id BIGINT;",
            "ALTER TABLE dashboard_channel_meta ADD COLUMN IF NOT EXISTS leaderboard_hash TEXT;",
            "ALTER TABLE dashboard_channel_meta ADD COLUMN IF NOT EXISTS quicklinks_hash TEXT;",
            "ALTER TABLE dashboard_channel_meta ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();",
        ]:
            await self._execute(s)
        await self.detect_schema()

    async def get_character_by_name(self, guild_id: int, name: str, include_archived: bool = True) -> Optional[Dict[str, Any]]:
        return await self.get_character_by_guild_name(guild_id, name, include_archived)

    async def get_character_by_guild_name(self, guild_id: int, name: str, include_archived: bool = True) -> Optional[Dict[str, Any]]:
        name = (name or "").strip()
        if not name:
            return None
        where = "guild_id=%s AND lower(name)=lower(%s)"
        params: List[Any] = [guild_id, name]
        if not include_archived:
            where += " AND COALESCE(archived, FALSE)=FALSE"
        return await self._fetchone(f"SELECT * FROM characters WHERE {where} ORDER BY character_id ASC LIMIT 1;", tuple(params))

    async def get_character_by_id(self, guild_id: int, character_id: int, include_archived: bool = True) -> Optional[Dict[str, Any]]:
        where = "guild_id=%s AND character_id=%s"
        params: List[Any] = [guild_id, int(character_id)]
        if not include_archived:
            where += " AND COALESCE(archived, FALSE)=FALSE"
        return await self._fetchone(f"SELECT * FROM characters WHERE {where} LIMIT 1;", tuple(params))

    async def create_character(self, guild_id: int, user_id: int, name: str, kingdom: Optional[str]) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("Character name cannot be empty.")
        if await self.get_character_by_guild_name(guild_id, name, include_archived=True):
            raise ValueError("A character with that name already exists in this guild.")
        row = await self._fetchone(
            "INSERT INTO characters (guild_id, user_id, name, kingdom, archived, updated_at) VALUES (%s, %s, %s, %s, FALSE, NOW()) RETURNING character_id;",
            (guild_id, user_id, name, (kingdom or "").strip() or None),
        )
        if not row or row.get("character_id") is None:
            raise RuntimeError("Failed to create character.")
        cid = int(row["character_id"])
        await self._execute("INSERT INTO players (guild_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (guild_id, user_id))
        return cid

    async def delete_character(self, guild_id: int, character_id: int) -> bool:
        await self._execute("DELETE FROM abilities WHERE guild_id=%s AND character_id=%s;", (guild_id, int(character_id)))
        await self._execute("DELETE FROM dashboard_character_messages WHERE guild_id=%s AND character_id=%s;", (guild_id, int(character_id)))
        return (await self._execute("DELETE FROM characters WHERE guild_id=%s AND character_id=%s;", (guild_id, int(character_id)))) > 0

    async def rename_character(self, guild_id: int, character_id: int, new_name: str) -> bool:
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("New name cannot be empty.")
        collision = await self._fetchone("SELECT 1 FROM characters WHERE guild_id=%s AND lower(name)=lower(%s) AND character_id<>%s LIMIT 1;", (guild_id, new_name, int(character_id)))
        if collision:
            raise ValueError("A character with that name already exists in this guild.")
        row = await self.get_character_by_id(guild_id, character_id, include_archived=True)
        if not row:
            return False
        old_name = str(row.get("name") or "")
        await self._execute("UPDATE characters SET name=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;", (new_name, guild_id, int(character_id)))
        try:
            await self._execute(f"UPDATE abilities SET {self.abilities_char_col}=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;", (new_name, guild_id, int(character_id)))
        except Exception:
            await self._execute(f"UPDATE abilities SET {self.abilities_char_col}=%s, updated_at=NOW() WHERE guild_id=%s AND {self.abilities_char_col}=%s;", (new_name, guild_id, old_name))
        return True

    async def set_character_kingdom(self, guild_id: int, character_id: int, kingdom: Optional[str]) -> bool:
        return (await self._execute("UPDATE characters SET kingdom=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;", (((kingdom or "").strip() or None), guild_id, int(character_id)))) > 0

    async def get_character_state(self, guild_id: int, character_id: int) -> Dict[str, Any]:
        row = await self._fetchone(
            "SELECT name, user_id, character_id, legacy_plus, legacy_minus, lifetime_plus, lifetime_minus, influence_plus, influence_minus, ability_stars, kingdom, COALESCE(archived, FALSE) AS archived FROM characters WHERE guild_id=%s AND character_id=%s LIMIT 1;",
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
            "kingdom": str(row.get("kingdom") or ""),
            "archived": bool(row.get("archived") or False),
        }

    async def get_character_message_entry(self, guild_id: int, character_id: int) -> Dict[str, Any]:
        return (await self._fetchone("SELECT channel_id, message_id, content_hash, template_version, updated_at FROM dashboard_character_messages WHERE guild_id=%s AND character_id=%s;", (guild_id, int(character_id)))) or {}

    async def set_character_message_entry(self, guild_id: int, character_id: int, channel_id: int, message_id: int, content_hash_value: str) -> None:
        await self._execute(
            "INSERT INTO dashboard_character_messages (guild_id, character_id, channel_id, message_id, content_hash, template_version, updated_at) VALUES (%s, %s, %s, %s, %s, %s, NOW()) ON CONFLICT (guild_id, character_id) DO UPDATE SET channel_id=EXCLUDED.channel_id, message_id=EXCLUDED.message_id, content_hash=EXCLUDED.content_hash, template_version=EXCLUDED.template_version, updated_at=NOW();",
            (guild_id, int(character_id), channel_id, message_id, content_hash_value, DASHBOARD_TEMPLATE_VERSION),
        )

    async def clear_character_message_entry(self, guild_id: int, character_id: int) -> None:
        await self._execute("DELETE FROM dashboard_character_messages WHERE guild_id=%s AND character_id=%s;", (guild_id, int(character_id)))

    async def clear_character_message_entries_for_guild(self, guild_id: int) -> None:
        await self._execute("DELETE FROM dashboard_character_messages WHERE guild_id=%s;", (guild_id,))

    async def get_dashboard_meta(self, guild_id: int) -> Dict[str, Any]:
        return (await self._fetchone("SELECT channel_id, leaderboard_message_id, quicklinks_message_id, leaderboard_hash, quicklinks_hash, updated_at FROM dashboard_channel_meta WHERE guild_id=%s;", (guild_id,))) or {}

    async def set_dashboard_meta(self, guild_id: int, channel_id: int, leaderboard_message_id: Optional[int], quicklinks_message_id: Optional[int], leaderboard_hash: Optional[str], quicklinks_hash: Optional[str]) -> None:
        await self._execute(
            "INSERT INTO dashboard_channel_meta (guild_id, channel_id, leaderboard_message_id, quicklinks_message_id, leaderboard_hash, quicklinks_hash, updated_at) VALUES (%s, %s, %s, %s, %s, %s, NOW()) ON CONFLICT (guild_id) DO UPDATE SET channel_id=EXCLUDED.channel_id, leaderboard_message_id=EXCLUDED.leaderboard_message_id, quicklinks_message_id=EXCLUDED.quicklinks_message_id, leaderboard_hash=EXCLUDED.leaderboard_hash, quicklinks_hash=EXCLUDED.quicklinks_hash, updated_at=NOW();",
            (guild_id, channel_id, leaderboard_message_id, quicklinks_message_id, leaderboard_hash, quicklinks_hash),
        )

    async def clear_dashboard_meta(self, guild_id: int) -> None:
        await self._execute("DELETE FROM dashboard_channel_meta WHERE guild_id=%s;", (guild_id,))

    async def list_abilities_for_character(self, guild_id: int, character_id: int) -> List[Tuple[str, int]]:
        expr = "COALESCE(upgrade_level, 0) AS upgrade_level" if self.abilities_level_col == "upgrade_level" else "COALESCE(level, 0) AS upgrade_level"
        rows = await self._fetchall(f"SELECT ability_name, {expr} FROM abilities WHERE guild_id=%s AND character_id=%s ORDER BY created_at ASC, ability_name ASC;", (guild_id, int(character_id)))
        return [(str(r.get("ability_name") or "").strip(), safe_int(r.get("upgrade_level"), 0)) for r in rows if str(r.get("ability_name") or "").strip()]

    async def add_ability(self, guild_id: int, character_id: int, ability_name: str) -> None:
        ability_name = (ability_name or "").strip()
        if not ability_name:
            raise ValueError("Ability name cannot be empty.")
        st = await self.get_character_state(guild_id, character_id)
        cap = 2 + clamp(st["ability_stars"], 0, MAX_ABILITY_STARS)
        current = await self.list_abilities_for_character(guild_id, character_id)
        if len(current) >= cap:
            raise ValueError(f"Ability capacity reached ({len(current)}/{cap}). Earn more Ability Stars to add abilities.")
        if any(nm.lower() == ability_name.lower() for nm, _ in current):
            raise ValueError("That ability already exists for this character.")
        await self._execute(
            "INSERT INTO abilities (guild_id, user_id, character_id, character_name, ability_name, upgrade_level, level, upgrades, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, 0, 0, 0, NOW(), NOW());",
            (guild_id, st["user_id"], int(character_id), st["name"], ability_name),
        )

    async def rename_ability(self, guild_id: int, character_id: int, old_ability: str, new_ability: str) -> bool:
        old_ability = (old_ability or "").strip()
        new_ability = (new_ability or "").strip()
        if not old_ability or not new_ability:
            raise ValueError("Ability names cannot be empty.")
        current = await self.list_abilities_for_character(guild_id, character_id)
        if any(nm.lower() == new_ability.lower() for nm, _ in current):
            raise ValueError("That new ability name already exists for this character.")
        return (await self._execute("UPDATE abilities SET ability_name=%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s AND ability_name=%s;", (new_ability, guild_id, int(character_id), old_ability))) > 0

    async def upgrade_ability(self, guild_id: int, character_id: int, ability_name: str, pay_positive: int, pay_negative: int) -> Tuple[int, int]:
        ability_name = (ability_name or "").strip()
        pay_positive = max(0, int(pay_positive))
        pay_negative = max(0, int(pay_negative))
        if pay_positive + pay_negative != MINOR_UPGRADE_COST:
            raise ValueError(f"Each upgrade costs exactly {MINOR_UPGRADE_COST} total points (+ and - may split).")
        conn = self._require_conn()
        level_col = self.abilities_level_col
        max_level = 5
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute("SELECT legacy_plus, legacy_minus FROM characters WHERE guild_id=%s AND character_id=%s FOR UPDATE;", (guild_id, int(character_id)))
                char_row = await cur.fetchone()
                if not char_row:
                    raise ValueError("Character not found.")
                legacy_plus = safe_int(char_row.get("legacy_plus"), 0)
                legacy_minus = safe_int(char_row.get("legacy_minus"), 0)
                if legacy_plus < pay_positive:
                    raise ValueError(f"Not enough available positive points (need {pay_positive}, have {legacy_plus}).")
                if legacy_minus < pay_negative:
                    raise ValueError(f"Not enough available negative points (need {pay_negative}, have {legacy_minus}).")
                await cur.execute(f"SELECT COALESCE({level_col}, 0) AS cur_level FROM abilities WHERE guild_id=%s AND character_id=%s AND ability_name=%s ORDER BY created_at ASC LIMIT 1 FOR UPDATE;", (guild_id, int(character_id), ability_name))
                row = await cur.fetchone()
                if not row:
                    raise ValueError("Ability not found. Add it first with /ability_add.")
                cur_level = safe_int(row.get("cur_level"), 0)
                if cur_level >= max_level:
                    raise ValueError("Upgrade limit reached (5/5).")
                await cur.execute("UPDATE characters SET legacy_plus=legacy_plus-%s, legacy_minus=legacy_minus-%s, updated_at=NOW() WHERE guild_id=%s AND character_id=%s;", (pay_positive, pay_negative, guild_id, int(character_id)))
                new_level = cur_level + 1
                sets = [f"{level_col}=%s", "upgrades=COALESCE(upgrades,0)+1", "updated_at=NOW()"]
                params: List[Any] = [new_level]
                if level_col != "upgrade_level" and "upgrade_level" in self.abilities_cols:
                    sets.append("upgrade_level=%s")
                    params.append(new_level)
                if level_col != "level" and "level" in self.abilities_cols:
                    sets.append("level=%s")
                    params.append(new_level)
                params.extend([guild_id, int(character_id), ability_name])
                await cur.execute(f"UPDATE abilities SET {', '.join(sets)} WHERE guild_id=%s AND character_id=%s AND ability_name=%s;", tuple(params))
                return new_level, max_level

    async def _fetchall_first_success(self, attempts: Sequence[Tuple[str, Sequence[Any]]]) -> List[Dict[str, Any]]:
        for sql, params in attempts:
            try:
                return await self._fetchall(sql, params)
            except Exception:
                continue
        return []

    async def list_noble_titles_for_character(self, guild_id: int, character_name: str) -> List[str]:
        attempts = [
            ("SELECT tier_name, custom_name FROM economy.character_assets WHERE guild_id=%s AND lower(character_name)=lower(%s) AND lower(asset_type)='title' ORDER BY created_at ASC, id ASC;", (guild_id, character_name)),
            ("SELECT tier_name, custom_name FROM character_assets WHERE guild_id=%s AND lower(character_name)=lower(%s) AND lower(asset_type)='title' ORDER BY created_at ASC, id ASC;", (guild_id, character_name)),
            ("SELECT tier AS tier_name, asset_name AS custom_name FROM econ_assets WHERE guild_id=%s AND lower(character_name)=lower(%s) AND lower(asset_type)='title' ORDER BY created_at ASC;", (guild_id, character_name)),
        ]
        rows = await self._fetchall_first_success(attempts)
        out: List[str] = []
        for r in rows:
            tier_name = str(r.get("tier_name") or "").strip()
            custom_name = str(r.get("custom_name") or "").strip()
            if tier_name and custom_name:
                out.append(f"{tier_name} of {custom_name}")
            elif tier_name:
                out.append(tier_name)
            elif custom_name:
                out.append(custom_name)
        return out

    async def list_tourney_laurels_for_character(self, guild_id: int, character_id: int) -> List[str]:
        event_map = {"archery": "the Butts", "joust": "the Lists", "hunt": "the Great Hunt", "duel": "the Duel", "horse_race": "the Horse Race", "grand_melee": "the Grand Melee"}
        attempts = [
            ("SELECT e.event_type, t.name AS tournament_name, a.id AS award_id FROM tourney.awards a JOIN tourney.events e ON e.id = a.event_id JOIN tourney.tournaments t ON t.id = a.tournament_id WHERE a.guild_id=%s AND a.character_id=%s ORDER BY a.id DESC;", (guild_id, int(character_id))),
            ("SELECT e.event_type, t.name AS tournament_name, a.id AS award_id FROM awards a JOIN events e ON e.id = a.event_id JOIN tournaments t ON t.id = a.tournament_id WHERE a.guild_id=%s AND a.character_id=%s ORDER BY a.id DESC;", (guild_id, int(character_id))),
        ]
        rows = await self._fetchall_first_success(attempts)
        out: List[str] = []
        for r in rows:
            ceremonial_name = event_map.get(str(r.get("event_type") or "").strip().lower())
            tournament_name = str(r.get("tournament_name") or "").strip()
            if ceremonial_name and tournament_name:
                out.append(f"Champion of {ceremonial_name}, {tournament_name}")
        return out

    async def get_leaderboard_rows(self, guild_id: int) -> List[Dict[str, Any]]:
        rows = await self._fetchall(
            "SELECT c.character_id, c.user_id, c.name, COALESCE(c.ability_stars, 0) AS ability_stars, COALESCE(c.influence_plus, 0) AS influence_plus, COALESCE(c.influence_minus, 0) AS influence_minus, COALESCE(COUNT(a.ability_name), 0) AS total_abilities, COALESCE(SUM(COALESCE(a.upgrade_level, a.level, 0)), 0) AS total_ability_upgrades FROM characters c LEFT JOIN abilities a ON a.guild_id = c.guild_id AND a.character_id = c.character_id WHERE c.guild_id=%s AND COALESCE(c.archived, FALSE)=FALSE GROUP BY c.character_id, c.user_id, c.name, c.ability_stars, c.influence_plus, c.influence_minus ORDER BY c.name ASC, c.character_id ASC;",
            (guild_id,),
        )
        for row in rows:
            row["ability_stars"] = safe_int(row.get("ability_stars"), 0)
            row["influence_plus"] = safe_int(row.get("influence_plus"), 0)
            row["influence_minus"] = safe_int(row.get("influence_minus"), 0)
            row["total_abilities"] = safe_int(row.get("total_abilities"), 0)
            row["total_ability_upgrades"] = safe_int(row.get("total_ability_upgrades"), 0)
            row["total_influence_stars"] = row["influence_plus"] + row["influence_minus"]
        return rows

@dataclass
class CharacterCard:
    name: str
    character_id: int
    user_id: int
    kingdom: str
    noble_titles: List[str]
    legacy_plus: int
    legacy_minus: int
    lifetime_plus: int
    lifetime_minus: int
    ability_stars: int
    infl_plus: int
    infl_minus: int
    abilities: List[Tuple[str, int]]
    tourney_laurels: List[str]

async def build_character_card(db: Database, guild_id: int, character_id: int) -> CharacterCard:
    st = await db.get_character_state(guild_id, character_id)
    return CharacterCard(
        name=st["name"],
        character_id=st["character_id"],
        user_id=st["user_id"],
        kingdom=st["kingdom"],
        noble_titles=await db.list_noble_titles_for_character(guild_id, st["name"]),
        legacy_plus=st["legacy_plus"],
        legacy_minus=st["legacy_minus"],
        lifetime_plus=st["lifetime_plus"],
        lifetime_minus=st["lifetime_minus"],
        ability_stars=st["ability_stars"],
        infl_plus=st["influence_plus"],
        infl_minus=st["influence_minus"],
        abilities=await db.list_abilities_for_character(guild_id, character_id),
        tourney_laurels=await db.list_tourney_laurels_for_character(guild_id, character_id),
    )

def render_character_embed(card: CharacterCard) -> discord.Embed:
    net_lifetime = card.lifetime_plus - card.lifetime_minus
    desc_lines = [
        f"🏰 Kingdom: {card.kingdom}" if card.kingdom else "🏰 Kingdom:",
        f"👑 Noble Titles: {' | '.join(card.noble_titles)}" if card.noble_titles else "👑 Noble Titles:",
        "",
        f"✨ Legacy: +{card.legacy_plus}/-{card.legacy_minus}",
        f"📜 Lifetime: +{card.lifetime_plus}/-{card.lifetime_minus}",
        "",
        f"⭐ Ability Stars: {render_ability_star_bar(card.ability_stars)}",
        f"🜂 Influence Stars: {render_influence_star_bar(card.infl_minus, card.infl_plus)}",
        "",
        render_reputation_block(net_lifetime),
        "",
        "🧠 Abilities: " + (" | ".join(f"{nm} ({lvl})" for nm, lvl in card.abilities) if card.abilities else "_none set_"),
        "",
        "🏅 Tourney Laurels",
    ]
    if card.tourney_laurels:
        desc_lines.extend(card.tourney_laurels)
    else:
        desc_lines.append("None yet")
    return discord.Embed(title=f"◆ {card.name}", description="\n".join(desc_lines), color=EMBED_COLOR)

def render_leaderboard_embed(guild: discord.Guild, rows: List[Dict[str, Any]]) -> discord.Embed:
    influential = sorted(rows, key=lambda r: (-safe_int(r.get("total_influence_stars"), 0), str(r.get("name") or "").lower(), safe_int(r.get("character_id"), 0)))
    powerful = sorted(rows, key=lambda r: (-safe_int(r.get("total_abilities"), 0), -safe_int(r.get("total_ability_upgrades"), 0), str(r.get("name") or "").lower(), safe_int(r.get("character_id"), 0)))
    inf_lines = ["(Top 5 by Total Influence Stars)", ""]
    for idx, row in enumerate(influential[:5], start=1):
        inf_lines.append(f"{idx}. {row['name']} - {safe_int(row.get('total_influence_stars'), 0)}")
    if len(inf_lines) == 2:
        inf_lines.append("No characters yet.")
    pow_lines = ["(Top 5 by # of Abilities & Upgrades)", ""]
    for idx, row in enumerate(powerful[:5], start=1):
        pow_lines.append(f"{idx}. {row['name']} - {safe_int(row.get('total_abilities'), 0)} Abilities (+{safe_int(row.get('total_ability_upgrades'), 0)} upgrades)")
    if len(pow_lines) == 2:
        pow_lines.append("No characters yet.")
    embed = discord.Embed(title="🏆 Legacy Leaderboard", description="A quick look at the most influential and most powerful figures in the Legacy system.", color=EMBED_COLOR)
    embed.add_field(name="🌟 Most Influential", value="\n".join(inf_lines), inline=False)
    embed.add_field(name="⚔️ Most Powerful", value="\n".join(pow_lines), inline=False)
    return embed

async def render_quicklinks_embed(db: Database, guild: discord.Guild, channel_id: int, leaderboard_message_id: int) -> discord.Embed:
    chars = await db.list_active_characters_for_guild(guild.id)
    parts = [f"🏆 [LEADERBOARD]({build_message_link(guild.id, channel_id, leaderboard_message_id)}) 🏆", ""]
    link_parts: List[str] = []
    for row in chars:
        entry = await db.get_character_message_entry(guild.id, int(row["character_id"]))
        mid = safe_int(entry.get("message_id"), 0)
        if mid:
            link_parts.append(f"[{str(row['name'])}]({build_message_link(guild.id, channel_id, mid)})")
    description = "\n".join(parts) + (" | ".join(link_parts) if link_parts else "No character links available yet.")
    if len(description) > 4000:
        description = description[:3950].rstrip() + " ..."
    return discord.Embed(title="🧭 Quick Links", description=description, color=EMBED_COLOR)

async def get_dashboard_channel(guild: discord.Guild) -> Optional[discord.TextChannel]:
    ch_id = safe_int(os.getenv("DASHBOARD_CHANNEL_ID"), DEFAULT_DASHBOARD_CHANNEL_ID)
    ch = guild.get_channel(ch_id)
    if ch is None:
        try:
            ch = await guild.fetch_channel(ch_id)
        except Exception:
            ch = None
    return ch if isinstance(ch, discord.TextChannel) else None

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
        if ch:
            await ch.send(text, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        LOG.exception("Failed to write to command log channel")

async def _safe_fetch_message(channel: discord.TextChannel, message_id: int) -> Optional[discord.Message]:
    try:
        return await channel.fetch_message(message_id)
    except Exception:
        return None

async def refresh_character_embed(client: "VilyraBotClient", guild: discord.Guild, character_id: int) -> str:
    db = client.db
    channel = await get_dashboard_channel(guild)
    if not channel:
        return "failed"
    me = guild.me or (guild.get_member(client.user.id) if client.user else None)
    if me:
        perms = channel.permissions_for(me)
        if not (perms.view_channel and perms.send_messages and perms.embed_links):
            return "failed"
    card = await build_character_card(db, guild.id, character_id)
    embed = render_character_embed(card)
    new_hash = embed_hash(embed)
    entry = await db.get_character_message_entry(guild.id, character_id)
    old_hash = str(entry.get("content_hash") or "")
    old_tv = safe_int(entry.get("template_version"), 0)
    old_mid = safe_int(entry.get("message_id"), 0)
    old_msg = await _safe_fetch_message(channel, old_mid) if old_mid else None
    if old_msg and old_hash == new_hash and old_tv == DASHBOARD_TEMPLATE_VERSION:
        return "skipped"
    if old_msg:
        await client.dashboard_limiter.wait()
        await old_msg.edit(embed=embed)
        await db.set_character_message_entry(guild.id, character_id, channel.id, old_msg.id, new_hash)
        return "updated"
    await client.dashboard_limiter.wait()
    msg = await channel.send(embed=embed)
    await db.set_character_message_entry(guild.id, character_id, channel.id, msg.id, new_hash)
    return "created"

async def delete_character_embed_if_exists(client: "VilyraBotClient", guild: discord.Guild, character_id: int) -> None:
    db = client.db
    channel = await get_dashboard_channel(guild)
    if not channel:
        await db.clear_character_message_entry(guild.id, character_id)
        return
    entry = await db.get_character_message_entry(guild.id, character_id)
    old_mid = safe_int(entry.get("message_id"), 0)
    if old_mid:
        msg = await _safe_fetch_message(channel, old_mid)
        if msg:
            try:
                await client.dashboard_limiter.wait()
                await msg.delete()
            except Exception:
                pass
    await db.clear_character_message_entry(guild.id, character_id)

async def refresh_dashboard_navigation(client: "VilyraBotClient", guild: discord.Guild, *, force_repost: bool, dirty_leaderboard: bool, dirty_quicklinks: bool) -> str:
    db = client.db
    channel = await get_dashboard_channel(guild)
    if not channel:
        return "failed"
    me = guild.me or (guild.get_member(client.user.id) if client.user else None)
    if me:
        perms = channel.permissions_for(me)
        if not (perms.view_channel and perms.send_messages and perms.embed_links):
            return "failed"
    meta = await db.get_dashboard_meta(guild.id)
    old_leaderboard_id = safe_int(meta.get("leaderboard_message_id"), 0)
    old_quicklinks_id = safe_int(meta.get("quicklinks_message_id"), 0)
    old_leaderboard_hash = str(meta.get("leaderboard_hash") or "")
    old_quicklinks_hash = str(meta.get("quicklinks_hash") or "")
    leaderboard_msg = None if force_repost or not old_leaderboard_id else await _safe_fetch_message(channel, old_leaderboard_id)
    quicklinks_msg = None if force_repost or not old_quicklinks_id else await _safe_fetch_message(channel, old_quicklinks_id)
    rows = await db.get_leaderboard_rows(guild.id)
    leaderboard_embed = render_leaderboard_embed(guild, rows)
    leaderboard_hash = embed_hash(leaderboard_embed)
    if force_repost or leaderboard_msg is None:
        await client.dashboard_limiter.wait()
        leaderboard_msg = await channel.send(embed=leaderboard_embed)
    elif dirty_leaderboard and leaderboard_hash != old_leaderboard_hash:
        await client.dashboard_limiter.wait()
        await leaderboard_msg.edit(embed=leaderboard_embed)
    if leaderboard_msg is None:
        return "failed"
    quicklinks_embed = await render_quicklinks_embed(db, guild, channel.id, leaderboard_msg.id)
    quicklinks_hash = embed_hash(quicklinks_embed)
    if force_repost or quicklinks_msg is None:
        await client.dashboard_limiter.wait()
        quicklinks_msg = await channel.send(embed=quicklinks_embed)
    elif dirty_quicklinks and quicklinks_hash != old_quicklinks_hash:
        await client.dashboard_limiter.wait()
        await quicklinks_msg.edit(embed=quicklinks_embed)
    if force_repost:
        for old_id in [old_leaderboard_id, old_quicklinks_id]:
            if old_id:
                old_msg = await _safe_fetch_message(channel, old_id)
                if old_msg:
                    try:
                        await client.dashboard_limiter.wait()
                        await old_msg.delete()
                    except Exception:
                        pass
    await db.set_dashboard_meta(guild.id, channel.id, leaderboard_msg.id if leaderboard_msg else None, quicklinks_msg.id if quicklinks_msg else None, leaderboard_hash, quicklinks_hash)
    return "updated"

async def run_initialize_refresh(client: "VilyraBotClient", guild: discord.Guild) -> str:
    db = client.db
    channel = await get_dashboard_channel(guild)
    if not channel:
        return "Initialize failed: dashboard channel not found."
    await db.clear_character_message_entries_for_guild(guild.id)
    await db.clear_dashboard_meta(guild.id)
    chars = await db.list_active_characters_for_guild(guild.id)
    counts = {"updated": 0, "created": 0, "skipped": 0, "cleared": 0, "failed": 0}
    nav_status = await refresh_dashboard_navigation(client, guild, force_repost=True, dirty_leaderboard=True, dirty_quicklinks=False)
    await asyncio.sleep(BOOTSTRAP_NAV_PAUSE_SECONDS)
    for idx, row in enumerate(chars, start=1):
        cid = int(row["character_id"])
        try:
            status = await refresh_character_embed(client, guild, cid)
            counts[status if status in counts else "failed"] += 1
        except Exception:
            LOG.exception("initialize refresh_character_embed failed for character_id=%s", cid)
            counts["failed"] += 1
        await asyncio.sleep(BOOTSTRAP_PLAYER_PAUSE_SECONDS)
        if idx % BOOTSTRAP_BATCH_SIZE == 0 and idx < len(chars):
            await asyncio.sleep(BOOTSTRAP_BATCH_PAUSE_SECONDS)
    await asyncio.sleep(BOOTSTRAP_NAV_PAUSE_SECONDS)
    nav_status = await refresh_dashboard_navigation(client, guild, force_repost=True, dirty_leaderboard=True, dirty_quicklinks=True)
    return (
        "Initialize complete: "
        f"{counts['updated']} updated, "
        f"{counts['created']} created, "
        f"{counts['skipped']} skipped, "
        f"{counts['cleared']} cleared, "
        f"{counts['failed']} failed. "
        f"Leaderboards/navigation: {nav_status}."
    )

async def run_maintenance_refresh(client: "VilyraBotClient", guild: discord.Guild) -> str:
    chars = await client.db.list_active_characters_for_guild(guild.id)
    counts = {"updated": 0, "created": 0, "skipped": 0, "cleared": 0, "failed": 0}
    for row in chars:
        cid = int(row["character_id"])
        try:
            status = await refresh_character_embed(client, guild, cid)
            counts[status if status in counts else "failed"] += 1
        except Exception:
            LOG.exception("maintenance refresh_character_embed failed for character_id=%s", cid)
            counts["failed"] += 1
        await asyncio.sleep(MAINTENANCE_PLAYER_PAUSE_SECONDS)
    nav_status = await refresh_dashboard_navigation(client, guild, force_repost=False, dirty_leaderboard=True, dirty_quicklinks=True)
    return (
        "Maintenance complete: "
        f"{counts['updated']} updated, "
        f"{counts['created']} created, "
        f"{counts['skipped']} skipped, "
        f"{counts['cleared']} cleared, "
        f"{counts['failed']} failed. "
        f"Leaderboards/navigation: {nav_status}."
    )

async def refresh_character_and_schedule_nav(client: "VilyraBotClient", guild: discord.Guild, character_id: int, *, dirty_leaderboard: bool, dirty_quicklinks: bool, structural_rebuild: bool = False) -> str:
    status = await refresh_character_embed(client, guild, character_id)
    if dirty_leaderboard or dirty_quicklinks or structural_rebuild:
        client.mark_navigation_dirty(guild.id, dirty_leaderboard=dirty_leaderboard, dirty_quicklinks=dirty_quicklinks, force_repost=structural_rebuild)
        client.ensure_navigation_rebuild(guild)
    return status

async def autocomplete_character(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    try:
        guild = interaction.guild
        if guild is None:
            return []
        rows = await interaction.client.db.list_all_characters_for_guild(guild.id, include_archived=True, name_filter=(current or "").strip(), limit=25)
        out: List[app_commands.Choice[str]] = []
        for r in rows:
            cid = int(r.get("character_id") or 0)
            nm = str(r.get("name") or "").strip()
            if cid > 0 and nm:
                label = f"{nm} - #{cid}"
                out.append(app_commands.Choice(name=label[:100], value=str(cid)))
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
    row = await interaction.client.db.get_character_by_name(guild.id, t, include_archived=True)
    if not row:
        raise ValueError("Character not found.")
    return int(row["character_id"]), row

async def autocomplete_ability(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    try:
        guild = interaction.guild
        if guild is None:
            return []
        char_token = getattr(interaction.namespace, "character", None) or getattr(interaction.namespace, "character_name", None)
        if not char_token:
            return []
        cid, _ = await resolve_character(interaction, str(char_token))
        abilities = await interaction.client.db.list_abilities_for_character(guild.id, cid)
        q = (current or "").strip().lower()
        out: List[app_commands.Choice[str]] = []
        for nm, lvl in abilities:
            if q and q not in nm.lower():
                continue
            out.append(app_commands.Choice(name=f"{nm} ({lvl})"[:100], value=nm))
            if len(out) >= 25:
                break
        return out
    except Exception:
        LOG.exception("Ability autocomplete failed")
        return []

def in_guild_only(func=None):
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False
        return True
    decorator = app_commands.check(predicate)
    return decorator(func) if callable(func) else (lambda f: decorator(f))

def _parse_env_id_set(*names: str) -> set[int]:
    out: set[int] = set()
    for name in names:
        raw = os.getenv(name, "") or ""
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                out.add(int(part))
    return out

def staff_only(func=None):
    staff_user_ids = _parse_env_id_set("STAFF_USER_IDS")
    staff_role_ids = _parse_env_id_set("STAFF_ROLES_IDS", "STAFF_ROLE_IDS")
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False
        member = interaction.user
        if not isinstance(member, discord.Member):
            member = interaction.guild.get_member(interaction.user.id)  # type: ignore
        if interaction.user.id in staff_user_ids or interaction.guild.owner_id == interaction.user.id:
            return True
        if isinstance(member, discord.Member):
            if staff_role_ids and any(role.id in staff_role_ids for role in member.roles):
                return True
            perms = member.guild_permissions
            if perms.administrator or perms.manage_guild:
                return True
        await safe_reply(interaction, "You don't have permission to use this command.")
        return False
    decorator = app_commands.check(predicate)
    return decorator(func) if callable(func) else (lambda f: decorator(f))

@app_commands.command(name="set_server_rank", description="(Staff) Set a player's server rank.")
@in_guild_only()
@staff_only
@app_commands.describe(user="Player to update", rank="New server rank")
@app_commands.choices(rank=RANK_CHOICES)
async def set_server_rank(interaction: discord.Interaction, user: discord.Member, rank: app_commands.Choice[str]) -> None:
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        await interaction.client.db.set_player_rank(interaction.guild.id, user.id, str(rank.value))
        await log_command(interaction, f"🏷️ {interaction.user.mention} set server rank for {user.mention} → **{rank.value}**")
        await safe_reply(interaction, f"✅ Server rank for {user.mention} set to **{rank.value}**.")
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
            "• **/character_rename** — Rename a character.",
            "• **/set_char_kingdom** — Change a character's kingdom.",
            "• **/award_legacy** — Award legacy points (+/-) and lifetime totals.",
            "• **/convert_stars** — Spend 10 legacy points to add a star.",
            "• **/ability_add** — Add an ability.",
            "• **/ability_rename** — Rename an ability.",
            "• **/ability_upgrade** — Upgrade an ability.",
            "• **/set_available_legacy** — Set available legacy points.",
            "• **/set_lifetime_legacy** — Set lifetime legacy points.",
            "• **/refresh_dashboard mode:** initialize / maintenance",
            "",
            "Note: **/char_card** is a private, ephemeral lookup and is not logged.",
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
        status = await run_initialize_refresh(interaction.client, interaction.guild)
        await safe_reply(interaction, f"✅ Created **{character_name}** for {user.mention}.\n{status}")
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
        name = str(row["name"])
        ok = await run_db(interaction.client.db.delete_character(interaction.guild.id, cid), "delete_character")
        if not ok:
            raise RuntimeError("Character not found.")
        await run_db(delete_character_embed_if_exists(interaction.client, interaction.guild, cid), "delete_character_embed_if_exists")
        interaction.client.mark_navigation_dirty(interaction.guild.id, dirty_leaderboard=True, dirty_quicklinks=True, force_repost=False)
        interaction.client.ensure_navigation_rebuild(interaction.guild)
        await log_command(interaction, f"🗑️ {interaction.user.mention} deleted **{name}** `#{cid}`")
        await safe_reply(interaction, f"✅ Deleted **{name}**.")
    except Exception as e:
        LOG.exception("character_delete failed")
        await send_error(interaction, e)

@app_commands.command(name="character_rename", description="(Staff) Rename a character.")
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
        ok = await run_db(interaction.client.db.rename_character(interaction.guild.id, cid, new_name), "rename_character")
        if not ok:
            raise RuntimeError("Character not found.")
        await run_db(refresh_character_and_schedule_nav(interaction.client, interaction.guild, cid, dirty_leaderboard=True, dirty_quicklinks=True, structural_rebuild=False), "refresh_character_and_schedule_nav")
        await log_command(interaction, f"✏️ {interaction.user.mention} renamed **{old}** `#{cid}` → **{new_name.strip()}**")
        await safe_reply(interaction, f"✅ Renamed **{old}** → **{new_name.strip()}**.")
    except Exception as e:
        LOG.exception("character_rename failed")
        await send_error(interaction, e)

@app_commands.command(name="set_char_kingdom", description="(Staff) Set a character's kingdom.")
@in_guild_only()
@staff_only
@app_commands.describe(character="Character (autocomplete)", kingdom="New kingdom")
@app_commands.choices(kingdom=KINGDOM_CHOICES)
@app_commands.autocomplete(character=autocomplete_character)
async def set_char_kingdom(interaction: discord.Interaction, character: str, kingdom: app_commands.Choice[str]):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, row = await resolve_character(interaction, character)
        ok = await run_db(interaction.client.db.set_character_kingdom(interaction.guild.id, cid, kingdom.value), "set_character_kingdom")
        if not ok:
            raise RuntimeError("Character not found.")
        await run_db(refresh_character_embed(interaction.client, interaction.guild, cid), "refresh_character_embed")
        await log_command(interaction, f"🏰 {interaction.user.mention} set kingdom for **{row['name']}** `#{cid}` → **{kingdom.value}**")
        await safe_reply(interaction, f"✅ Set **{row['name']}** to kingdom **{kingdom.value}**.")
    except Exception as e:
        LOG.exception("set_char_kingdom failed")
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
        await run_db(refresh_character_embed(interaction.client, interaction.guild, cid), "refresh_character_embed")
        await log_command(interaction, f"🏅 {interaction.user.mention} awarded legacy to **{st['name']}** `#{cid}`: +{pos}/-{neg}")
        await safe_reply(interaction, f"✅ Awarded **{st['name']}**: +{pos}/-{neg}.")
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
async def convert_stars(interaction: discord.Interaction, character: str, star_type: app_commands.Choice[str], spend_plus: int, spend_minus: int):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        cid, _ = await resolve_character(interaction, character)
        await run_db(interaction.client.db.convert_stars(interaction.guild.id, cid, star_type.value, spend_plus, spend_minus), "convert_stars")
        st = await run_db(interaction.client.db.get_character_state(interaction.guild.id, cid), "get_character_state")
        await run_db(refresh_character_and_schedule_nav(interaction.client, interaction.guild, cid, dirty_leaderboard=True, dirty_quicklinks=False, structural_rebuild=False), "refresh_character_and_schedule_nav")
        await log_command(interaction, f"⭐ {interaction.user.mention} converted legacy to {star_type.name} for **{st['name']}** `#{cid}` (spent +{spend_plus}/-{spend_minus})")
        await safe_reply(interaction, f"✅ Added **{star_type.name}** to **{st['name']}**.")
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
        await run_db(refresh_character_and_schedule_nav(interaction.client, interaction.guild, cid, dirty_leaderboard=True, dirty_quicklinks=False, structural_rebuild=False), "refresh_character_and_schedule_nav")
        await log_command(interaction, f"➕ {interaction.user.mention} added ability **{ability_name.strip()}** to **{st['name']}** `#{cid}`")
        await safe_reply(interaction, f"✅ Added ability **{ability_name.strip()}** to **{st['name']}**.")
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
        await run_db(refresh_character_embed(interaction.client, interaction.guild, cid), "refresh_character_embed")
        await log_command(interaction, f"✏️ {interaction.user.mention} renamed ability on **{st['name']}** `#{cid}`: **{ability}** → **{new_ability_name.strip()}**")
        await safe_reply(interaction, f"✅ Renamed ability **{ability}** → **{new_ability_name.strip()}** for **{st['name']}**.")
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
        await run_db(refresh_character_and_schedule_nav(interaction.client, interaction.guild, cid, dirty_leaderboard=True, dirty_quicklinks=False, structural_rebuild=False), "refresh_character_and_schedule_nav")
        await log_command(interaction, f"⬆️ {interaction.user.mention} upgraded **{ability}** for **{st['name']}** `#{cid}` to {new_level}/{max_level}")
        await safe_reply(interaction, f"✅ Upgraded **{ability}** for **{st['name']}** to {new_level}/{max_level}.")
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
        await run_db(refresh_character_embed(interaction.client, interaction.guild, cid), "refresh_character_embed")
        await log_command(interaction, f"🧮 {interaction.user.mention} set AVAILABLE legacy for **{st['name']}** `#{cid}` to +{positive}/-{negative}")
        await safe_reply(interaction, f"✅ Set available legacy for **{st['name']}** to +{positive}/-{negative}.")
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
        await run_db(refresh_character_embed(interaction.client, interaction.guild, cid), "refresh_character_embed")
        await log_command(interaction, f"📈 {interaction.user.mention} set LIFETIME legacy for **{st['name']}** `#{cid}` to +{positive}/-{negative}")
        await safe_reply(interaction, f"✅ Set lifetime legacy for **{st['name']}** to +{positive}/-{negative}.")
    except Exception as e:
        LOG.exception("set_lifetime_legacy failed")
        await send_error(interaction, e)

@app_commands.command(name="refresh_dashboard", description="(Staff) Refresh the dashboard in initialize or maintenance mode.")
@in_guild_only()
@staff_only
@app_commands.describe(mode="Initialize for slow rebuild from scratch, or maintenance for lighter upkeep")
@app_commands.choices(mode=REFRESH_MODE_CHOICES)
async def refresh_dashboard(interaction: discord.Interaction, mode: app_commands.Choice[str]):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        status = await (run_initialize_refresh(interaction.client, interaction.guild) if mode.value == "initialize" else run_maintenance_refresh(interaction.client, interaction.guild))
        await log_command(interaction, f"🔄 {interaction.user.mention} ran /refresh_dashboard mode:{mode.value} ({status})")
        await safe_reply(interaction, status)
    except Exception as e:
        LOG.exception("refresh_dashboard failed")
        await send_error(interaction, e)

@app_commands.command(name="char_card", description="Show a character card privately.")
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
        text = f"{owner_mention}\n\n{render_character_embed(card).description}"
        await safe_reply(interaction, text)
    except Exception as e:
        LOG.exception("char_card failed")
        await send_error(interaction, e)

class VilyraBotClient(discord.Client):
    def __init__(self, db: Database) -> None:
        intents = discord.Intents.default()
        intents.members = True
        super().__init__(intents=intents)
        self.db = db
        self.tree = app_commands.CommandTree(self)
        self.dashboard_limiter = SimpleRateLimiter(DASHBOARD_EDIT_MIN_INTERVAL)
        self._did_sync = False
        self._nav_dirty: Dict[int, Dict[str, bool]] = {}
        self._nav_tasks: Dict[int, asyncio.Task] = {}

    def mark_navigation_dirty(self, guild_id: int, *, dirty_leaderboard: bool, dirty_quicklinks: bool, force_repost: bool = False) -> None:
        state = self._nav_dirty.setdefault(guild_id, {"leaderboard": False, "quicklinks": False, "force_repost": False})
        if dirty_leaderboard:
            state["leaderboard"] = True
        if dirty_quicklinks:
            state["quicklinks"] = True
        if force_repost:
            state["force_repost"] = True

    def ensure_navigation_rebuild(self, guild: discord.Guild) -> None:
        task = self._nav_tasks.get(guild.id)
        if task and not task.done():
            return
        async def runner() -> None:
            try:
                await asyncio.sleep(NAV_REBUILD_DEBOUNCE_SECONDS)
                state = self._nav_dirty.pop(guild.id, {"leaderboard": False, "quicklinks": False, "force_repost": False})
                if not state["leaderboard"] and not state["quicklinks"] and not state["force_repost"]:
                    return
                await refresh_dashboard_navigation(self, guild, force_repost=bool(state["force_repost"]), dirty_leaderboard=bool(state["leaderboard"]), dirty_quicklinks=bool(state["quicklinks"]))
            except Exception:
                LOG.exception("Debounced navigation rebuild failed for guild_id=%s", guild.id)
            finally:
                self._nav_tasks.pop(guild.id, None)
        self._nav_tasks[guild.id] = asyncio.create_task(runner())

    async def setup_hook(self) -> None:
        for cmd in [staff_commands, set_server_rank, character_create, character_delete, character_rename, set_char_kingdom, award_legacy, convert_stars, ability_add, ability_rename, ability_upgrade, set_available_legacy, set_lifetime_legacy, refresh_dashboard, char_card]:
            self.tree.add_command(cmd)
        names = [c.name for c in self.tree.get_commands()]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise RuntimeError(f"Duplicate command name(s) detected: {dupes}")
        await self._sync_commands()

    async def _sync_commands(self) -> None:
        if self._did_sync:
            return
        self._did_sync = True
        try:
            gid = safe_int(os.getenv("GUILD_ID"), 0)
            if not gid:
                await self.tree.sync()
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
            allow_global_reset = (os.getenv("ALLOW_GLOBAL_COMMAND_RESET") or "").strip().lower() in ("1", "true", "yes", "y", "on")
            if allow_global_reset:
                await self.http.bulk_upsert_global_commands(self.application_id, [])
            allow_reset = (os.getenv("ALLOW_COMMAND_RESET") or "").strip().lower() in ("1", "true", "yes", "y", "on")
            guild_obj = discord.Object(id=gid)
            self.tree.copy_global_to(guild=guild_obj)
            if allow_reset and self.application_id:
                await self.http.bulk_upsert_guild_commands(self.application_id, gid, [])
            await self.tree.sync(guild=guild_obj)
        except Exception:
            LOG.exception("Command sync failed")

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")
        do_refresh = (os.getenv("STARTUP_REFRESH") or "").strip().lower() in ("1", "true", "yes", "y", "on")
        if not do_refresh:
            LOG.info("Startup refresh disabled (set STARTUP_REFRESH=true to enable).")
            return
        for g in list(self.guilds):
            try:
                status = await run_maintenance_refresh(self, g)
                LOG.info("Startup dashboard refresh: %s", status)
            except Exception:
                LOG.exception("Startup dashboard refresh failed")

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
