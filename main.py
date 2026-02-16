import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import discord
from discord import app_commands

import psycopg
from psycopg.rows import dict_row
from psycopg import errors as pg_errors


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

PLAYER_POST_SOFT_LIMIT = 1900

SERVER_RANKS = [
    "Newcomer",
    "Apprentice",
    "Adventurer",
    "Sentinel",
    "Champion",
    "Legend",
    "Sovereign",
]

BORDER_LEN = 60
PLAYER_BORDER = "═" * BORDER_LEN
CHAR_SEPARATOR = "-" * BORDER_LEN
CHAR_HEADER_LEFT = "꧁•⊹٭ "
CHAR_HEADER_RIGHT = " ٭⊹•꧂"


LOG = logging.getLogger("VilyraBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VilyraBot: %(message)s")


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
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True, thinking=True)
    except Exception:
        pass


async def safe_followup(interaction: discord.Interaction, content: str) -> None:
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=True)
        else:
            await interaction.response.send_message(content, ephemeral=True)
    except Exception:
        LOG.exception("Failed to send response/followup")


async def run_db(coro, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=db_timeout())
    except asyncio.TimeoutError as e:
        raise RuntimeError(f"Database operation timed out ({label}).") from e


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

    return f"- {''.join(neg_slots)} | {''.join(pos_slots)} +"


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

    bar_line = f"[{''.join(bar)}]"

    center_col = 1 + center_idx
    left_text = "FEARED ←"
    right_text = "→ LOVED"

    prefix = left_text + " "
    pad_left = max(0, center_col - len(prefix))
    explainer = prefix + (" " * pad_left) + "│" + (" " * pad_left) + " " + right_text

    return f"{explainer}
{bar_line}"


class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg.AsyncConnection] = None
        self.characters_name_col: str = "character_name"
        self.abilities_name_col: str = "ability_name"
        self.abilities_level_col: str = "upgrade_level"

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

    async def _execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        conn = self._require_conn()
        async with conn.cursor() as cur:
            await cur.execute(sql, params)

    async def _fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        conn = self._require_conn()
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            return list(rows or [])

    async def _fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        conn = self._require_conn()
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()

    async def _has_column(self, table: str, column: str) -> bool:
        row = await self._fetchone(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s AND column_name=%s
            LIMIT 1;
            """,
            (table, column),
        )
        return bool(row)

    async def detect_compat_columns(self) -> None:
        if await self._has_column("characters", "character_name"):
            self.characters_name_col = "character_name"
        elif await self._has_column("characters", "name"):
            self.characters_name_col = "name"
        else:
            self.characters_name_col = "character_name"

        if await self._has_column("abilities", "ability_name"):
            self.abilities_name_col = "ability_name"
        elif await self._has_column("abilities", "ability"):
            self.abilities_name_col = "ability"
        else:
            self.abilities_name_col = "ability_name"

        if await self._has_column("abilities", "upgrade_level"):
            self.abilities_level_col = "upgrade_level"
        elif await self._has_column("abilities", "level"):
            self.abilities_level_col = "level"
        else:
            self.abilities_level_col = "upgrade_level"

        LOG.info(
            "Schema compat detected: characters.%s, abilities.%s, abilities.%s",
            self.characters_name_col, self.abilities_name_col, self.abilities_level_col,
        )

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
            CREATE TABLE IF NOT EXISTS characters (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT  NULL,
                name           TEXT  NULL,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS character_name TEXT;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS name TEXT;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS legacy_points (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                avail_pos      INT   NOT NULL DEFAULT 0,
                avail_neg      INT   NOT NULL DEFAULT 0,
                life_pos       INT   NOT NULL DEFAULT 0,
                life_neg       INT   NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, character_name)
            );
            """
        )

        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS stars (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                ability_stars  INT NOT NULL DEFAULT 0,
                infl_pos       INT NOT NULL DEFAULT 0,
                infl_neg       INT NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, character_name)
            );
            """
        )

        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                ability_name   TEXT NULL,
                ability        TEXT NULL,
                upgrade_level  INT NOT NULL DEFAULT 0,
                level          INT NULL,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS ability_name TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS ability TEXT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS upgrade_level INT NOT NULL DEFAULT 0;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS level INT;")
        await self._execute("ALTER TABLE abilities ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_messages (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                channel_id    BIGINT NOT NULL,
                message_ids   TEXT,
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id)
            );
            """
        )

        await self.detect_compat_columns()
        LOG.info("Database schema initialized / updated")

    async def get_player_rank(self, guild_id: int, user_id: int) -> str:
        row = await self._fetchone("SELECT server_rank FROM players WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))
        return row["server_rank"] if row and row.get("server_rank") else "Newcomer"

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

    async def add_character(self, guild_id: int, user_id: int, character_name: str) -> None:
        character_name = character_name.strip()
        if not character_name:
            raise ValueError("Character name cannot be empty.")
        await self._execute(
            "INSERT INTO characters (guild_id, user_id, character_name, name) VALUES (%s, %s, %s, %s);",
            (guild_id, user_id, character_name, character_name),
        )
        await self._execute(
            "INSERT INTO legacy_points (guild_id, user_id, character_name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
            (guild_id, user_id, character_name),
        )
        await self._execute(
            "INSERT INTO stars (guild_id, user_id, character_name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
            (guild_id, user_id, character_name),
        )
        await self._execute(
            "INSERT INTO players (guild_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (guild_id, user_id),
        )

    async def get_character(self, guild_id: int, user_id: int, character_name: str) -> Optional[Dict[str, Any]]:
        col = self.characters_name_col
        return await self._fetchone(
            f"SELECT {col} AS character_name FROM characters WHERE guild_id=%s AND user_id=%s AND {col}=%s LIMIT 1;",
            (guild_id, user_id, character_name.strip()),
        )

    async def list_characters(self, guild_id: int, user_id: int) -> List[str]:
        col = self.characters_name_col
        try:
            rows = await self._fetchall(
                f"SELECT {col} AS character_name FROM characters WHERE guild_id=%s AND user_id=%s ORDER BY created_at ASC, {col} ASC;",
                (guild_id, user_id),
            )
            return [r["character_name"] for r in rows if r.get("character_name")]
        except pg_errors.UndefinedColumn:
            fallback = "name" if col == "character_name" else "character_name"
            rows = await self._fetchall(
                f"SELECT {fallback} AS character_name FROM characters WHERE guild_id=%s AND user_id=%s ORDER BY created_at ASC, {fallback} ASC;",
                (guild_id, user_id),
            )
            self.characters_name_col = fallback
            return [r["character_name"] for r in rows if r.get("character_name")]

    async def list_player_ids(self, guild_id: int) -> List[int]:
        col = self.characters_name_col
        rows = await self._fetchall(
            f"SELECT DISTINCT user_id FROM characters WHERE guild_id=%s AND {col} IS NOT NULL ORDER BY user_id ASC;",
            (guild_id,),
        )
        return [int(r["user_id"]) for r in rows]

    async def get_legacy(self, guild_id: int, user_id: int, character_name: str) -> Dict[str, int]:
        row = await self._fetchone(
            "SELECT avail_pos, avail_neg, life_pos, life_neg FROM legacy_points WHERE guild_id=%s AND user_id=%s AND character_name=%s;",
            (guild_id, user_id, character_name.strip()),
        )
        if not row:
            return {"avail_pos": 0, "avail_neg": 0, "life_pos": 0, "life_neg": 0}
        return {k: safe_int(row.get(k), 0) for k in ("avail_pos", "avail_neg", "life_pos", "life_neg")}

    async def award_legacy(self, guild_id: int, user_id: int, character_name: str, pos: int = 0, neg: int = 0) -> Dict[str, int]:
        pos = max(0, int(pos))
        neg = max(0, int(neg))
        cur = await self.get_legacy(guild_id, user_id, character_name)
        new = dict(cur)
        new["avail_pos"] += pos
        new["life_pos"] += pos
        new["avail_neg"] += neg
        new["life_neg"] += neg
        await self._execute(
            """
            INSERT INTO legacy_points (guild_id, user_id, character_name, avail_pos, avail_neg, life_pos, life_neg)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (guild_id, user_id, character_name)
            DO UPDATE SET avail_pos=EXCLUDED.avail_pos, avail_neg=EXCLUDED.avail_neg,
                          life_pos=EXCLUDED.life_pos, life_neg=EXCLUDED.life_neg;
            """,
            (guild_id, user_id, character_name.strip(), new["avail_pos"], new["avail_neg"], new["life_pos"], new["life_neg"]),
        )
        return new

    async def spend_legacy(self, guild_id: int, user_id: int, character_name: str, pool: str, amount: int) -> Dict[str, int]:
        amount = max(0, int(amount))
        pool = pool.strip().lower()
        if pool not in ("positive", "negative"):
            raise ValueError("pool must be positive or negative")
        cur = await self.get_legacy(guild_id, user_id, character_name)
        new = dict(cur)
        if pool == "positive":
            if new["avail_pos"] < amount:
                raise ValueError(f"Not enough available positive points (need {amount}, have {new['avail_pos']})")
            new["avail_pos"] -= amount
        else:
            if new["avail_neg"] < amount:
                raise ValueError(f"Not enough available negative points (need {amount}, have {new['avail_neg']})")
            new["avail_neg"] -= amount
        await self._execute(
            "UPDATE legacy_points SET avail_pos=%s, avail_neg=%s WHERE guild_id=%s AND user_id=%s AND character_name=%s;",
            (new["avail_pos"], new["avail_neg"], guild_id, user_id, character_name.strip()),
        )
        return new

    async def reset_legacy(self, guild_id: int, user_id: int, character_name: str, target: str,
                           avail_pos: Optional[int], avail_neg: Optional[int],
                           life_pos: Optional[int], life_neg: Optional[int]) -> Dict[str, int]:
        target = target.strip().lower()
        if target not in ("available", "lifetime", "both"):
            raise ValueError("target must be available, lifetime, or both")
        cur = await self.get_legacy(guild_id, user_id, character_name)
        new = dict(cur)
        if target in ("available", "both"):
            if avail_pos is not None:
                new["avail_pos"] = max(0, int(avail_pos))
            if avail_neg is not None:
                new["avail_neg"] = max(0, int(avail_neg))
        if target in ("lifetime", "both"):
            if life_pos is not None:
                new["life_pos"] = max(0, int(life_pos))
            if life_neg is not None:
                new["life_neg"] = max(0, int(life_neg))
        await self._execute(
            """
            INSERT INTO legacy_points (guild_id, user_id, character_name, avail_pos, avail_neg, life_pos, life_neg)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (guild_id, user_id, character_name)
            DO UPDATE SET avail_pos=EXCLUDED.avail_pos, avail_neg=EXCLUDED.avail_neg,
                          life_pos=EXCLUDED.life_pos, life_neg=EXCLUDED.life_neg;
            """,
            (guild_id, user_id, character_name.strip(), new["avail_pos"], new["avail_neg"], new["life_pos"], new["life_neg"]),
        )
        return new

    async def get_stars(self, guild_id: int, user_id: int, character_name: str) -> Dict[str, int]:
        row = await self._fetchone(
            "SELECT ability_stars, infl_pos, infl_neg FROM stars WHERE guild_id=%s AND user_id=%s AND character_name=%s;",
            (guild_id, user_id, character_name.strip()),
        )
        if not row:
            return {"ability_stars": 0, "infl_pos": 0, "infl_neg": 0}
        return {k: safe_int(row.get(k), 0) for k in ("ability_stars", "infl_pos", "infl_neg")}

    async def set_stars(self, guild_id: int, user_id: int, character_name: str,
                        ability_stars: Optional[int] = None, infl_pos: Optional[int] = None, infl_neg: Optional[int] = None) -> Dict[str, int]:
        cur = await self.get_stars(guild_id, user_id, character_name)
        new = dict(cur)
        if ability_stars is not None:
            new["ability_stars"] = clamp(int(ability_stars), 0, MAX_ABILITY_STARS)
        if infl_pos is not None:
            new["infl_pos"] = clamp(int(infl_pos), 0, MAX_INFL_STARS_TOTAL)
        if infl_neg is not None:
            new["infl_neg"] = clamp(int(infl_neg), 0, MAX_INFL_STARS_TOTAL)
        if new["infl_pos"] + new["infl_neg"] > MAX_INFL_STARS_TOTAL:
            raise ValueError("Total influence stars (pos+neg) cannot exceed 5")
        await self._execute(
            """
            INSERT INTO stars (guild_id, user_id, character_name, ability_stars, infl_pos, infl_neg)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (guild_id, user_id, character_name)
            DO UPDATE SET ability_stars=EXCLUDED.ability_stars, infl_pos=EXCLUDED.infl_pos, infl_neg=EXCLUDED.infl_neg;
            """,
            (guild_id, user_id, character_name.strip(), new["ability_stars"], new["infl_pos"], new["infl_neg"]),
        )
        return new

    async def convert_star(self, guild_id: int, user_id: int, character_name: str,
                           star_type: str, pool: Optional[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
        star_type = star_type.strip().lower()
        pool = pool.strip().lower() if pool else None

        legacy = await self.get_legacy(guild_id, user_id, character_name)
        stars = await self.get_stars(guild_id, user_id, character_name)

        if star_type == "influence_positive":
            legacy = await self.spend_legacy(guild_id, user_id, character_name, "positive", STAR_COST)
            if stars["infl_pos"] + stars["infl_neg"] >= MAX_INFL_STARS_TOTAL:
                raise ValueError("Influence stars already at max total (5)")
            stars = await self.set_stars(guild_id, user_id, character_name, infl_pos=stars["infl_pos"] + 1)
        elif star_type == "influence_negative":
            legacy = await self.spend_legacy(guild_id, user_id, character_name, "negative", STAR_COST)
            if stars["infl_pos"] + stars["infl_neg"] >= MAX_INFL_STARS_TOTAL:
                raise ValueError("Influence stars already at max total (5)")
            stars = await self.set_stars(guild_id, user_id, character_name, infl_neg=stars["infl_neg"] + 1)
        elif star_type == "ability":
            if pool not in ("positive", "negative"):
                raise ValueError("For ability stars, pool must be positive or negative.")
            legacy = await self.spend_legacy(guild_id, user_id, character_name, pool, STAR_COST)
            if stars["ability_stars"] >= MAX_ABILITY_STARS:
                raise ValueError("Ability stars already at max (5)")
            stars = await self.set_stars(guild_id, user_id, character_name, ability_stars=stars["ability_stars"] + 1)
        else:
            raise ValueError("star_type must be ability, influence_positive, or influence_negative")

        return legacy, stars

    async def list_abilities(self, guild_id: int, user_id: int, character_name: str) -> List[Dict[str, Any]]:
        name_col = self.abilities_name_col
        lvl_col = self.abilities_level_col
        try:
            return await self._fetchall(
                f"""
                SELECT {name_col} AS ability_name, COALESCE({lvl_col}, 0) AS upgrade_level
                FROM abilities
                WHERE guild_id=%s AND user_id=%s AND character_name=%s
                ORDER BY created_at ASC, {name_col} ASC;
                """,
                (guild_id, user_id, character_name.strip()),
            )
        except pg_errors.UndefinedColumn:
            self.abilities_name_col = "ability_name"
            self.abilities_level_col = "upgrade_level"
            return await self._fetchall(
                """
                SELECT ability_name AS ability_name, COALESCE(upgrade_level, 0) AS upgrade_level
                FROM abilities
                WHERE guild_id=%s AND user_id=%s AND character_name=%s
                ORDER BY created_at ASC, ability_name ASC;
                """,
                (guild_id, user_id, character_name.strip()),
            )

    async def add_ability(self, guild_id: int, user_id: int, character_name: str, ability_name: str) -> None:
        ability_name = ability_name.strip()
        if not ability_name:
            raise ValueError("Ability name cannot be empty.")
        stars = await self.get_stars(guild_id, user_id, character_name)
        existing = await self.list_abilities(guild_id, user_id, character_name)
        cap = 2 + clamp(stars["ability_stars"], 0, MAX_ABILITY_STARS)
        if len(existing) >= cap:
            raise ValueError(f"Ability capacity reached ({len(existing)}/{cap}). Earn more Ability Stars to add abilities.")
        await self._execute(
            "INSERT INTO abilities (guild_id, user_id, character_name, ability_name, ability, upgrade_level) VALUES (%s, %s, %s, %s, %s, 0);",
            (guild_id, user_id, character_name.strip(), ability_name, ability_name),
        )

    async def upgrade_ability(self, guild_id: int, user_id: int, character_name: str, ability_name: str, pool: str) -> Dict[str, Any]:
        ability_name = ability_name.strip()
        pool = pool.strip().lower()
        if pool not in ("positive", "negative"):
            raise ValueError("pool must be positive or negative")

        name_col = self.abilities_name_col
        lvl_col = self.abilities_level_col
        row = await self._fetchone(
            f"""
            SELECT COALESCE({lvl_col}, 0) AS upgrade_level
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND character_name=%s AND {name_col}=%s
            LIMIT 1;
            """,
            (guild_id, user_id, character_name.strip(), ability_name),
        )
        if not row:
            raise ValueError("Ability not found. Add it first with /add_ability.")

        stars = await self.get_stars(guild_id, user_id, character_name)
        max_upgrades = 2 + (2 * clamp(stars["ability_stars"], 0, MAX_ABILITY_STARS))
        cur_level = safe_int(row.get("upgrade_level"), 0)
        if cur_level >= max_upgrades:
            raise ValueError(f"Upgrade limit reached ({cur_level}/{max_upgrades}).")

        await self.spend_legacy(guild_id, user_id, character_name, pool, MINOR_UPGRADE_COST)
        new_level = cur_level + 1

        await self._execute(
            f"UPDATE abilities SET {lvl_col}=%s, upgrade_level=%s WHERE guild_id=%s AND user_id=%s AND character_name=%s AND {name_col}=%s;",
            (new_level, new_level, guild_id, user_id, character_name.strip(), ability_name),
        )
        return {"ability_name": ability_name, "upgrade_level": new_level, "max_upgrades": max_upgrades}

    async def get_dashboard_message_ids(self, guild_id: int, user_id: int) -> List[int]:
        row = await self._fetchone("SELECT message_ids FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))
        return parse_ids(row["message_ids"]) if row and row.get("message_ids") else []

    async def set_dashboard_message_ids(self, guild_id: int, user_id: int, channel_id: int, ids: List[int]) -> None:
        await self._execute(
            """
            INSERT INTO dashboard_messages (guild_id, user_id, channel_id, message_ids, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (guild_id, user_id)
            DO UPDATE SET channel_id=EXCLUDED.channel_id, message_ids=EXCLUDED.message_ids, updated_at=NOW();
            """,
            (guild_id, user_id, channel_id, fmt_ids(ids) if ids else None),
        )

    async def clear_dashboard_message_ids(self, guild_id: int, user_id: int) -> None:
        await self._execute("DELETE FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))


@dataclass
class CharacterCard:
    character_name: str
    avail_pos: int
    avail_neg: int
    life_pos: int
    life_neg: int
    ability_stars: int
    infl_pos: int
    infl_neg: int
    abilities: List[Tuple[str, int]]


async def build_character_card(db: Database, guild_id: int, user_id: int, character_name: str) -> CharacterCard:
    legacy = await db.get_legacy(guild_id, user_id, character_name)
    stars = await db.get_stars(guild_id, user_id, character_name)
    abilities_rows = await db.list_abilities(guild_id, user_id, character_name)
    abilities = [(r["ability_name"], safe_int(r.get("upgrade_level"), 0)) for r in abilities_rows if r.get("ability_name")]
    return CharacterCard(
        character_name=character_name,
        avail_pos=legacy["avail_pos"],
        avail_neg=legacy["avail_neg"],
        life_pos=legacy["life_pos"],
        life_neg=legacy["life_neg"],
        ability_stars=stars["ability_stars"],
        infl_pos=stars["infl_pos"],
        infl_neg=stars["infl_neg"],
        abilities=abilities,
    )


def render_character_block(card: CharacterCard) -> str:
    net_lifetime = card.life_pos - card.life_neg
    lines: List[str] = []
    lines.append(f"{CHAR_HEADER_LEFT}{card.character_name}{CHAR_HEADER_RIGHT}")
    lines.append(f"Legacy Points: +{card.avail_pos}/-{card.avail_neg} | Lifetime: +{card.life_pos}/-{card.life_neg}")
    lines.append(f"Ability Stars: {render_ability_star_bar(card.ability_stars)}")
    lines.append(f"Influence Stars: {render_influence_star_bar(card.infl_neg, card.infl_pos)}")
    lines.append(render_reputation_block(net_lifetime))
    if card.abilities:
        parts = [f"{name} ({lvl})" for name, lvl in card.abilities]
        lines.append("Abilities: " + " | ".join(parts))
    else:
        lines.append("Abilities: _none set_")
    return "
".join(lines).strip()


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

    content = "
".join(lines).rstrip()
    if len(content) > PLAYER_POST_SOFT_LIMIT:
        truncated = content[:PLAYER_POST_SOFT_LIMIT - 40]
        cut = truncated.rfind("
")
        if cut > 0:
            truncated = truncated[:cut]
        content = truncated.rstrip() + "

…(truncated: too many characters to fit in one post)"
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

    chars = await db.list_characters(guild.id, user_id)
    if not chars:
        stored_ids = await db.get_dashboard_message_ids(guild.id, user_id)
        for mid in stored_ids:
            try:
                m = await channel.fetch_message(mid)
                await m.delete()
            except Exception:
                pass
        await db.clear_dashboard_message_ids(guild.id, user_id)
        return f"No characters for user_id={user_id}; dashboard entry cleared."

    content = await render_player_post(db, guild, user_id)
    stored_ids = await db.get_dashboard_message_ids(guild.id, user_id)
    msg: Optional[discord.Message] = None
    if stored_ids:
        try:
            msg = await channel.fetch_message(stored_ids[0])
        except Exception:
            msg = None

    if msg is None:
        msg = await channel.send(content)
        await db.set_dashboard_message_ids(guild.id, user_id, channel.id, [msg.id])
        return f"Dashboard created for user_id={user_id}."
    else:
        await msg.edit(content=content)
        if len(stored_ids) > 1:
            for extra_id in stored_ids[1:]:
                try:
                    extra_msg = await channel.fetch_message(extra_id)
                    await extra_msg.delete()
                except Exception:
                    pass
            await db.set_dashboard_message_ids(guild.id, user_id, channel.id, [msg.id])
        return f"Dashboard updated for user_id={user_id}."


async def refresh_all_dashboards(client: "VilyraBotClient", guild: discord.Guild) -> str:
    user_ids = await client.db.list_player_ids(guild.id)
    if not user_ids:
        return "No players with characters yet."
    ok = 0
    for uid in user_ids:
        await refresh_player_dashboard(client, guild, uid)
        ok += 1
    return f"Refreshed dashboards for {ok} player(s)."


def staff_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild:
            return False
        if isinstance(interaction.user, discord.Member) and is_staff(interaction.user):
            return True
        await safe_followup(interaction, "Staff only (Guardian/Warden).")
        return False
    return app_commands.check(predicate)


def in_guild_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_followup(interaction, "This command can only be used in a server.")
            return False
        return True
    return app_commands.check(predicate)


async def require_character(db: Database, guild_id: int, user_id: int, character_name: str) -> None:
    if not await db.get_character(guild_id, user_id, character_name):
        raise ValueError("Character not found for that user.")


@app_commands.command(name="character_add", description="(Staff) Alias for /add_character (kept for compatibility).")
@in_guild_only()
@staff_only()
async def character_add(interaction: discord.Interaction, user: discord.Member, character_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        await run_db(interaction.client.db.add_character(interaction.guild.id, user.id, character_name.strip()), "add_character")
        status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
        await safe_followup(interaction, f"Character added. {status}")
    except Exception as e:
        await safe_followup(interaction, f"Add character failed: {e}")


class VilyraBotClient(discord.Client):
    def __init__(self, db: Database, **kwargs: Any):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True
        super().__init__(intents=intents, **kwargs)
        self.tree = app_commands.CommandTree(self)
        self.db = db

    async def setup_hook(self) -> None:
        try:
            gid = safe_int(os.getenv("GUILD_ID"), 0)
            if gid:
                await self.tree.sync(guild=discord.Object(id=gid))
            else:
                await self.tree.sync()
        except Exception:
            LOG.exception("Command sync failed")

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")
        for g in list(self.guilds):
            try:
                await refresh_all_dashboards(self, g)
            except Exception:
                LOG.exception("Startup dashboard refresh failed")


async def main_async() -> None:
    token = env("DISCORD_TOKEN")
    dsn = env("DATABASE_URL")
    db = Database(dsn)
    await db.connect()
    await db.init_schema()
    client = VilyraBotClient(db=db)
    client.tree.add_command(character_add)
    await client.start(token)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
