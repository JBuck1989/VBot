import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import discord
from discord import app_commands

import psycopg
from psycopg.rows import dict_row


# -----------------------------
# CONFIG
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

DASHBOARD_PAGE_LIMIT = 1900  # < 2000 Discord limit

SERVER_RANKS = [
    "Newcomer",
    "Apprentice",
    "Adventurer",
    "Sentinel",
    "Champion",
    "Legend",
    "Sovereign",
]

PLAYER_DIVIDER_LINE = "════════════════════════════════════"
CHAR_HEADER_LEFT = "꧁•⊹٭ "
CHAR_HEADER_RIGHT = " ٭⊹•꧂"


# -----------------------------
# LOGGING
# -----------------------------

LOG = logging.getLogger("VilyraBot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] VilyraBot: %(message)s"
)


# -----------------------------
# UTILS
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


def render_ability_star_bar(n: int) -> str:
    n = clamp(int(n), 0, MAX_ABILITY_STARS)
    return "★" * n + "☆" * (MAX_ABILITY_STARS - n)


def render_influence_star_bar(neg: int, pos: int) -> str:
    # neg fills right-to-left; pos fills left-to-right; total max 5
    neg = clamp(int(neg), 0, MAX_INFL_STARS_TOTAL)
    pos = clamp(int(pos), 0, MAX_INFL_STARS_TOTAL)

    neg_slots = ["☆"] * MAX_INFL_STARS_TOTAL
    for i in range(neg):
        neg_slots[MAX_INFL_STARS_TOTAL - 1 - i] = "★"

    pos_slots = ["☆"] * MAX_INFL_STARS_TOTAL
    for i in range(pos):
        pos_slots[i] = "★"

    return f"- {''.join(neg_slots)} | {''.join(pos_slots)} +"


def render_reputation_bar(net_lifetime: int) -> str:
    """
    Compact mobile-friendly bar:
    -100 .. +100 mapped to 21 positions across a line with center '|'
    We render as:  -------------------|-------------------  with a marker '▲'.
    """
    net = clamp(int(net_lifetime), REP_MIN, REP_MAX)
    # 41-char line: 19 left, center '|', 19 right
    left_len = 19
    right_len = 19
    # Map [-100..100] -> [0..(left+right)] inclusive
    span = left_len + right_len
    pos = int(round((net - REP_MIN) / (REP_MAX - REP_MIN) * span))
    pos = clamp(pos, 0, span)

    # Build baseline line
    left = ["-"] * left_len
    right = ["-"] * right_len

    # Marker placement: pos in [0..span], where 0 is far left, span is far right
    if pos < left_len:
        left[pos] = "▲"
    elif pos == left_len:
        # exactly center: place on the center '|' by showing marker above it
        pass
    else:
        rpos = pos - left_len - 1  # positions after center
        if 0 <= rpos < right_len:
            right[rpos] = "▲"

    line = f"{''.join(left)}|{''.join(right)}"
    label = "FEARED         |         LOVED"
    # If exactly center, show marker as a separate line with center alignment
    if pos == left_len:
        marker_line = (" " * left_len) + "▲" + (" " * right_len)
        return f"{label}\n{marker_line}\n{line}  ({net:+d})"
    return f"{label}\n{line}  ({net:+d})"


def split_pages(text: str, limit: int = DASHBOARD_PAGE_LIMIT) -> List[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]

    pages: List[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind("\n", 0, limit)
        if cut == -1 or cut < 200:
            cut = limit
        pages.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        pages.append(remaining)
    return pages


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


# -----------------------------
# DATABASE
# -----------------------------

class Database:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._conn: Optional[psycopg.AsyncConnection] = None

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
            row = await cur.fetchone()
            return row

    async def _list_public_tables(self) -> List[str]:
        rows = await self._fetchall(
            """
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname='public'
            ORDER BY tablename;
            """
        )
        return [r["tablename"] for r in rows]

    async def init_schema(self) -> None:
        # Fail-fast safety if DB is empty and bootstrapping not allowed
        allow = os.getenv("ALLOW_EMPTY_DB_BOOTSTRAP", "1").strip() in ("1", "true", "TRUE", "yes", "YES")
        existing_tables = await self._list_public_tables()
        if not existing_tables and not allow:
            raise RuntimeError(
                "Connected database has no tables and ALLOW_EMPTY_DB_BOOTSTRAP is not enabled. "
                "Refusing to initialize a potentially-wrong empty DB."
            )
        if not existing_tables and allow:
            LOG.warning("DB has no tables; bootstrapping schema (ALLOW_EMPTY_DB_BOOTSTRAP=1).")

        # Players (server rank)
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

        # Characters
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                character_name TEXT  NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id, character_name)
            );
            """
        )

        # Legacy Points (available + lifetime; pos + neg)
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS legacy_points (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                avail_pos     INT   NOT NULL DEFAULT 0,
                avail_neg     INT   NOT NULL DEFAULT 0,
                life_pos      INT   NOT NULL DEFAULT 0,
                life_neg      INT   NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, character_name)
            );
            """
        )

        # Stars
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS stars (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                ability_stars INT NOT NULL DEFAULT 0,
                infl_pos      INT NOT NULL DEFAULT 0,
                infl_neg      INT NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, character_name)
            );
            """
        )

        # Abilities with upgrade level (minor upgrades)
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
                guild_id       BIGINT NOT NULL,
                user_id        BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                ability_name   TEXT NOT NULL,
                upgrade_level  INT NOT NULL DEFAULT 0,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id, character_name, ability_name)
            );
            """
        )

        # Dashboard messages: one per user, can be multiple IDs for paging
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

        LOG.info("Database schema initialized / updated")

    # ---------- players ----------

    async def get_player_rank(self, guild_id: int, user_id: int) -> str:
        row = await self._fetchone(
            "SELECT server_rank FROM players WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )
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

    async def list_player_ids(self, guild_id: int) -> List[int]:
        rows = await self._fetchall(
            """
            SELECT DISTINCT user_id FROM characters
            WHERE guild_id=%s
            ORDER BY user_id ASC;
            """,
            (guild_id,),
        )
        return [int(r["user_id"]) for r in rows]

    # ---------- characters ----------

    async def add_character(self, guild_id: int, user_id: int, character_name: str) -> None:
        character_name = character_name.strip()
        await self._execute(
            """
            INSERT INTO characters (guild_id, user_id, character_name)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (guild_id, user_id, character_name),
        )
        # Seed associated rows
        await self._execute(
            """
            INSERT INTO legacy_points (guild_id, user_id, character_name)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (guild_id, user_id, character_name),
        )
        await self._execute(
            """
            INSERT INTO stars (guild_id, user_id, character_name)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (guild_id, user_id, character_name),
        )
        # Ensure player exists
        await self._execute(
            """
            INSERT INTO players (guild_id, user_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (guild_id, user_id),
        )

    async def get_character(self, guild_id: int, user_id: int, character_name: str) -> Optional[Dict[str, Any]]:
        return await self._fetchone(
            """
            SELECT guild_id, user_id, character_name, created_at
            FROM characters
            WHERE guild_id=%s AND user_id=%s AND character_name=%s;
            """,
            (guild_id, user_id, character_name),
        )

    async def list_characters(self, guild_id: int, user_id: int) -> List[str]:
        rows = await self._fetchall(
            """
            SELECT character_name
            FROM characters
            WHERE guild_id=%s AND user_id=%s
            ORDER BY character_name ASC;
            """,
            (guild_id, user_id),
        )
        return [r["character_name"] for r in rows]

    # ---------- legacy points ----------

    async def get_legacy(self, guild_id: int, user_id: int, character_name: str) -> Dict[str, int]:
        row = await self._fetchone(
            """
            SELECT avail_pos, avail_neg, life_pos, life_neg
            FROM legacy_points
            WHERE guild_id=%s AND user_id=%s AND character_name=%s;
            """,
            (guild_id, user_id, character_name),
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
            (guild_id, user_id, character_name, new["avail_pos"], new["avail_neg"], new["life_pos"], new["life_neg"]),
        )
        return new

    async def spend_legacy(self, guild_id: int, user_id: int, character_name: str, pool: str, amount: int) -> Dict[str, int]:
        """
        Spend from available points only (never touches lifetime).
        pool: "positive" or "negative"
        amount: positive int
        """
        amount = max(0, int(amount))
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
            """
            UPDATE legacy_points
            SET avail_pos=%s, avail_neg=%s
            WHERE guild_id=%s AND user_id=%s AND character_name=%s;
            """,
            (new["avail_pos"], new["avail_neg"], guild_id, user_id, character_name),
        )
        return new

    async def reset_legacy(self, guild_id: int, user_id: int, character_name: str, target: str,
                           avail_pos: Optional[int], avail_neg: Optional[int],
                           life_pos: Optional[int], life_neg: Optional[int]) -> Dict[str, int]:
        """
        target: "available", "lifetime", "both"
        Provide the corresponding values (ints). If None, keep current.
        """
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
            (guild_id, user_id, character_name, new["avail_pos"], new["avail_neg"], new["life_pos"], new["life_neg"]),
        )
        return new

    # ---------- stars ----------

    async def get_stars(self, guild_id: int, user_id: int, character_name: str) -> Dict[str, int]:
        row = await self._fetchone(
            """
            SELECT ability_stars, infl_pos, infl_neg
            FROM stars
            WHERE guild_id=%s AND user_id=%s AND character_name=%s;
            """,
            (guild_id, user_id, character_name),
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

        # enforce influence total <= 5
        total = new["infl_pos"] + new["infl_neg"]
        if total > MAX_INFL_STARS_TOTAL:
            raise ValueError("Total influence stars (pos+neg) cannot exceed 5")

        await self._execute(
            """
            INSERT INTO stars (guild_id, user_id, character_name, ability_stars, infl_pos, infl_neg)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (guild_id, user_id, character_name)
            DO UPDATE SET ability_stars=EXCLUDED.ability_stars, infl_pos=EXCLUDED.infl_pos, infl_neg=EXCLUDED.infl_neg;
            """,
            (guild_id, user_id, character_name, new["ability_stars"], new["infl_pos"], new["infl_neg"]),
        )
        return new

    async def convert_star(self, guild_id: int, user_id: int, character_name: str,
                           star_type: str, pool: Optional[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        star_type: "ability", "influence_positive", "influence_negative"
        pool: for ability star spending, must be "positive" or "negative"
        Deducts 10 from available pool, increments relevant star counter.
        Returns (legacy, stars).
        """
        star_type = star_type.strip().lower()

        legacy = await self.get_legacy(guild_id, user_id, character_name)
        stars = await self.get_stars(guild_id, user_id, character_name)

        if star_type == "influence_positive":
            # spend positive pool
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
                raise ValueError("For ability stars, pool must be positive or negative (which points are being spent).")
            legacy = await self.spend_legacy(guild_id, user_id, character_name, pool, STAR_COST)
            if stars["ability_stars"] >= MAX_ABILITY_STARS:
                raise ValueError("Ability stars already at max (5)")
            stars = await self.set_stars(guild_id, user_id, character_name, ability_stars=stars["ability_stars"] + 1)

        else:
            raise ValueError("star_type must be ability, influence_positive, or influence_negative")

        return legacy, stars

    # ---------- abilities ----------

    async def list_abilities(self, guild_id: int, user_id: int, character_name: str) -> List[Dict[str, Any]]:
        return await self._fetchall(
            """
            SELECT ability_name, upgrade_level
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND character_name=%s
            ORDER BY created_at ASC, ability_name ASC;
            """,
            (guild_id, user_id, character_name),
        )

    async def add_ability(self, guild_id: int, user_id: int, character_name: str, ability_name: str) -> None:
        ability_name = ability_name.strip()
        # capacity = 2 + ability_stars
        stars = await self.get_stars(guild_id, user_id, character_name)
        existing = await self.list_abilities(guild_id, user_id, character_name)
        cap = 2 + clamp(stars["ability_stars"], 0, MAX_ABILITY_STARS)
        if len(existing) >= cap:
            raise ValueError(f"Ability capacity reached ({len(existing)}/{cap}). Earn more Ability Stars to add abilities.")
        await self._execute(
            """
            INSERT INTO abilities (guild_id, user_id, character_name, ability_name, upgrade_level)
            VALUES (%s, %s, %s, %s, 0)
            ON CONFLICT DO NOTHING;
            """,
            (guild_id, user_id, character_name, ability_name),
        )

    async def upgrade_ability(self, guild_id: int, user_id: int, character_name: str, ability_name: str, pool: str) -> Dict[str, Any]:
        ability_name = ability_name.strip()
        if pool not in ("positive", "negative"):
            raise ValueError("pool must be positive or negative")

        # ensure ability exists
        row = await self._fetchone(
            """
            SELECT ability_name, upgrade_level
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND character_name=%s AND ability_name=%s;
            """,
            (guild_id, user_id, character_name, ability_name),
        )
        if not row:
            raise ValueError("Ability not found on this character. Add it first with /add_ability.")

        stars = await self.get_stars(guild_id, user_id, character_name)
        max_upgrades = 2 + (2 * clamp(stars["ability_stars"], 0, MAX_ABILITY_STARS))
        cur_level = safe_int(row.get("upgrade_level"), 0)
        if cur_level >= max_upgrades:
            raise ValueError(f"Upgrade limit reached for this ability ({cur_level}/{max_upgrades}).")

        # spend 5 points from chosen pool, then increment upgrade level
        await self.spend_legacy(guild_id, user_id, character_name, pool, MINOR_UPGRADE_COST)
        new_level = cur_level + 1
        await self._execute(
            """
            UPDATE abilities
            SET upgrade_level=%s
            WHERE guild_id=%s AND user_id=%s AND character_name=%s AND ability_name=%s;
            """,
            (new_level, guild_id, user_id, character_name, ability_name),
        )
        return {"ability_name": ability_name, "upgrade_level": new_level, "max_upgrades": max_upgrades}

    # ---------- dashboard messages ----------

    async def get_dashboard_message_ids(self, guild_id: int, user_id: int) -> List[int]:
        row = await self._fetchone(
            "SELECT message_ids FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )
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
        await self._execute(
            "DELETE FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )


# -----------------------------
# DASHBOARD RENDERING
# -----------------------------

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
    abilities: List[Tuple[str, int]]  # (name, upgrade_level)


async def build_character_card(db: Database, guild_id: int, user_id: int, character_name: str) -> CharacterCard:
    legacy = await db.get_legacy(guild_id, user_id, character_name)
    stars = await db.get_stars(guild_id, user_id, character_name)
    abilities_rows = await db.list_abilities(guild_id, user_id, character_name)
    abilities = [(r["ability_name"], safe_int(r.get("upgrade_level"), 0)) for r in abilities_rows]

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
    lines.append(
        f"Legacy Points: +{card.avail_pos}/ -{card.avail_neg} | Lifetime: +{card.life_pos}/-{card.life_neg}"
    )
    lines.append(f"Ability Stars: {render_ability_star_bar(card.ability_stars)}")
    lines.append(f"Influence Stars: {render_influence_star_bar(card.infl_neg, card.infl_pos)}")
    lines.append(render_reputation_bar(net_lifetime))

    if card.abilities:
        # Compact: Ability (level) | Ability (level)
        ability_parts = [f"{name} ({lvl})" for name, lvl in card.abilities]
        lines.append("Abilities: " + " | ".join(ability_parts))
    else:
        lines.append("Abilities: _none set_")

    return "\n".join(lines).strip()


async def render_player_card(db: Database, guild: discord.Guild, user_id: int) -> str:
    member = guild.get_member(user_id)
    display = member.display_name if member else f"User {user_id}"
    rank = await db.get_player_rank(guild.id, user_id)

    chars = await db.list_characters(guild.id, user_id)
    if not chars:
        return ""

    lines: List[str] = []
    lines.append(f"{PLAYER_DIVIDER_LINE}")
    lines.append(f"**{display}** | Server Rank **{rank}**")
    lines.append("")

    for cname in chars:
        card = await build_character_card(db, guild.id, user_id, cname)
        lines.append(render_character_block(card))
        lines.append("")

    return "\n".join(lines).strip()


# -----------------------------
# DASHBOARD POSTING
# -----------------------------

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
        return f"Dashboard channel not found or not a text channel (ID {safe_int(os.getenv('DASHBOARD_CHANNEL_ID'), DEFAULT_DASHBOARD_CHANNEL_ID)})."

    # Permission check
    me = guild.me or (guild.get_member(client.user.id) if client.user else None)
    if me:
        perms = channel.permissions_for(me)
        if not (perms.view_channel and perms.send_messages):
            return f"Missing permissions in <#{channel.id}>: need View Channel + Send Messages."

    # If user has no characters, delete dashboard messages for that user if they exist
    chars = await db.list_characters(guild.id, user_id)
    if not chars:
        # Try delete old messages
        old_ids = await db.get_dashboard_message_ids(guild.id, user_id)
        for mid in old_ids:
            try:
                m = await channel.fetch_message(mid)
                await m.delete()
            except Exception:
                pass
        await db.clear_dashboard_message_ids(guild.id, user_id)
        return f"No characters for user {user_id}; dashboard entries cleared."

    content = await render_player_card(db, guild, user_id)
    if not content:
        return f"No content rendered for user {user_id}."

    pages = split_pages(content, DASHBOARD_PAGE_LIMIT)

    stored_ids = await db.get_dashboard_message_ids(guild.id, user_id)
    messages: List[discord.Message] = []

    # Load existing messages
    for mid in stored_ids:
        try:
            m = await channel.fetch_message(mid)
            messages.append(m)
        except Exception:
            continue

    # Create missing messages
    while len(messages) < len(pages):
        m = await channel.send("Creating player dashboard…")
        messages.append(m)

    # Edit pages
    for idx, page in enumerate(pages):
        await messages[idx].edit(content=page)

    # Delete extras
    for extra in messages[len(pages):]:
        try:
            await extra.delete()
        except Exception:
            pass

    final_ids = [m.id for m in messages[:len(pages)]]
    await db.set_dashboard_message_ids(guild.id, user_id, channel.id, final_ids)

    return f"Dashboard updated for <@{user_id}> ({len(pages)} message(s))."


async def refresh_all_dashboards(client: "VilyraBotClient", guild: discord.Guild) -> str:
    db = client.db
    user_ids = await db.list_player_ids(guild.id)
    if not user_ids:
        return "No players with characters yet."
    results: List[str] = []
    for uid in user_ids:
        res = await refresh_player_dashboard(client, guild, uid)
        results.append(res)
    return " | ".join(results[:5]) + (f" (+{len(results)-5} more)" if len(results) > 5 else "")


# -----------------------------
# DISCORD CLIENT
# -----------------------------

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
            await self.tree.sync()
            LOG.info("Guild sync succeeded: %s commands", len(self.tree.get_commands()))
        except Exception:
            LOG.exception("Command sync failed")

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")
        # Startup refresh (non-fatal)
        for g in list(self.guilds):
            try:
                status = await refresh_all_dashboards(self, g)
                LOG.info("Startup dashboard refresh: %s", status)
            except Exception:
                LOG.exception("Startup dashboard refresh failed")


# -----------------------------
# COMMAND CHECKS
# -----------------------------

def staff_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild:
            return False
        if isinstance(interaction.user, discord.Member) and is_staff(interaction.user):
            return True
        await interaction.response.send_message("Staff only (Guardian/Warden).", ephemeral=True)
        return False
    return app_commands.check(predicate)


def in_guild_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return False
        return True
    return app_commands.check(predicate)


async def require_character(db: Database, guild_id: int, user_id: int, character_name: str) -> None:
    if not await db.get_character(guild_id, user_id, character_name):
        raise ValueError("Character not found for that user.")


# -----------------------------
# COMMANDS — STAFF
# -----------------------------

@app_commands.command(name="set_server_rank", description="(Staff) Set a player's server rank.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", rank="Newcomer/Apprentice/Adventurer/Sentinel/Champion/Legend/Sovereign")
async def set_server_rank(interaction: discord.Interaction, user: discord.Member, rank: str):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    rank = rank.strip()
    if rank not in SERVER_RANKS:
        await interaction.followup.send(f"Invalid rank. Options: {', '.join(SERVER_RANKS)}", ephemeral=True)
        return

    await interaction.client.db.set_player_rank(interaction.guild.id, user.id, rank)
    await log_to_channel(interaction.guild, f"🏷️ {interaction.user.mention} set server rank for {user.mention} to **{rank}**")
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Rank set. {status}", ephemeral=True)


@app_commands.command(name="add_character", description="(Staff) Add a character under a player.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", character_name="Character name")
async def add_character(interaction: discord.Interaction, user: discord.Member, character_name: str):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    await interaction.client.db.add_character(interaction.guild.id, user.id, character_name)

    await log_to_channel(interaction.guild, f"➕ {interaction.user.mention} added character **{character_name}** under {user.mention}")
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Character added. {status}", ephemeral=True)


@app_commands.command(name="award_legacy_points", description="(Staff) Award positive and/or negative legacy points to a character.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", character_name="Character name", positive="Positive points to add", negative="Negative points to add")
async def award_legacy_points(interaction: discord.Interaction, user: discord.Member, character_name: str, positive: int = 0, negative: int = 0):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    await require_character(interaction.client.db, interaction.guild.id, user.id, character_name)

    if positive < 0 or negative < 0:
        await interaction.followup.send("Points must be >= 0.", ephemeral=True)
        return
    if positive == 0 and negative == 0:
        await interaction.followup.send("Provide positive and/or negative points to award.", ephemeral=True)
        return

    new = await interaction.client.db.award_legacy(interaction.guild.id, user.id, character_name, pos=positive, neg=negative)
    await log_to_channel(
        interaction.guild,
        f"🏅 {interaction.user.mention} awarded **{character_name}** ({user.mention}) legacy points: +{positive} / -{negative} | now avail +{new['avail_pos']}/-{new['avail_neg']} life +{new['life_pos']}/-{new['life_neg']}",
    )
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Awarded. {status}", ephemeral=True)


@app_commands.command(name="convert_star", description="(Staff) Convert legacy points to a star (cost: 10).")
@in_guild_only()
@staff_only()
@app_commands.describe(
    user="The player",
    character_name="Character name",
    star_type="ability / influence_positive / influence_negative",
    pool="For ability stars only: which points are being spent (positive/negative)",
)
async def convert_star(interaction: discord.Interaction, user: discord.Member, character_name: str, star_type: str, pool: Optional[str] = None):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    star_type = star_type.strip().lower()
    pool = pool.strip().lower() if pool else None

    await require_character(interaction.client.db, interaction.guild.id, user.id, character_name)

    try:
        legacy, stars = await interaction.client.db.convert_star(interaction.guild.id, user.id, character_name, star_type, pool)
    except Exception as e:
        await interaction.followup.send(f"Convert failed: {e}", ephemeral=True)
        return

    await log_to_channel(
        interaction.guild,
        f"⭐ {interaction.user.mention} converted points -> **{star_type}** for **{character_name}** ({user.mention}) | avail +{legacy['avail_pos']}/-{legacy['avail_neg']} | stars ability={stars['ability_stars']} infl -{stars['infl_neg']} +{stars['infl_pos']}",
    )
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Converted. {status}", ephemeral=True)


@app_commands.command(name="add_ability", description="(Staff) Add an ability to a character (requires capacity from Ability Stars).")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", character_name="Character name", ability_name="Ability name")
async def add_ability(interaction: discord.Interaction, user: discord.Member, character_name: str, ability_name: str):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    ability_name = ability_name.strip()

    await require_character(interaction.client.db, interaction.guild.id, user.id, character_name)

    try:
        await interaction.client.db.add_ability(interaction.guild.id, user.id, character_name, ability_name)
    except Exception as e:
        await interaction.followup.send(f"Add ability failed: {e}", ephemeral=True)
        return

    await log_to_channel(interaction.guild, f"🧩 {interaction.user.mention} added ability **{ability_name}** to **{character_name}** ({user.mention})")
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Ability added. {status}", ephemeral=True)


@app_commands.command(name="upgrade_ability", description="(Staff) Spend 5 legacy points to apply a minor ability upgrade.")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", character_name="Character name", ability_name="Ability name", pool="Which points to spend (positive/negative)")
async def upgrade_ability(interaction: discord.Interaction, user: discord.Member, character_name: str, ability_name: str, pool: str):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    ability_name = ability_name.strip()
    pool = pool.strip().lower()

    await require_character(interaction.client.db, interaction.guild.id, user.id, character_name)

    try:
        res = await interaction.client.db.upgrade_ability(interaction.guild.id, user.id, character_name, ability_name, pool)
    except Exception as e:
        await interaction.followup.send(f"Upgrade failed: {e}", ephemeral=True)
        return

    await log_to_channel(
        interaction.guild,
        f"🔧 {interaction.user.mention} upgraded **{ability_name}** on **{character_name}** ({user.mention}) -> level {res['upgrade_level']} (max {res['max_upgrades']}); spent {MINOR_UPGRADE_COST} {pool} points",
    )
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Upgraded to level {res['upgrade_level']}/{res['max_upgrades']}. {status}", ephemeral=True)


@app_commands.command(name="reset_legacy_points", description="(Staff) Reset available and/or lifetime legacy point totals (for corrections).")
@in_guild_only()
@staff_only()
@app_commands.describe(
    user="The player",
    character_name="Character name",
    target="available / lifetime / both",
    avail_pos="New available positive (omit to keep)",
    avail_neg="New available negative (omit to keep)",
    life_pos="New lifetime positive (omit to keep)",
    life_neg="New lifetime negative (omit to keep)",
)
async def reset_legacy_points(
    interaction: discord.Interaction,
    user: discord.Member,
    character_name: str,
    target: str,
    avail_pos: Optional[int] = None,
    avail_neg: Optional[int] = None,
    life_pos: Optional[int] = None,
    life_neg: Optional[int] = None,
):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    target = target.strip().lower()

    await require_character(interaction.client.db, interaction.guild.id, user.id, character_name)

    try:
        new = await interaction.client.db.reset_legacy(interaction.guild.id, user.id, character_name, target, avail_pos, avail_neg, life_pos, life_neg)
    except Exception as e:
        await interaction.followup.send(f"Reset failed: {e}", ephemeral=True)
        return

    await log_to_channel(
        interaction.guild,
        f"🧾 {interaction.user.mention} reset legacy points for **{character_name}** ({user.mention}) target={target} -> avail +{new['avail_pos']}/-{new['avail_neg']} life +{new['life_pos']}/-{new['life_neg']}",
    )
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Reset complete. {status}", ephemeral=True)


@app_commands.command(name="reset_ability_stars", description="(Staff) Set a character's Ability Star total (0-5).")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", character_name="Character name", ability_stars="0..5")
async def reset_ability_stars(interaction: discord.Interaction, user: discord.Member, character_name: str, ability_stars: int):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    await require_character(interaction.client.db, interaction.guild.id, user.id, character_name)

    try:
        new = await interaction.client.db.set_stars(interaction.guild.id, user.id, character_name, ability_stars=ability_stars)
    except Exception as e:
        await interaction.followup.send(f"Reset failed: {e}", ephemeral=True)
        return

    await log_to_channel(interaction.guild, f"🌟 {interaction.user.mention} set Ability Stars for **{character_name}** ({user.mention}) to {new['ability_stars']}")
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Ability Stars set to {new['ability_stars']}. {status}", ephemeral=True)


@app_commands.command(name="reset_influence_stars", description="(Staff) Set a character's Influence Stars (pos/neg; total max 5).")
@in_guild_only()
@staff_only()
@app_commands.describe(user="The player", character_name="Character name", positive="0..5", negative="0..5")
async def reset_influence_stars(interaction: discord.Interaction, user: discord.Member, character_name: str, positive: int, negative: int):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    await require_character(interaction.client.db, interaction.guild.id, user.id, character_name)

    try:
        new = await interaction.client.db.set_stars(interaction.guild.id, user.id, character_name, infl_pos=positive, infl_neg=negative)
    except Exception as e:
        await interaction.followup.send(f"Reset failed: {e}", ephemeral=True)
        return

    await log_to_channel(interaction.guild, f"⚖️ {interaction.user.mention} set Influence Stars for **{character_name}** ({user.mention}) to -{new['infl_neg']} +{new['infl_pos']}")
    status = await refresh_player_dashboard(interaction.client, interaction.guild, user.id)
    await interaction.followup.send(f"Influence Stars set. {status}", ephemeral=True)


@app_commands.command(name="refresh_dashboard", description="(Staff) Force refresh the whole dashboard.")
@in_guild_only()
@staff_only()
async def refresh_dashboard(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    status = await refresh_all_dashboards(interaction.client, interaction.guild)
    await log_to_channel(interaction.guild, f"🔄 {interaction.user.mention} refreshed the dashboard")
    await interaction.followup.send(status, ephemeral=True)


# -----------------------------
# COMMANDS — ANY USER
# -----------------------------

@app_commands.command(name="char_card", description="Show a character's card ephemerally (quick-check).")
@in_guild_only()
@app_commands.describe(character_name="Character name", user="(Optional) Only staff can look up another user.")
async def char_card(interaction: discord.Interaction, character_name: str, user: Optional[discord.Member] = None):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None

    character_name = character_name.strip()
    target_user = user or interaction.user

    # non-staff cannot look up other users
    if user is not None:
        if not (isinstance(interaction.user, discord.Member) and is_staff(interaction.user)):
            await interaction.followup.send("You can only look up your own characters.", ephemeral=True)
            return

    try:
        await require_character(interaction.client.db, interaction.guild.id, target_user.id, character_name)
        card = await build_character_card(interaction.client.db, interaction.guild.id, target_user.id, character_name)
        block = render_character_block(card)
        await interaction.followup.send(block, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"Lookup failed: {e}", ephemeral=True)


# -----------------------------
# MAIN
# -----------------------------

async def main_async() -> None:
    token = env("DISCORD_TOKEN")
    dsn = env("DATABASE_URL")

    db = Database(dsn)
    await db.connect()
    await db.init_schema()

    client = VilyraBotClient(db=db)

    # Register commands
    client.tree.add_command(set_server_rank)
    client.tree.add_command(add_character)
    client.tree.add_command(award_legacy_points)
    client.tree.add_command(convert_star)
    client.tree.add_command(add_ability)
    client.tree.add_command(upgrade_ability)
    client.tree.add_command(reset_legacy_points)
    client.tree.add_command(reset_ability_stars)
    client.tree.add_command(reset_influence_stars)
    client.tree.add_command(refresh_dashboard)
    client.tree.add_command(char_card)

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
