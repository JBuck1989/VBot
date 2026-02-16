import asyncio
import logging
import os
import re
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import discord
from discord import app_commands

import psycopg
from psycopg import rows as psy_rows


# -----------------------------
# Config
# -----------------------------
DASHBOARD_CHANNEL_ID: int = 1469879866655768738
ALLOWED_ROLE_NAMES = {"guardian", "wardens", "warden", "guardians"}  # allow minor variants
BOARD_HEADER_SEPARATOR = "────────────────────────"

# Dashboard bar settings
BAR_MAX_ABS = 100  # shown range; values beyond clamp to terminal end
BAR_DASHES_PER_SIDE = 20  # each dash = 5 points (because 100 / 20 = 5)
BAR_POINTS_PER_DASH = BAR_MAX_ABS // BAR_DASHES_PER_SIDE  # 5
BAR_CENTER_CHAR = "|"
BAR_LINE_CHAR = "—"  # em dash to look solid; stable on mobile
BAR_MARKER_CHAR = "◆"

# Stars
STAR_EMPTY = "☆"
STAR_FILLED = "★"

# Limits
MAX_ABILITIES_PER_CHARACTER = 7
MAX_ABILITY_STARS_TOTAL = 5
MAX_INFLUENCE_STARS_TOTAL = 5

# Discord message limits: keep each player post comfortably below 2000.
PLAYER_POST_SOFT_LIMIT = 1800

# Logging
LOG = logging.getLogger("VilyraBot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# -----------------------------
# Utilities
# -----------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def clamp(n: int, lo: int, hi: int) -> int:
    return lo if n < lo else hi if n > hi else n


def norm_name(s: str) -> str:
    # Normalize character names for DB keying and duplicate prevention, but keep original display separately.
    return re.sub(r"\s+", " ", s.strip())


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def is_allowed_member(member: discord.Member) -> bool:
    if member.guild_permissions.administrator:
        return True
    for r in getattr(member, "roles", []):
        if r and r.name and r.name.strip().lower() in ALLOWED_ROLE_NAMES:
            return True
    return False


async def log_action(guild: discord.Guild, text: str) -> None:
    cid = os.getenv("COMMAND_LOG_CHANNEL_ID")
    if not cid:
        return
    try:
        channel_id = int(cid)
    except Exception:
        return
    ch = guild.get_channel(channel_id)
    if ch is None:
        try:
            ch = await guild.fetch_channel(channel_id)
        except Exception:
            return
    if isinstance(ch, (discord.TextChannel, discord.Thread)):
        try:
            await ch.send(text, allowed_mentions=discord.AllowedMentions.none())
        except Exception:
            pass


# -----------------------------
# Rendering
# -----------------------------
def render_ability_stars_line(ability_stars: int) -> str:
    ability_stars = clamp(ability_stars, 0, MAX_ABILITY_STARS_TOTAL)
    return f"{STAR_FILLED * ability_stars}{STAR_EMPTY * (MAX_ABILITY_STARS_TOTAL - ability_stars)}"


def render_influence_stars_line(minus: int, plus: int) -> str:
    """
    Render: -☆☆☆☆☆|☆☆☆☆☆+
    Fill from the center outward:
      minus fills RIGHT->LEFT (adjacent to | first)
      plus fills LEFT->RIGHT (adjacent to | first)
    """
    minus = clamp(minus, 0, MAX_INFLUENCE_STARS_TOTAL)
    plus = clamp(plus, 0, MAX_INFLUENCE_STARS_TOTAL)
    # Enforce total <= 5 defensively
    if minus + plus > MAX_INFLUENCE_STARS_TOTAL:
        over = (minus + plus) - MAX_INFLUENCE_STARS_TOTAL
        if plus >= minus:
            plus = max(0, plus - over)
        else:
            minus = max(0, minus - over)

    left = [STAR_EMPTY] * MAX_INFLUENCE_STARS_TOTAL
    # fill from right to left for minus
    for i in range(minus):
        idx = (MAX_INFLUENCE_STARS_TOTAL - 1) - i
        left[idx] = STAR_FILLED

    right = [STAR_EMPTY] * MAX_INFLUENCE_STARS_TOTAL
    # fill from left to right for plus
    for i in range(plus):
        if i < MAX_INFLUENCE_STARS_TOTAL:
            right[i] = STAR_FILLED

    return f"-{''.join(left)}{BAR_CENTER_CHAR}{''.join(right)}+"


def _dash_offset_from_net(net: int) -> int:
    """
    Convert net points (-100..+100) to an integer dash offset in [-20..+20].
    Each dash represents 5 points.
    We move 1 dash per 5 points, truncating toward 0 for small values.
    """
    net = clamp(net, -BAR_MAX_ABS, BAR_MAX_ABS)
    if net > 0:
        return int(math.floor(net / BAR_POINTS_PER_DASH))
    if net < 0:
        return int(math.ceil(net / BAR_POINTS_PER_DASH))
    return 0


def render_influence_bar_lines(net: int, has_any_lifetime: bool) -> Tuple[str, str]:
    """
    Two-line bar:
    FEARED           <- | ->          LOVED
    ————————————◆———————|———————————————  (marker embedded, optional)

    - BAR_DASHES_PER_SIDE dashes on each side of center.
    - Marker replaces a dash (never placed on the center |).
    - If has_any_lifetime is False, no marker.
    """
    line1 = "FEARED           <- | ->          LOVED"

    left = [BAR_LINE_CHAR] * BAR_DASHES_PER_SIDE
    right = [BAR_LINE_CHAR] * BAR_DASHES_PER_SIDE

    if has_any_lifetime:
        off = _dash_offset_from_net(net)

        if off == 0:
            if right:
                right[0] = BAR_MARKER_CHAR
        elif off < 0:
            idx = BAR_DASHES_PER_SIDE - 1 + off  # off is negative
            idx = clamp(idx, 0, BAR_DASHES_PER_SIDE - 1)
            left[idx] = BAR_MARKER_CHAR
        else:
            idx = off - 1
            idx = clamp(idx, 0, BAR_DASHES_PER_SIDE - 1)
            right[idx] = BAR_MARKER_CHAR

    line2 = f"{''.join(left)}{BAR_CENTER_CHAR}{''.join(right)}"
    return line1, line2


def format_character_header(name: str) -> str:
    return f"__***꧁•⊹٭ {name} ٭⊹•꧂***__"


def format_player_header(display_name: str, rank: str) -> str:
    return f"__**Player: {display_name} | Server Rank: {rank}**__"


def render_abilities_inline(abilities: List[Tuple[str, int]]) -> str:
    if not abilities:
        return "—"
    parts = []
    for n, u in abilities:
        n = n.strip()
        if not n:
            continue
        parts.append(f"{n} ({u})")
    return ", ".join(parts) if parts else "—"


# -----------------------------
# Database
# -----------------------------
class Database:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn: Optional[psycopg.AsyncConnection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        sslmode = os.getenv("SSLMODE", "require")
        dsn = self.dsn
        if "sslmode=" not in dsn:
            joiner = "&" if "?" in dsn else "?"
            dsn = f"{dsn}{joiner}sslmode={sslmode}"

        last_exc: Optional[Exception] = None
        for attempt in range(1, 11):
            try:
                self.conn = await psycopg.AsyncConnection.connect(
                    dsn,
                    autocommit=True,
                    row_factory=psyc_rows.dict_row,
                )
                LOG.info("PostgreSQL async connection established (autocommit=True)")
                return
            except Exception as e:
                last_exc = e
                LOG.warning("DB connect attempt %s failed: %s", attempt, repr(e))
                await asyncio.sleep(min(2.5 * attempt, 12.0))
        raise last_exc or RuntimeError("DB connection failed")

    async def close(self) -> None:
        if self.conn:
            try:
                await self.conn.close()
            except Exception:
                pass
            self.conn = None

    async def _execute(self, sql: str, params: Tuple[Any, ...] = ()) -> None:
        if not self.conn:
            await self.connect()
        assert self.conn is not None
        async with self._lock:
            async with self.conn.cursor() as cur:
                await cur.execute(sql, params)

    async def _fetchall(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        if not self.conn:
            await self.connect()
        assert self.conn is not None
        async with self._lock:
            async with self.conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                return list(rows or [])

    async def _fetchone(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        rows = await self._fetchall(sql, params)
        return rows[0] if rows else None

    async def init_schema(self) -> None:
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS players (
              guild_id    BIGINT NOT NULL,
              user_id     BIGINT NOT NULL,
              server_rank TEXT NOT NULL DEFAULT '—',
              archived    BOOLEAN NOT NULL DEFAULT FALSE,
              created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (guild_id, user_id)
            );
            """
        )
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
              guild_id        BIGINT NOT NULL,
              user_id         BIGINT NOT NULL,
              name            TEXT   NOT NULL,
              legacy_plus     INTEGER NOT NULL DEFAULT 0,
              legacy_minus    INTEGER NOT NULL DEFAULT 0,
              lifetime_plus   INTEGER NOT NULL DEFAULT 0,
              lifetime_minus  INTEGER NOT NULL DEFAULT 0,
              ability_stars   INTEGER NOT NULL DEFAULT 0,
              influence_plus  INTEGER NOT NULL DEFAULT 0,
              influence_minus INTEGER NOT NULL DEFAULT 0,
              archived        BOOLEAN NOT NULL DEFAULT FALSE,
              created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (guild_id, user_id, name)
            );
            """
        )
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
              guild_id        BIGINT NOT NULL,
              user_id         BIGINT NOT NULL,
              character_name  TEXT   NOT NULL,
              ability_name    TEXT   NOT NULL,
              upgrades        INTEGER NOT NULL DEFAULT 0,
              created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (guild_id, user_id, character_name, ability_name)
            );
            """
        )
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_posts (
              guild_id   BIGINT NOT NULL,
              user_id    BIGINT NOT NULL,
              message_id BIGINT NOT NULL,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (guild_id, user_id)
            );
            """
        )
        LOG.info("Database schema initialized / updated")

    async def ensure_player(self, guild_id: int, user_id: int) -> None:
        await self._execute(
            """
            INSERT INTO players (guild_id, user_id) VALUES (%s, %s)
            ON CONFLICT (guild_id, user_id) DO UPDATE
              SET updated_at = now();
            """,
            (guild_id, user_id),
        )

    async def set_player_rank(self, guild_id: int, user_id: int, rank: str) -> None:
        await self._execute(
            """
            INSERT INTO players (guild_id, user_id, server_rank) VALUES (%s, %s, %s)
            ON CONFLICT (guild_id, user_id) DO UPDATE
              SET server_rank = EXCLUDED.server_rank,
                  archived = FALSE,
                  updated_at = now();
            """,
            (guild_id, user_id, rank),
        )

    async def archive_player(self, guild_id: int, user_id: int, archived: bool) -> None:
        await self._execute(
            """
            UPDATE players SET archived = %s, updated_at = now()
            WHERE guild_id = %s AND user_id = %s;
            """,
            (archived, guild_id, user_id),
        )

    async def get_player_rank(self, guild_id: int, user_id: int) -> str:
        row = await self._fetchone(
            "SELECT server_rank FROM players WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )
        return (row or {}).get("server_rank") or "—"

    async def create_character(self, guild_id: int, user_id: int, name: str) -> None:
        await self.ensure_player(guild_id, user_id)
        await self._execute(
            """
            INSERT INTO characters (guild_id, user_id, name)
            VALUES (%s, %s, %s)
            ON CONFLICT (guild_id, user_id, name) DO UPDATE
              SET archived = FALSE,
                  updated_at = now();
            """,
            (guild_id, user_id, name),
        )

    async def archive_character(self, guild_id: int, user_id: int, name: str, archived: bool) -> None:
        await self._execute(
            """
            UPDATE characters SET archived = %s, updated_at = now()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (archived, guild_id, user_id, name),
        )

    async def delete_character(self, guild_id: int, user_id: int, name: str) -> None:
        await self._execute(
            """
            DELETE FROM abilities
            WHERE guild_id=%s AND user_id=%s AND character_name=%s;
            """,
            (guild_id, user_id, name),
        )
        await self._execute(
            """
            DELETE FROM characters
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (guild_id, user_id, name),
        )

    async def list_characters_for_user(self, guild_id: int, user_id: int, include_archived: bool = False) -> List[Dict[str, Any]]:
        if include_archived:
            sql = """
                SELECT *
                FROM characters
                WHERE guild_id=%s AND user_id=%s
                ORDER BY name ASC;
            """
            params = (guild_id, user_id)
        else:
            sql = """
                SELECT *
                FROM characters
                WHERE guild_id=%s AND user_id=%s AND archived=FALSE
                ORDER BY name ASC;
            """
            params = (guild_id, user_id)
        return await self._fetchall(sql, params)

    async def get_character(self, guild_id: int, user_id: int, name: str) -> Optional[Dict[str, Any]]:
        return await self._fetchone(
            """
            SELECT *
            FROM characters
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (guild_id, user_id, name),
        )

    async def list_all_active_characters(self, guild_id: int) -> List[Dict[str, Any]]:
        return await self._fetchall(
            """
            SELECT *
            FROM characters
            WHERE guild_id=%s AND archived=FALSE
            ORDER BY user_id ASC, name ASC;
            """,
            (guild_id,),
        )

    async def add_legacy_points(self, guild_id: int, user_id: int, name: str, plus: int, minus: int) -> None:
        await self._execute(
            """
            UPDATE characters
            SET legacy_plus = legacy_plus + %s,
                legacy_minus = legacy_minus + %s,
                lifetime_plus = lifetime_plus + %s,
                lifetime_minus = lifetime_minus + %s,
                updated_at = now()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (plus, minus, plus, minus, guild_id, user_id, name),
        )

    async def remove_legacy_points(self, guild_id: int, user_id: int, name: str, plus: int, minus: int) -> None:
        await self._execute(
            """
            UPDATE characters
            SET legacy_plus = GREATEST(0, legacy_plus - %s),
                legacy_minus = GREATEST(0, legacy_minus - %s),
                updated_at = now()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (plus, minus, guild_id, user_id, name),
        )

    async def set_ability_stars(self, guild_id: int, user_id: int, name: str, stars: int) -> None:
        stars = clamp(stars, 0, MAX_ABILITY_STARS_TOTAL)
        await self._execute(
            """
            UPDATE characters
            SET ability_stars = %s,
                updated_at = now()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (stars, guild_id, user_id, name),
        )

    async def set_influence_stars(self, guild_id: int, user_id: int, name: str, minus: int, plus: int) -> None:
        minus = clamp(minus, 0, MAX_INFLUENCE_STARS_TOTAL)
        plus = clamp(plus, 0, MAX_INFLUENCE_STARS_TOTAL)
        if minus + plus > MAX_INFLUENCE_STARS_TOTAL:
            over = (minus + plus) - MAX_INFLUENCE_STARS_TOTAL
            if plus >= minus:
                plus = max(0, plus - over)
            else:
                minus = max(0, minus - over)
        await self._execute(
            """
            UPDATE characters
            SET influence_minus=%s,
                influence_plus=%s,
                updated_at=now()
            WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (minus, plus, guild_id, user_id, name),
        )

    async def list_abilities(self, guild_id: int, user_id: int, character_name: str) -> List[Tuple[str, int]]:
        rows = await self._fetchall(
            """
            SELECT ability_name, upgrades
            FROM abilities
            WHERE guild_id=%s AND user_id=%s AND character_name=%s
            ORDER BY ability_name ASC;
            """,
            (guild_id, user_id, character_name),
        )
        out: List[Tuple[str, int]] = []
        for r in rows:
            out.append((r.get("ability_name") or "", safe_int(r.get("upgrades"), 0)))
        return out

    async def add_ability(self, guild_id: int, user_id: int, character_name: str, ability_name: str) -> None:
        existing = await self.list_abilities(guild_id, user_id, character_name)
        if len(existing) >= MAX_ABILITIES_PER_CHARACTER:
            raise ValueError(f"Max abilities reached ({MAX_ABILITIES_PER_CHARACTER}).")
        await self._execute(
            """
            INSERT INTO abilities (guild_id, user_id, character_name, ability_name, upgrades)
            VALUES (%s, %s, %s, %s, 0)
            ON CONFLICT (guild_id, user_id, character_name, ability_name) DO UPDATE
              SET updated_at=now();
            """,
            (guild_id, user_id, character_name, ability_name),
        )

    async def upgrade_ability(self, guild_id: int, user_id: int, character_name: str, ability_name: str, delta: int) -> None:
        await self._execute(
            """
            INSERT INTO abilities (guild_id, user_id, character_name, ability_name, upgrades)
            VALUES (%s, %s, %s, %s, GREATEST(0, %s))
            ON CONFLICT (guild_id, user_id, character_name, ability_name) DO UPDATE
              SET upgrades = GREATEST(0, abilities.upgrades + %s),
                  updated_at = now();
            """,
            (guild_id, user_id, character_name, ability_name, delta, delta),
        )

    async def get_dashboard_message_id(self, guild_id: int, user_id: int) -> Optional[int]:
        row = await self._fetchone(
            "SELECT message_id FROM dashboard_posts WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )
        if not row:
            return None
        return safe_int(row.get("message_id"), 0) or None

    async def set_dashboard_message_id(self, guild_id: int, user_id: int, message_id: int) -> None:
        await self._execute(
            """
            INSERT INTO dashboard_posts (guild_id, user_id, message_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (guild_id, user_id) DO UPDATE
              SET message_id = EXCLUDED.message_id,
                  updated_at = now();
            """,
            (guild_id, user_id, message_id),
        )


# -----------------------------
# Bot
# -----------------------------
@dataclass
class CharacterSnapshot:
    name: str
    legacy_plus: int
    legacy_minus: int
    lifetime_plus: int
    lifetime_minus: int
    ability_stars: int
    influence_minus: int
    influence_plus: int
    abilities: List[Tuple[str, int]]

    @property
    def net_lifetime(self) -> int:
        return int(self.lifetime_plus - self.lifetime_minus)

    @property
    def has_any_lifetime(self) -> bool:
        return (self.lifetime_plus + self.lifetime_minus) > 0


class VilyraBot(discord.Client):
    def __init__(self, db: Database):
        intents = discord.Intents.default()
        intents.guilds = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

# Register all app commands declared below with @app_commands.command
for cmd in (
    character_create,
    character_archive,
    character_delete,
    rank_set,
    legacy_add,
    legacy_remove,
    ability_stars_set,
    influence_stars_set,
    ability_add,
    ability_upgrade,
    refresh_dashboard,
    card,
):
    try:
        self.tree.add_command(cmd)
    except Exception:
        # Avoid hard-crashing if Discord caches an older signature during rapid deploys.
        pass
        self.db = db

    async def setup_hook(self) -> None:
        await self.db.connect()
        await self.db.init_schema()
        await self.tree.sync()
        LOG.info("Guild sync succeeded: %s commands", len(self.tree.get_commands()))

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, getattr(self.user, "id", "?"))
        guilds = list(self.guilds)
        if guilds:
            await refresh_all_safe(self, guilds[0], who="on_ready")


# -----------------------------
# Dashboard build + refresh
# -----------------------------
async def fetch_member_safe(guild: discord.Guild, user_id: int) -> Optional[discord.Member]:
    m = guild.get_member(user_id)
    if m is not None:
        return m
    try:
        return await guild.fetch_member(user_id)
    except Exception:
        return None


async def build_player_post(
    guild: discord.Guild,
    user_id: int,
    rank: str,
    chars: List[CharacterSnapshot],
) -> str:
    member = await fetch_member_safe(guild, user_id)
    display = member.display_name if member else f"User {user_id}"

    lines: List[str] = []
    lines.append(format_player_header(display, rank))

    for idx, c in enumerate(chars):
        if idx == 0:
            lines.append("")
        lines.append(format_character_header(c.name))
        lines.append(
            f"Legacy Points: +{c.legacy_plus} / -{c.legacy_minus}  ·  Lifetime: +{c.lifetime_plus} / -{c.lifetime_minus}"
        )
        lines.append(f"Ability Stars: {render_ability_stars_line(c.ability_stars)}")
        lines.append(f"Influence Stars: {render_influence_stars_line(c.influence_minus, c.influence_plus)}")
        b1, b2 = render_influence_bar_lines(c.net_lifetime, c.has_any_lifetime)
        lines.append(b1)
        lines.append(b2)
        lines.append(f"Abilities: {render_abilities_inline(c.abilities)}")

        if idx != len(chars) - 1:
            lines.append(BOARD_HEADER_SEPARATOR)

        if sum(len(x) + 1 for x in lines) > PLAYER_POST_SOFT_LIMIT:
            if lines and lines[-1].startswith("Abilities: "):
                lines[-1] = "Abilities: (too long to display)"
            break

    return "\n".join(lines).strip()


async def refresh_dashboard_board(bot: VilyraBot, guild: discord.Guild, who: str = "unknown") -> None:
    rows = await bot.db.list_all_active_characters(guild.id)

    grouped: Dict[int, List[CharacterSnapshot]] = {}
    for r in rows:
        uid = safe_int(r.get("user_id"))
        if not uid:
            continue
        name = r.get("name") or ""
        abilities = await bot.db.list_abilities(guild.id, uid, name)

        snap = CharacterSnapshot(
            name=name,
            legacy_plus=safe_int(r.get("legacy_plus")),
            legacy_minus=safe_int(r.get("legacy_minus")),
            lifetime_plus=safe_int(r.get("lifetime_plus")),
            lifetime_minus=safe_int(r.get("lifetime_minus")),
            ability_stars=safe_int(r.get("ability_stars")),
            influence_minus=safe_int(r.get("influence_minus")),
            influence_plus=safe_int(r.get("influence_plus")),
            abilities=abilities,
        )
        grouped.setdefault(uid, []).append(snap)

    # Archive players/characters if member left server
    for uid in list(grouped.keys()):
        member = await fetch_member_safe(guild, uid)
        if member is None:
            await bot.db.archive_player(guild.id, uid, True)
            chars = await bot.db.list_characters_for_user(guild.id, uid, include_archived=False)
            for c in chars:
                await bot.db.archive_character(guild.id, uid, c.get("name") or "", True)
            grouped.pop(uid, None)

    ch = guild.get_channel(DASHBOARD_CHANNEL_ID)
    if ch is None:
        ch = await guild.fetch_channel(DASHBOARD_CHANNEL_ID)
    if not isinstance(ch, discord.TextChannel):
        raise RuntimeError("Dashboard channel is not a text channel")

    for uid, chars in grouped.items():
        await bot.db.ensure_player(guild.id, uid)
        rank = await bot.db.get_player_rank(guild.id, uid)
        chars.sort(key=lambda c: c.name.lower())
        content = await build_player_post(guild, uid, rank, chars)

        msg_id = await bot.db.get_dashboard_message_id(guild.id, uid)
        msg: Optional[discord.Message] = None

        if msg_id:
            try:
                msg = await ch.fetch_message(msg_id)
            except Exception:
                msg = None

        if msg is None:
            msg = await ch.send(content, allowed_mentions=discord.AllowedMentions.none())
            await bot.db.set_dashboard_message_id(guild.id, uid, msg.id)
        else:
            if msg.content != content:
                await msg.edit(content=content, allowed_mentions=discord.AllowedMentions.none())

    await log_action(guild, f"✅ Dashboard refreshed ({who}).")


async def refresh_all_safe(bot: VilyraBot, guild: discord.Guild, who: str = "unknown") -> None:
    try:
        await refresh_dashboard_board(bot, guild, who=who)
    except Exception as e:
        LOG.exception("refresh_all_safe failed")
        await log_action(guild, f"❌ Dashboard update failed ({who}): {type(e).__name__}: {e}")


# -----------------------------
# Commands
# -----------------------------
def require_guardian_warden():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            return False
        if is_allowed_member(interaction.user):
            return True
        raise app_commands.CheckFailure("You do not have permission to use this command.")
    return app_commands.check(predicate)


@app_commands.command(name="character_create", description="Create (or unarchive) a character for a user.")
@require_guardian_warden()
@app_commands.describe(member="Owner of the character", name="Character name")
async def character_create(interaction: discord.Interaction, member: discord.Member, name: str):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    name_n = norm_name(name)
    await interaction.client.db.create_character(interaction.guild.id, member.id, name_n)
    await log_action(interaction.guild, f"🧾 {interaction.user.mention} created/unarchived character **{name_n}** for {member.mention}.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"character_create by {interaction.user.id}")
    await interaction.followup.send(f"✅ Character **{name_n}** created/unarchived for **{member.display_name}**.", ephemeral=True)


@app_commands.command(name="character_archive", description="Archive or unarchive a character (removes/shows it on the dashboard).")
@require_guardian_warden()
@app_commands.describe(member="Owner of the character", name="Character name", archived="True to archive, False to unarchive")
async def character_archive(interaction: discord.Interaction, member: discord.Member, name: str, archived: bool):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    name_n = norm_name(name)
    await interaction.client.db.archive_character(interaction.guild.id, member.id, name_n, archived)
    await log_action(interaction.guild, f"🗃️ {interaction.user.mention} set archived={archived} for **{name_n}** ({member.mention}).")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"character_archive by {interaction.user.id}")
    await interaction.followup.send(f"✅ Archived={archived} for **{name_n}**.", ephemeral=True)


@app_commands.command(name="character_delete", description="Delete a character permanently (also deletes its abilities).")
@require_guardian_warden()
@app_commands.describe(member="Owner of the character", name="Character name")
async def character_delete(interaction: discord.Interaction, member: discord.Member, name: str):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    name_n = norm_name(name)
    await interaction.client.db.delete_character(interaction.guild.id, member.id, name_n)
    await log_action(interaction.guild, f"🗑️ {interaction.user.mention} deleted character **{name_n}** ({member.mention}).")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"character_delete by {interaction.user.id}")
    await interaction.followup.send(f"✅ Deleted **{name_n}**.", ephemeral=True)


@app_commands.command(name="rank_set", description="Set a player's server rank (stored and shown on their dashboard post).")
@require_guardian_warden()
@app_commands.describe(member="The member to set rank for", rank="Rank text (e.g., Guardian, Warden, etc.)")
async def rank_set(interaction: discord.Interaction, member: discord.Member, rank: str):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    rank_clean = rank.strip() or "—"
    await interaction.client.db.set_player_rank(interaction.guild.id, member.id, rank_clean)
    await log_action(interaction.guild, f"🏷️ {interaction.user.mention} set rank **{rank_clean}** for {member.mention}.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"rank_set by {interaction.user.id}")
    await interaction.followup.send(f"✅ Rank set to **{rank_clean}** for **{member.display_name}**.", ephemeral=True)


@app_commands.command(name="legacy_add", description="Add legacy points to a character (also increments lifetime).")
@require_guardian_warden()
@app_commands.describe(member="Owner", name="Character name", plus="Positive points to add", minus="Negative points to add")
async def legacy_add(interaction: discord.Interaction, member: discord.Member, name: str, plus: int = 0, minus: int = 0):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    name_n = norm_name(name)
    plus_i = max(0, int(plus or 0))
    minus_i = max(0, int(minus or 0))
    await interaction.client.db.add_legacy_points(interaction.guild.id, member.id, name_n, plus_i, minus_i)
    await log_action(interaction.guild, f"➕ {interaction.user.mention} added legacy to **{name_n}** ({member.mention}): +{plus_i} / -{minus_i}.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"legacy_add by {interaction.user.id}")
    await interaction.followup.send(f"✅ Added to **{name_n}**: +{plus_i} / -{minus_i}.", ephemeral=True)


@app_commands.command(name="legacy_remove", description="Remove legacy points from current totals (does not change lifetime).")
@require_guardian_warden()
@app_commands.describe(member="Owner", name="Character name", plus="Positive points to remove", minus="Negative points to remove")
async def legacy_remove(interaction: discord.Interaction, member: discord.Member, name: str, plus: int = 0, minus: int = 0):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    name_n = norm_name(name)
    plus_i = max(0, int(plus or 0))
    minus_i = max(0, int(minus or 0))
    await interaction.client.db.remove_legacy_points(interaction.guild.id, member.id, name_n, plus_i, minus_i)
    await log_action(interaction.guild, f"➖ {interaction.user.mention} removed legacy from **{name_n}** ({member.mention}): +{plus_i} / -{minus_i}.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"legacy_remove by {interaction.user.id}")
    await interaction.followup.send(f"✅ Removed from **{name_n}**: +{plus_i} / -{minus_i} (clamped at 0).", ephemeral=True)


@app_commands.command(name="ability_stars_set", description="Set total ability stars for a character (0–5).")
@require_guardian_warden()
@app_commands.describe(member="Owner", name="Character name", stars="Total ability stars (0-5)")
async def ability_stars_set(interaction: discord.Interaction, member: discord.Member, name: str, stars: int):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    name_n = norm_name(name)
    stars_i = clamp(int(stars), 0, MAX_ABILITY_STARS_TOTAL)
    await interaction.client.db.set_ability_stars(interaction.guild.id, member.id, name_n, stars_i)
    await log_action(interaction.guild, f"✨ {interaction.user.mention} set ability stars for **{name_n}** ({member.mention}) to {stars_i}.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"ability_stars_set by {interaction.user.id}")
    await interaction.followup.send(f"✅ Ability stars for **{name_n}** set to **{stars_i}**.", ephemeral=True)


@app_commands.command(name="influence_stars_set", description="Set influence stars for a character (total max 5; split +/-).")
@require_guardian_warden()
@app_commands.describe(member="Owner", name="Character name", minus="Negative influence stars (0-5)", plus="Positive influence stars (0-5)")
async def influence_stars_set(interaction: discord.Interaction, member: discord.Member, name: str, minus: int, plus: int):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    name_n = norm_name(name)
    minus_i = clamp(int(minus), 0, MAX_INFLUENCE_STARS_TOTAL)
    plus_i = clamp(int(plus), 0, MAX_INFLUENCE_STARS_TOTAL)
    if minus_i + plus_i > MAX_INFLUENCE_STARS_TOTAL:
        await interaction.followup.send("❌ Total influence stars cannot exceed 5.", ephemeral=True)
        return
    await interaction.client.db.set_influence_stars(interaction.guild.id, member.id, name_n, minus_i, plus_i)
    await log_action(interaction.guild, f"⭐ {interaction.user.mention} set influence stars for **{name_n}** ({member.mention}) to -{minus_i} / +{plus_i}.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"influence_stars_set by {interaction.user.id}")
    await interaction.followup.send(f"✅ Influence stars for **{name_n}** set to -{minus_i} / +{plus_i}.", ephemeral=True)


@app_commands.command(name="ability_add", description="Add a custom ability to a character (max 7).")
@require_guardian_warden()
@app_commands.describe(member="Owner", character="Character name", ability="Ability name")
async def ability_add(interaction: discord.Interaction, member: discord.Member, character: str, ability: str):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    c = norm_name(character)
    a = norm_name(ability)
    if not a:
        await interaction.followup.send("❌ Ability name cannot be empty.", ephemeral=True)
        return
    try:
        await interaction.client.db.add_ability(interaction.guild.id, member.id, c, a)
    except ValueError as ve:
        await interaction.followup.send(f"❌ {ve}", ephemeral=True)
        return
    await log_action(interaction.guild, f"🧠 {interaction.user.mention} added ability **{a}** to **{c}** ({member.mention}).")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"ability_add by {interaction.user.id}")
    await interaction.followup.send(f"✅ Added ability **{a}** to **{c}**.", ephemeral=True)


@app_commands.command(name="ability_upgrade", description="Adjust upgrades for a character ability (+1/-1 etc).")
@require_guardian_warden()
@app_commands.describe(member="Owner", character="Character name", ability="Ability name", delta="Change in upgrades (e.g., 1 or -1)")
async def ability_upgrade(interaction: discord.Interaction, member: discord.Member, character: str, ability: str, delta: int):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    c = norm_name(character)
    a = norm_name(ability)
    d = int(delta or 0)
    await interaction.client.db.upgrade_ability(interaction.guild.id, member.id, c, a, d)
    await log_action(interaction.guild, f"🛠️ {interaction.user.mention} adjusted upgrades for **{a}** on **{c}** ({member.mention}) by {d}.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"ability_upgrade by {interaction.user.id}")
    await interaction.followup.send(f"✅ Updated **{a}** upgrades by {d} on **{c}**.", ephemeral=True)


@app_commands.command(name="refresh_dashboard", description="Refresh the dashboard now.")
@require_guardian_warden()
async def refresh_dashboard(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    await refresh_all_safe(interaction.client, interaction.guild, who=f"dashboard_refresh by {interaction.user.id}")
    await interaction.followup.send("✅ Dashboard refreshed.", ephemeral=True)


@app_commands.command(name="card", description="Show your character card (ephemeral).")
@app_commands.describe(name="Your character name")
async def card(interaction: discord.Interaction, name: str):
    await interaction.response.defer(ephemeral=True)
    assert interaction.guild is not None
    requester = interaction.user
    if not isinstance(requester, discord.Member):
        await interaction.followup.send("❌ This command must be used in a server.", ephemeral=True)
        return
    name_n = norm_name(name)

    row = await interaction.client.db.get_character(interaction.guild.id, requester.id, name_n)
    if row is None and is_allowed_member(requester):
        all_rows = await interaction.client.db._fetchall(
            "SELECT * FROM characters WHERE guild_id=%s AND name=%s LIMIT 1;",
            (interaction.guild.id, name_n),
        )
        row = all_rows[0] if all_rows else None

    if row is None:
        await interaction.followup.send("❌ Character not found (or you don’t own it).", ephemeral=True)
        return

    uid = safe_int(row.get("user_id"))
    abilities = await interaction.client.db.list_abilities(interaction.guild.id, uid, name_n)

    snap = CharacterSnapshot(
        name=row.get("name") or name_n,
        legacy_plus=safe_int(row.get("legacy_plus")),
        legacy_minus=safe_int(row.get("legacy_minus")),
        lifetime_plus=safe_int(row.get("lifetime_plus")),
        lifetime_minus=safe_int(row.get("lifetime_minus")),
        ability_stars=safe_int(row.get("ability_stars")),
        influence_minus=safe_int(row.get("influence_minus")),
        influence_plus=safe_int(row.get("influence_plus")),
        abilities=abilities,
    )

    lines: List[str] = []
    lines.append(format_character_header(snap.name))
    lines.append(f"Legacy Points: +{snap.legacy_plus} / -{snap.legacy_minus}  ·  Lifetime: +{snap.lifetime_plus} / -{snap.lifetime_minus}")
    lines.append(f"Ability Stars: {render_ability_stars_line(snap.ability_stars)}")
    lines.append(f"Influence Stars: {render_influence_stars_line(snap.influence_minus, snap.influence_plus)}")
    b1, b2 = render_influence_bar_lines(snap.net_lifetime, snap.has_any_lifetime)
    lines.append(b1)
    lines.append(b2)
    lines.append(f"Abilities: {render_abilities_inline(snap.abilities)}")

    await interaction.followup.send("\n".join(lines), ephemeral=True)


def main() -> None:
    token = os.getenv("DISCORD_TOKEN")
    dsn = os.getenv("DATABASE_URL")
    if not token:
        raise RuntimeError("DISCORD_TOKEN is not set")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    db = Database(dsn)
    client = VilyraBot(db=db)
    client.run(token)


if __name__ == "__main__":
    main()
