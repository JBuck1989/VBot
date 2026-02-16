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

MAX_STARS = 5
REP_MIN = -100
REP_MAX = 100

PLAYER_DIVIDER_LINE = "════════════════════════════════════"
SEP_LINE = "────────────────────────"

# Keep under Discord 2000 hard limit with breathing room
DASHBOARD_PAGE_LIMIT = 1900


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


async def log_action(guild: Optional[discord.Guild], text: str) -> None:
    if not guild:
        return
    ch_id = safe_int(os.getenv("COMMAND_LOG_CHANNEL_ID"), DEFAULT_COMMAND_LOG_CHANNEL_ID)
    try:
        ch = guild.get_channel(ch_id) or await guild.fetch_channel(ch_id)
        if isinstance(ch, discord.TextChannel):
            await ch.send(text, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        LOG.exception("Failed to write to command log channel")


def render_ability_stars(n: int) -> str:
    n = clamp(int(n), 0, MAX_STARS)
    return "★" * n + "☆" * (MAX_STARS - n)


def render_influence_stars(minus: int, plus: int) -> str:
    minus = clamp(int(minus), 0, MAX_STARS)
    plus = clamp(int(plus), 0, MAX_STARS)

    neg = ["☆"] * MAX_STARS
    for i in range(minus):
        neg[MAX_STARS - 1 - i] = "★"

    pos = ["☆"] * MAX_STARS
    for i in range(plus):
        pos[i] = "★"

    return f"-{''.join(neg)}|{''.join(pos)}+"


def render_reputation_bar(net: int) -> Tuple[str, str]:
    top = "FEARED           <- | ->          LOVED"
    net = clamp(int(net), REP_MIN, REP_MAX)
    pos = int(round((net - REP_MIN) / (REP_MAX - REP_MIN) * 20))
    pos = clamp(pos, 0, 20)

    bar = ["-"] * 20
    if pos == 0:
        bar[0] = "│"
    elif pos >= 20:
        bar[-1] = "│"
    else:
        bar[pos - 1] = "│"
    return top, f"[{''.join(bar)}]  {net:+d}"


def split_pages(text: str, limit: int = DASHBOARD_PAGE_LIMIT) -> List[str]:
    """
    Split text into pages <= limit.
    Prefer splitting on double-newline boundaries, then single newline, then hard split.
    """
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
            cut = limit  # hard split as last resort
        page = remaining[:cut].rstrip()
        pages.append(page)
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
        self._abilities_col: str = "ability"

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

    async def _detect_abilities_column(self) -> str:
        rows = await self._fetchall(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'abilities';
            """
        )
        cols = {r["column_name"] for r in rows}
        for candidate in ("ability", "ability_name", "abilityKey", "ability_key", "skill", "skill_name"):
            if candidate in cols:
                return candidate
        return "ability"

    async def init_schema(self) -> None:
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                name          TEXT   NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id, name)
            );
            """
        )

        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                name          TEXT   NOT NULL,
                ability       TEXT   NOT NULL,
                stars         INT    NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, name, ability)
            );
            """
        )

        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS influence (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                name          TEXT   NOT NULL,
                minus_stars   INT    NOT NULL DEFAULT 0,
                plus_stars    INT    NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, name)
            );
            """
        )

        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS reputation (
                guild_id      BIGINT NOT NULL,
                user_id       BIGINT NOT NULL,
                name          TEXT   NOT NULL,
                net           INT    NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, name)
            );
            """
        )

        # dashboard_state now supports multiple messages via message_ids (comma-separated)
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_state (
                guild_id      BIGINT PRIMARY KEY,
                channel_id    BIGINT NOT NULL,
                message_id    BIGINT,
                message_ids   TEXT
            );
            """
        )
        # Add column in case table existed from earlier versions
        await self._execute("ALTER TABLE dashboard_state ADD COLUMN IF NOT EXISTS message_ids TEXT;")

        self._abilities_col = await self._detect_abilities_column()
        LOG.info("Database schema initialized / updated (abilities column=%s)", self._abilities_col)

    # -------- characters --------

    async def add_character(self, guild_id: int, user_id: int, name: str) -> None:
        await self._execute(
            "INSERT INTO characters (guild_id, user_id, name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
            (guild_id, user_id, name),
        )
        await self._execute(
            "INSERT INTO influence (guild_id, user_id, name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
            (guild_id, user_id, name),
        )
        await self._execute(
            "INSERT INTO reputation (guild_id, user_id, name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
            (guild_id, user_id, name),
        )

    async def remove_character(self, guild_id: int, user_id: int, name: str) -> None:
        await self._execute("DELETE FROM abilities WHERE guild_id=%s AND user_id=%s AND name=%s;", (guild_id, user_id, name))
        await self._execute("DELETE FROM influence WHERE guild_id=%s AND user_id=%s AND name=%s;", (guild_id, user_id, name))
        await self._execute("DELETE FROM reputation WHERE guild_id=%s AND user_id=%s AND name=%s;", (guild_id, user_id, name))
        await self._execute("DELETE FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s;", (guild_id, user_id, name))

    async def get_character(self, guild_id: int, user_id: int, name: str) -> Optional[Dict[str, Any]]:
        return await self._fetchone(
            "SELECT guild_id, user_id, name, created_at FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (guild_id, user_id, name),
        )

    async def list_characters_for_user(self, guild_id: int, user_id: int) -> List[Dict[str, Any]]:
        return await self._fetchall(
            "SELECT name, created_at FROM characters WHERE guild_id=%s AND user_id=%s ORDER BY name ASC;",
            (guild_id, user_id),
        )

    async def list_users_with_characters(self, guild_id: int) -> List[int]:
        rows = await self._fetchall(
            "SELECT DISTINCT user_id FROM characters WHERE guild_id=%s ORDER BY user_id ASC;",
            (guild_id,)
        )
        return [int(r["user_id"]) for r in rows]

    # -------- abilities --------

    async def set_ability_stars(self, guild_id: int, user_id: int, name: str, ability: str, stars: int) -> None:
        stars = clamp(int(stars), 0, MAX_STARS)
        col = self._abilities_col

        try:
            await self._execute(
                f"""
                INSERT INTO abilities (guild_id, user_id, name, {col}, stars)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (guild_id, user_id, name, {col})
                DO UPDATE SET stars = EXCLUDED.stars;
                """,
                (guild_id, user_id, name, ability, stars),
            )
        except psycopg.Error:
            # Fallback for weird legacy constraints
            await self._execute(
                f"DELETE FROM abilities WHERE guild_id=%s AND user_id=%s AND name=%s AND {col}=%s;",
                (guild_id, user_id, name, ability),
            )
            await self._execute(
                f"INSERT INTO abilities (guild_id, user_id, name, {col}, stars) VALUES (%s, %s, %s, %s, %s);",
                (guild_id, user_id, name, ability, stars),
            )

    async def get_abilities(self, guild_id: int, user_id: int, name: str) -> List[Dict[str, Any]]:
        col = self._abilities_col
        return await self._fetchall(
            f"SELECT {col} AS ability, stars FROM abilities WHERE guild_id=%s AND user_id=%s AND name=%s ORDER BY {col} ASC;",
            (guild_id, user_id, name),
        )

    # -------- influence --------

    async def get_influence(self, guild_id: int, user_id: int, name: str) -> Dict[str, int]:
        row = await self._fetchone(
            "SELECT minus_stars, plus_stars FROM influence WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (guild_id, user_id, name),
        )
        if not row:
            return {"minus_stars": 0, "plus_stars": 0}
        return {"minus_stars": safe_int(row.get("minus_stars"), 0), "plus_stars": safe_int(row.get("plus_stars"), 0)}

    async def add_influence(self, guild_id: int, user_id: int, name: str, plus: int = 0, minus: int = 0) -> Dict[str, int]:
        cur = await self.get_influence(guild_id, user_id, name)
        new_minus = clamp(cur["minus_stars"] + int(minus), 0, MAX_STARS)
        new_plus = clamp(cur["plus_stars"] + int(plus), 0, MAX_STARS)
        await self._execute(
            """
            INSERT INTO influence (guild_id, user_id, name, minus_stars, plus_stars)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (guild_id, user_id, name)
            DO UPDATE SET minus_stars = EXCLUDED.minus_stars, plus_stars = EXCLUDED.plus_stars;
            """,
            (guild_id, user_id, name, new_minus, new_plus),
        )
        return {"minus_stars": new_minus, "plus_stars": new_plus}

    async def remove_influence(self, guild_id: int, user_id: int, name: str, plus: int = 0, minus: int = 0) -> Dict[str, int]:
        return await self.add_influence(guild_id, user_id, name, plus=-int(plus), minus=-int(minus))

    # -------- reputation --------

    async def get_reputation(self, guild_id: int, user_id: int, name: str) -> int:
        row = await self._fetchone(
            "SELECT net FROM reputation WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (guild_id, user_id, name),
        )
        if not row:
            return 0
        return clamp(safe_int(row.get("net"), 0), REP_MIN, REP_MAX)

    async def set_reputation(self, guild_id: int, user_id: int, name: str, net: int) -> int:
        net = clamp(int(net), REP_MIN, REP_MAX)
        await self._execute(
            """
            INSERT INTO reputation (guild_id, user_id, name, net)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (guild_id, user_id, name)
            DO UPDATE SET net = EXCLUDED.net;
            """,
            (guild_id, user_id, name, net),
        )
        return net

    # -------- dashboard state --------

    async def get_dashboard_state(self, guild_id: int) -> Dict[str, Any]:
        row = await self._fetchone("SELECT guild_id, channel_id, message_id, message_ids FROM dashboard_state WHERE guild_id=%s;", (guild_id,))
        if not row:
            ch_id = safe_int(os.getenv("DASHBOARD_CHANNEL_ID"), DEFAULT_DASHBOARD_CHANNEL_ID)
            await self._execute(
                "INSERT INTO dashboard_state (guild_id, channel_id, message_id, message_ids) VALUES (%s, %s, NULL, NULL) ON CONFLICT DO NOTHING;",
                (guild_id, ch_id),
            )
            return {"guild_id": guild_id, "channel_id": ch_id, "message_id": None, "message_ids": None}
        return row

    async def set_dashboard_messages(self, guild_id: int, channel_id: int, message_ids: List[int]) -> None:
        # Keep legacy message_id as first entry for backward compatibility.
        first = message_ids[0] if message_ids else None
        await self._execute(
            """
            INSERT INTO dashboard_state (guild_id, channel_id, message_id, message_ids)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (guild_id)
            DO UPDATE SET channel_id=EXCLUDED.channel_id, message_id=EXCLUDED.message_id, message_ids=EXCLUDED.message_ids;
            """,
            (guild_id, channel_id, first, fmt_ids(message_ids) if message_ids else None),
        )


# -----------------------------
# DASHBOARD RENDERING
# -----------------------------

@dataclass
class CharacterSnapshot:
    name: str
    abilities: List[Tuple[str, int]]
    minus_stars: int
    plus_stars: int
    reputation: int


async def build_character_snapshot(db: Database, guild_id: int, user_id: int, name: str) -> CharacterSnapshot:
    abilities_rows = await db.get_abilities(guild_id, user_id, name)
    abilities = [(r["ability"], safe_int(r.get("stars"), 0)) for r in abilities_rows]
    inf = await db.get_influence(guild_id, user_id, name)
    rep = await db.get_reputation(guild_id, user_id, name)
    return CharacterSnapshot(name=name, abilities=abilities, minus_stars=inf["minus_stars"], plus_stars=inf["plus_stars"], reputation=rep)


async def render_dashboard_post(db: Database, guild: discord.Guild) -> str:
    user_ids = await db.list_users_with_characters(guild.id)
    if not user_ids:
        return "No characters found yet.\n\nUse `/character_add` to get started."

    lines: List[str] = []
    lines.append("## Vilyra Legacy Dashboard")
    lines.append("*(auto-updated; staff can force refresh with `/dashboard_refresh`)*")
    lines.append("")

    for uid in user_ids:
        member = guild.get_member(uid)
        display = member.display_name if member else f"User {uid}"
        lines.append(PLAYER_DIVIDER_LINE)
        lines.append(f"**{display}** (`{uid}`)")

        chars = await db.list_characters_for_user(guild.id, uid)
        if not chars:
            lines.append("_No characters._")
            continue

        for c in chars:
            name = c["name"]
            snap = await build_character_snapshot(db, guild.id, uid, name)
            lines.append(SEP_LINE)
            lines.append(f"### {snap.name}")
            lines.append(f"Influence: {render_influence_stars(snap.minus_stars, snap.plus_stars)}")
            top, bar = render_reputation_bar(snap.reputation)
            lines.append("Reputation:")
            lines.append(f"{top}\n{bar}")

            if snap.abilities:
                lines.append("Abilities:")
                for ability, stars in snap.abilities:
                    lines.append(f"- **{ability}**: {render_ability_stars(stars)}")
            else:
                lines.append("Abilities: _none set_")

        lines.append("")

    return "\n".join(lines).strip()


async def refresh_dashboard_board(client: "VilyraBotClient", guild: discord.Guild) -> str:
    """
    Updates/creates the dashboard messages. Returns a human-readable status.
    Does NOT raise for normal failures; it returns the error string.
    """
    db = client.db
    state = await db.get_dashboard_state(guild.id)
    channel_id = safe_int(state.get("channel_id"), safe_int(os.getenv("DASHBOARD_CHANNEL_ID"), DEFAULT_DASHBOARD_CHANNEL_ID))
    stored_ids = parse_ids(state.get("message_ids")) or ([] if not state.get("message_id") else [safe_int(state.get("message_id"))])

    # Fetch channel
    channel = guild.get_channel(channel_id)
    if channel is None:
        try:
            channel = await guild.fetch_channel(channel_id)
        except Exception:
            channel = None

    if not isinstance(channel, discord.TextChannel):
        msg = f"Dashboard channel {channel_id} not found or not a text channel."
        LOG.error(msg)
        return msg

    # Permission check (common silent failure)
    me = guild.me or guild.get_member(client.user.id) if client.user else None
    if me:
        perms = channel.permissions_for(me)
        if not (perms.view_channel and perms.send_messages):
            msg = f"Missing permissions in <#{channel.id}>: need View Channel + Send Messages."
            LOG.error(msg)
            return msg

    try:
        content = await render_dashboard_post(db, guild)
        pages = split_pages(content, DASHBOARD_PAGE_LIMIT)
        header = f"**Dashboard Pages:** {len(pages)}"

        # Ensure we have the right number of messages
        messages: List[discord.Message] = []

        # Try load existing messages
        for mid in stored_ids:
            try:
                m = await channel.fetch_message(mid)
                messages.append(m)
            except Exception:
                continue

        # Create missing messages
        while len(messages) < len(pages):
            m = await channel.send("Creating dashboard…")
            messages.append(m)

        # Edit pages into messages (first message includes header)
        for idx, page in enumerate(pages):
            body = page
            if idx == 0:
                body = f"{header}\n\n{page}"
            await messages[idx].edit(content=body)

        # Delete extra old messages (if dashboard shrank)
        for extra in messages[len(pages):]:
            try:
                await extra.delete()
            except Exception:
                pass

        final_ids = [m.id for m in messages[:len(pages)]]
        await db.set_dashboard_messages(guild.id, channel.id, final_ids)

        ok = f"Dashboard posted to <#{channel.id}> in {len(pages)} message(s)."
        LOG.info(ok)
        return ok

    except discord.HTTPException as e:
        # Most common: content too long / missing perms / bad request
        msg = f"Discord HTTP error while posting dashboard: {e}"
        LOG.exception(msg)
        return msg
    except Exception as e:
        msg = f"Board update failed: {type(e).__name__}: {e}"
        LOG.exception("❌ Board update failed")
        return msg


async def refresh_all_safe(client: "VilyraBotClient", guild: discord.Guild, who: str = "startup") -> str:
    try:
        return await refresh_dashboard_board(client, guild)
    except Exception as e:
        LOG.exception("refresh_all_safe failed (who=%s)", who)
        return f"refresh_all_safe failed: {type(e).__name__}: {e}"


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
        for guild in list(self.guilds):
            status = await refresh_all_safe(self, guild, who="on_ready")
            if "posted to" not in status.lower():
                LOG.warning("Startup dashboard refresh issue (guild=%s): %s", guild.id, status)


# -----------------------------
# COMMANDS
# -----------------------------

def staff_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if not interaction.guild:
            return False
        if isinstance(interaction.user, discord.Member) and is_staff(interaction.user):
            return True
        await interaction.response.send_message("Staff only.", ephemeral=True)
        return False
    return app_commands.check(predicate)


def in_guild_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await interaction.response.send_message("This command can only be used in a server.", ephemeral=True)
            return False
        return True
    return app_commands.check(predicate)


async def ensure_character_exists(db: Database, guild_id: int, user_id: int, name: str) -> bool:
    return (await db.get_character(guild_id, user_id, name)) is not None


@app_commands.command(name="character_add", description="Add a character to your roster.")
@in_guild_only()
async def character_add(interaction: discord.Interaction, name: str):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None
    name = name.strip()
    await interaction.client.db.add_character(interaction.guild.id, interaction.user.id, name)
    await log_action(interaction.guild, f"✅ {interaction.user.mention} added character **{name}**")
    status = await refresh_all_safe(interaction.client, interaction.guild, who="character_add")
    await interaction.followup.send(f"Added character **{name}**.\n{status}", ephemeral=True)


@app_commands.command(name="character_remove", description="Remove one of your characters (deletes associated stats).")
@in_guild_only()
async def character_remove(interaction: discord.Interaction, name: str):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None
    name = name.strip()
    if not await ensure_character_exists(interaction.client.db, interaction.guild.id, interaction.user.id, name):
        await interaction.followup.send("That character name wasn't found on your roster.", ephemeral=True)
        return
    await interaction.client.db.remove_character(interaction.guild.id, interaction.user.id, name)
    await log_action(interaction.guild, f"🗑️ {interaction.user.mention} removed character **{name}**")
    status = await refresh_all_safe(interaction.client, interaction.guild, who="character_remove")
    await interaction.followup.send(f"Removed character **{name}**.\n{status}", ephemeral=True)


@app_commands.command(name="ability_set", description="Set an ability's star level for one of your characters.")
@in_guild_only()
async def ability_set(interaction: discord.Interaction, character: str, ability: str, stars: int):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None
    character = character.strip()
    ability = ability.strip()
    if not await ensure_character_exists(interaction.client.db, interaction.guild.id, interaction.user.id, character):
        await interaction.followup.send("That character name wasn't found on your roster.", ephemeral=True)
        return
    stars = clamp(int(stars), 0, MAX_STARS)
    await interaction.client.db.set_ability_stars(interaction.guild.id, interaction.user.id, character, ability, stars)
    await log_action(interaction.guild, f"⭐ {interaction.user.mention} set **{character}** ability **{ability}** to {stars} stars")
    status = await refresh_all_safe(interaction.client, interaction.guild, who="ability_set")
    await interaction.followup.send(f"Set **{ability}** for **{character}** to {render_ability_stars(stars)}.\n{status}", ephemeral=True)


@app_commands.command(name="influence_stars_add", description="(Staff) Add influence stars to a character.")
@in_guild_only()
@staff_only()
async def influence_stars_add(interaction: discord.Interaction, user: discord.Member, character: str, plus: int = 0, minus: int = 0):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None
    character = character.strip()
    if not await ensure_character_exists(interaction.client.db, interaction.guild.id, user.id, character):
        await interaction.followup.send("That character name wasn't found on that user's roster.", ephemeral=True)
        return
    res = await interaction.client.db.add_influence(interaction.guild.id, user.id, character, plus=plus, minus=minus)
    await log_action(interaction.guild, f"➕ {interaction.user.mention} added influence to **{character}** ({user.mention}): {res}")
    status = await refresh_all_safe(interaction.client, interaction.guild, who="influence_stars_add")
    await interaction.followup.send(
        f"Updated influence for **{character}**: {render_influence_stars(res['minus_stars'], res['plus_stars'])}\n{status}",
        ephemeral=True,
    )


@app_commands.command(name="influence_stars_remove", description="(Staff) Remove influence stars from a character.")
@in_guild_only()
@staff_only()
async def influence_stars_remove(interaction: discord.Interaction, user: discord.Member, character: str, plus: int = 0, minus: int = 0):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None
    character = character.strip()
    if not await ensure_character_exists(interaction.client.db, interaction.guild.id, user.id, character):
        await interaction.followup.send("That character name wasn't found on that user's roster.", ephemeral=True)
        return
    res = await interaction.client.db.remove_influence(interaction.guild.id, user.id, character, plus=plus, minus=minus)
    await log_action(interaction.guild, f"➖ {interaction.user.mention} removed influence from **{character}** ({user.mention}): {res}")
    status = await refresh_all_safe(interaction.client, interaction.guild, who="influence_stars_remove")
    await interaction.followup.send(
        f"Updated influence for **{character}**: {render_influence_stars(res['minus_stars'], res['plus_stars'])}\n{status}",
        ephemeral=True,
    )


@app_commands.command(name="reputation_set", description="(Staff) Set reputation net value (-100..+100) for a character.")
@in_guild_only()
@staff_only()
async def reputation_set(interaction: discord.Interaction, user: discord.Member, character: str, net: int):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None
    character = character.strip()
    if not await ensure_character_exists(interaction.client.db, interaction.guild.id, user.id, character):
        await interaction.followup.send("That character name wasn't found on that user's roster.", ephemeral=True)
        return
    net = clamp(int(net), REP_MIN, REP_MAX)
    new_net = await interaction.client.db.set_reputation(interaction.guild.id, user.id, character, net)
    await log_action(interaction.guild, f"📈 {interaction.user.mention} set reputation for **{character}** ({user.mention}) to {new_net:+d}")
    status = await refresh_all_safe(interaction.client, interaction.guild, who="reputation_set")
    await interaction.followup.send(f"Reputation for **{character}** set to {new_net:+d}.\n{status}", ephemeral=True)


@app_commands.command(name="dashboard_refresh", description="(Staff) Force refresh the dashboard message.")
@in_guild_only()
@staff_only()
async def dashboard_refresh(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    assert interaction.guild is not None
    status = await refresh_all_safe(interaction.client, interaction.guild, who="dashboard_refresh")
    await interaction.followup.send(status, ephemeral=True)


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

    client.tree.add_command(character_add)
    client.tree.add_command(character_remove)
    client.tree.add_command(ability_set)
    client.tree.add_command(influence_stars_add)
    client.tree.add_command(influence_stars_remove)
    client.tree.add_command(reputation_set)
    client.tree.add_command(dashboard_refresh)

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
