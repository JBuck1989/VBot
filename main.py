import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import discord
from discord import app_commands

import psycopg
from psycopg.rows import dict_row


# -----------------------------
# CONFIG (edit these)
# -----------------------------
DASHBOARD_CHANNEL_ID = 1469879866655768738
COMMAND_LOG_CHANNEL_ID = 1469879960729817098

GUARDIAN_ROLE_NAME = "Guardian"
WARDEN_ROLE_NAME = "Warden"

# Influence star caps (display only)
MAX_STARS = 5

# Reputation bar display range; values outside clamp to ends
REP_MIN = -100
REP_MAX = 100

# Dashboard formatting
SEP_LINE = "────────────────────────"


# -----------------------------
# LOGGING
# -----------------------------
LOG = logging.getLogger("VilyraBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VilyraBot: %(message)s")


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


def is_staff(member: discord.abc.User | discord.Member) -> bool:
    # Interactions always provide a Member in guild context; but be defensive
    roles = getattr(member, "roles", []) or []
    for r in roles:
        if getattr(r, "name", "") in (GUARDIAN_ROLE_NAME, WARDEN_ROLE_NAME):
            return True
    return False


async def log_action(guild: discord.Guild, text: str) -> None:
    if not COMMAND_LOG_CHANNEL_ID:
        return
    try:
        ch = guild.get_channel(COMMAND_LOG_CHANNEL_ID) or await guild.fetch_channel(COMMAND_LOG_CHANNEL_ID)
        if isinstance(ch, discord.TextChannel):
            await ch.send(text, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        # Never crash on logging
        LOG.exception("Failed to write to command log channel")


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def render_ability_stars(n: int) -> str:
    n = clamp(int(n), 0, MAX_STARS)
    return "★" * n + "☆" * (MAX_STARS - n)


def render_influence_stars(minus: int, plus: int) -> str:
    minus = clamp(int(minus), 0, MAX_STARS)
    plus = clamp(int(plus), 0, MAX_STARS)

    neg = ["☆"] * MAX_STARS
    # fill from center outward => right-to-left
    for i in range(minus):
        neg[MAX_STARS - 1 - i] = "★"

    pos = ["☆"] * MAX_STARS
    # fill from center outward => left-to-right
    for i in range(plus):
        pos[i] = "★"

    return f"-{''.join(neg)}|{''.join(pos)}+"


def render_reputation_bar(net: int) -> Tuple[str, str]:
    """
    Returns 2 lines:
      FEARED           <- | ->          LOVED
      --------------------|--------------------
    With a bold indicator embedded on the dash line.
    """
    top = "FEARED           <- | ->          LOVED"

    # Build base line
    left = ["-"] * 20
    right = ["-"] * 20
    center = "|"

    # Clamp to display range
    v = clamp(int(net), REP_MIN, REP_MAX)

    # Map [-100..+100] -> 0..40 (41 positions including center)
    # pos==20 => center
    pos = int(round(((v - REP_MIN) / (REP_MAX - REP_MIN)) * 40))
    pos = clamp(pos, 0, 40)

    if pos == 20:
        center = "●"
    elif pos < 20:
        left[pos] = "●"
    else:
        # pos 21..40 -> right 0..19
        right[pos - 21] = "●"

    bottom = "".join(left) + center + "".join(right)
    return top, bottom


def fmt_character_header(name: str) -> str:
    # Bold + italic + underline
    # Example: __***꧁•⊹٭ Elarion Vaelith ٭⊹•꧂***__
    return f"__***꧁•⊹٭ {name} ٭⊹•꧂***__"


def fmt_player_header(player_name: str, rank: str) -> str:
    # Underlined + bold per requirement
    return f"__**Player: {player_name} | Server Rank: {rank}**__"


def fmt_abilities(abilities: List[Tuple[str, int]]) -> str:
    if not abilities:
        return "—"
    parts = []
    for (n, u) in abilities[:7]:
        parts.append(f"{n} ({safe_int(u, 0)})")
    return ", ".join(parts)


# -----------------------------
# DATABASE
# -----------------------------
class Database:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn: Optional[psycopg.AsyncConnection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._conn and not self._conn.closed:
            return
        LOG.info("Connecting to PostgreSQL...")
        self._conn = await psycopg.AsyncConnection.connect(self.dsn, row_factory=dict_row)
        await self._conn.set_autocommit(True)
        LOG.info("PostgreSQL async connection established (autocommit=True)")

    async def close(self) -> None:
        if self._conn and not self._conn.closed:
            await self._conn.close()

    async def _execute(self, sql: str, params: Tuple[Any, ...] = ()) -> None:
        await self.connect()
        assert self._conn
        async with self._lock:
            async with self._conn.cursor() as cur:
                await cur.execute(sql, params)

    async def _fetchall(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        await self.connect()
        assert self._conn
        async with self._lock:
            async with self._conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                return list(rows or [])

    async def _fetchone(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        rows = await self._fetchall(sql, params)
        return rows[0] if rows else None

    async def init_schema(self) -> None:
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS characters (
              guild_id BIGINT NOT NULL,
              user_id BIGINT NOT NULL,
              name TEXT NOT NULL,
              archived BOOLEAN NOT NULL DEFAULT FALSE,
              legacy_plus BIGINT NOT NULL DEFAULT 0,
              legacy_minus BIGINT NOT NULL DEFAULT 0,
              lifetime_plus BIGINT NOT NULL DEFAULT 0,
              lifetime_minus BIGINT NOT NULL DEFAULT 0,
              ability_stars INT NOT NULL DEFAULT 0,
              influence_minus INT NOT NULL DEFAULT 0,
              influence_plus INT NOT NULL DEFAULT 0,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (guild_id, user_id, name)
            );
            """
        )
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS abilities (
              guild_id BIGINT NOT NULL,
              user_id BIGINT NOT NULL,
              character_name TEXT NOT NULL,
              ability_name TEXT NOT NULL,
              upgrades INT NOT NULL DEFAULT 0,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (guild_id, user_id, character_name, ability_name)
            );
            """
        )
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS user_ranks (
              guild_id BIGINT NOT NULL,
              user_id BIGINT NOT NULL,
              rank TEXT NOT NULL,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (guild_id, user_id)
            );
            """
        )
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_posts (
              guild_id BIGINT NOT NULL,
              user_id BIGINT NOT NULL,
              message_id BIGINT NOT NULL,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (guild_id, user_id)
            );
            """
        )
        LOG.info("Database schema initialized / updated")

    # ---- characters
    async def upsert_character(self, guild_id: int, user_id: int, name: str) -> None:
        await self._execute(
            """
            INSERT INTO characters (guild_id, user_id, name, archived, updated_at)
            VALUES (%s, %s, %s, FALSE, now())
            ON CONFLICT (guild_id, user_id, name) DO UPDATE
              SET archived = FALSE,
                  updated_at = now();
            """,
            (guild_id, user_id, name),
        )

    async def set_archived(self, guild_id: int, user_id: int, name: str, archived: bool) -> None:
        await self._execute(
            """
            UPDATE characters
               SET archived=%s,
                   updated_at=now()
             WHERE guild_id=%s AND user_id=%s AND name=%s;
            """,
            (archived, guild_id, user_id, name),
        )

    async def delete_character(self, guild_id: int, user_id: int, name: str) -> None:
        # delete abilities first
        await self._execute(
            "DELETE FROM abilities WHERE guild_id=%s AND user_id=%s AND character_name=%s;",
            (guild_id, user_id, name),
        )
        await self._execute(
            "DELETE FROM characters WHERE guild_id=%s AND user_id=%s AND name=%s;",
            (guild_id, user_id, name),
        )

    async def get_characters_for_user(self, guild_id: int, user_id: int, include_archived: bool = False) -> List[Dict[str, Any]]:
        if include_archived:
            return await self._fetchall(
                """
                SELECT * FROM characters
                 WHERE guild_id=%s AND user_id=%s
                 ORDER BY lower(name) ASC;
                """,
                (guild_id, user_id),
            )
        return await self._fetchall(
            """
            SELECT * FROM characters
             WHERE guild_id=%s AND user_id=%s AND archived=FALSE
             ORDER BY lower(name) ASC;
            """,
            (guild_id, user_id),
        )

    async def get_all_active_characters(self, guild_id: int) -> List[Dict[str, Any]]:
        return await self._fetchall(
            """
            SELECT * FROM characters
             WHERE guild_id=%s AND archived=FALSE
             ORDER BY user_id, lower(name);
            """,
            (guild_id,),
        )

    async def add_legacy(self, guild_id: int, user_id: int, name: str, amount: int) -> None:
        amt = abs(int(amount))
        await self._execute(
            """
            UPDATE characters
               SET legacy_plus = legacy_plus + %s,
                   lifetime_plus = lifetime_plus + %s,
                   updated_at = now()
             WHERE guild_id=%s AND user_id=%s AND name=%s AND archived=FALSE;
            """,
            (amt, amt, guild_id, user_id, name),
        )

    async def remove_legacy(self, guild_id: int, user_id: int, name: str, amount: int) -> None:
        amt = abs(int(amount))
        await self._execute(
            """
            UPDATE characters
               SET legacy_minus = legacy_minus + %s,
                   lifetime_minus = lifetime_minus + %s,
                   updated_at = now()
             WHERE guild_id=%s AND user_id=%s AND name=%s AND archived=FALSE;
            """,
            (amt, amt, guild_id, user_id, name),
        )

    async def set_ability_stars(self, guild_id: int, user_id: int, name: str, stars: int) -> None:
        stars = clamp(int(stars), 0, MAX_STARS)
        await self._execute(
            """
            UPDATE characters
               SET ability_stars=%s,
                   updated_at=now()
             WHERE guild_id=%s AND user_id=%s AND name=%s AND archived=FALSE;
            """,
            (stars, guild_id, user_id, name),
        )

    async def set_influence_stars(self, guild_id: int, user_id: int, name: str, minus: int, plus: int) -> None:
        minus = clamp(int(minus), 0, MAX_STARS)
        plus = clamp(int(plus), 0, MAX_STARS)
        await self._execute(
            """
            UPDATE characters
               SET influence_minus=%s,
                   influence_plus=%s,
                   updated_at=now()
             WHERE guild_id=%s AND user_id=%s AND name=%s AND archived=FALSE;
            """,
            (minus, plus, guild_id, user_id, name),
        )

    # ---- abilities
    async def ability_add(self, guild_id: int, user_id: int, character_name: str, ability_name: str) -> None:
        await self._execute(
            """
            INSERT INTO abilities (guild_id, user_id, character_name, ability_name, upgrades)
            VALUES (%s, %s, %s, %s, 0)
            ON CONFLICT (guild_id, user_id, character_name, ability_name) DO NOTHING;
            """,
            (guild_id, user_id, character_name, ability_name),
        )

    async def ability_upgrade(self, guild_id: int, user_id: int, character_name: str, ability_name: str, delta: int) -> None:
        delta = int(delta)
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

    async def get_abilities(self, guild_id: int, user_id: int, character_name: str) -> List[Tuple[str, int]]:
        rows = await self._fetchall(
            """
            SELECT ability_name, upgrades
              FROM abilities
             WHERE guild_id=%s AND user_id=%s AND character_name=%s
             ORDER BY lower(ability_name) ASC;
            """,
            (guild_id, user_id, character_name),
        )
        return [(str(r["ability_name"]), safe_int(r["upgrades"], 0)) for r in rows]

    # ---- ranks
    async def set_rank(self, guild_id: int, user_id: int, rank: str) -> None:
        await self._execute(
            """
            INSERT INTO user_ranks (guild_id, user_id, rank)
            VALUES (%s, %s, %s)
            ON CONFLICT (guild_id, user_id) DO UPDATE
              SET rank=EXCLUDED.rank, updated_at=now();
            """,
            (guild_id, user_id, rank),
        )

    async def get_rank(self, guild_id: int, user_id: int) -> str:
        row = await self._fetchone("SELECT rank FROM user_ranks WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))
        if not row:
            return "—"
        return str(row.get("rank") or "—")

    # ---- dashboard posts
    async def get_dashboard_message_id(self, guild_id: int, user_id: int) -> Optional[int]:
        row = await self._fetchone(
            "SELECT message_id FROM dashboard_posts WHERE guild_id=%s AND user_id=%s;",
            (guild_id, user_id),
        )
        if not row:
            return None
        mid = safe_int(row.get("message_id"), 0)
        return mid or None

    async def set_dashboard_message_id(self, guild_id: int, user_id: int, message_id: int) -> None:
        await self._execute(
            """
            INSERT INTO dashboard_posts (guild_id, user_id, message_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (guild_id, user_id) DO UPDATE
              SET message_id=EXCLUDED.message_id, updated_at=now();
            """,
            (guild_id, user_id, message_id),
        )


# -----------------------------
# SNAPSHOTS + RENDERING
# -----------------------------
@dataclass
class CharacterSnapshot:
    user_id: int
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


async def build_character_snapshot(db: Database, guild_id: int, user_id: int, row: Dict[str, Any]) -> CharacterSnapshot:
    name = str(row["name"])
    abilities = await db.get_abilities(guild_id, user_id, name)
    return CharacterSnapshot(
        user_id=user_id,
        name=name,
        legacy_plus=safe_int(row.get("legacy_plus")),
        legacy_minus=safe_int(row.get("legacy_minus")),
        lifetime_plus=safe_int(row.get("lifetime_plus")),
        lifetime_minus=safe_int(row.get("lifetime_minus")),
        ability_stars=safe_int(row.get("ability_stars")),
        influence_minus=safe_int(row.get("influence_minus")),
        influence_plus=safe_int(row.get("influence_plus")),
        abilities=abilities,
    )


def render_character_block(c: CharacterSnapshot) -> str:
    lines: List[str] = []
    lines.append(fmt_character_header(c.name))
    lines.append("")
    lines.append(f"Legacy Points: +{c.legacy_plus} / -{c.legacy_minus}  ·  Lifetime: +{c.lifetime_plus} / -{c.lifetime_minus}")
    lines.append(f"Ability Stars: {render_ability_stars(c.ability_stars)}")
    lines.append(f"Influence Stars: {render_influence_stars(c.influence_minus, c.influence_plus)}")
    top, bar = render_reputation_bar(c.net_lifetime)
    lines.append(top)
    lines.append(bar)
    lines.append(f"Abilities: {fmt_abilities(c.abilities)}")
    return "\n".join(lines)


async def render_dashboard_post(
    db: Database,
    guild: discord.Guild,
    user_id: int,
    user_fallback: str,
) -> Optional[str]:
    # Pull active characters for user; if none, no dashboard post
    chars = await db.get_characters_for_user(guild.id, user_id, include_archived=False)
    if not chars:
        return None

    # Resolve display name without pinging; we prefer fetch_member (REST) so it works without privileged intents.
    display_name = user_fallback
    try:
        m = guild.get_member(user_id)
        if m is None:
            m = await guild.fetch_member(user_id)
        if m:
            display_name = m.display_name
    except Exception:
        pass

    rank = await db.get_rank(guild.id, user_id)

    out: List[str] = []
    out.append(fmt_player_header(display_name, rank))

    snapshots: List[CharacterSnapshot] = []
    for r in chars:
        snapshots.append(await build_character_snapshot(db, guild.id, user_id, r))

    for i, c in enumerate(snapshots):
        out.append(render_character_block(c))
        if i != len(snapshots) - 1:
            out.append(SEP_LINE)

    # Protect from accidental pings
    return "\n".join(out)


# -----------------------------
# BOT
# -----------------------------
class VilyraBot(discord.Client):
    def __init__(self, db: Database):
        intents = discord.Intents.default()
        intents.guilds = True
        super().__init__(intents=intents)
        self.db = db
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        await self.db.connect()
        await self.db.init_schema()
        await self.tree.sync()
        LOG.info("Guild sync succeeded: %s commands", len(self.tree.get_commands()))

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, getattr(self.user, "id", "?"))
        # Auto-refresh dashboard once on startup
        guilds = list(self.guilds)
        if guilds:
            await refresh_all_safe(self, guilds[0], who="on_ready")


# -----------------------------
# COMMANDS
# -----------------------------
@app_commands.command(name="character_create", description="Create (or unarchive) a character for a user.")
@app_commands.describe(user="Owner of the character", name="Character name")
async def character_create(interaction: discord.Interaction, user: discord.Member, name: str):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.upsert_character(interaction.guild_id, user.id, name)
    await log_action(interaction.guild, f"✅ {interaction.user} created/unarchived **{name}** for **{user}**.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"character_create by {interaction.user.id}")
    await interaction.followup.send(f"Created/unarchived **{name}** for **{user.display_name}**.", ephemeral=True)


@app_commands.command(name="character_archive", description="Archive a character (removes from dashboard; can be restored later).")
@app_commands.describe(user="Owner of the character", name="Character name")
async def character_archive(interaction: discord.Interaction, user: discord.Member, name: str):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.set_archived(interaction.guild_id, user.id, name, True)
    await log_action(interaction.guild, f"📦 {interaction.user} archived **{name}** for **{user}**.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"character_archive by {interaction.user.id}")
    await interaction.followup.send(f"Archived **{name}** for **{user.display_name}**.", ephemeral=True)


@app_commands.command(name="character_delete", description="DELETE a character permanently (cannot be undone).")
@app_commands.describe(user="Owner of the character", name="Character name")
async def character_delete(interaction: discord.Interaction, user: discord.Member, name: str):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.delete_character(interaction.guild_id, user.id, name)
    await log_action(interaction.guild, f"🗑️ {interaction.user} DELETED **{name}** for **{user}**.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"character_delete by {interaction.user.id}")
    await interaction.followup.send(f"Deleted **{name}** for **{user.display_name}**.", ephemeral=True)


@app_commands.command(name="rank_set", description="Set a player's server rank label (stored; displayed on dashboard).")
@app_commands.describe(user="Player", rank="Rank label (e.g., Guardian, Warden, etc.)")
async def rank_set(interaction: discord.Interaction, user: discord.Member, rank: str):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.set_rank(interaction.guild_id, user.id, rank)
    await log_action(interaction.guild, f"🏷️ {interaction.user} set rank for **{user}** to **{rank}**.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"rank_set by {interaction.user.id}")
    await interaction.followup.send(f"Rank for **{user.display_name}** set to **{rank}**.", ephemeral=True)


@app_commands.command(name="legacy_add", description="Add legacy points (+) to a character.")
@app_commands.describe(user="Owner", character="Character name", amount="Amount to add (positive number)")
async def legacy_add(interaction: discord.Interaction, user: discord.Member, character: str, amount: int):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.add_legacy(interaction.guild_id, user.id, character, amount)
    await log_action(interaction.guild, f"➕ {interaction.user} added **{abs(int(amount))}** legacy to **{character}** (owner: {user}).")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"legacy_add by {interaction.user.id}")
    await interaction.followup.send(f"Added **{abs(int(amount))}** legacy to **{character}**.", ephemeral=True)


@app_commands.command(name="legacy_remove", description="Remove legacy points (-) from a character.")
@app_commands.describe(user="Owner", character="Character name", amount="Amount to remove (positive number)")
async def legacy_remove(interaction: discord.Interaction, user: discord.Member, character: str, amount: int):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.remove_legacy(interaction.guild_id, user.id, character, amount)
    await log_action(interaction.guild, f"➖ {interaction.user} removed **{abs(int(amount))}** legacy from **{character}** (owner: {user}).")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"legacy_remove by {interaction.user.id}")
    await interaction.followup.send(f"Removed **{abs(int(amount))}** legacy from **{character}**.", ephemeral=True)


@app_commands.command(name="ability_stars_set", description="Set a character's ability stars (0–5).")
@app_commands.describe(user="Owner", character="Character name", stars="0–5")
async def ability_stars_set(interaction: discord.Interaction, user: discord.Member, character: str, stars: int):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.set_ability_stars(interaction.guild_id, user.id, character, stars)
    await log_action(interaction.guild, f"⭐ {interaction.user} set ability stars for **{character}** (owner: {user}) to **{clamp(stars,0,5)}**.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"ability_stars_set by {interaction.user.id}")
    await interaction.followup.send(f"Ability stars for **{character}** set to **{clamp(stars,0,5)}**.", ephemeral=True)


@app_commands.command(name="influence_stars_set", description="Set a character's influence stars (minus and plus, each 0–5).")
@app_commands.describe(user="Owner", character="Character name", minus="0–5", plus="0–5")
async def influence_stars_set(interaction: discord.Interaction, user: discord.Member, character: str, minus: int, plus: int):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.set_influence_stars(interaction.guild_id, user.id, character, minus, plus)
    await log_action(interaction.guild, f"🌗 {interaction.user} set influence stars for **{character}** (owner: {user}) to **-{clamp(minus,0,5)} / +{clamp(plus,0,5)}**.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"influence_stars_set by {interaction.user.id}")
    await interaction.followup.send(f"Influence stars for **{character}** set to -{clamp(minus,0,5)} / +{clamp(plus,0,5)}.", ephemeral=True)


@app_commands.command(name="ability_custom_add", description="Add a custom ability to a character (starts at 0 upgrades).")
@app_commands.describe(user="Owner", character="Character name", ability="Ability name")
async def ability_custom_add(interaction: discord.Interaction, user: discord.Member, character: str, ability: str):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.ability_add(interaction.guild_id, user.id, character, ability)
    await log_action(interaction.guild, f"🧩 {interaction.user} added ability **{ability}** to **{character}** (owner: {user}).")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"ability_custom_add by {interaction.user.id}")
    await interaction.followup.send(f"Added ability **{ability}** to **{character}**.", ephemeral=True)


@app_commands.command(name="ability_custom_upgrade", description="Adjust upgrade count for a custom ability (can be negative; min 0).")
@app_commands.describe(user="Owner", character="Character name", ability="Ability name", delta="Change in upgrades (e.g., +1 or -1)")
async def ability_custom_upgrade(interaction: discord.Interaction, user: discord.Member, character: str, ability: str, delta: int):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await interaction.client.db.ability_upgrade(interaction.guild_id, user.id, character, ability, delta)
    await log_action(interaction.guild, f"🔧 {interaction.user} adjusted upgrades for **{ability}** on **{character}** (owner: {user}) by **{delta:+d}**.")
    await refresh_all_safe(interaction.client, interaction.guild, who=f"ability_custom_upgrade by {interaction.user.id}")
    await interaction.followup.send(f"Upgrades for **{ability}** on **{character}** adjusted by **{delta:+d}**.", ephemeral=True)


@app_commands.command(name="refresh_dashboard", description="Force-refresh the entire dashboard.")
async def refresh_dashboard(interaction: discord.Interaction):
    if not is_staff(interaction.user):
        return await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)

    await interaction.response.defer(ephemeral=True)
    await refresh_all_safe(interaction.client, interaction.guild, who=f"dashboard_refresh by {interaction.user.id}")
    await log_action(interaction.guild, f"🔄 {interaction.user} refreshed the dashboard.")
    await interaction.followup.send("Dashboard refreshed.", ephemeral=True)


@app_commands.command(name="card", description="View one of your character cards (ephemeral).")
@app_commands.describe(character="Character name")
async def card(interaction: discord.Interaction, character: str):
    await interaction.response.defer(ephemeral=True)
    rows = await interaction.client.db.get_characters_for_user(interaction.guild_id, interaction.user.id, include_archived=False)
    row = next((r for r in rows if str(r["name"]).lower() == character.lower()), None)
    if not row:
        return await interaction.followup.send("Character not found (or archived).", ephemeral=True)

    snap = await build_character_snapshot(interaction.client.db, interaction.guild_id, interaction.user.id, row)
    await interaction.followup.send(render_character_block(snap), ephemeral=True, allowed_mentions=discord.AllowedMentions.none())


# -----------------------------
# DASHBOARD REFRESH
# -----------------------------
async def refresh_dashboard_board(client: VilyraBot, guild: discord.Guild, who: str) -> None:
    ch = guild.get_channel(DASHBOARD_CHANNEL_ID) or await guild.fetch_channel(DASHBOARD_CHANNEL_ID)
    if not isinstance(ch, discord.TextChannel):
        raise RuntimeError("Dashboard channel ID is not a text channel.")

    # Group active characters by user_id
    rows = await client.db.get_all_active_characters(guild.id)
    by_user: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        uid = safe_int(r.get("user_id"))
        by_user.setdefault(uid, []).append(r)

    # We also want dashboard pages for users who have a rank set even if no characters? Spec didn’t require.
    # So we only render users with active characters.

    for user_id, char_rows in by_user.items():
        # We want a stable fallback string for display name if fetch fails
        fallback = f"User {user_id}"
        try:
            # Try to pull a safe name (not mention)
            m = guild.get_member(user_id)
            if m is None:
                m = await guild.fetch_member(user_id)
            if m:
                fallback = m.display_name
        except Exception:
            pass

        content = await render_dashboard_post(client.db, guild, user_id, fallback)
        if not content:
            continue

        # Ensure no accidental pings
        allowed_mentions = discord.AllowedMentions.none()

        existing_id = await client.db.get_dashboard_message_id(guild.id, user_id)
        if existing_id:
            try:
                msg = await ch.fetch_message(existing_id)
                await msg.edit(content=content, allowed_mentions=allowed_mentions)
            except discord.NotFound:
                existing_id = None
            except Exception:
                LOG.exception("Failed to edit dashboard post for user %s", user_id)

        if not existing_id:
            msg = await ch.send(content, allowed_mentions=allowed_mentions)
            await client.db.set_dashboard_message_id(guild.id, user_id, msg.id)

    await log_action(guild, f"✅ Dashboard updated ({who}).")


async def refresh_all_safe(client: VilyraBot, guild: discord.Guild, who: str) -> None:
    try:
        await refresh_dashboard_board(client, guild, who=who)
    except Exception as e:
        LOG.exception("refresh_all_safe failed")
        try:
            await log_action(guild, f"❌ Board update failed ({who}): {type(e).__name__}: {e}")
        except Exception:
            pass


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    token = env("DISCORD_TOKEN")
    dsn = env("DATABASE_URL")

    db = Database(dsn)
    client = VilyraBot(db)

    # Register commands on the instance tree (not on the class)
    client.tree.add_command(character_create)
    client.tree.add_command(character_archive)
    client.tree.add_command(character_delete)
    client.tree.add_command(rank_set)
    client.tree.add_command(legacy_add)
    client.tree.add_command(legacy_remove)
    client.tree.add_command(ability_stars_set)
    client.tree.add_command(influence_stars_set)
    client.tree.add_command(ability_custom_add)
    client.tree.add_command(ability_custom_upgrade)
    client.tree.add_command(refresh_dashboard)
    client.tree.add_command(card)

    client.run(token)


if __name__ == "__main__":
    main()
