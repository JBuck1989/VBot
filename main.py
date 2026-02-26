# VB_v104 — Vilyra Legacy Bot (Railway + Postgres) — FULL REPLACEMENT
# Fixes:
# - Correct class/method indentation (setup_hook/on_ready now actually run)
# - /staff_commands is registered + synced reliably
# - Guarded command cleanup via ALLOW_COMMAND_RESET (+ ALLOWED_GUILD_IDS)
# - Startup refresh is feature-flagged to avoid rate limits (default OFF)
# - Character autocomplete pulls directly from characters table (works with your schema)
# - No destructive DB ops (additive-only ALTER IF NOT EXISTS)

from __future__ import annotations

import os
import asyncio
import hashlib
import logging
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

# Reduce Discord 429s
DASHBOARD_EDIT_MIN_INTERVAL = float(os.getenv("DASHBOARD_EDIT_MIN_INTERVAL", "1.5"))
STARTUP_REFRESH_ENABLED = (os.getenv("FEATURE_STARTUP_REFRESH") or "").strip().lower() in ("1", "true", "yes", "y", "on")

# Dashboard rendering
DASHBOARD_TEMPLATE_VERSION = 4
PLAYER_POST_SOFT_LIMIT = 1900


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


def db_timeout() -> float:
    return float(os.getenv("DB_TIMEOUT_SECONDS", "10"))


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


async def safe_reply(interaction: discord.Interaction, content: str, *, ephemeral: bool = True) -> None:
    try:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=ephemeral)
        else:
            await interaction.response.send_message(content, ephemeral=ephemeral)
    except Exception:
        LOG.exception("Failed to reply")


async def defer_ephemeral(interaction: discord.Interaction) -> None:
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)
    except Exception:
        # If already acknowledged, ignore
        pass


async def send_error(interaction: discord.Interaction, error: Exception | str) -> None:
    msg = str(error)
    await safe_reply(interaction, f"❌ {msg}", ephemeral=True)


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


class SimpleRateLimiter:
    def __init__(self, min_interval: float):
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
# Database
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

        LOG.info(
            "Detected characters columns: %s",
            ", ".join(sorted(self.characters_cols)) if self.characters_cols else "(none)",
        )
        LOG.info(
            "Detected abilities columns: %s",
            ", ".join(sorted(self.abilities_cols)) if self.abilities_cols else "(none)",
        )

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
        # Additive-only safeguards
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS archived BOOLEAN NOT NULL DEFAULT FALSE;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS kingdom TEXT;")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
        await self._execute("ALTER TABLE characters ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")

        # players + dashboard_messages for this bot
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

        await self.detect_schema()
        LOG.info("Database schema initialized / updated")

    # ---- Characters / Autocomplete ----

    async def list_characters(self, guild_id: int, user_id: int, include_archived: bool = False) -> List[str]:
        where = "guild_id=%s AND user_id=%s"
        params: List[Any] = [guild_id, user_id]
        if not include_archived:
            where += " AND COALESCE(archived, FALSE)=FALSE"

        rows = await self._fetchall(
            f"""
            SELECT name
            FROM characters
            WHERE {where}
            ORDER BY created_at ASC, name ASC;
            """,
            tuple(params),
        )
        return [str(r["name"]) for r in rows if r and r.get("name")]

    async def list_all_characters_for_guild(
        self,
        guild_id: int,
        include_archived: bool = True,
        name_filter: str = "",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        name_filter = (name_filter or "").strip()
        lim = max(1, min(int(limit or 25), 50))

        where = ["guild_id=%s"]
        params: List[Any] = [guild_id]

        if not include_archived:
            where.append("COALESCE(archived, FALSE)=FALSE")

        if name_filter:
            where.append("name ILIKE %s")
            params.append(f"%{name_filter}%")

        sql = f"""
            SELECT user_id, name, COALESCE(archived, FALSE) AS archived
            FROM characters
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(archived, FALSE) ASC, name ASC
            LIMIT {lim};
        """
        return await self._fetchall(sql, tuple(params))

    async def list_player_ids(self, guild_id: int) -> List[int]:
        # prefer players, fallback to characters
        try:
            rows = await self._fetchall("SELECT user_id FROM players WHERE guild_id=%s ORDER BY user_id ASC;", (guild_id,))
            ids = [int(r["user_id"]) for r in rows if r and r.get("user_id") is not None]
            if ids:
                return ids
        except Exception:
            LOG.debug("players table missing/empty; falling back", exc_info=True)

        rows2 = await self._fetchall(
            "SELECT DISTINCT user_id FROM characters WHERE guild_id=%s ORDER BY user_id ASC;",
            (guild_id,),
        )
        return [int(r["user_id"]) for r in rows2 if r and r.get("user_id") is not None]

    async def get_player_rank(self, guild_id: int, user_id: int) -> str:
        row = await self._fetchone("SELECT server_rank FROM players WHERE guild_id=%s AND user_id=%s;", (guild_id, user_id))
        return str(row["server_rank"]) if row and row.get("server_rank") else "Newcomer"


# -----------------------------
# Autocomplete + Character resolver
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
        out: List[app_commands.Choice[str]] = []
        for r in rows:
            name = str(r.get("name") or "").strip()
            uid = int(r.get("user_id") or 0)
            if not name or uid <= 0:
                continue
            label = name
            if len(label) > 100:
                label = label[:97] + "..."
            # value carries disambiguator; names are unique in your server, but keep uid for safety
            out.append(app_commands.Choice(name=label, value=f"{uid}:{name}"))
        return out
    except Exception:
        LOG.exception("Character autocomplete failed")
        return []


async def resolve_character_input(interaction: discord.Interaction, token: str) -> Tuple[int, str]:
    t = (token or "").strip()
    guild = interaction.guild
    if guild is None:
        raise ValueError("This command must be used in a server.")

    # Preferred: 'user_id:Name'
    if ":" in t:
        left, right = t.split(":", 1)
        if left.strip().isdigit() and right.strip():
            return int(left.strip()), right.strip()

    # Fallback: plain name (names are unique in this server per user statement)
    if not t:
        raise ValueError("Character name is required.")

    rows = await interaction.client.db.list_all_characters_for_guild(
        guild.id,
        include_archived=True,
        name_filter=t,
        limit=25,
    )
    exact = [r for r in rows if str(r.get("name") or "").lower() == t.lower()]
    if len(exact) == 1:
        return int(exact[0]["user_id"]), str(exact[0]["name"])
    raise ValueError("Please select the character from the autocomplete list.")


# -----------------------------
# Guards
# -----------------------------

def in_guild_only():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await safe_reply(interaction, "This command can only be used in a server.")
            return False
        return True
    return app_commands.check(predicate)


def staff_only():
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

        # allowlist
        if interaction.user.id in staff_user_ids:
            return True

        # owner / perms
        if interaction.guild.owner_id == interaction.user.id:
            return True

        member = interaction.user
        if not isinstance(member, discord.Member):
            member = interaction.guild.get_member(interaction.user.id)  # type: ignore

        if isinstance(member, discord.Member):
            perms = member.guild_permissions
            if perms.administrator or perms.manage_guild:
                return True

        await safe_reply(interaction, "You don't have permission to use this command.")
        return False

    return app_commands.check(predicate)


# -----------------------------
# Dashboard (lightweight)
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


async def render_player_post(db: Database, guild: discord.Guild, user_id: int) -> str:
    member = guild.get_member(user_id)
    nickname = member.display_name if member else f"User {user_id}"
    rank = await db.get_player_rank(guild.id, user_id)
    chars = await db.list_characters(guild.id, user_id, include_archived=False)
    if not chars:
        return ""

    lines: List[str] = ["═" * 20, f"__***{nickname}***__", f"__***Server Rank: {rank}***__", ""]
    for cname in chars:
        lines.append(f"• **{cname}**")
    lines.append("═" * 20)

    content = "\n".join(lines).rstrip()
    return content[:PLAYER_POST_SOFT_LIMIT]


async def refresh_player_dashboard(client: "VilyraBotClient", guild: discord.Guild, user_id: int) -> str:
    db = client.db
    channel = await get_dashboard_channel(guild)
    if not channel:
        return "Dashboard channel not found."

    content = await render_player_post(db, guild, user_id)
    if not content:
        return "skipped"

    new_hash = content_hash(content)
    # dashboard_messages table is present but we keep updates minimal
    row = await db._fetchone(
        "SELECT message_ids, content_hash, COALESCE(template_version,0) AS tv FROM dashboard_messages WHERE guild_id=%s AND user_id=%s;",
        (guild.id, user_id),
    )
    stored_ids = parse_ids(row["message_ids"]) if row and row.get("message_ids") else []
    stored_hash = str(row["content_hash"]) if row and row.get("content_hash") else None
    stored_tv = int(row["tv"]) if row else 0

    if stored_hash == new_hash and stored_tv == DASHBOARD_TEMPLATE_VERSION:
        return "skipped"

    msg: Optional[discord.Message] = None
    if stored_ids:
        try:
            msg = await channel.fetch_message(stored_ids[0])
        except Exception:
            msg = None

    await client.dashboard_limiter.wait()
    if msg is None:
        msg = await channel.send(content)
        await db._execute(
            """
            INSERT INTO dashboard_messages (guild_id,user_id,channel_id,message_ids,content_hash,template_version,updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,NOW())
            ON CONFLICT (guild_id,user_id)
            DO UPDATE SET channel_id=EXCLUDED.channel_id, message_ids=EXCLUDED.message_ids,
                          content_hash=EXCLUDED.content_hash, template_version=EXCLUDED.template_version, updated_at=NOW();
            """,
            (guild.id, user_id, channel.id, fmt_ids([msg.id]), new_hash, DASHBOARD_TEMPLATE_VERSION),
        )
        return "created"

    await msg.edit(content=content)
    await db._execute(
        "UPDATE dashboard_messages SET channel_id=%s, message_ids=%s, content_hash=%s, template_version=%s, updated_at=NOW() WHERE guild_id=%s AND user_id=%s;",
        (channel.id, fmt_ids([msg.id]), new_hash, DASHBOARD_TEMPLATE_VERSION, guild.id, user_id),
    )
    return "updated"


async def refresh_all_dashboards(client: "VilyraBotClient", guild: discord.Guild) -> str:
    user_ids = await client.db.list_player_ids(guild.id)
    if not user_ids:
        return "No players with characters yet."

    ok = 0
    for uid in user_ids:
        try:
            await refresh_player_dashboard(client, guild, uid)
        except Exception:
            LOG.exception("Dashboard refresh failed for user_id=%s", uid)
        ok += 1
        await asyncio.sleep(0.25)
    return f"Refreshed dashboards for {ok} player(s)."


# -----------------------------
# Commands
# -----------------------------

@app_commands.command(name="staff_commands", description="(Staff) Show the staff command list.")
@in_guild_only()
@staff_only()
async def staff_commands_cmd(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    # Static, clear list (avoids drift if Discord caching delays updates)
    items: List[Tuple[str, str]] = [
        ("/staff_commands", "Show this list."),
        ("/refresh_dashboard", "Force-refresh all dashboard posts."),
        ("/debug_characters", "Show character counts for this guild."),
        # Add other commands you keep in this bot here.
    ]
    lines = ["**Staff Commands**", ""]
    for cmd, desc in items:
        lines.append(f"• **{cmd}** — {desc}")
    await safe_reply(interaction, "\n".join(lines), ephemeral=True)


@app_commands.command(name="refresh_dashboard", description="(Staff) Force refresh all dashboards now.")
@in_guild_only()
@staff_only()
async def refresh_dashboard_cmd(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        status = await refresh_all_dashboards(interaction.client, interaction.guild)
        await log_to_channel(interaction.guild, f"🔄 {interaction.user.mention} refreshed the dashboard")
        await safe_reply(interaction, status, ephemeral=True)
    except Exception as e:
        LOG.exception("refresh_dashboard failed")
        await send_error(interaction, e)


@app_commands.command(name="debug_characters", description="(Staff) Debug: show character counts for this guild.")
@in_guild_only()
@staff_only()
async def debug_characters_cmd(interaction: discord.Interaction):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        rows = await interaction.client.db._fetchall(
            """
            SELECT COALESCE(archived, FALSE) AS archived, COUNT(*) AS n
            FROM characters
            WHERE guild_id=%s
            GROUP BY COALESCE(archived, FALSE)
            ORDER BY COALESCE(archived, FALSE);
            """,
            (interaction.guild.id,),
        )
        total = 0
        parts: List[str] = []
        for r in rows:
            n = int(r["n"])
            total += n
            parts.append(f"archived={bool(r['archived'])}: {n}")
        msg = " | ".join(parts) if parts else "no rows"
        await safe_reply(interaction, f"Guild {interaction.guild.id}: characters total={total} ({msg})", ephemeral=True)
    except Exception as e:
        LOG.exception("debug_characters failed")
        await send_error(interaction, e)


@app_commands.command(name="char_lookup", description="Lookup a character (autocomplete proof-of-life).")
@in_guild_only()
@app_commands.autocomplete(character_name=autocomplete_character_guild)
async def char_lookup_cmd(interaction: discord.Interaction, character_name: str):
    await defer_ephemeral(interaction)
    try:
        assert interaction.guild is not None
        uid, name = await resolve_character_input(interaction, character_name)
        await safe_reply(interaction, f"✅ Found **{name}** (owner user_id={uid}).", ephemeral=True)
    except Exception as e:
        await send_error(interaction, e)


# -----------------------------
# Discord Client
# -----------------------------

class VilyraBotClient(discord.Client):
    def __init__(self, db: Database):
        intents = discord.Intents.default()
        intents.members = True
        super().__init__(intents=intents)

        self.db = db
        self.tree = app_commands.CommandTree(self)
        self.dashboard_limiter = SimpleRateLimiter(DASHBOARD_EDIT_MIN_INTERVAL)
        self._synced = False

    async def setup_hook(self) -> None:
        # Register commands (single source of truth)
        self.tree.add_command(staff_commands_cmd)
        self.tree.add_command(refresh_dashboard_cmd)
        self.tree.add_command(debug_characters_cmd)
        self.tree.add_command(char_lookup_cmd)

        names = [c.name for c in self.tree.get_commands()]
        LOG.info("Command tree prepared: %s command(s); GUILD_ID=%s", len(names), safe_int(os.getenv("GUILD_ID"), 0))

        await self._sync_commands_guarded()

    async def _sync_commands_guarded(self) -> None:
        # Clean up duplicates by syncing only to the configured guild.
        # If you want to delete old stale commands in Discord, set:
        #   ALLOW_COMMAND_RESET=true
        #   ALLOWED_GUILD_IDS=<your guild id>
        try:
            gid = safe_int(os.getenv("GUILD_ID"), 0)
            allow_reset = (os.getenv("ALLOW_COMMAND_RESET") or "").strip().lower() in ("1", "true", "yes", "y", "on")
            raw_allow = (os.getenv("ALLOWED_GUILD_IDS") or "").strip()
            allowed: Optional[set[int]] = None
            if raw_allow:
                try:
                    allowed = {int(x.strip()) for x in raw_allow.split(",") if x.strip()}
                except Exception:
                    allowed = None

            if gid:
                if allowed and gid not in allowed:
                    LOG.error("GUILD_ID %s not in ALLOWED_GUILD_IDS; skipping guild sync/reset.", gid)
                    return

                guild_obj = discord.Object(id=gid)
                self.tree.copy_global_to(guild=guild_obj)
                if allow_reset and getattr(self, "application_id", None):
                    # Delete ALL guild commands then re-sync
                    await self.http.bulk_upsert_guild_commands(self.application_id, gid, [])
                    LOG.warning("Performed hard guild command reset (ALLOW_COMMAND_RESET=true) for guild %s", gid)

                synced = await self.tree.sync(guild=guild_obj)
                LOG.info("Guild command sync complete: %s commands (hard_reset=%s)", len(synced), allow_reset)
                self._synced = True
                return

            # No GUILD_ID: sync globally (not recommended while cleaning duplicates)
            synced = await self.tree.sync()
            LOG.info("Global command sync complete: %s commands", len(synced))
            self._synced = True

        except Exception:
            LOG.exception("Command sync/reset failed")

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "unknown")
        # Only do startup refresh if explicitly enabled (prevents rate-limit storms)
        if STARTUP_REFRESH_ENABLED:
            LOG.info("Startup dashboard refresh enabled: beginning for %d guild(s)...", len(list(self.guilds)))
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
