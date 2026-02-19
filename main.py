import os
import json
import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import discord
from discord import app_commands

import asyncpg

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


# -------------------------
# Version / Config
# -------------------------
VERSION = "EconBot_v30"

TZ_NAME = "America/Chicago"
TZ = ZoneInfo(TZ_NAME) if ZoneInfo else dt.timezone.utc

BASE_DAILY_INCOME_VAL = 10  # 10 Cinths == 10 Val == 1 Arce

# Denominations (in Val)
DENOMS: List[Tuple[str, int]] = [
    ("NOVIR", 10_000),
    ("ORIN", 1_000),
    ("ELSH", 100),
    ("ARCE", 10),
    ("CINTH", 1),
]


def _get(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def _req_token() -> str:
    v = _get("DISCORD_TOKEN")
    if not v:
        raise RuntimeError("Missing required env var: DISCORD_TOKEN")
    return v


def _parse_int(v: str | None) -> int | None:
    try:
        return int(v) if v is not None and str(v).strip() != "" else None
    except Exception:
        return None


def _parse_int_list(v: str | None) -> list[int]:
    if not v:
        return []
    out: list[int] = []
    for part in v.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def _parse_int_set(v: str | None) -> set[int]:
    return set(_parse_int_list(v))


DISCORD_TOKEN = _req_token()
DATABASE_URL = _get("DATABASE_URL")

GUILD_ID = _parse_int(_get("GUILD_ID"))
LEGACY_SOURCE_GUILD_ID = _parse_int(_get("LEGACY_SOURCE_GUILD_ID")) or GUILD_ID

BANK_CHANNEL_ID = _parse_int(_get("BANK_CHANNEL_ID"))
ECON_LOG_CHANNEL_ID = _parse_int(_get("ECON_LOG_CHANNEL_ID"))

BANK_MESSAGE_IDS = _parse_int_list(_get("BANK_MESSAGE_IDS"))

STAFF_ROLE_IDS = _parse_int_set(
    _get("STAFF_ROLE_IDS")
    or _get("Staff_Role_IDs")
    or _get("STAFF_ROLE_IDs")
    or _get("STAFF_ROLE_IDS")
)


# -------------------------
# Asset Catalog (from uploaded Asset Table.xlsx)
# Keys: (asset_type, secondary_type, tier_name)
# Costs are absolute (exact to spreadsheet) in Val.
# -------------------------
ASSET_CATALOG: List[Dict[str, object]] = [
    {"asset_type": 'Business', "secondary_type": 'Guild Trade Workshop', "tier": 1, "tier_name": 'Apprentice', "cost_val": 300, "income_val": 50},
    {"asset_type": 'Business', "secondary_type": 'Guild Trade Workshop', "tier": 2, "tier_name": 'Journeyman', "cost_val": 600, "income_val": 100},
    {"asset_type": 'Business', "secondary_type": 'Guild Trade Workshop', "tier": 3, "tier_name": 'Leased Workshop', "cost_val": 1200, "income_val": 150},
    {"asset_type": 'Business', "secondary_type": 'Guild Trade Workshop', "tier": 4, "tier_name": 'Small Workshop', "cost_val": 2000, "income_val": 200},
    {"asset_type": 'Business', "secondary_type": 'Guild Trade Workshop', "tier": 5, "tier_name": 'Large Workshop', "cost_val": 3000, "income_val": 250},
    {"asset_type": 'Business', "secondary_type": 'Market Stall', "tier": 1, "tier_name": 'Consignment Arrangement', "cost_val": 300, "income_val": 50},
    {"asset_type": 'Business', "secondary_type": 'Market Stall', "tier": 2, "tier_name": 'Small Alley Stand', "cost_val": 600, "income_val": 100},
    {"asset_type": 'Business', "secondary_type": 'Market Stall', "tier": 3, "tier_name": 'Market Stall', "cost_val": 1200, "income_val": 150},
    {"asset_type": 'Business', "secondary_type": 'Market Stall', "tier": 4, "tier_name": 'Small Shop', "cost_val": 2000, "income_val": 200},
    {"asset_type": 'Business', "secondary_type": 'Market Stall', "tier": 5, "tier_name": 'Large Shop', "cost_val": 3000, "income_val": 250},
    {"asset_type": 'Business', "secondary_type": 'Farm/Ranch', "tier": 1, "tier_name": 'Subsistence Surplus', "cost_val": 300, "income_val": 50},
    {"asset_type": 'Business', "secondary_type": 'Farm/Ranch', "tier": 2, "tier_name": 'Leased Fields', "cost_val": 600, "income_val": 100},
    {"asset_type": 'Business', "secondary_type": 'Farm/Ranch', "tier": 3, "tier_name": 'Owned Acre', "cost_val": 1200, "income_val": 150},
    {"asset_type": 'Business', "secondary_type": 'Farm/Ranch', "tier": 4, "tier_name": 'Small Fields and Barn', "cost_val": 2000, "income_val": 200},
    {"asset_type": 'Business', "secondary_type": 'Farm/Ranch', "tier": 5, "tier_name": 'Large Fields and Barn', "cost_val": 3000, "income_val": 250},
    {"asset_type": 'Business', "secondary_type": 'Tavern/Inn', "tier": 1, "tier_name": 'One-Room Flophouse', "cost_val": 300, "income_val": 50},
    {"asset_type": 'Business', "secondary_type": 'Tavern/Inn', "tier": 2, "tier_name": 'Leased Establishment', "cost_val": 600, "income_val": 100},
    {"asset_type": 'Business', "secondary_type": 'Tavern/Inn', "tier": 3, "tier_name": 'Small Tavern', "cost_val": 1200, "income_val": 150},
    {"asset_type": 'Business', "secondary_type": 'Tavern/Inn', "tier": 4, "tier_name": 'Large Tavern', "cost_val": 2000, "income_val": 200},
    {"asset_type": 'Business', "secondary_type": 'Tavern/Inn', "tier": 5, "tier_name": 'Large Tavern and Inn', "cost_val": 3000, "income_val": 250},
    {"asset_type": 'Business', "secondary_type": 'Warehouse/Trade House', "tier": 1, "tier_name": 'Small Storage Shed', "cost_val": 300, "income_val": 50},
    {"asset_type": 'Business', "secondary_type": 'Warehouse/Trade House', "tier": 2, "tier_name": 'Large Storage Shed', "cost_val": 600, "income_val": 100},
    {"asset_type": 'Business', "secondary_type": 'Warehouse/Trade House', "tier": 3, "tier_name": 'Small Trading Post', "cost_val": 1200, "income_val": 150},
    {"asset_type": 'Business', "secondary_type": 'Warehouse/Trade House', "tier": 4, "tier_name": 'Large Trading Post', "cost_val": 2000, "income_val": 200},
    {"asset_type": 'Business', "secondary_type": 'Warehouse/Trade House', "tier": 5, "tier_name": 'Large Warehouse and Trading Post', "cost_val": 3000, "income_val": 250},
    {"asset_type": 'Holdings', "secondary_type": 'House', "tier": 1, "tier_name": 'Shack', "cost_val": 600, "income_val": 0},
    {"asset_type": 'Holdings', "secondary_type": 'House', "tier": 2, "tier_name": 'Hut', "cost_val": 1200, "income_val": 0},
    {"asset_type": 'Holdings', "secondary_type": 'House', "tier": 3, "tier_name": 'House', "cost_val": 2000, "income_val": 0},
    {"asset_type": 'Holdings', "secondary_type": 'House', "tier": 4, "tier_name": 'Lodge', "cost_val": 3000, "income_val": 0},
    {"asset_type": 'Holdings', "secondary_type": 'House', "tier": 5, "tier_name": 'Mansion', "cost_val": 5000, "income_val": 0},
    {"asset_type": 'Holdings', "secondary_type": 'Village', "tier": 1, "tier_name": 'Chartered Assembly', "cost_val": 1200, "income_val": 100},
    {"asset_type": 'Holdings', "secondary_type": 'Village', "tier": 2, "tier_name": 'Hamlet', "cost_val": 2400, "income_val": 200},
    {"asset_type": 'Holdings', "secondary_type": 'Village', "tier": 3, "tier_name": 'Village', "cost_val": 4800, "income_val": 300},
    {"asset_type": 'Holdings', "secondary_type": 'Village', "tier": 4, "tier_name": 'Town', "cost_val": 9600, "income_val": 400},
    {"asset_type": 'Holdings', "secondary_type": 'Village', "tier": 5, "tier_name": 'Small City', "cost_val": 15000, "income_val": 500},
    {"asset_type": 'Enchantments', "secondary_type": 'Weapons', "tier": 1, "tier_name": 'Hit +1 / Dmg +1d4', "cost_val": 300, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Weapons', "tier": 2, "tier_name": 'Hit +1 / Dmg +1d6', "cost_val": 600, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Weapons', "tier": 3, "tier_name": 'Hit +2 / Dmg +1d8', "cost_val": 1200, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Weapons', "tier": 4, "tier_name": 'Hit +2 / Dmg +1d10', "cost_val": 2400, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Weapons', "tier": 5, "tier_name": 'Hit +2 / Dmg +1d12', "cost_val": 4800, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Armor', "tier": 1, "tier_name": 'AC +1', "cost_val": 300, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Armor', "tier": 2, "tier_name": 'AC +2', "cost_val": 600, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Armor', "tier": 3, "tier_name": 'AC +2 / Adv Magic Atk', "cost_val": 1200, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Armor', "tier": 4, "tier_name": 'AC +2 / Adv Magic and Melee Atk', "cost_val": 2400, "income_val": 0},
    {"asset_type": 'Enchantments', "secondary_type": 'Armor', "tier": 5, "tier_name": 'AC +3 / Adv Magic and Melee Atk', "cost_val": 4800, "income_val": 0},
]


def _catalog_key(a: Dict[str, object]) -> Tuple[str, str, str]:
    return (str(a["asset_type"]), str(a["secondary_type"]), str(a["tier_name"]))


CATALOG_BY_KEY: Dict[Tuple[str, str, str], Dict[str, object]] = { _catalog_key(a): a for a in ASSET_CATALOG }


# -------------------------
# DB
# -------------------------
_pool: asyncpg.Pool | None = None


async def db_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set; database features are disabled.")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    return _pool


async def init_db() -> None:
    if not DATABASE_URL:
        print(f"[{VERSION}] DATABASE_URL missing; running without DB.")
        return

    pool = await db_pool()
    async with pool.acquire() as con:
        await con.execute("CREATE SCHEMA IF NOT EXISTS economy")

        # Balances (stored in Val)
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS economy.balances (
                guild_id BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                balance_val BIGINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, character_name)
            )
            """
        )

        # Income claims (daily, per character)
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS economy.income_claims (
                guild_id BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                claim_date DATE NOT NULL,
                claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, character_name, claim_date)
            )
            """
        )

        # Owned assets (named by player; upgrades replace tier fields but keep asset_name)
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS economy.assets_owned (
                guild_id BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                character_user_id BIGINT NOT NULL,
                asset_name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                secondary_type TEXT NOT NULL,
                tier_name TEXT NOT NULL,
                tier INTEGER NOT NULL,
                cost_val BIGINT NOT NULL,
                income_val BIGINT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, character_name, character_user_id, asset_name)
            )
            """
        )

        # Action log
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS economy.ledger (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                guild_id BIGINT NOT NULL,
                character_name TEXT NOT NULL,
                character_user_id BIGINT NOT NULL,
                actor_user_id BIGINT NOT NULL,
                kind TEXT NOT NULL,
                delta_val BIGINT NOT NULL,
                reason TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )


# -------------------------
# Legacy characters table helpers
# -------------------------
@dataclass
class Character:
    guild_id: int
    user_id: int
    name: str
    archived: bool


async def fetch_character_by_name(legacy_guild_id: int, name: str) -> Character | None:
    """Legacy source: public.characters. Character ID is name (unique per guild)."""
    if not DATABASE_URL:
        return None
    pool = await db_pool()
    async with pool.acquire() as con:
        row = await con.fetchrow(
            """
            SELECT guild_id, user_id, name, archived
            FROM characters
            WHERE guild_id=$1 AND name=$2
            LIMIT 1
            """,
            int(legacy_guild_id),
            str(name),
        )
        if not row:
            return None
        return Character(
            guild_id=int(row["guild_id"]),
            user_id=int(row["user_id"]),
            name=str(row["name"]),
            archived=bool(row["archived"]),
        )


async def search_characters(legacy_guild_id: int, query: str, limit: int = 25) -> List[str]:
    if not DATABASE_URL:
        return []
    q = (query or "").strip()
    if not q:
        q = ""
    pool = await db_pool()
    async with pool.acquire() as con:
        rows = await con.fetch(
            """
            SELECT name
            FROM characters
            WHERE guild_id=$1
              AND archived=FALSE
              AND name ILIKE $2
            ORDER BY name ASC
            LIMIT $3
            """,
            int(legacy_guild_id),
            f"%{q}%",
            int(limit),
        )
        return [str(r["name"]) for r in rows]


# -------------------------
# Economy helpers
# -------------------------
async def get_balance_val(guild_id: int, character_name: str) -> int:
    if not DATABASE_URL:
        return 0
    pool = await db_pool()
    async with pool.acquire() as con:
        row = await con.fetchrow(
            "SELECT balance_val FROM economy.balances WHERE guild_id=$1 AND character_name=$2",
            int(guild_id),
            str(character_name),
        )
        return int(row["balance_val"]) if row else 0


async def set_balance_val(guild_id: int, character_name: str, new_val: int) -> None:
    if not DATABASE_URL:
        return
    pool = await db_pool()
    async with pool.acquire() as con:
        await con.execute(
            """
            INSERT INTO economy.balances (guild_id, character_name, balance_val, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (guild_id, character_name)
            DO UPDATE SET balance_val=EXCLUDED.balance_val, updated_at=NOW()
            """,
            int(guild_id),
            str(character_name),
            int(new_val),
        )


async def add_balance_val(guild_id: int, character_name: str, delta: int) -> int:
    """Adds delta, never below 0. Returns updated balance."""
    if not DATABASE_URL:
        return 0
    pool = await db_pool()
    async with pool.acquire() as con:
        async with con.transaction():
            cur = await con.fetchrow(
                "SELECT balance_val FROM economy.balances WHERE guild_id=$1 AND character_name=$2 FOR UPDATE",
                int(guild_id),
                str(character_name),
            )
            cur_val = int(cur["balance_val"]) if cur else 0
            new_val = cur_val + int(delta)
            if new_val < 0:
                new_val = 0
            await con.execute(
                """
                INSERT INTO economy.balances (guild_id, character_name, balance_val, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (guild_id, character_name)
                DO UPDATE SET balance_val=EXCLUDED.balance_val, updated_at=NOW()
                """,
                int(guild_id),
                str(character_name),
                int(new_val),
            )
            return int(new_val)


async def sum_asset_income_val(guild_id: int, character_name: str) -> int:
    if not DATABASE_URL:
        return 0
    pool = await db_pool()
    async with pool.acquire() as con:
        row = await con.fetchrow(
            """
            SELECT COALESCE(SUM(income_val), 0) AS s
            FROM economy.assets_owned
            WHERE guild_id=$1 AND character_name=$2
            """,
            int(guild_id),
            str(character_name),
        )
        return int(row["s"]) if row else 0


async def fetch_assets_owned(guild_id: int, character_name: str) -> List[Dict[str, object]]:
    if not DATABASE_URL:
        return []
    pool = await db_pool()
    async with pool.acquire() as con:
        rows = await con.fetch(
            """
            SELECT asset_name, asset_type, secondary_type, tier_name, tier, cost_val, income_val, character_user_id
            FROM economy.assets_owned
            WHERE guild_id=$1 AND character_name=$2
            ORDER BY asset_type, secondary_type, tier DESC, asset_name
            """,
            int(guild_id),
            str(character_name),
        )
        return [dict(r) for r in rows]


async def ledger_write(
    guild_id: int,
    character_name: str,
    character_user_id: int,
    actor_user_id: int,
    kind: str,
    delta_val: int,
    reason: str,
    details: Dict[str, object] | None = None,
) -> None:
    if not DATABASE_URL:
        return
    pool = await db_pool()
    async with pool.acquire() as con:
        await con.execute(
            """
            INSERT INTO economy.ledger (guild_id, character_name, character_user_id, actor_user_id, kind, delta_val, reason, details_json)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            int(guild_id),
            str(character_name),
            int(character_user_id),
            int(actor_user_id),
            str(kind),
            int(delta_val),
            str(reason),
            json.dumps(details or {}, ensure_ascii=False),
        )


# -------------------------
# Formatting
# -------------------------
UNIT_LABELS = {
    "CINTH": "Cinth",
    "ARCE": "Arce",
    "ELSH": "Elsh",
    "ORIN": "Orin",
    "NOVIR": "Novir",
}


def breakdown_denoms(val: int) -> List[Tuple[str, int]]:
    remaining = int(max(0, val))
    parts: List[Tuple[str, int]] = []
    for unit, unit_val in DENOMS:
        if remaining <= 0:
            break
        count = remaining // unit_val
        if count:
            parts.append((unit, int(count)))
            remaining -= count * unit_val
    if not parts:
        parts = [("CINTH", 0)]
    return parts


def format_money(val: int) -> str:
    parts = breakdown_denoms(val)
    out = []
    for unit, count in parts:
        label = UNIT_LABELS.get(unit, unit.title())
        plural = "" if count == 1 else "s"
        out.append(f"{count} {label}{plural}")
    return ", ".join(out)


def format_assets_lines(assets: List[Dict[str, object]]) -> str:
    if not assets:
        return "*(None)*"
    lines = []
    for a in assets:
        asset_name = str(a.get("asset_name", "")).strip()
        tier_name = str(a.get("tier_name", "")).strip()
        lines.append(f"{asset_name} — {tier_name}")
    return "\n".join(lines)


# -------------------------
# Discord setup
# -------------------------
intents = discord.Intents.none()
intents.guilds = True

client = discord.Client(intents=intents, allowed_mentions=discord.AllowedMentions.none())
tree = app_commands.CommandTree(client)


@tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    msg: str
    if isinstance(error, app_commands.CheckFailure):
        msg = str(error) or "You don’t have permission to use that command."
    elif isinstance(error, app_commands.CommandInvokeError) and getattr(error, "original", None) is not None:
        orig = error.original
        msg = f"Command error: {type(orig).__name__}: {orig}"
    else:
        msg = f"Command error: {type(error).__name__}: {error}"

    try:
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
        else:
            await interaction.response.send_message(msg, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
    except Exception:
        pass


def require_admin():
    """Staff check using STAFF_ROLE_IDS; if not configured, fall back to admin perms."""

    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            raise app_commands.CheckFailure("This command can only be used in a server.")

        member = interaction.user if isinstance(interaction.user, discord.Member) else None
        if member is None:
            member = interaction.guild.get_member(interaction.user.id)

        if member is None:
            raise app_commands.CheckFailure("I couldn't verify your roles/permissions. Try again in a moment.")

        if STAFF_ROLE_IDS:
            if any(r.id in STAFF_ROLE_IDS for r in getattr(member, "roles", [])):
                return True
            raise app_commands.CheckFailure("You don’t have permission to use that command.")

        perms = getattr(member, "guild_permissions", None)
        if perms and (perms.administrator or perms.manage_guild):
            return True

        raise app_commands.CheckFailure(
            "Staff roles are not configured (STAFF_ROLE_IDS). Ask an admin to set STAFF_ROLE_IDS in Railway Variables."
        )

    return app_commands.check(predicate)


# -------------------------
# Autocomplete
# -------------------------
async def character_name_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    if not LEGACY_SOURCE_GUILD_ID:
        return []
    names = await search_characters(int(LEGACY_SOURCE_GUILD_ID), current or "", limit=25)
    return [app_commands.Choice(name=n, value=n) for n in names[:25]]


async def owned_asset_name_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    # expects "character" to be in namespace
    char = str(getattr(interaction.namespace, "character", "") or "").strip()
    if not char or not LEGACY_SOURCE_GUILD_ID:
        return []
    assets = await fetch_assets_owned(int(LEGACY_SOURCE_GUILD_ID), char)
    cur = (current or "").lower().strip()
    names = []
    for a in assets:
        nm = str(a.get("asset_name", "")).strip()
        if not nm:
            continue
        if not cur or cur in nm.lower():
            names.append(nm)
    names = sorted(set(names))[:25]
    return [app_commands.Choice(name=n, value=n) for n in names]


# -------------------------
# Commands
# -------------------------
COMMAND_HELP_LINES = [
    "Player Commands",
    "/balance — Show a character’s current money and owned assets. Anyone can use this; it matches the public bank dashboard.",
    "/income — Claim daily income for one of YOUR characters (once per day, Chicago time). Adds base income plus income from owned assets.",
    "",
    "Staff Commands",
    "/econ_adjust — Staff-only. Add or subtract money from a character. The bot will not allow balances to go negative.",
    "/econ_set_balance — Staff-only. Set a character’s balance to an exact amount (non-negative). Useful for corrections.",
    "/econ_purchase_new — Staff-only. Buy a new asset at a chosen tier (cost is tier 1 + ... + target tier, per NEW Asset Table).",
    "/econ_refresh_bank — Staff-only. Force-refresh the Bank of Vilyra dashboard.",
    "/econ_commands — Staff-only. Shows this command list with short descriptions (kept updated as we add features).",
]



@tree.command(name="econ_commands", description="Staff: show EconBot command list and what each command does.")
@require_admin()
async def econ_commands_cmd(interaction: discord.Interaction):
    await interaction.response.defer(thinking=False, ephemeral=True)

    em = discord.Embed(title="EconBot Commands", description="Quick reference (kept updated as we build).")
    public = []
    staff = []
    for name, desc in COMMAND_HELP_LINES:
        line = f"**/{name}** — {desc}"
        if name in {"balance", "income"}:
            public.append(line)
        else:
            staff.append(line)

    if public:
        em.add_field(name="Player Commands", value="\n".join(public), inline=False)
    if staff:
        em.add_field(name="Staff Commands", value="\n".join(staff), inline=False)

    em.set_footer(text=VERSION)
    await interaction.followup.send(embed=em, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())


@tree.command(name="balance", description="Show a character’s money + assets.")
@app_commands.describe(character="Pick a character")
@app_commands.autocomplete(character=character_name_autocomplete)
async def balance_cmd(interaction: discord.Interaction, character: str):
    await interaction.response.defer(thinking=False, ephemeral=True)

    if not LEGACY_SOURCE_GUILD_ID:
        await interaction.followup.send("LEGACY_SOURCE_GUILD_ID is not set.", ephemeral=True)
        return

    ch = await fetch_character_by_name(int(LEGACY_SOURCE_GUILD_ID), character)
    if not ch or ch.archived:
        await interaction.followup.send("Character not found.", ephemeral=True)
        return

    bal = await get_balance_val(ch.guild_id, ch.name)
    assets = await fetch_assets_owned(ch.guild_id, ch.name)

    em = discord.Embed(title=f"{ch.name}", description=f"**Balance:** {format_money(bal)}")
    em.add_field(name="__*Assets*__", value=format_assets_lines(assets), inline=False)
    em.set_footer(text=VERSION)
    await interaction.followup.send(embed=em, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())


@tree.command(name="income", description="Owner-only: claim daily income for your character (Chicago time).")
@app_commands.describe(character="Pick one of your characters")
@app_commands.autocomplete(character=character_name_autocomplete)
async def income_cmd(interaction: discord.Interaction, character: str):
    await interaction.response.defer(thinking=False, ephemeral=True)

    if not (DATABASE_URL and LEGACY_SOURCE_GUILD_ID):
        await interaction.followup.send("Database is not configured.", ephemeral=True)
        return

    ch = await fetch_character_by_name(int(LEGACY_SOURCE_GUILD_ID), character)
    if not ch or ch.archived:
        await interaction.followup.send("Character not found.", ephemeral=True)
        return

    if interaction.user.id != ch.user_id:
        await interaction.followup.send("You can only claim income for characters you own.", ephemeral=True)
        return

    today = dt.datetime.now(TZ).date()

    pool = await db_pool()
    async with pool.acquire() as con:
        async with con.transaction():
            exists = await con.fetchrow(
                "SELECT 1 FROM economy.income_claims WHERE guild_id=$1 AND character_name=$2 AND claim_date=$3",
                int(ch.guild_id),
                str(ch.name),
                today,
            )
            if exists:
                await interaction.followup.send("Income already claimed for today.", ephemeral=True)
                return

            asset_income = await sum_asset_income_val(ch.guild_id, ch.name)
            total_income = int(BASE_DAILY_INCOME_VAL + asset_income)

            # Apply
            cur = await con.fetchrow(
                "SELECT balance_val FROM economy.balances WHERE guild_id=$1 AND character_name=$2 FOR UPDATE",
                int(ch.guild_id),
                str(ch.name),
            )
            cur_val = int(cur["balance_val"]) if cur else 0
            new_val = cur_val + total_income

            await con.execute(
                """
                INSERT INTO economy.balances (guild_id, character_name, balance_val, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (guild_id, character_name)
                DO UPDATE SET balance_val=EXCLUDED.balance_val, updated_at=NOW()
                """,
                int(ch.guild_id),
                str(ch.name),
                int(new_val),
            )

            await con.execute(
                "INSERT INTO economy.income_claims (guild_id, character_name, claim_date) VALUES ($1,$2,$3)",
                int(ch.guild_id),
                str(ch.name),
                today,
            )

            await con.execute(
                """
                INSERT INTO economy.ledger (guild_id, character_name, character_user_id, actor_user_id, kind, delta_val, reason, details_json)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                """,
                int(ch.guild_id),
                str(ch.name),
                int(ch.user_id),
                int(interaction.user.id),
                "INCOME",
                int(total_income),
                "Daily income",
                json.dumps({"base": BASE_DAILY_INCOME_VAL, "assets": asset_income}, ensure_ascii=False),
            )

    await interaction.followup.send(
        f"Income claimed: **{format_money(total_income)}**. New balance: **{format_money(new_val)}**.",
        ephemeral=True,
        allowed_mentions=discord.AllowedMentions.none(),
    )

    asyncio.create_task(rebuild_dashboard())


@tree.command(name="econ_adjust", description="Staff: add/subtract currency (never negative).")
@require_admin()
@app_commands.describe(character="Pick a character", delta_val="Delta in Val (negative subtracts)", reason="Why")
@app_commands.autocomplete(character=character_name_autocomplete)
async def econ_adjust_cmd(interaction: discord.Interaction, character: str, delta_val: int, reason: str):
    await interaction.response.defer(thinking=False, ephemeral=True)

    if not (DATABASE_URL and LEGACY_SOURCE_GUILD_ID):
        await interaction.followup.send("Database is not configured.", ephemeral=True)
        return

    ch = await fetch_character_by_name(int(LEGACY_SOURCE_GUILD_ID), character)
    if not ch or ch.archived:
        await interaction.followup.send("Character not found.", ephemeral=True)
        return

    cur = await get_balance_val(ch.guild_id, ch.name)
    if cur + int(delta_val) < 0:
        await interaction.followup.send("Insufficient funds (this action would make the balance negative).", ephemeral=True)
        return

    new_val = await add_balance_val(ch.guild_id, ch.name, int(delta_val))
    await ledger_write(ch.guild_id, ch.name, ch.user_id, interaction.user.id, "ADJUST", int(delta_val), reason, {})

    await interaction.followup.send(
        f"Updated **{ch.name}**: {format_money(cur)} → **{format_money(new_val)}**.",
        ephemeral=True,
        allowed_mentions=discord.AllowedMentions.none(),
    )

    asyncio.create_task(rebuild_dashboard())


@tree.command(name="econ_set_balance", description="Staff: set exact balance (non-negative).")
@require_admin()
@app_commands.describe(character="Pick a character", balance_val="Exact balance in Val", reason="Why")
@app_commands.autocomplete(character=character_name_autocomplete)
async def econ_set_balance_cmd(interaction: discord.Interaction, character: str, balance_val: int, reason: str):
    await interaction.response.defer(thinking=False, ephemeral=True)

    if not (DATABASE_URL and LEGACY_SOURCE_GUILD_ID):
        await interaction.followup.send("Database is not configured.", ephemeral=True)
        return

    if int(balance_val) < 0:
        await interaction.followup.send("Balance cannot be negative.", ephemeral=True)
        return

    ch = await fetch_character_by_name(int(LEGACY_SOURCE_GUILD_ID), character)
    if not ch or ch.archived:
        await interaction.followup.send("Character not found.", ephemeral=True)
        return

    cur = await get_balance_val(ch.guild_id, ch.name)
    await set_balance_val(ch.guild_id, ch.name, int(balance_val))
    await ledger_write(ch.guild_id, ch.name, ch.user_id, interaction.user.id, "SET_BALANCE", int(balance_val) - cur, reason, {"set_to": int(balance_val)})

    await interaction.followup.send(
        f"Set **{ch.name}** balance: {format_money(cur)} → **{format_money(int(balance_val))}**.",
        ephemeral=True,
        allowed_mentions=discord.AllowedMentions.none(),
    )

    asyncio.create_task(rebuild_dashboard())


# Helper: build tier choices from the NEW Asset Table catalog
def _tier_choice_value(asset_type: str, tier_name: str) -> str:
    return f"{asset_type}||{tier_name}"

def _parse_tier_choice_value(val: str) -> tuple[str, str]:
    if "||" not in val:
        raise ValueError("Invalid tier selection")
    a, t = val.split("||", 1)
    return a.strip(), t.strip()

async def tier_choice_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    q = (current or "").lower().strip()
    out: list[app_commands.Choice[str]] = []
    try:
        for row in ASSET_CATALOG:
            label = f"{row['asset_type']} — {row['tier_name']}"
            if q and q not in label.lower():
                continue
            out.append(app_commands.Choice(name=label[:100], value=_tier_choice_value(row['asset_type'], row['tier_name'])[:100]))
            if len(out) >= 25:
                break
        return out
    except Exception:
        return []

def cumulative_tier_cost(asset_type: str, target_tier: int) -> int:
    # Sum cost to acquire for tiers 1..target_tier of the given asset_type.
    costs = [r['cost_val'] for r in ASSET_CATALOG if r['asset_type'] == asset_type and int(r['tier']) <= int(target_tier)]
    return int(sum(costs))

@tree.command(name="econ_purchase_new", description="Staff: buy a new asset for a character (tier 1..target tier cost, no negative balances).")
@require_admin()
@app_commands.describe(character="Character name", tier_choice="Asset Type + Tier (from NEW Asset Table)", asset_name="Custom name for this asset (unique per character)")
@app_commands.autocomplete(character=character_name_autocomplete, tier_choice=tier_choice_autocomplete)
async def econ_purchase_new(interaction: discord.Interaction, character: str, tier_choice: str, asset_name: str):
    await interaction.response.defer(ephemeral=True)
    asset_name = (asset_name or "").strip()
    if not asset_name:
        await interaction.followup.send("Asset name cannot be blank.", ephemeral=True)
        return

    try:
        asset_type, tier_name = _parse_tier_choice_value(tier_choice)
    except Exception:
        await interaction.followup.send("Invalid tier selection.", ephemeral=True)
        return

    row = next((r for r in ASSET_CATALOG if r['asset_type'] == asset_type and r['tier_name'] == tier_name), None)
    if not row:
        await interaction.followup.send("That asset tier was not found in the NEW Asset Table.", ephemeral=True)
        return

    target_tier = int(row['tier'])
    total_cost = cumulative_tier_cost(asset_type, target_tier)

    async with pool.acquire() as con:
        # Find character (legacy table) to get owner user_id
        ch = await get_character_by_name(con, character)
        if not ch:
            await interaction.followup.send(f"Character not found: {character}", ephemeral=True)
            return

        owner_user_id = int(ch['user_id'])
        # Fetch current balance
        bal = await get_balance(con, owner_user_id, character)
        current_val = int(bal['balance_val']) if bal else 0

        if current_val < total_cost:
            delta = total_cost - current_val
            await interaction.followup.send(
                f"Insufficient funds.\n"
                f"Available: **{format_value(current_val)}**\n"
                f"Total cost: **{format_value(total_cost)}**\n"
                f"Shortfall: **{format_value(delta)}**",
                ephemeral=True,
            )
            return

        # Enforce unique asset_name per (user_id, character, asset_name)
        exists = await con.fetchval(
            "SELECT 1 FROM econ_assets_owned WHERE guild_id=$1 AND user_id=$2 AND character_name=$3 AND asset_name=$4",
            LEGACY_SOURCE_GUILD_ID, owner_user_id, character, asset_name
        )
        if exists:
            await interaction.followup.send(
                "That character already has an asset with that name. Asset names must be unique per character.",
                ephemeral=True,
            )
            return

        # Debit and insert asset
        new_balance = current_val - total_cost
        await set_balance(con, owner_user_id, character, new_balance)

        await con.execute(
            """INSERT INTO econ_assets_owned
                (guild_id, user_id, character_name, asset_name, asset_type, secondary_type, tier, tier_name, cost_val, income_val)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)""",
            LEGACY_SOURCE_GUILD_ID, owner_user_id, character, asset_name,
            asset_type, asset_type, target_tier, tier_name, int(row['cost_val']), int(row['income_val'])
        )

        details = {
            "action": "purchase_new",
            "asset_type": asset_type,
            "tier_name": tier_name,
            "target_tier": target_tier,
            "total_cost": total_cost,
            "asset_name": asset_name,
        }
        await log_ledger(
            con,
            user_id=owner_user_id,
            character_name=character,
            action="purchase_new",
            delta_val=-total_cost,
            by_user_id=int(interaction.user.id),
            details=details,
        )

    # Refresh dashboard + confirm
    await refresh_bank_dashboard()
    await interaction.followup.send(
        f"Purchased **{asset_name}** — **{asset_type} / {tier_name}** for **{format_value(total_cost)}**.\n"
        f"New balance: **{format_value(new_balance)}**",
        ephemeral=True,
    )

@tree.command(name="econ_refresh_bank", description="Staff: force-refresh the Bank of Vilyra dashboard.")
@require_admin()
async def econ_refresh_bank(interaction: discord.Interaction):
    await interaction.response.defer(thinking=False, ephemeral=True)
    await rebuild_dashboard()
    await interaction.followup.send("Bank of Vilyra dashboard refreshed.", ephemeral=True)



# -------------------------
# Dashboard
# -------------------------

async def rebuild_dashboard():
    """Rebuild the Bank of Vilyra dashboard by editing existing messages (no pings)."""
    if not BANK_CHANNEL_ID or not BANK_MESSAGE_IDS:
        return

    guild = client.get_guild(int(LEGACY_SOURCE_GUILD_ID or COMMANDS_GUILD_ID or 0))
    if guild is None:
        try:
            guild = await client.fetch_guild(int(LEGACY_SOURCE_GUILD_ID or COMMANDS_GUILD_ID or 0))
        except Exception:
            guild = None

    bank_channel = None
    if guild:
        bank_channel = guild.get_channel(int(BANK_CHANNEL_ID))
    if bank_channel is None:
        try:
            bank_channel = await client.fetch_channel(int(BANK_CHANNEL_ID))
        except Exception:
            return

    # Fetch characters and balances/assets
    pool = await db_pool()
    async with pool.acquire() as con:
        chars = await con.fetch(
            """
            SELECT name, user_id
            FROM characters
            WHERE guild_id=$1 AND COALESCE(archived, FALSE)=FALSE
            ORDER BY user_id NULLS LAST, name ASC
            """,
            int(LEGACY_SOURCE_GUILD_ID or COMMANDS_GUILD_ID or 0),
        )

        # balances
        bals = await con.fetch(
            "SELECT character_name, balance_val FROM econ_balances WHERE guild_id=$1",
            int(LEGACY_SOURCE_GUILD_ID or COMMANDS_GUILD_ID or 0),
        )
        bal_map = {str(r["character_name"]): int(r["balance_val"]) for r in bals}

        # assets
        assets_rows = await con.fetch(
            """
            SELECT character_name, asset_name, tier_name, secondary_type
            FROM econ_assets
            WHERE guild_id=$1
            ORDER BY character_name ASC, asset_name ASC
            """,
            int(LEGACY_SOURCE_GUILD_ID or COMMANDS_GUILD_ID or 0),
        )
        assets_by_char: Dict[str, List[Dict[str, str]]] = {}
        for r in assets_rows:
            cn = str(r["character_name"])
            assets_by_char.setdefault(cn, []).append(
                {
                    "asset_name": str(r["asset_name"]),
                    "tier_name": str(r["tier_name"]),
                    "secondary_type": str(r["secondary_type"]),
                }
            )

    # Resolve display names (NO mentions)
    display_cache: Dict[Optional[int], str] = {None: "Unassigned"}
    async def get_display_name(uid: Optional[int]) -> str:
        if uid in display_cache:
            return display_cache[uid]
        if not guild or uid is None:
            display_cache[uid] = f"User {uid}" if uid is not None else "Unassigned"
            return display_cache[uid]
        try:
            member = guild.get_member(int(uid))
            if member is None:
                member = await guild.fetch_member(int(uid))
            display_cache[uid] = member.display_name
        except Exception:
            display_cache[uid] = f"User {uid}"
        return display_cache[uid]

    # Build grouped blocks by user
    grouped: Dict[Optional[int], List[str]] = {}
    for row in chars:
        name = str(row["name"])
        uid = row["user_id"]
        bal = bal_map.get(name, 0)
        lines: List[str] = []
        lines.append(f"• **{name}** — {format_money(bal)}")

        arows = assets_by_char.get(name, [])
        if arows:
            lines.append("  *__Assets__*")
            for a in arows[:25]:
                # requested format: [Name] - [Tier Name]
                lines.append(f"  {a['asset_name']} - {a['tier_name']}")

        grouped.setdefault(uid, []).extend(lines)
        grouped.setdefault(uid, []).append("")  # spacer

    # Turn grouped blocks into pages
    blocks: List[str] = []
    for uid, lines in grouped.items():
        disp = await get_display_name(uid)
        header = f"__**{disp}**__"
        body = "\n".join([ln for ln in lines if ln is not None]).rstrip()
        blocks.append(f"{header}\n{body}".rstrip())

    pages: List[str] = []
    cur = ""
    for b in blocks:
        if not b.strip():
            continue
        if len(cur) + len(b) + 2 > 3500:
            pages.append(cur.rstrip())
            cur = b + "\n\n"
        else:
            cur += b + "\n\n"
    if cur.strip():
        pages.append(cur.rstrip())

    # Pad/truncate to available bank message ids
    pages = pages[: len(BANK_MESSAGE_IDS)]
    while len(pages) < len(BANK_MESSAGE_IDS):
        pages.append("")

    # Edit each existing message
    for i, mid in enumerate(BANK_MESSAGE_IDS):
        content = pages[i] or " "
        try:
            msg = await bank_channel.fetch_message(int(mid))
            await msg.edit(content=content, allowed_mentions=discord.AllowedMentions.none())
        except Exception:
            # if we can't fetch/edit, just skip
            continue

async def on_ready():
    try:
        await init_db()
    except Exception as e:
        print(f"[{VERSION}] DB init failed: {e}")
        raise

    # Sync commands (guild-scoped for fast iteration)
    if GUILD_ID:
        guild_obj = discord.Object(id=int(GUILD_ID))
        # IMPORTANT: When changing a command from a simple command to a Group with subcommands
        # (like /econ_purchase -> /econ_purchase new|upgrade), Discord can keep the old shape
        # unless we delete the guild commands first. We do a "clear + sync empty" pass, then
        # sync the current definitions.
        try:
            tree.clear_commands(guild=guild_obj)
            await tree.sync(guild=guild_obj)  # pushes an empty command set -> deletes old guild commands
        except Exception as e:
            print(f"[{VERSION}] Guild command clear failed (continuing): {e}")

        try:
            tree.copy_global_to(guild=guild_obj)
            await tree.sync(guild=guild_obj)
        except Exception as e:
            print(f"[{VERSION}] Command sync failed: {e}")
    else:
        try:
            await tree.sync()
        except Exception as e:
            print(f"[{VERSION}] Global command sync failed: {e}")

    print(f"[test] Starting {VERSION}…")
    print(f"[test] Logged in as {client.user} (commands guild: {GUILD_ID or 'GLOBAL'}; legacy source guild: {LEGACY_SOURCE_GUILD_ID})")

    asyncio.create_task(rebuild_dashboard())


def main():
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
