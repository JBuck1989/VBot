import os
import logging
import asyncio
import discord
from discord import app_commands

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger("CommandPurger")

# -----------------------------
# Env helpers
# -----------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

# -----------------------------
# Config
# -----------------------------
DISCORD_TOKEN = _env_str("DISCORD_TOKEN")
GUILD_ID_STR = _env_str("GUILD_ID")  # your server ID (recommended)
PURGE = _env_bool("PURGE_COMMANDS", True)

# PURGE_SCOPE can be: "guild", "global", "both"
PURGE_SCOPE = _env_str("PURGE_SCOPE", "guild").lower()

# After purging, exit the process (recommended)
EXIT_AFTER_PURGE = _env_bool("EXIT_AFTER_PURGE", True)

# -----------------------------
# Discord Client
# -----------------------------
intents = discord.Intents.none()  # no privileged intents needed for command sync
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

@client.event
async def on_ready():
    LOG.info("Logged in as %s (ID: %s)", client.user, client.user.id)

async def purge_guild_commands(guild_id: int) -> int:
    """Clear and sync commands scoped to a specific guild."""
    guild = discord.Object(id=guild_id)
    tree.clear_commands(guild=guild)
    synced = await tree.sync(guild=guild)
    # When clearing, sync() returns the currently registered commands after sync.
    # After a clear, this should be 0.
    return len(synced)

async def purge_global_commands() -> int:
    """Clear and sync global commands."""
    tree.clear_commands(guild=None)
    synced = await tree.sync()
    return len(synced)

async def do_purge():
    if not PURGE:
        LOG.info("PURGE_COMMANDS is disabled. Nothing to do.")
        return

    guild_id = int(GUILD_ID_STR) if GUILD_ID_STR.isdigit() else None

    if PURGE_SCOPE not in ("guild", "global", "both"):
        raise ValueError("PURGE_SCOPE must be one of: guild, global, both")

    if PURGE_SCOPE in ("guild", "both"):
        if guild_id is None:
            raise RuntimeError("PURGE_SCOPE includes 'guild' but GUILD_ID is not set or invalid.")
        LOG.info("Purging GUILD commands for guild_id=%s ...", guild_id)
        count = await purge_guild_commands(guild_id)
        LOG.info("Guild purge complete. Commands now registered in guild: %s", count)

    if PURGE_SCOPE in ("global", "both"):
        LOG.info("Purging GLOBAL commands ...")
        count = await purge_global_commands()
        LOG.info("Global purge complete. Commands now registered globally: %s", count)

    LOG.info("✅ Purge finished.")

class PurgeBot(discord.Client):
    async def setup_hook(self) -> None:
        # Run purge as early as possible after login handshake begins.
        # setup_hook is awaited by discord.py before on_ready.
        try:
            await do_purge()
        except Exception as e:
            LOG.exception("❌ Purge failed: %s", e)
        finally:
            if EXIT_AFTER_PURGE:
                LOG.info("Exiting after purge (EXIT_AFTER_PURGE=True).")
                # Close the client cleanly and stop the process.
                await self.close()

# Replace the base client with PurgeBot instance
client = PurgeBot(intents=intents)
tree = app_commands.CommandTree(client)

def main():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN is missing.")
    LOG.info("Starting Command Purger with PURGE_SCOPE=%s, EXIT_AFTER_PURGE=%s", PURGE_SCOPE, EXIT_AFTER_PURGE)
    client.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
