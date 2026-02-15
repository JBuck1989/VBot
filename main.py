import os
import logging
import asyncio
from typing import Optional

import discord
from discord import app_commands

# Optional local-dev support; safe on Railway
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LOG = logging.getLogger("FinBot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()


class FinBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.none()  # no privileged intents needed
        intents.guilds = True  # needed for app commands
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # Sync commands globally (can take time to appear), OR to one guild for instant iteration.
        # For fastest testing, set GUILD_ID in Railway env vars to your server ID.
        guild_id = os.getenv("GUILD_ID", "").strip()
        if guild_id.isdigit():
            g = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=g)
            synced = await self.tree.sync(guild=g)
            LOG.info("Guild sync succeeded: %s commands", len(synced))
        else:
            synced = await self.tree.sync()
            LOG.info("Global sync succeeded: %s commands", len(synced))

    async def on_ready(self) -> None:
        LOG.info("Logged in as %s (ID: %s)", self.user, self.user.id)


client = FinBot()


@client.tree.command(name="ping", description="Health check")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("✅ Pong", ephemeral=True)


def main() -> None:
    if not DISCORD_TOKEN:
        LOG.error("DISCORD_TOKEN is missing. Set it in Railway Variables.")
        raise SystemExit(1)

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()

