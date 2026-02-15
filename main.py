# main.py
"""
PURGE / RESET SCRIPT (run once)

What this does:
- Logs in as your bot
- Deletes ALL GLOBAL application commands for the bot
- Deletes ALL GUILD (server) application commands for the bot (for one or more guilds you specify)
- Optionally syncs an empty command tree to those guilds (fast)
- Exits cleanly

How to run:
1) Set env var DISCORD_TOKEN
2) Set env var GUILD_IDS (recommended) as comma-separated IDs, e.g.:
   GUILD_IDS=1324470539506548766
   or
   GUILD_IDS=1324470539506548766,123456789012345678
3) Deploy/run once. Watch logs for "PURGE COMPLETE", then replace main.py with your real bot.

Will this delete/purge old commands?
✅ Yes, for BOTH global and the guild(s) you list.

Does it have to stay in the code permanently?
❌ No. Use it once, then remove/replace.
"""

import os
import asyncio
import logging
from typing import List, Optional

import discord
from discord import app_commands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("CommandPurger")


def _parse_guild_ids(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise SystemExit(f"Invalid GUILD_IDS entry: {part!r} (must be integer IDs)")
    return out


class PurgeClient(discord.Client):
    def __init__(self):
        # No privileged intents required for command deletion.
        intents = discord.Intents.none()
        super().__init__(intents=intents)

        # Even though we won't be using the command tree to register commands,
        # having a tree attached is useful for syncing an empty set.
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # Called before on_ready; good place to do app-command operations.
        await self._purge_commands_then_exit()

    async def _purge_commands_then_exit(self) -> None:
        token_present = bool(os.getenv("DISCORD_TOKEN"))
        if not token_present:
            raise SystemExit("Missing env var DISCORD_TOKEN")

        guild_ids = _parse_guild_ids(os.getenv("GUILD_IDS"))

        # Ensure we have an application ID available
        app_info = await self.application_info()
        log.info("Logged in application: %s (ID: %s)", app_info.name, app_info.id)

        # 1) Delete GLOBAL commands
        try:
            global_cmds = await self.http.get_global_commands(app_info.id)
            log.info("Found %d global commands", len(global_cmds))
            deleted = 0
            for cmd in global_cmds:
                cmd_id = cmd.get("id")
                cmd_name = cmd.get("name")
                if not cmd_id:
                    continue
                await self.http.delete_global_command(app_info.id, cmd_id)
                deleted += 1
                log.info("Deleted global command: %s (ID: %s)", cmd_name, cmd_id)
            log.info("Deleted %d global commands", deleted)
        except Exception:
            log.exception("Failed while deleting global commands")
            # Continue; we still want to attempt guild purge.

        # 2) Delete GUILD commands (recommended because they disappear quickly)
        if not guild_ids:
            log.warning(
                "No GUILD_IDS provided. Guild commands will NOT be purged.\n"
                "Set env var GUILD_IDS to your server ID(s) for a complete wipe."
            )
        else:
            for gid in guild_ids:
                try:
                    guild_cmds = await self.http.get_guild_commands(app_info.id, gid)
                    log.info("Guild %s: found %d commands", gid, len(guild_cmds))
                    deleted = 0
                    for cmd in guild_cmds:
                        cmd_id = cmd.get("id")
                        cmd_name = cmd.get("name")
                        if not cmd_id:
                            continue
                        await self.http.delete_guild_command(app_info.id, gid, cmd_id)
                        deleted += 1
                        log.info("Guild %s: deleted command %s (ID: %s)", gid, cmd_name, cmd_id)
                    log.info("Guild %s: deleted %d commands", gid, deleted)

                    # 3) Sync EMPTY tree to the guild to force immediate "no commands"
                    # Clear any accidental commands in the local tree (should be empty anyway)
                    self.tree.clear_commands(guild=discord.Object(id=gid))
                    synced = await self.tree.sync(guild=discord.Object(id=gid))
                    log.info("Guild %s: synced empty tree (%d commands)", gid, len(synced))

                except Exception:
                    log.exception("Guild %s: failed during purge/sync", gid)

        log.info("✅ PURGE COMPLETE. Shutting down now.")
        await self.close()


def main() -> None:
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Missing env var DISCORD_TOKEN")

    # Optional: GUILD_IDS=comma,separated,ids
    # If you only have one server, set it to that server's ID.
    client = PurgeClient()
    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
