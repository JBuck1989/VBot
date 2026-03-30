# TM_v090_main.txt
# Tournament Bot – Staff-only core management, ephemeral admin workflow by default,
# with public announcement / bracket posting / public match narration for staff-led testing.

import os
import json
import random
import logging
import asyncio
from typing import List, Optional

import discord
from discord import app_commands
from discord.ext import commands
import psycopg
from psycopg.rows import dict_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] TourneyBot: %(message)s")
log = logging.getLogger("TourneyBot")

XP_THRESHOLDS = [0, 12, 30, 60, 105, 170]
EVENT_XP_RULES = {
    "joust": {
        "round_participation": {"riding": 1, "weapon": 1},
        "runner_up": {"composure": 1},
        "champion": {"riding": 1, "weapon": 1, "composure": 1},
    },
    "archery": {
        "round_participation": {"archery": 1},
        "runner_up": {"composure": 1},
        "champion": {"archery": 1, "composure": 1},
    },
    "grand_melee": {
        "round_participation": {"duel": 1, "stamina": 1},
        "runner_up": {"weapon": 1},
        "champion": {"duel": 1, "stamina": 1, "weapon": 1},
    },
    "duel": {
        "round_participation": {"weapon": 1, "duel": 1},
        "runner_up": {"stamina": 1},
        "champion": {"weapon": 1, "duel": 1, "stamina": 1},
    },
    "horse_race": {
        "round_participation": {"riding": 1, "stamina": 1},
        "runner_up": {"composure": 1},
        "champion": {"riding": 1, "stamina": 1, "composure": 1},
    },
    "hunt": {
        "round_participation": {"archery": 1, "stamina": 1},
        "runner_up": {"composure": 1},
        "champion": {"archery": 1, "stamina": 1, "composure": 1},
    },
}

EVENT_LABELS = {
    "joust": "Joust",
    "archery": "Archery",
    "grand_melee": "Grand Melee",
    "duel": "Duel",
    "horse_race": "Horse Race",
    "hunt": "Hunt",
}


EVENT_FORMAT_MAP = {
    "joust": "head_to_head",
    "duel": "head_to_head",
    "archery": "scored_round",
    "horse_race": "scored_round",
    "hunt": "scored_round",
    "grand_melee": "free_for_all",
}


EVENT_DESCRIPTION_TEXTS = {
    "joust": {
        "contest": [
            "The lists stand ready beneath snapping pennants while armored riders lower their lances and salute the gallery.",
            "In this contest, two champions meet across the tilt and thunder toward one another in a storm of splintering ash, ringing plate, and hard-won nerve.",
        ],
        "rules": [
            "Each pass rewards the rider who keeps the straighter seat, the steadier heart, and the truer point.",
            "A clean strike, a shattered lance, or a rider hurled from the saddle may decide the matter in an instant; otherwise, the marshals judge the contest by accumulated advantage across the passes.",
            "Hold your seat, strike true, and trust the saddle beneath you.",
        ],
        "skills": "This event relies on your Riding, Weapon, and Composure.",
    },
    "duel": {
        "contest": [
            "The duel ring is cleared, the witnesses hushed, and two blades are raised for a contest of nerve, measure, and exacting skill.",
            "This is not a wild brawl but a formal clash of timing, footwork, and cleanly claimed touches beneath the judges’ eyes.",
        ],
        "rules": [
            "Each exchange favors the duelist who seizes initiative, finds the better line of attack, and endures the answering pressure without yielding ground.",
            "Touches are counted across the bout, and if neither duelist breaks the deadlock cleanly, the judges award victory to the superior showing in control and bladecraft.",
            "Precision, endurance, and iron self-command matter as much as courage here.",
        ],
        "skills": "This event relies on your Duel, Weapon, Stamina, Precision, and Composure.",
    },
    "archery": {
        "contest": [
            "The butts are raised, the range is stilled, and the archers step to the line beneath the watch of the butts-master and the gathered crowd.",
            "Here victory belongs not to the loudest boast, but to the surest eye and the hand that can repeat excellence arrow after arrow.",
        ],
        "rules": [
            "Each round is judged by the strength of your volleys, with tighter groupings and truer shots earning the greater share of honor.",
            "After the field has loosed and the arrows are counted, only the strongest half advance to the next round until the final championship line is called.",
            "A steady hand, a calm breath, and the discipline to repeat a perfect release will carry the day.",
        ],
        "skills": "This event relies on your Archery and Composure.",
    },
    "horse_race": {
        "contest": [
            "The course-master drops the flag and the riders break from the line in a rush of hooves, shouted wagers, and flying turf.",
            "This is a contest of pace, judgment, and the bond between rider and mount across every punishing leg of the track.",
        ],
        "rules": [
            "Each round measures how well a rider keeps speed without surrendering control, and how much strength remains when the course begins to bite back.",
            "When the riders have finished the heat, the top half advance until only the final championship run remains.",
            "A racer who can balance daring with restraint will often outlast the one who burns too bright too soon.",
        ],
        "skills": "This event relies on your Riding, Stamina, and Composure.",
    },
    "hunt": {
        "contest": [
            "The horns sound, the gates are opened, and the hunters vanish into wood, brush, and broken ground in pursuit of prize and prestige.",
            "This is a contest of keen eyes, tireless movement, and the patience to read sign where others see only wilderness.",
        ],
        "rules": [
            "Each round rewards the hunter who best tracks the trail, keeps their strength for the long chase, and takes the decisive shot when the moment finally comes.",
            "After every cast, the strongest half of the field advance until the last hunt decides the master of the day.",
            "Those who waste effort early or lose their nerve when the quarry breaks will be left behind.",
        ],
        "skills": "This event relies on your Archery, Stamina, and Composure.",
    },
    "grand_melee": {
        "contest": [
            "The ring is opened to the full field at once, and what follows is not a gentleman’s pairing but a roaring crush of steel, shields, bruises, and desperate opportunity.",
            "The Grand Melee is a test of survival as much as victory: only those who can endure the chaos, seize their moments, and remain standing will continue.",
        ],
        "rules": [
            "All combatants enter together, and the judges halt the fighting only when the field has been thinned enough for rest, wounds, and order to be restored.",
            "After each halt, the strongest half continue into the next round until only two remain for the final championship clash.",
            "Boldness matters, but so does stamina; glory belongs to the fighter who can survive the press and still strike with purpose when the ring narrows.",
        ],
        "skills": "This event relies on your Duel, Stamina, and Weapon.",
    },
}

def build_event_description_embed(tournament_name: str, event_row: dict) -> discord.Embed:
    event_type = event_row["event_type"]
    text = EVENT_DESCRIPTION_TEXTS[event_type]
    embed = discord.Embed(
        title=f"{clean_display_name(event_row['name'])} — Event Description",
        color=discord.Color.blurple(),
        description=f"**Tournament:** {clean_display_name(tournament_name)}\n**Event Type:** {EVENT_LABELS.get(event_type, clean_display_name(event_type))}",
    )
    embed.add_field(name="The Contest", value="\n".join(text["contest"]), inline=False)
    embed.add_field(name="Rules of the Day", value="\n".join(text["rules"]), inline=False)
    embed.add_field(name="Skill of the Competitor", value=text["skills"], inline=False)
    return embed

STATUS_LABELS = {
    "draft": "Draft",
    "seeded": "Seeded",
    "active": "Active",
    "completed": "Completed",
    "ready_to_finalize": "Ready to Finalize",
}

KINGDOM_CHOICES = [
    app_commands.Choice(name="Setrathiel", value="Setrathiel"),
    app_commands.Choice(name="Velarith", value="Velarith"),
    app_commands.Choice(name="Lyvik", value="Lyvik"),
    app_commands.Choice(name="Baelon", value="Baelon"),
    app_commands.Choice(name="Avalea", value="Avalea"),
]

SEASON_CHOICES = [
    app_commands.Choice(name="Lirael", value="Lirael"),
    app_commands.Choice(name="Solarae", value="Solarae"),
    app_commands.Choice(name="Faelith", value="Faelith"),
    app_commands.Choice(name="Vaelune", value="Vaelune"),
]

def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable missing: {name}")
    return value

def parse_role_ids() -> set[int]:
    raw = getenv_required("TOURNEY_ADMIN_ROLE_IDS")
    return {int(x.strip()) for x in raw.split(",") if x.strip()}

def user_is_admin(interaction: discord.Interaction) -> bool:
    member = interaction.user
    if not isinstance(member, discord.Member):
        return False
    return any(role.id in ADMIN_ROLE_IDS for role in member.roles)

async def deny_if_not_admin(interaction: discord.Interaction) -> bool:
    if not user_is_admin(interaction):
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return True
    return False

def actor_can_use_character(interaction: discord.Interaction, character_row: dict | None) -> bool:
    if not character_row:
        return False
    if user_is_admin(interaction):
        return True
    return character_row.get("user_id") == interaction.user.id

def get_db():
    return psycopg.connect(BOT_STATE["db"], row_factory=dict_row)

def rank_from_xp(xp_total: int) -> int:
    rank = 0
    for idx, threshold in enumerate(XP_THRESHOLDS):
        if xp_total >= threshold:
            rank = idx
    return min(rank, 5)

def preflight_checks():
    log.info("[TM_v090] Starting preflight checks")
    token = getenv_required("DISCORD_TOKEN")
    db = getenv_required("DATABASE_URL")
    guild = int(getenv_required("GUILD_ID"))
    _ = parse_role_ids()
    conn = psycopg.connect(db, row_factory=dict_row)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'tourney'")
            if not cur.fetchone():
                raise RuntimeError("tourney schema missing")
    finally:
        conn.close()
    log.info("[TM_v090] Preflight checks complete")
    return {"token": token, "db": db, "guild": guild}

BOT_STATE = preflight_checks()
ADMIN_ROLE_IDS = parse_role_ids()
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)
tree = bot.tree

def roll_2d6():
    d1 = random.randint(1, 6)
    d2 = random.randint(1, 6)
    return d1, d2, d1 + d2

def get_tournament_by_name(conn, guild_id: int, tournament_name: str):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM tourney.tournaments WHERE guild_id = %s AND name = %s ORDER BY id DESC LIMIT 1", (guild_id, tournament_name))
        return cur.fetchone()

def get_event_by_name(conn, guild_id: int, tournament_id: int, event_name: str):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM tourney.events WHERE guild_id = %s AND tournament_id = %s AND name = %s ORDER BY id DESC LIMIT 1", (guild_id, tournament_id, event_name))
        return cur.fetchone()

def clean_display_name(name: str) -> str:
    if name is None:
        return ""
    return str(name).replace("\\n", " ").replace("\n", " ").replace("\r", " ").strip()

def fetch_effective_skills(conn, guild_id: int, character_id: int):
    with conn.cursor() as cur:
        cur.execute("SELECT strength, agility, precision, endurance, presence FROM tourney.character_profiles WHERE guild_id = %s AND character_id = %s LIMIT 1", (guild_id, character_id))
        profile = cur.fetchone()
        if not profile:
            raise RuntimeError(f"No tournament profile for character_id={character_id}")
        cur.execute("SELECT skill_code, rank_bonus FROM tourney.character_skill_xp WHERE guild_id = %s AND character_id = %s", (guild_id, character_id))
        bonuses = {row["skill_code"]: row["rank_bonus"] for row in cur.fetchall()}
    base = {
        "riding": profile["agility"] + profile["endurance"],
        "weapon": profile["strength"] + profile["agility"],
        "archery": profile["precision"] + profile["agility"],
        "duel": profile["strength"] + profile["agility"] + profile["precision"],
        "stamina": profile["endurance"] + profile["strength"],
        "composure": profile["presence"] + profile["endurance"],
    }
    return {k: v + bonuses.get(k, 0) for k, v in base.items()}

def compute_seed_score(event_type: str, effective: dict) -> int:
    if event_type == "joust":
        return (effective["riding"] * 3) + (effective["weapon"] * 3) + (effective["composure"] * 2)
    if event_type == "archery":
        return (effective["archery"] * 4) + (effective["composure"] * 2)
    if event_type == "grand_melee":
        return (effective["duel"] * 3) + (effective["stamina"] * 3)
    if event_type == "duel":
        return (effective["duel"] * 3) + (effective["weapon"] * 2) + (effective["stamina"] * 2)
    if event_type == "horse_race":
        return (effective["riding"] * 4) + (effective["stamina"] * 2)
    if event_type == "hunt":
        return (effective["archery"] * 3) + (effective["stamina"] * 2) + (effective["composure"] * 2)
    return 0

def resolve_joust_scoring(strike_total: int, opp_stability_total: int):
    diff = strike_total - opp_stability_total
    if diff >= 8: return 5, "unhorsed", True
    if diff >= 5: return 3, "lance break", False
    if diff >= 2: return 2, "solid hit", False
    if diff >= 0: return 1, "glancing hit", False
    return 0, "miss", False

def log_roll(cur, match_id, guild_id, character_id, phase_code, roll_formula, die_1, die_2, base_total, modifier_total, final_total, detail):
    cur.execute("""
        INSERT INTO tourney.match_rolls
        (match_id, guild_id, character_id, phase_code, roll_formula, die_1, die_2, base_total, modifier_total, final_total, detail_json, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
    """, (match_id, guild_id, character_id, phase_code, roll_formula, die_1, die_2, base_total, modifier_total, final_total, json.dumps(detail)))

def joust_pass_title(pass_no: int) -> str:
    titles = {
        1: "First Pass — The herald drops the pennant and the lances thunder forward!",
        2: "Second Pass — The riders wheel about as the lists tremble beneath hooves!",
        3: "Third Pass — The crowd rises as steel, wood, and honor meet!",
        4: "Fourth Pass — Dust rolls across the tilt as neither rider yields!",
        5: "Fifth Pass — The tension deepens; victory hangs by a heartbeat!",
        6: "Sixth Pass — Endurance and nerve now matter as much as strength!",
        7: "Final Pass — Glory and ruin ride together down the lists!",
    }
    return titles.get(pass_no, "Another pass begins beneath the eyes of the crowd!")

def joust_result_text(attacker_name: str, defender_name: str, result: str) -> str:
    attacker_name = clean_display_name(attacker_name)
    defender_name = clean_display_name(defender_name)
    options_map = {
        "miss": [
            f"{attacker_name}'s lance skids wide of {defender_name}, finding only empty air.",
            f"{attacker_name} fails to find the mark as {defender_name} thunders past.",
            f"{attacker_name}'s strike goes astray, splintering harmlessly aside.",
        ],
        "glancing hit": [
            f"{attacker_name} clips {defender_name} with a glancing strike.",
            f"{attacker_name}'s lance scrapes across {defender_name}'s guard.",
            f"{attacker_name} lands only a shallow blow against {defender_name}.",
        ],
        "solid hit": [
            f"{attacker_name} lands a solid strike upon {defender_name}!",
            f"{attacker_name}'s lance slams true into {defender_name}'s defenses.",
            f"{attacker_name} drives home a clean and forceful blow against {defender_name}.",
        ],
        "lance break": [
            f"{attacker_name}'s lance explodes into splinters against {defender_name}!",
            f"{attacker_name} shatters the lance in a ringing, thunderous hit!",
            f"{attacker_name} strikes with such force that the lance bursts apart on impact!",
        ],
        "unhorsed": [
            f"{attacker_name} delivers a devastating strike that tears {defender_name} from the saddle!",
            f"{attacker_name} strikes with terrible force, hurling {defender_name} earthward!",
            f"{attacker_name} unseats {defender_name} in a storm of splintered wood and dust!",
        ],
    }
    return random.choice(options_map.get(result, [f"{attacker_name} presses the attack against {defender_name}."]))

def duel_round_title(round_no: int) -> str:
    titles = {
        1: "First Exchange — Steel sings as the duelists test one another's measure!",
        2: "Second Exchange — Boots scrape and blades flash in a deadly rhythm!",
        3: "Third Exchange — The watching crowd falls silent for the deciding touches!",
        4: "Sudden Exchange — Honor and victory balance on a razor's edge!",
    }
    return titles.get(round_no, "Another exchange begins as both duelists circle for the opening.")

def duel_touch_text(attacker_name: str, defender_name: str, result: str) -> str:
    attacker_name = clean_display_name(attacker_name)
    defender_name = clean_display_name(defender_name)
    options_map = {
        "clean_touch": [
            f"{attacker_name} slips past the guard and scores a clean touch on {defender_name}!",
            f"{attacker_name}'s blade darts in true, marking a clean touch against {defender_name}.",
            f"{attacker_name} threads the opening and claims the touch on {defender_name}!",
        ],
        "glancing_touch": [
            f"{attacker_name} brushes {defender_name} with a light scoring touch.",
            f"{attacker_name} finds only the narrowest opening but still lands a touch.",
            f"{attacker_name} clips {defender_name} with a glancing point.",
        ],
        "parried": [
            f"{defender_name} turns aside the attack with a sharp parry.",
            f"{defender_name} reads the line and closes it before the touch can land.",
            f"{defender_name} deflects the strike cleanly and resets the measure.",
        ],
    }
    return random.choice(options_map.get(result, [f"{attacker_name} presses forward against {defender_name}."]))

def get_or_create_skill_row(cur, guild_id, character_id, skill_code):
    cur.execute("SELECT id, xp_total, rank_bonus FROM tourney.character_skill_xp WHERE guild_id = %s AND character_id = %s AND skill_code = %s LIMIT 1", (guild_id, character_id, skill_code))
    row = cur.fetchone()
    if row:
        return row
    cur.execute("""
        INSERT INTO tourney.character_skill_xp
        (guild_id, character_id, skill_code, xp_total, rank_bonus, updated_at)
        VALUES (%s, %s, %s, 0, 0, NOW())
        RETURNING id, xp_total, rank_bonus
    """, (guild_id, character_id, skill_code))
    return cur.fetchone()

def award_xp_packet(cur, guild_id, character_id, packet, reason_code, tournament_id, event_id, source_match_id=None, notes=None):
    for skill_code, xp_change in packet.items():
        row = get_or_create_skill_row(cur, guild_id, character_id, skill_code)
        old_xp = row["xp_total"]
        old_rank = row["rank_bonus"]
        new_xp = old_xp + xp_change
        new_rank = rank_from_xp(new_xp)
        cur.execute("UPDATE tourney.character_skill_xp SET xp_total = %s, rank_bonus = %s, updated_at = NOW() WHERE id = %s", (new_xp, new_rank, row["id"]))
        cur.execute("""
            INSERT INTO tourney.character_skill_xp_log
            (guild_id, character_id, skill_code, xp_change, reason_code, source_tournament_id, source_event_id, source_match_id, old_rank_bonus, new_rank_bonus, notes, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (guild_id, character_id, skill_code, xp_change, reason_code, tournament_id, event_id, source_match_id, old_rank, new_rank, notes))

def update_record_counter(cur, guild_id, character_id, event_type, field_name, increment):
    cur.execute("SELECT id, entries_count, wins_count, losses_count, championships_count, runner_up_count, points_for, points_against FROM tourney.records WHERE guild_id = %s AND character_id = %s AND event_type = %s LIMIT 1", (guild_id, character_id, event_type))
    row = cur.fetchone()
    if not row:
        defaults = {"entries_count": 0, "wins_count": 0, "losses_count": 0, "championships_count": 0, "runner_up_count": 0, "points_for": 0, "points_against": 0}
        defaults[field_name] += increment
        cur.execute("""
            INSERT INTO tourney.records
            (guild_id, character_id, event_type, entries_count, wins_count, losses_count, championships_count, runner_up_count, points_for, points_against, last_played_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (guild_id, character_id, event_type, defaults["entries_count"], defaults["wins_count"], defaults["losses_count"], defaults["championships_count"], defaults["runner_up_count"], defaults["points_for"], defaults["points_against"]))
        return
    cur.execute(f"UPDATE tourney.records SET {field_name} = %s, last_played_at = NOW() WHERE id = %s", (row[field_name] + increment, row["id"]))

def auto_seed_event(cur, conn, guild_id, event_row):
    cur.execute("""
        SELECT en.id AS entry_id, en.character_id, c.name
        FROM tourney.entries en
        JOIN characters c ON en.character_id = c.character_id
        WHERE en.guild_id = %s AND en.event_id = %s AND en.registration_status = 'registered'
        ORDER BY c.name
    """, (guild_id, event_row["id"]))
    entrants = cur.fetchall()
    scored = []
    for entrant in entrants:
        effective = fetch_effective_skills(conn, guild_id, entrant["character_id"])
        score = compute_seed_score(event_row["event_type"], effective)
        primary_1 = effective["riding"] if event_row["event_type"] in ("joust", "horse_race") else effective["archery"] if event_row["event_type"] in ("archery", "hunt") else effective["duel"]
        primary_2 = effective["weapon"] if event_row["event_type"] in ("joust", "duel") else effective["stamina"] if event_row["event_type"] in ("grand_melee", "horse_race", "hunt") else effective["composure"]
        scored.append({"entry_id": entrant["entry_id"], "character_id": entrant["character_id"], "name": clean_display_name(entrant["name"]), "score": score, "primary_1": primary_1, "primary_2": primary_2})
    scored.sort(key=lambda x: (x["score"], x["primary_1"], x["primary_2"], x["name"]), reverse=True)
    for idx, entrant in enumerate(scored, start=1):
        cur.execute("UPDATE tourney.entries SET seed = %s WHERE id = %s", (idx, entrant["entry_id"]))
    return scored

def build_first_round_pairings(cur, guild_id: int, event_id: int):
    cur.execute("""
        SELECT en.id AS entry_id, en.character_id, en.user_id, en.seed
        FROM tourney.entries en
        WHERE en.guild_id = %s
          AND en.event_id = %s
          AND en.registration_status = 'registered'
        ORDER BY en.seed ASC NULLS LAST, en.id ASC
    """, (guild_id, event_id))
    seeded_entries = cur.fetchall()
    if len(seeded_entries) < 2:
        return [], None

    working = list(seeded_entries)
    pairings = []
    bye_entry = None

    while len(working) > 1:
        top = working.pop(0)
        preferred_idx = None
        for idx in range(len(working) - 1, -1, -1):
            candidate = working[idx]
            if candidate["user_id"] != top["user_id"]:
                preferred_idx = idx
                break
        if preferred_idx is None:
            preferred_idx = len(working) - 1

        opponent = working.pop(preferred_idx)
        pairings.append((top, opponent))

    if len(working) == 1:
        bye_entry = working[0]

    return pairings, bye_entry



def build_match_result_embeds(match_row: dict, match_id: int, winner_name: str, loser_name: str, summary_lines: List[str], public: bool = False):
    color = discord.Color.gold() if public else discord.Color.dark_gold()
    event_label = EVENT_LABELS.get(match_row.get('event_type'), match_row.get('event_type', 'Match'))
    title = f"{'⚔️ Public Result' if public else '⚔️ Match Result'} — {event_label}"

    main_embed = discord.Embed(title=title, color=color)
    main_embed.description = (
        f"**Tournament:** {clean_display_name(match_row['tournament_name'])}\n"
        f"**Event:** {clean_display_name(match_row['event_name'])}\n"
        f"**Round:** {match_row['round_number']}\n"
        f"**Match ID:** `{match_id}`"
    )
    main_embed.add_field(name="Victor of the Field", value=f"**{clean_display_name(winner_name)}**", inline=True)
    main_embed.add_field(name="Defeated This Round", value=f"**{clean_display_name(loser_name)}**", inline=True)
    main_embed.add_field(name="Judgment", value="The result is entered into the lists.", inline=True)

    body_lines = [clean_display_name(line) for line in summary_lines if clean_display_name(line).strip()]
    if not body_lines:
        return [main_embed]

    body_chunks = chunk_lines_for_embed(body_lines, max_len=1000)
    if body_chunks:
        main_embed.add_field(name="Arena Chronicle", value=body_chunks[0], inline=False)

    embeds = [main_embed]
    remaining = body_chunks[1:]
    if remaining:
        extra_embed = None
        field_count = 0
        part_no = 2
        for chunk in remaining:
            if extra_embed is None or field_count >= 5:
                extra_embed = discord.Embed(
                    title=f"{title} — Continued",
                    color=color,
                    description=(
                        f"**Tournament:** {clean_display_name(match_row['tournament_name'])}\n"
                        f"**Event:** {clean_display_name(match_row['event_name'])}\n"
                        f"**Round:** {match_row['round_number']}"
                    ),
                )
                embeds.append(extra_embed)
                field_count = 0
            extra_embed.add_field(name=f"Arena Chronicle ({part_no})", value=chunk, inline=False)
            field_count += 1
            part_no += 1

    return embeds[:10]


def archery_arrow_result(total: int):
    if total >= 22:
        return 5, "Perfect Bullseye"
    if total >= 19:
        return 4, "Bullseye"
    if total >= 16:
        return 3, "Inner Ring"
    if total >= 13:
        return 2, "Outer Ring"
    return 0, "Miss"

def archery_round_title(round_no: int, entrant_count: int) -> str:
    if entrant_count == 2:
        return "Championship Round — The final two archers step to the line beneath a breathless hush."
    return f"Round {round_no} — {entrant_count} archers take the line as the crowd settles into tense silence."

def archery_arrow_text(name: str, arrow_no: int, result_label: str) -> str:
    name = clean_display_name(name)
    ordinal = {1: "First", 2: "Second", 3: "Third"}.get(arrow_no, f"Arrow {arrow_no}")
    options = {
        "Perfect Bullseye": [
            f"{ordinal} arrow — {name} splits the center so cleanly that the butts-master lifts both brows in disbelief. **PERFECT BULLSEYE!**",
            f"{ordinal} arrow — {name}'s shaft buries itself dead-center. **PERFECT BULLSEYE!**",
        ],
        "Bullseye": [
            f"{ordinal} arrow — {name} sends a beautiful shot into the heart of the target. **BULLSEYE!**",
            f"{ordinal} arrow — {name} draws, looses, and finds the center ring. **BULLSEYE!**",
        ],
        "Inner Ring": [
            f"{ordinal} arrow — {name} lands a strong shot in the inner ring.",
            f"{ordinal} arrow — {name}'s aim holds true, striking the inner ring.",
        ],
        "Outer Ring": [
            f"{ordinal} arrow — {name} catches the outer ring and keeps the score alive.",
            f"{ordinal} arrow — {name}'s shaft clips the outer ring.",
        ],
        "Miss": [
            f"{ordinal} arrow — {name}'s shot goes wide, drawing a groan from the rail.",
            f"{ordinal} arrow — {name} misjudges the line and misses clean.",
        ],
    }
    return random.choice(options.get(result_label, [f"{ordinal} arrow — {name} looses a shot."]))

def build_archery_round_embed(tournament_name: str, event_name: str, round_no: int, rankings: list, champion_name: str | None = None, public: bool = False, summary_lines: list[str] | None = None):
    color = discord.Color.gold() if public else discord.Color.dark_teal()
    title = f"{clean_display_name(event_name)} — {'Public Results' if public else 'Round Results'}"
    embed = discord.Embed(title=title, color=color)
    embed.description = f"**Tournament:** {clean_display_name(tournament_name)}\n**Round:** {round_no}"

    lines = []
    advancing = []
    eliminated = []
    champion_label = "🏹 Champion of the Butts"
    for idx, row in enumerate(rankings, start=1):
        marker = "👑 " if champion_name and clean_display_name(row["name"]) == clean_display_name(champion_name) else ""
        status = row.get("round_status", "")
        status_text = f" — {status}" if status else ""
        best_arrow_points = row.get("best_arrow_points", row.get("best_segment_points", 0))
        display_name = clean_display_name(row["name"])
        lines.append(
            f"{idx}. {marker}**{display_name}** — {row['total_points']} pts "
            f"(best arrow: {best_arrow_points}){status_text}"
        )
        if "ADVANCES" in status:
            advancing.append(display_name)
        elif "ELIMINATED" in status or row.get("eliminated"):
            eliminated.append(display_name)

    standings_chunks = chunk_lines_for_embed(lines if lines else ["No scores recorded."])
    for i, chunk in enumerate(standings_chunks, start=1):
        embed.add_field(name="Standings" if i == 1 else f"Standings ({i})", value=chunk, inline=False)

    if advancing:
        adv_chunks = chunk_lines_for_embed([f"• **{name}**" for name in advancing], max_len=900)
        for i, chunk in enumerate(adv_chunks, start=1):
            embed.add_field(name="Advancing" if i == 1 else f"Advancing ({i})", value=chunk, inline=False)

    if eliminated:
        elim_chunks = chunk_lines_for_embed([f"• {name}" for name in eliminated], max_len=900)
        for i, chunk in enumerate(elim_chunks, start=1):
            embed.add_field(name="Eliminated" if i == 1 else f"Eliminated ({i})", value=chunk, inline=False)

    if summary_lines:
        field_focus = summary_lines[:18]
        crowd_focus = summary_lines[18:]
        for i, chunk in enumerate(chunk_lines_for_embed(field_focus, max_len=900)[:4], start=1):
            embed.add_field(name="Field Narration" if i == 1 else f"Field Narration ({i})", value=chunk, inline=False)
        if crowd_focus:
            for i, chunk in enumerate(chunk_lines_for_embed(crowd_focus, max_len=900)[:2], start=1):
                embed.add_field(name="Crowd and Judges" if i == 1 else f"Crowd and Judges ({i})", value=chunk, inline=False)

    if champion_name:
        embed.add_field(name=champion_label, value=clean_display_name(champion_name), inline=False)
    return embed



def fetch_scored_round_rankings(conn, match_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                mp.character_id,
                c.name,
                mp.final_position,
                mp.eliminated,
                m.winner_character_id,
                e.event_type,
                COALESCE(SUM(CASE WHEN mr.detail_json ? 'points' THEN (mr.detail_json->>'points')::int ELSE 0 END), 0) AS total_points,
                COALESCE(MAX(CASE WHEN mr.detail_json ? 'points' THEN (mr.detail_json->>'points')::int ELSE 0 END), 0) AS best_segment_points
            FROM tourney.match_participants mp
            JOIN tourney.matches m ON mp.match_id = m.id
            JOIN tourney.events e ON m.event_id = e.id
            JOIN characters c ON mp.character_id = c.character_id
            LEFT JOIN tourney.match_rolls mr ON mr.match_id = mp.match_id AND mr.character_id = mp.character_id
            WHERE mp.match_id = %s
            GROUP BY mp.character_id, c.name, mp.final_position, mp.eliminated, m.winner_character_id, e.event_type
            ORDER BY mp.final_position ASC NULLS LAST, total_points DESC, best_segment_points DESC, c.name ASC
        """, (match_id,))
        rows = cur.fetchall()
    rankings = []
    total_rows = len(rows)
    for row in rows:
        item = dict(row)
        if total_rows == 2 and item["winner_character_id"] == item["character_id"]:
            item["round_status"] = "Champion"
        elif total_rows == 2 and item["final_position"] == 2:
            item["round_status"] = "Runner-up"
        elif item["eliminated"]:
            item["round_status"] = "Eliminated"
        else:
            item["round_status"] = "Advances"
        rankings.append(item)
    return rankings

def get_latest_scored_round_match(conn, guild_id: int, event_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.*
            FROM tourney.matches m
            WHERE m.guild_id = %s
              AND m.event_id = %s
              AND m.match_type = 'scored_round'
            ORDER BY m.round_number DESC, m.id DESC
            LIMIT 1
        """, (guild_id, event_id))
        return cur.fetchone()

def resolve_archery_shoot_off(cur, conn, guild_id: int, match_id: int, round_no: int, tied_rows: list, purpose_text: str):
    notes = [f"**The judges call for a shoot-off to decide {purpose_text}.**"]
    attempts = 0
    ordered = tied_rows[:]
    while attempts < 3:
        attempts += 1
        shot_rows = []
        for row in ordered:
            effective = fetch_effective_skills(conn, guild_id, row["character_id"])
            d1, d2, base = roll_2d6()
            modifier = effective["archery"]
            final_total = base + modifier
            points, result_label = archery_arrow_result(final_total)
            log_roll(
                cur,
                match_id,
                guild_id,
                row["character_id"],
                f"round_{round_no}_shootoff_{attempts}",
                "2d6 + archery",
                d1,
                d2,
                base,
                modifier,
                final_total,
                {"skill": "archery", "points": points, "result": result_label, "shoot_off": True},
            )
            notes.append(f"Shoot-off {attempts} — **{clean_display_name(row['name'])}** scores `{final_total}`.")
            shot_rows.append({
                "character_id": row["character_id"],
                "name": row["name"],
                "shoot_total": final_total,
                "tie_stat": row["tie_stat"],
            })
        shot_rows.sort(key=lambda x: (-x["shoot_total"], -x["tie_stat"], clean_display_name(x["name"])))
        if len(shot_rows) == 1 or shot_rows[0]["shoot_total"] != shot_rows[1]["shoot_total"]:
            name_to_row = {r["character_id"]: r for r in ordered}
            ordered = [name_to_row[r["character_id"]] for r in shot_rows]
            return ordered, notes
        ordered = [r for r in ordered if r["character_id"] in [s["character_id"] for s in shot_rows]]
    notes.append("**Even the shoot-off cannot separate them cleanly; the judges defer to steadier composure and prior marksmanship.**")
    ordered.sort(key=lambda x: (-x["total_points"], -x["best_segment_points"], -x["tie_stat"], clean_display_name(x["name"])))
    return ordered, notes

def horse_race_leg_result(total: int):
    if total >= 24:
        return 5, "Thunderous Lead"
    if total >= 21:
        return 4, "Strong Surge"
    if total >= 17:
        return 3, "Steady Gallop"
    if total >= 14:
        return 2, "Held Pace"
    return 0, "Stumble"

def horse_race_round_title(round_no: int, entrant_count: int) -> str:
    if entrant_count == 2:
        return "Championship Heat — The final two riders lower themselves into the saddle as the crowd presses against the rails."
    return f"Heat {round_no} — {entrant_count} riders gather at the line while the course-master raises the pennant."

def horse_race_leg_text(name: str, leg_no: int, result_label: str) -> str:
    name = clean_display_name(name)
    leg_name = {1: "Break from the Line", 2: "Middle Stretch", 3: "Final Sprint"}.get(leg_no, f"Leg {leg_no}")
    options = {
        "Thunderous Lead": [
            f"{leg_name} — {name} explodes forward in a spray of turf and takes a commanding lead!",
            f"{leg_name} — {name} and mount devour the course, thunder rolling beneath iron-shod hooves!",
        ],
        "Strong Surge": [
            f"{leg_name} — {name} urges the mount onward and gains hard, valuable ground.",
            f"{leg_name} — {name} drives into a powerful surge that rattles the rail.",
        ],
        "Steady Gallop": [
            f"{leg_name} — {name} keeps a strong, disciplined pace through the field.",
            f"{leg_name} — {name} rides cleanly and holds position with a steady gallop.",
        ],
        "Held Pace": [
            f"{leg_name} — {name} holds the line, neither breaking away nor falling fully behind.",
            f"{leg_name} — {name} keeps the mount composed and in contention.",
        ],
        "Stumble": [
            f"{leg_name} — {name}'s mount loses precious ground and the crowd gasps at the falter.",
            f"{leg_name} — {name} hits a rough patch of course and stumbles back in the pack.",
        ],
    }
    return random.choice(options.get(result_label, [f"{leg_name} — {name} drives onward down the course."]))

def resolve_horse_race_sprint_off(cur, conn, guild_id: int, match_id: int, round_no: int, tied_rows: list, purpose_text: str):
    notes = [f"**The course-master orders a sprint-off to decide {purpose_text}.**"]
    attempts = 0
    ordered = tied_rows[:]
    while attempts < 3:
        attempts += 1
        sprint_rows = []
        for row in ordered:
            effective = fetch_effective_skills(conn, guild_id, row["character_id"])
            d1, d2, base = roll_2d6()
            modifier = effective["riding"] + (effective["stamina"] // 2)
            final_total = base + modifier
            points, result_label = horse_race_leg_result(final_total)
            log_roll(
                cur,
                match_id,
                guild_id,
                row["character_id"],
                f"round_{round_no}_sprintoff_{attempts}",
                "2d6 + riding + stamina_half",
                d1,
                d2,
                base,
                modifier,
                final_total,
                {"skill": "riding", "points": points, "result": result_label, "sprint_off": True},
            )
            notes.append(f"Sprint-off {attempts} — **{clean_display_name(row['name'])}** clocks `{final_total}`.")
            sprint_rows.append({
                "character_id": row["character_id"],
                "name": row["name"],
                "sprint_total": final_total,
                "tie_stat": row["tie_stat"],
            })
        sprint_rows.sort(key=lambda x: (-x["sprint_total"], -x["tie_stat"], clean_display_name(x["name"])))
        if len(sprint_rows) == 1 or sprint_rows[0]["sprint_total"] != sprint_rows[1]["sprint_total"]:
            name_to_row = {r["character_id"]: r for r in ordered}
            ordered = [name_to_row[r["character_id"]] for r in sprint_rows]
            return ordered, notes
        ordered = [r for r in ordered if r["character_id"] in [s["character_id"] for s in sprint_rows]]
    notes.append("**Even the sprint-off cannot cleanly break the deadlock; the judges defer to endurance and prior pace across the course.**")
    ordered.sort(key=lambda x: (-x["total_points"], -x["best_segment_points"], -x["tie_stat"], clean_display_name(x["name"])))
    return ordered, notes

def hunt_leg_result(total: int):
    if total >= 24:
        return 5, "Perfect Sign"
    if total >= 21:
        return 4, "Clean Strike"
    if total >= 17:
        return 3, "Strong Trail"
    if total >= 14:
        return 2, "Kept Pace"
    return 0, "Lost the Quarry"

def hunt_round_title(round_no: int, entrant_count: int) -> str:
    if entrant_count == 2:
        return "Championship Hunt — The final two hunters vanish into the green as the gallery waits on horn and hound."
    return f"Hunt {round_no} — {entrant_count} hunters fan into the wilds as horns sound beyond the tree line."

def hunt_leg_text(name: str, leg_no: int, result_label: str) -> str:
    name = clean_display_name(name)
    leg_name = {1: "Taking the Sign", 2: "Closing the Trail", 3: "The Final Cast"}.get(leg_no, f"Leg {leg_no}")
    options = {
        "Perfect Sign": [
            f"{leg_name} — {name} reads the woodland sign like a living ledger and moves with uncanny certainty!",
            f"{leg_name} — {name} finds the freshest sign at once, drawing impressed murmurs from the watching field.",
        ],
        "Clean Strike": [
            f"{leg_name} — {name} keeps the trail true and gains precious ground on the quarry.",
            f"{leg_name} — {name} guides the pursuit with sharp instinct and disciplined pace.",
        ],
        "Strong Trail": [
            f"{leg_name} — {name} follows well, holding a reliable line through brush and briar.",
            f"{leg_name} — {name} keeps the hunt orderly and strong.",
        ],
        "Kept Pace": [
            f"{leg_name} — {name} stays in contention, though without fully breaking ahead.",
            f"{leg_name} — {name} maintains the chase and does not yield the field.",
        ],
        "Lost the Quarry": [
            f"{leg_name} — {name} loses the clearest spoor and precious moments slip away.",
            f"{leg_name} — {name} is forced to recover the trail as the quarry gains distance.",
        ],
    }
    return random.choice(options.get(result_label, [f"{leg_name} — {name} presses on through the hunt."]))

def resolve_hunt_shoot_off(cur, conn, guild_id: int, match_id: int, round_no: int, tied_rows: list, purpose_text: str):
    notes = [f"**The master of the hunt calls for a final proving shot to decide {purpose_text}.**"]
    attempts = 0
    ordered = tied_rows[:]
    while attempts < 3:
        attempts += 1
        shot_rows = []
        for row in ordered:
            effective = fetch_effective_skills(conn, guild_id, row["character_id"])
            d1, d2, base = roll_2d6()
            modifier = effective["archery"] + (effective["composure"] // 2)
            final_total = base + modifier
            points, result_label = hunt_leg_result(final_total)
            log_roll(
                cur,
                match_id,
                guild_id,
                row["character_id"],
                f"round_{round_no}_huntoff_{attempts}",
                "2d6 + archery + composure_half",
                d1,
                d2,
                base,
                modifier,
                final_total,
                {"skill": "archery", "points": points, "result": result_label, "hunt_off": True},
            )
            notes.append(f"Hunt-off {attempts} — **{clean_display_name(row['name'])}** marks `{final_total}`.")
            shot_rows.append({
                "character_id": row["character_id"],
                "name": row["name"],
                "hunt_total": final_total,
                "tie_stat": row["tie_stat"],
            })
        shot_rows.sort(key=lambda x: (-x["hunt_total"], -x["tie_stat"], clean_display_name(x["name"])))
        if len(shot_rows) == 1 or shot_rows[0]["hunt_total"] != shot_rows[1]["hunt_total"]:
            name_to_row = {r["character_id"]: r for r in ordered}
            ordered = [name_to_row[r["character_id"]] for r in shot_rows]
            return ordered, notes
        ordered = [r for r in ordered if r["character_id"] in [s["character_id"] for s in shot_rows]]
    notes.append("**Even the proving shots fail to part them; the judges defer to steadier composure and stronger trailcraft across the day.**")
    ordered.sort(key=lambda x: (-x["total_points"], -x["best_segment_points"], -x["tie_stat"], clean_display_name(x["name"])))
    return ordered, notes





def grand_melee_phase_result(total: int):
    if total >= 24:
        return 5, "Dominant Clash"
    if total >= 21:
        return 4, "Decisive Push"
    if total >= 17:
        return 3, "Strong Press"
    if total >= 14:
        return 2, "Held Ground"
    return 0, "Driven Back"

def grand_melee_round_title(round_no: int, entrant_count: int) -> str:
    if entrant_count == 2:
        return "Championship Clash — The final two champions remain amid the battered ring as steel rises for the last exchange."
    return f"Melee Round {round_no} — {entrant_count} combatants crash together as the marshals struggle to follow every strike."

def grand_melee_phase_text(name: str, phase_no: int, result_label: str) -> str:
    name = clean_display_name(name)
    phase_name = {1: "Opening Clash", 2: "Thinning Circle", 3: "Final Press"}.get(phase_no, f"Phase {phase_no}")
    options = {
        "Dominant Clash": [
            f"{phase_name} — {name} breaks through the press like a storm, scattering opponents in all directions!",
            f"{phase_name} — {name} surges through the melee with brutal authority, drawing a roar from the stands.",
        ],
        "Decisive Push": [
            f"{phase_name} — {name} batters aside the line and gains commanding position.",
            f"{phase_name} — {name} turns defense into momentum and drives deeper into the fray.",
        ],
        "Strong Press": [
            f"{phase_name} — {name} holds strong and wins hard-fought ground.",
            f"{phase_name} — {name} fights cleanly through the crush of shields and blades.",
        ],
        "Held Ground": [
            f"{phase_name} — {name} remains standing in the chaos and gives little away.",
            f"{phase_name} — {name} weathers the press and stays in contention.",
        ],
        "Driven Back": [
            f"{phase_name} — {name} is forced backward beneath the crush of the melee.",
            f"{phase_name} — {name} loses ground as the circle tightens dangerously.",
        ],
    }
    return random.choice(options.get(result_label, [f"{phase_name} — {name} fights on through the chaos."]))

def resolve_grand_melee_tie(cur, conn, guild_id: int, match_id: int, round_no: int, tied_rows: list, purpose_text: str):
    notes = [f"**The marshals call for one last brutal exchange to decide {purpose_text}.**"]
    attempts = 0
    ordered = tied_rows[:]
    while attempts < 3:
        attempts += 1
        clash_rows = []
        for row in ordered:
            effective = fetch_effective_skills(conn, guild_id, row["character_id"])
            d1, d2, base = roll_2d6()
            modifier = effective["duel"] + (effective["stamina"] // 2)
            final_total = base + modifier
            points, result_label = grand_melee_phase_result(final_total)
            log_roll(
                cur,
                match_id,
                guild_id,
                row["character_id"],
                f"round_{round_no}_meleeoff_{attempts}",
                "2d6 + duel + stamina_half",
                d1,
                d2,
                base,
                modifier,
                final_total,
                {"skill": "duel", "points": points, "result": result_label, "melee_off": True},
            )
            notes.append(f"Clash-off {attempts} — **{clean_display_name(row['name'])}** scores `{final_total}`.")
            clash_rows.append({
                "character_id": row["character_id"],
                "name": row["name"],
                "clash_total": final_total,
                "tie_stat": row["tie_stat"],
            })
        clash_rows.sort(key=lambda x: (-x["clash_total"], -x["tie_stat"], clean_display_name(x["name"])))
        if len(clash_rows) == 1 or clash_rows[0]["clash_total"] != clash_rows[1]["clash_total"]:
            name_to_row = {r["character_id"]: r for r in ordered}
            ordered = [name_to_row[r["character_id"]] for r in clash_rows]
            return ordered, notes
        ordered = [r for r in ordered if r["character_id"] in [s["character_id"] for s in clash_rows]]
    notes.append("**Even that exchange fails to separate them; the judges defer to the stronger showing across the full melee.**")
    ordered.sort(key=lambda x: (-x["total_points"], -x["best_segment_points"], -x["tie_stat"], clean_display_name(x["name"])))
    return ordered, notes


def split_scored_round_summary(summary_lines: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    if not summary_lines:
        return [], [], []

    cleaned = [clean_display_name(line) for line in summary_lines]
    intro_lines: list[str] = []
    participant_blocks: list[list[str]] = []
    closing_lines: list[str] = []

    current_block: list[str] = []
    saw_participants = False

    def is_participant_start(text: str) -> bool:
        return text.startswith("**") and (
            "steps to the line" in text
            or "gathers the reins" in text
            or "slips into the green" in text
            or "lowers into the crush of the melee" in text
        )

    for line in cleaned:
        stripped = (line or "").strip()
        if not stripped:
            if current_block:
                participant_blocks.append(current_block)
                current_block = []
                saw_participants = True
            continue

        if not saw_participants and not is_participant_start(stripped) and not current_block:
            intro_lines.append(stripped)
            continue

        if is_participant_start(stripped):
            if current_block:
                participant_blocks.append(current_block)
            current_block = [stripped]
            saw_participants = True
        elif current_block:
            current_block.append(stripped)
        else:
            closing_lines.append(stripped)

    if current_block:
        participant_blocks.append(current_block)

    return intro_lines, participant_blocks, closing_lines

def pack_blocks_into_chunks(blocks: list[list[str]], max_len: int = 3500) -> list[str]:
    chunks: list[str] = []
    current = ""
    for block in blocks:
        block_text = "\n".join(block).strip()
        if not block_text:
            continue
        candidate = block_text if not current else current + "\n\n" + block_text
        if len(candidate) > max_len:
            if current:
                chunks.append(current)
                current = block_text
            else:
                line_chunks = chunk_lines_for_embed(block_text.split("\n"), max_len=max_len)
                chunks.extend(line_chunks[:-1])
                current = line_chunks[-1] if line_chunks else ""
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_scored_round_embeds(
    tournament_name: str,
    event_name: str,
    event_type: str,
    round_no: int,
    rankings: list,
    champion_name: str | None = None,
    public: bool = False,
    summary_lines: list[str] | None = None,
):
    color = discord.Color.gold() if public else discord.Color.dark_teal()
    title = f"{'📣' if public else '📜'} {clean_display_name(event_name)} — {'Official Scoreboard' if public else 'Round Results'}"

    if event_type == "archery":
        segment_label = "best arrow"
        champion_label = "🏹 Champion of the Butts"
        narrative_label = "Field Narration"
        reaction_label = "Judges and Gallery"
        round_flair = "The range narrows."
    elif event_type == "horse_race":
        segment_label = "best leg"
        champion_label = "🏇 Champion of the Course"
        narrative_label = "Race Narrative"
        reaction_label = "Course and Crowd"
        round_flair = "The course claims the slow and tests the bold."
    elif event_type == "hunt":
        segment_label = "best trail"
        champion_label = "🦌 Master of the Hunt"
        narrative_label = "Hunt Narrative"
        reaction_label = "Masters and Onlookers"
        round_flair = "The quarry runs and the field thins."
    else:
        segment_label = "best clash"
        champion_label = "⚔️ Champion of the Melee"
        narrative_label = "Field of Battle"
        reaction_label = "Judges and Crowd"
        round_flair = "Only the hardiest remain."

    lines = []
    advancing = []
    for idx, row in enumerate(rankings, start=1):
        marker = "👑 " if champion_name and clean_display_name(row["name"]) == clean_display_name(champion_name) else ""
        status = row.get("round_status", "")
        status_text = f" — {status}" if status else ""
        lines.append(
            f"{idx}. {marker}**{clean_display_name(row['name'])}** — {row['total_points']} pts "
            f"({segment_label}: {row['best_segment_points']}){status_text}"
        )
        if "ADVANCES" in status.upper():
            advancing.append(clean_display_name(row["name"]))

    main_embed = discord.Embed(title=title, color=color)
    main_embed.description = f"**Tournament:** {clean_display_name(tournament_name)}\n**Round:** {round_no}\n*{round_flair}*"

    for i, chunk in enumerate(chunk_lines_for_embed(lines if lines else ["No scores recorded."], max_len=1000), start=1):
        main_embed.add_field(name="Stylized Scoreboard" if i == 1 else f"Stylized Scoreboard ({i})", value=chunk, inline=False)

    if advancing:
        for i, chunk in enumerate(chunk_lines_for_embed([f"• **{name}**" for name in advancing], max_len=1000), start=1):
            main_embed.add_field(name="Advancing" if i == 1 else f"Advancing ({i})", value=chunk, inline=False)

    if champion_name:
        main_embed.add_field(name=champion_label, value=clean_display_name(champion_name), inline=False)

    embeds = [main_embed]

    if summary_lines:
        intro_lines, participant_blocks, closing_lines = split_scored_round_summary(summary_lines)

        if intro_lines:
            herald_embed = discord.Embed(
                title=f"{clean_display_name(event_name)} — Herald's Cry",
                color=color,
                description=f"**Tournament:** {clean_display_name(tournament_name)}\n**Round:** {round_no}",
            )
            for i, chunk in enumerate(chunk_lines_for_embed(intro_lines, max_len=1000), start=1):
                herald_embed.add_field(name="Opening Call" if i == 1 else f"Opening Call ({i})", value=chunk, inline=False)
            embeds.append(herald_embed)

        participant_chunks = pack_blocks_into_chunks(participant_blocks, max_len=3400)
        for i, chunk in enumerate(participant_chunks, start=1):
            narrative_embed = discord.Embed(
                title=f"{clean_display_name(event_name)} — {narrative_label}",
                color=color,
                description=f"**Tournament:** {clean_display_name(tournament_name)}\n**Round:** {round_no}\n**Part:** {i} of {len(participant_chunks)}",
            )
            for j, subchunk in enumerate(chunk_lines_for_embed(chunk.split("\n"), max_len=1000), start=1):
                narrative_embed.add_field(name=narrative_label if j == 1 else f"{narrative_label} ({j})", value=subchunk, inline=False)
            embeds.append(narrative_embed)

        if closing_lines:
            reaction_embed = discord.Embed(
                title=f"{clean_display_name(event_name)} — {reaction_label}",
                color=color,
                description=f"**Tournament:** {clean_display_name(tournament_name)}\n**Round:** {round_no}",
            )
            for i, chunk in enumerate(chunk_lines_for_embed(closing_lines, max_len=1000), start=1):
                reaction_embed.add_field(name=reaction_label if i == 1 else f"{reaction_label} ({i})", value=chunk, inline=False)
            embeds.append(reaction_embed)

    return embeds[:10]


def run_scored_round_and_store(conn, guild_id: int, tournament_row: dict, event_row: dict):
    if event_row["event_type"] not in ("archery", "horse_race", "hunt", "grand_melee") or event_row["format_type"] not in ("scored_round", "free_for_all"):
        return {"ok": False, "error": "This command currently supports Archery, Horse Race, Hunt, and Grand Melee."}

    with conn.cursor() as cur:
        cur.execute("""
            SELECT en.id AS entry_id, en.character_id, en.registration_status, c.name
            FROM tourney.entries en
            JOIN characters c ON en.character_id = c.character_id
            WHERE en.guild_id = %s
              AND en.event_id = %s
              AND en.registration_status IN ('registered', 'advanced')
            ORDER BY COALESCE(en.seed, 999999), c.name
        """, (guild_id, event_row["id"]))
        entrants = cur.fetchall()

        if len(entrants) < 2:
            noun = "archers" if event_row["event_type"] == "archery" else "riders" if event_row["event_type"] == "horse_race" else "hunters" if event_row["event_type"] == "hunt" else "combatants"
            return {"ok": False, "error": f"At least two active {noun} are required to run a round."}

        round_no = (event_row["round_number"] or 0) + 1
        cur.execute("""
            INSERT INTO tourney.matches
            (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, created_at)
            VALUES (%s, %s, %s, %s, 1, 'active', 'scored_round', NOW())
            RETURNING id
        """, (event_row["id"], tournament_row["id"], guild_id, round_no))
        match_id = cur.fetchone()["id"]

        for slot_no, entrant in enumerate(entrants, start=1):
            cur.execute(
                "INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, %s, FALSE)",
                (match_id, entrant["entry_id"], entrant["character_id"], slot_no),
            )

        if event_row["event_type"] == "archery":
            summary_lines = [archery_round_title(round_no, len(entrants)), ""]
        elif event_row["event_type"] == "horse_race":
            summary_lines = [horse_race_round_title(round_no, len(entrants)), ""]
        elif event_row["event_type"] == "hunt":
            summary_lines = [hunt_round_title(round_no, len(entrants)), ""]
        else:
            summary_lines = [grand_melee_round_title(round_no, len(entrants)), ""]

        results = []

        for entrant in entrants:
            effective = fetch_effective_skills(conn, guild_id, entrant["character_id"])
            tie_stat = effective["composure"] if event_row["event_type"] == "archery" else effective["stamina"] if event_row["event_type"] == "horse_race" else (effective["composure"] + effective["stamina"]) if event_row["event_type"] == "hunt" else (effective["stamina"] + effective["duel"])
            segment_points = []
            display_name = clean_display_name(entrant["name"])

            if event_row["event_type"] == "archery":
                summary_lines.append(f"**{display_name}** steps to the line.")
            elif event_row["event_type"] == "horse_race":
                summary_lines.append(f"**{display_name}** gathers the reins as the course opens ahead.")
            elif event_row["event_type"] == "hunt":
                summary_lines.append(f"**{display_name}** slips into the green with bow, hound, and keen eye.")
            else:
                summary_lines.append(f"**{display_name}** lowers into the crush of the melee as steel rings on every side.")

            for segment_no in range(1, 4):
                d1, d2, base = roll_2d6()
                if event_row["event_type"] == "archery":
                    modifier = effective["archery"] + (effective["composure"] // 2)
                    final_total = base + modifier
                    points, result_label = archery_arrow_result(final_total)
                    formula = "2d6 + archery + composure_half"
                    detail = {"skill": "archery", "points": points, "result": result_label, "segment_no": segment_no}
                    summary_lines.append(archery_arrow_text(entrant["name"], segment_no, result_label))
                elif event_row["event_type"] == "horse_race":
                    modifier = effective["riding"] + (effective["stamina"] // 2) + (effective["composure"] // 3)
                    final_total = base + modifier
                    points, result_label = horse_race_leg_result(final_total)
                    formula = "2d6 + riding + stamina_half + composure_third"
                    detail = {"skill": "riding", "points": points, "result": result_label, "segment_no": segment_no}
                    summary_lines.append(horse_race_leg_text(entrant["name"], segment_no, result_label))
                elif event_row["event_type"] == "hunt":
                    modifier = effective["archery"] + (effective["stamina"] // 2) + (effective["composure"] // 3)
                    final_total = base + modifier
                    points, result_label = hunt_leg_result(final_total)
                    formula = "2d6 + archery + stamina_half + composure_third"
                    detail = {"skill": "archery", "points": points, "result": result_label, "segment_no": segment_no}
                    summary_lines.append(hunt_leg_text(entrant["name"], segment_no, result_label))
                else:
                    modifier = effective["duel"] + (effective["stamina"] // 2) + (effective["weapon"] // 3)
                    final_total = base + modifier
                    points, result_label = grand_melee_phase_result(final_total)
                    momentum = 0
                    if base >= 10:
                        momentum = 1
                        points += 1
                    formula = "2d6 + duel + stamina_half + weapon_third"
                    detail = {"skill": "duel", "points": points, "result": result_label, "segment_no": segment_no, "momentum": momentum}
                    summary_lines.append(grand_melee_phase_text(entrant["name"], segment_no, result_label))
                    if momentum:
                        summary_lines.append(f"The crowd howls as **{display_name}** catches sudden momentum and turns the press in a heartbeat!")

                log_roll(
                    cur,
                    match_id,
                    guild_id,
                    entrant["character_id"],
                    f"round_{round_no}_segment_{segment_no}",
                    formula,
                    d1,
                    d2,
                    base,
                    modifier,
                    final_total,
                    detail,
                )
                segment_points.append(points)

            total_points = sum(segment_points)
            best_segment_points = max(segment_points) if segment_points else 0
            label = "Total score" if event_row["event_type"] == "archery" else "Overall pace" if event_row["event_type"] == "horse_race" else "Overall hunt score" if event_row["event_type"] == "hunt" else "Overall melee score"
            summary_lines.append(f"{label}: **{display_name} — {total_points}**")
            summary_lines.append("")
            results.append({
                "entry_id": entrant["entry_id"],
                "character_id": entrant["character_id"],
                "name": display_name,
                "total_points": total_points,
                "best_segment_points": best_segment_points,
                "tie_stat": tie_stat,
            })

        results.sort(key=lambda x: (-x["total_points"], -x["best_segment_points"], -x["tie_stat"], clean_display_name(x["name"])))

        if len(results) == 2:
            if (
                results[0]["total_points"] == results[1]["total_points"]
                and results[0]["best_segment_points"] == results[1]["best_segment_points"]
                and results[0]["tie_stat"] == results[1]["tie_stat"]
            ):
                if event_row["event_type"] == "archery":
                    results, shoot_notes = resolve_archery_shoot_off(cur, conn, guild_id, match_id, round_no, results, "the championship")
                elif event_row["event_type"] == "horse_race":
                    results, shoot_notes = resolve_horse_race_sprint_off(cur, conn, guild_id, match_id, round_no, results, "the championship")
                elif event_row["event_type"] == "hunt":
                    results, shoot_notes = resolve_hunt_shoot_off(cur, conn, guild_id, match_id, round_no, results, "the championship")
                else:
                    results, shoot_notes = resolve_grand_melee_tie(cur, conn, guild_id, match_id, round_no, results, "the championship")
                summary_lines.extend(shoot_notes)
                summary_lines.append("")

            champion = results[0]
            runner_up = results[1]
            champion["round_status"] = "Champion"
            runner_up["round_status"] = "Runner-up"
            cur.execute("UPDATE tourney.entries SET registration_status = 'advanced' WHERE id = %s", (champion["entry_id"],))
            cur.execute("UPDATE tourney.entries SET registration_status = 'eliminated' WHERE id = %s", (runner_up["entry_id"],))
            cur.execute("UPDATE tourney.events SET round_number = %s, status = 'ready_to_finalize', updated_at = NOW() WHERE id = %s", (round_no, event_row["id"]))
            winner_character_id = champion["character_id"]
            champion_name = champion["name"]
            if event_row["event_type"] == "archery":
                summary_lines.append(f"**{champion_name.upper()} claims the final volley and stands champion of the butts!**")
            elif event_row["event_type"] == "horse_race":
                summary_lines.append(f"**{champion_name.upper()} breaks clear in the championship heat and claims the laurels of the course!**")
            elif event_row["event_type"] == "hunt":
                summary_lines.append(f"**{champion_name.upper()} brings the quarry to heel and is hailed master of the hunt!**")
            else:
                summary_lines.append(f"**{champion_name.upper()} stands alone at the heart of the ring and is hailed champion of the grand melee!**")
        else:
            advance_count = max(2, len(results) // 2)
            boundary = results[advance_count - 1]
            tied_boundary = [
                r for r in results
                if r["total_points"] == boundary["total_points"]
                and r["best_segment_points"] == boundary["best_segment_points"]
                and r["tie_stat"] == boundary["tie_stat"]
            ]
            if len(tied_boundary) > 1:
                tied_ids = {r["character_id"] for r in tied_boundary}
                inside = [r for r in results[:advance_count] if r["character_id"] in tied_ids]
                outside = [r for r in results[advance_count:] if r["character_id"] in tied_ids]
                if inside and outside:
                    if event_row["event_type"] == "archery":
                        reordered, shoot_notes = resolve_archery_shoot_off(cur, conn, guild_id, match_id, round_no, tied_boundary, "the cut line")
                    elif event_row["event_type"] == "horse_race":
                        reordered, shoot_notes = resolve_horse_race_sprint_off(cur, conn, guild_id, match_id, round_no, tied_boundary, "the cut line")
                    elif event_row["event_type"] == "hunt":
                        reordered, shoot_notes = resolve_hunt_shoot_off(cur, conn, guild_id, match_id, round_no, tied_boundary, "the cut line")
                    else:
                        reordered, shoot_notes = resolve_grand_melee_tie(cur, conn, guild_id, match_id, round_no, tied_boundary, "the cut line")
                    summary_lines.extend(shoot_notes)
                    summary_lines.append("")
                    tied_order = {r["character_id"]: i for i, r in enumerate(reordered)}
                    results.sort(key=lambda x: (
                        -x["total_points"],
                        -x["best_segment_points"],
                        -x["tie_stat"],
                        tied_order.get(x["character_id"], 999),
                        clean_display_name(x["name"]),
                    ))

            advancing = results[:advance_count]
            eliminated = results[advance_count:]
            advance_ids = {r["entry_id"] for r in advancing}
            for row in results:
                row["round_status"] = "Advances" if row["entry_id"] in advance_ids else "Eliminated"

            cur.execute("""
                UPDATE tourney.entries
                SET registration_status = CASE WHEN id = ANY(%s) THEN 'advanced' ELSE 'eliminated' END
                WHERE guild_id = %s
                  AND event_id = %s
                  AND registration_status IN ('registered', 'advanced')
            """, ([r["entry_id"] for r in advancing], guild_id, event_row["id"]))
            cur.execute("UPDATE tourney.events SET round_number = %s, status = 'active', updated_at = NOW() WHERE id = %s", (round_no, event_row["id"]))
            winner_character_id = results[0]["character_id"]
            champion_name = None
            if event_row["event_type"] == "archery":
                summary_lines.append("**The marshals raise their hands and announce the advancing archers.**")
                summary_lines.append("Advancing: " + ", ".join([f"**{r['name']}**" for r in advancing]))
                if eliminated:
                    summary_lines.append("Fallen from the line: " + ", ".join([f"**{r['name']}**" for r in eliminated]))
            elif event_row["event_type"] == "horse_race":
                summary_lines.append("**The course-master marks the riders who have earned the next heat.**")
                summary_lines.append("Advancing: " + ", ".join([f"**{r['name']}**" for r in advancing]))
                if eliminated:
                    summary_lines.append("Dropped from the running: " + ", ".join([f"**{r['name']}**" for r in eliminated]))
            elif event_row["event_type"] == "hunt":
                summary_lines.append("**The hunt-master names the hunters who will ride on into the next cast.**")
                summary_lines.append("Advancing: " + ", ".join([f"**{r['name']}**" for r in advancing]))
                if eliminated:
                    summary_lines.append("Lost from the chase: " + ", ".join([f"**{r['name']}**" for r in eliminated]))
            else:
                summary_lines.append("**The judges halt the melee for water, chirurgeons, and the grim sorting of those still fit to continue.**")
                summary_lines.append("The crowd pounds the rails as names are called and battered survivors are sent back to ready themselves for the next clash.")
                summary_lines.append("Advancing: " + ", ".join([f"**{r['name']}**" for r in advancing]))
                if eliminated:
                    for fallen in eliminated:
                        fallen["round_status"] = random.choice(["Overborne", "Yielded", "Pulled for Wounds", "Cast from the Ring"])
                    summary_lines.append("Removed from the melee: " + ", ".join([f"**{r['name']}** ({r['round_status']})" for r in eliminated]))
                    summary_lines.append("Some stagger out beneath their own strength, while others are led away under the careful hands of squires and chirurgeons.")

        cur.execute(
            "UPDATE tourney.matches SET status = 'completed', winner_character_id = %s, narrative_summary = %s, completed_at = NOW() WHERE id = %s",
            (winner_character_id, "\n".join(summary_lines), match_id)
        )
        cur.execute("""
            UPDATE tourney.match_participants
            SET final_position = ranked.pos,
                eliminated = CASE WHEN ranked.character_id = %s THEN FALSE ELSE ranked.elim END
            FROM (
                SELECT * FROM (VALUES %s) AS t(character_id, pos, elim)
            ) AS ranked
            WHERE match_id = %s AND tourney.match_participants.character_id = ranked.character_id
        """ % (
            winner_character_id,
            ", ".join(
                f"({r['character_id']}, {idx}, {'TRUE' if r.get('round_status') not in ('Advances', 'Champion') else 'FALSE'})"
                for idx, r in enumerate(results, start=1)
            ),
            match_id
        ))
        conn.commit()

    return {
        "ok": True,
        "match_id": match_id,
        "round_no": round_no,
        "summary_lines": summary_lines,
        "rankings": results,
        "champion_name": champion_name,
        "winner_name": results[0]["name"],
        "loser_name": results[1]["name"] if len(results) > 1 else "",
        "event_type": event_row["event_type"],
    }


def load_match_context(conn, guild_id: int, match_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.*, e.name AS event_name, e.event_type, t.name AS tournament_name
            FROM tourney.matches m
            JOIN tourney.events e ON m.event_id = e.id
            JOIN tourney.tournaments t ON m.tournament_id = t.id
            WHERE m.guild_id = %s AND m.id = %s
            LIMIT 1
        """, (guild_id, match_id))
        match_row = cur.fetchone()
        if not match_row:
            return None
        cur.execute("""
            SELECT mp.id, mp.entry_id, mp.character_id, mp.slot_number, c.name, en.user_id
            FROM tourney.match_participants mp
            JOIN characters c ON mp.character_id = c.character_id
            LEFT JOIN tourney.entries en ON mp.entry_id = en.id
            WHERE mp.match_id = %s
            ORDER BY mp.slot_number
        """, (match_id,))
        participants = cur.fetchall()
    return {"match_row": match_row, "participants": participants}

def resolve_match_and_store(conn, guild_id: int, match_id: int):
    ctx = load_match_context(conn, guild_id, match_id)
    if not ctx:
        return {"ok": False, "error": "Match not found."}
    match_row = ctx["match_row"]
    participants = ctx["participants"]
    if match_row["status"] == "completed":
        return {"ok": False, "error": "That match is already completed."}
    if len(participants) == 1:
        return {"ok": False, "error": "That match is a bye. It will be finalized automatically when the contested matches in that round are complete."}
    if len(participants) != 2:
        return {"ok": False, "error": "Current match runner only supports two-participant matches."}

    a, b = participants[0], participants[1]

    if match_row["event_type"] == "joust":
        a_skills = fetch_effective_skills(conn, guild_id, a["character_id"])
        b_skills = fetch_effective_skills(conn, guild_id, b["character_id"])
        a_score = 0
        b_score = 0
        winner_character_id = None
        summary_lines = []

        with conn.cursor() as cur:
            cur.execute("UPDATE tourney.matches SET status = 'active' WHERE id = %s", (match_id,))
            pass_no = 1
            while pass_no <= 7:
                a_s1, a_s2, a_base_stab = roll_2d6(); b_s1, b_s2, b_base_stab = roll_2d6()
                a_k1, a_k2, a_base_strike = roll_2d6(); b_k1, b_k2, b_base_strike = roll_2d6()
                a_stability = a_base_stab + a_skills["riding"]
                b_stability = b_base_stab + b_skills["riding"]
                a_mod = a_skills["composure"] // 4
                b_mod = b_skills["composure"] // 4
                a_strike = a_base_strike + a_skills["weapon"] + a_mod
                b_strike = b_base_strike + b_skills["weapon"] + b_mod

                log_roll(cur, match_id, guild_id, a["character_id"], f"pass_{pass_no}_stability", "2d6 + riding", a_s1, a_s2, a_base_stab, a_skills["riding"], a_stability, {"skill": "riding"})
                log_roll(cur, match_id, guild_id, b["character_id"], f"pass_{pass_no}_stability", "2d6 + riding", b_s1, b_s2, b_base_stab, b_skills["riding"], b_stability, {"skill": "riding"})
                log_roll(cur, match_id, guild_id, a["character_id"], f"pass_{pass_no}_strike", "2d6 + weapon + composure_mod", a_k1, a_k2, a_base_strike, a_skills["weapon"] + a_mod, a_strike, {"skill": "weapon", "composure_mod": a_mod})
                log_roll(cur, match_id, guild_id, b["character_id"], f"pass_{pass_no}_strike", "2d6 + weapon + composure_mod", b_k1, b_k2, b_base_strike, b_skills["weapon"] + b_mod, b_strike, {"skill": "weapon", "composure_mod": b_mod})

                a_points, a_result, a_unhorsed = resolve_joust_scoring(a_strike, b_stability)
                b_points, b_result, b_unhorsed = resolve_joust_scoring(b_strike, a_stability)
                a_score += a_points
                b_score += b_points
                summary_lines += [
                    joust_pass_title(pass_no),
                    joust_result_text(a["name"], b["name"], a_result),
                    joust_result_text(b["name"], a["name"], b_result),
                    f"Score after the pass: **{clean_display_name(a['name'])} {a_score} — {clean_display_name(b['name'])} {b_score}**",
                    ""
                ]

                if a_unhorsed and not b_unhorsed:
                    winner_character_id = a["character_id"]
                    summary_lines += [f"**{clean_display_name(a['name']).upper()} UNHORSES {clean_display_name(b['name']).upper()}!**", "The grandstand erupts as the fallen rider crashes into the dust and shattered lancewood skitters across the tilt."]
                    break
                if b_unhorsed and not a_unhorsed:
                    winner_character_id = b["character_id"]
                    summary_lines += [f"**{clean_display_name(b['name']).upper()} UNHORSES {clean_display_name(a['name']).upper()}!**", "A roar rolls over the lists as armor, splinters, and pride strike the earth together."]
                    break
                if pass_no >= 3 and a_score != b_score:
                    winner_character_id = a["character_id"] if a_score > b_score else b["character_id"]
                    summary_lines += [f"**The marshals call the tilt after pass {pass_no}; {clean_display_name(a['name']) if a_score > b_score else clean_display_name(b['name'])} holds the advantage.**"]
                    break
                if pass_no == 3 and a_score == b_score:
                    summary_lines += ["**The scores stand even after three passes — the lists demand more!**"]
                pass_no += 1

            if winner_character_id is None:
                a_total = a_skills["riding"] + a_skills["weapon"] + a_skills["composure"]
                b_total = b_skills["riding"] + b_skills["weapon"] + b_skills["composure"]
                winner_character_id = a["character_id"] if (a_total > b_total or (a_total == b_total and clean_display_name(a["name"]) < clean_display_name(b["name"]))) else b["character_id"]
                summary_lines += ["**At the limit of the contest, the judges confer and award victory by superior showing in the saddle and at strike.**"]

            cur.execute("UPDATE tourney.matches SET status = 'completed', winner_character_id = %s, narrative_summary = %s, completed_at = NOW() WHERE id = %s", (winner_character_id, "\n".join(summary_lines), match_id))
            cur.execute("UPDATE tourney.match_participants SET final_position = CASE WHEN character_id = %s THEN 1 ELSE 2 END, eliminated = CASE WHEN character_id = %s THEN FALSE ELSE TRUE END WHERE match_id = %s", (winner_character_id, winner_character_id, match_id))

        conn.commit()
        winner_name = clean_display_name(a["name"]) if winner_character_id == a["character_id"] else clean_display_name(b["name"])
        loser_name = clean_display_name(b["name"]) if winner_character_id == a["character_id"] else clean_display_name(a["name"])
        return {"ok": True, "winner_name": winner_name, "loser_name": loser_name, "summary_lines": summary_lines}

    if match_row["event_type"] == "duel":
        a_skills = fetch_effective_skills(conn, guild_id, a["character_id"])
        b_skills = fetch_effective_skills(conn, guild_id, b["character_id"])
        a_touches = 0
        b_touches = 0
        winner_character_id = None
        summary_lines = []

        with conn.cursor() as cur:
            cur.execute("SELECT strength, agility, precision, endurance, presence FROM tourney.character_profiles WHERE guild_id = %s AND character_id = %s LIMIT 1", (guild_id, a["character_id"]))
            a_profile = cur.fetchone()
            cur.execute("SELECT strength, agility, precision, endurance, presence FROM tourney.character_profiles WHERE guild_id = %s AND character_id = %s LIMIT 1", (guild_id, b["character_id"]))
            b_profile = cur.fetchone()

            cur.execute("UPDATE tourney.matches SET status = 'active' WHERE id = %s", (match_id,))
            round_no = 1
            while round_no <= 4:
                a_i1, a_i2, a_base_init = roll_2d6()
                b_i1, b_i2, b_base_init = roll_2d6()
                a_o1, a_o2, a_base_off = roll_2d6()
                b_o1, b_o2, b_base_off = roll_2d6()
                a_d1, a_d2, a_base_def = roll_2d6()
                b_d1, b_d2, b_base_def = roll_2d6()

                a_initiative_mod = a_profile["precision"] + (a_skills["composure"] // 3)
                b_initiative_mod = b_profile["precision"] + (b_skills["composure"] // 3)
                a_initiative = a_base_init + a_initiative_mod
                b_initiative = b_base_init + b_initiative_mod
                a_attack = a_base_off + a_skills["duel"] + (a_skills["weapon"] // 2)
                b_attack = b_base_off + b_skills["duel"] + (b_skills["weapon"] // 2)
                a_defense = a_base_def + a_skills["stamina"] + (a_skills["composure"] // 2)
                b_defense = b_base_def + b_skills["stamina"] + (b_skills["composure"] // 2)

                log_roll(cur, match_id, guild_id, a["character_id"], f"round_{round_no}_initiative", "2d6 + precision + composure_mod", a_i1, a_i2, a_base_init, a_initiative_mod, a_initiative, {"skill": "precision"})
                log_roll(cur, match_id, guild_id, b["character_id"], f"round_{round_no}_initiative", "2d6 + precision + composure_mod", b_i1, b_i2, b_base_init, b_initiative_mod, b_initiative, {"skill": "precision"})
                log_roll(cur, match_id, guild_id, a["character_id"], f"round_{round_no}_attack", "2d6 + duel + weapon_half", a_o1, a_o2, a_base_off, a_skills["duel"] + (a_skills["weapon"] // 2), a_attack, {"skill": "duel"})
                log_roll(cur, match_id, guild_id, b["character_id"], f"round_{round_no}_attack", "2d6 + duel + weapon_half", b_o1, b_o2, b_base_off, b_skills["duel"] + (b_skills["weapon"] // 2), b_attack, {"skill": "duel"})
                log_roll(cur, match_id, guild_id, a["character_id"], f"round_{round_no}_defense", "2d6 + stamina + composure_half", a_d1, a_d2, a_base_def, a_skills["stamina"] + (a_skills["composure"] // 2), a_defense, {"skill": "stamina"})
                log_roll(cur, match_id, guild_id, b["character_id"], f"round_{round_no}_defense", "2d6 + stamina + composure_half", b_d1, b_d2, b_base_def, b_skills["stamina"] + (b_skills["composure"] // 2), b_defense, {"skill": "stamina"})

                summary_lines.append(duel_round_title(round_no if round_no <= 3 else 4))

                first, second = (a, b) if (a_initiative > b_initiative or (a_initiative == b_initiative and clean_display_name(a["name"]) < clean_display_name(b["name"]))) else (b, a)

                if first["character_id"] == a["character_id"]:
                    diff1 = a_attack - b_defense
                else:
                    diff1 = b_attack - a_defense

                if diff1 >= 5:
                    if first["character_id"] == a["character_id"]:
                        a_touches += 1
                    else:
                        b_touches += 1
                    summary_lines.append(duel_touch_text(first["name"], second["name"], "clean_touch"))
                elif diff1 >= 2:
                    if first["character_id"] == a["character_id"]:
                        a_touches += 1
                    else:
                        b_touches += 1
                    summary_lines.append(duel_touch_text(first["name"], second["name"], "glancing_touch"))
                else:
                    summary_lines.append(duel_touch_text(first["name"], second["name"], "parried"))

                if second["character_id"] == a["character_id"]:
                    diff2 = a_attack - b_defense - 1
                else:
                    diff2 = b_attack - a_defense - 1

                if diff2 >= 5:
                    if second["character_id"] == a["character_id"]:
                        a_touches += 1
                    else:
                        b_touches += 1
                    summary_lines.append(duel_touch_text(second["name"], first["name"], "clean_touch"))
                elif diff2 >= 2:
                    if second["character_id"] == a["character_id"]:
                        a_touches += 1
                    else:
                        b_touches += 1
                    summary_lines.append(duel_touch_text(second["name"], first["name"], "glancing_touch"))
                else:
                    summary_lines.append(duel_touch_text(second["name"], first["name"], "parried"))

                summary_lines += [f"Touches after the exchange: **{clean_display_name(a['name'])} {a_touches} — {clean_display_name(b['name'])} {b_touches}**", ""]

                if round_no == 3 and a_touches != b_touches:
                    winner_character_id = a["character_id"] if a_touches > b_touches else b["character_id"]
                    summary_lines.append(f"**Three exchanges are complete; {clean_display_name(a['name']) if a_touches > b_touches else clean_display_name(b['name'])} holds the superior measure before the assembled court.**")
                    break

                if round_no == 3 and a_touches == b_touches:
                    summary_lines.append("**The duel stands even after three exchanges — one final sudden exchange will decide it!**")

                if round_no == 4:
                    if a_touches != b_touches:
                        winner_character_id = a["character_id"] if a_touches > b_touches else b["character_id"]
                    else:
                        a_total = a_skills["duel"] + a_skills["weapon"] + a_skills["composure"]
                        b_total = b_skills["duel"] + b_skills["weapon"] + b_skills["composure"]
                        winner_character_id = a["character_id"] if (a_total > b_total or (a_total == b_total and clean_display_name(a["name"]) < clean_display_name(b["name"]))) else b["character_id"]
                        summary_lines.append("**Even after the final exchange, the touches remain equal; the judges award the duel on superior control and bladecraft.**")
                    break

                round_no += 1

            cur.execute("UPDATE tourney.matches SET status = 'completed', winner_character_id = %s, narrative_summary = %s, completed_at = NOW() WHERE id = %s", (winner_character_id, "\n".join(summary_lines), match_id))
            cur.execute("UPDATE tourney.match_participants SET final_position = CASE WHEN character_id = %s THEN 1 ELSE 2 END, eliminated = CASE WHEN character_id = %s THEN FALSE ELSE TRUE END WHERE match_id = %s", (winner_character_id, winner_character_id, match_id))

        conn.commit()
        winner_name = clean_display_name(a["name"]) if winner_character_id == a["character_id"] else clean_display_name(b["name"])
        loser_name = clean_display_name(b["name"]) if winner_character_id == a["character_id"] else clean_display_name(a["name"])
        return {"ok": True, "winner_name": winner_name, "loser_name": loser_name, "summary_lines": summary_lines}

    return {"ok": False, "error": "That event type is not live yet. Current match runner supports joust and duel."}

def build_public_announcement_embed(tournament_row: dict, event_rows: list):
    embed = discord.Embed(title=f"{clean_display_name(tournament_row['name'])} — Tournament Announcement", color=discord.Color.gold())
    embed.description = "Hear ye, hear ye! The lists are prepared and the tournament is called."
    embed.add_field(name="Status", value=STATUS_LABELS.get(tournament_row["status"], tournament_row["status"]))
    embed.add_field(name="Division", value=tournament_row["division"] or "open")
    embed.add_field(name="Season", value=tournament_row["season_label"] or "—")
    embed.add_field(name="Host Location", value=tournament_row["host_location"] or "—", inline=False)
    if tournament_row["notes"]:
        embed.add_field(name="Notes", value=tournament_row["notes"], inline=False)
    if event_rows:
        lines = [f"• **{clean_display_name(row['name'])}** — {EVENT_LABELS.get(row['event_type'], row['event_type'])} (`{STATUS_LABELS.get(row['status'], row['status'])}`)" for row in event_rows]
        embed.add_field(name="Events", value="\n".join(lines), inline=False)
    return embed



def get_event_settings(settings_json):
    if settings_json is None:
        return {}
    if isinstance(settings_json, dict):
        return dict(settings_json)
    try:
        return json.loads(settings_json)
    except Exception:
        return {}

def set_event_settings(cur, event_id: int, settings: dict):
    cur.execute(
        "UPDATE tourney.events SET settings_json = %s::jsonb, updated_at = NOW() WHERE id = %s",
        (json.dumps(settings), event_id)
    )


def get_public_board_ids(settings: dict):
    channel_id = settings.get("public_standings_channel_id") or settings.get("public_bracket_channel_id")
    message_id = settings.get("public_standings_message_id") or settings.get("public_bracket_message_id")
    return channel_id, message_id

def set_public_board_ids(settings: dict, channel_id: int | str, message_id: int | str):
    settings["public_standings_channel_id"] = str(channel_id)
    settings["public_standings_message_id"] = str(message_id)
    settings["public_bracket_channel_id"] = str(channel_id)
    settings["public_bracket_message_id"] = str(message_id)
    return settings

def build_no_results_standings_embed(tournament_name: str, event_name: str, format_type: str, entrant_names: list[str] | None = None):
    embed = discord.Embed(title=f"{clean_display_name(event_name)} — Public Standings", color=discord.Color.blue())
    embed.description = f"**Tournament:** {clean_display_name(tournament_name)}"

    entrants = [clean_display_name(name) for name in (entrant_names or []) if clean_display_name(name)]
    if entrants:
        entry_lines = [f"• **{name}**" for name in entrants]
        for i, chunk in enumerate(chunk_lines_for_embed(entry_lines), start=1):
            embed.add_field(name="Registered Entrants" if i == 1 else f"Registered Entrants ({i})", value=chunk, inline=False)

    if format_type == "head_to_head":
        embed.add_field(name="Status", value="No bracket has been generated yet. Entrants will appear here as they gather for the lists.", inline=False)
    elif format_type == "scored_round":
        embed.add_field(name="Status", value="No round has been run yet. The field stands ready.", inline=False)
    else:
        embed.add_field(name="Status", value="No melee round has been run yet. The marshals have not yet called the first clash.", inline=False)
    return embed



def build_event_intro_embed(tournament_name: str, event_name: str, event_type: str):
    embed = discord.Embed(title=f"{clean_display_name(event_name)} — Opening Call", color=discord.Color.gold())
    embed.description = f"**Tournament:** {clean_display_name(tournament_name)}"
    if event_type == "joust":
        text = [
            "The tilt is dressed, the pennants fly, and the lists await the thunder of hooves.",
            "When the bracket is set, each pairing will ride beneath the eyes of the crowd until only one champion remains.",
        ]
    elif event_type == "duel":
        text = [
            "The duel ground is marked and the judges stand ready to count each touch and turning of the blade.",
            "Each round will narrow the field until two duelists remain for the deciding exchange.",
        ]
    elif event_type == "archery":
        text = [
            "The targets are raised, the range is cleared, and the hush before the first string settles over the butts.",
            "Each round will cut the field until the finest two archers stand for the final volley.",
        ]
    elif event_type == "horse_race":
        text = [
            "The course lies open, the rails are lined, and the riders gather for the opening heat.",
            "With each round the field will be thinned until two riders remain for the championship run.",
        ]
    elif event_type == "hunt":
        text = [
            "The horns are readied, the hounds strain at the leash, and the first sign of the quarry is sought beneath the trees.",
            "Each cast will narrow the field until two hunters remain to contest the master's prize.",
        ]
    else:
        text = [
            "The grand melee begins as all combatants enter the ring together beneath the roar of the stands.",
            "When the field is thinned, the judges will halt the fighting for rest, chirurgeon checks, and the counting of those still fit to continue.",
            "Round by round the melee will narrow until two remain for the crowning clash.",
        ]
    for i, chunk in enumerate(chunk_lines_for_embed(text, max_len=900), start=1):
        embed.add_field(name="Herald's Call" if i == 1 else f"Herald's Call ({i})", value=chunk, inline=False)
    return embed


def build_public_bracket_embed(tournament_name: str, event_name: str, rows: list):
    clean_t = clean_display_name(tournament_name)
    clean_e = clean_display_name(event_name)
    embed = discord.Embed(title=f"⚔️ {clean_e} — Official Board", color=discord.Color.blue())
    embed.description = f"**Tournament:** {clean_t}\n**Status:** Active competition"

    lines = []
    for row in rows:
        slot_1 = clean_display_name(row["slot_1_name"])
        slot_2 = clean_display_name(row["slot_2_name"] or "BYE")
        status = clean_display_name(row["status"]).title()
        line = f"**Round {row['round_number']} • Match {row['match_id']}**\n{slot_1} vs {slot_2}\nStatus: `{status}`"
        if row.get("winner_name"):
            line += f"\nVictor: **{clean_display_name(row['winner_name'])}**"
        lines.append(line)

    if not lines:
        lines = ["No matches have been posted yet."]

    for i, chunk in enumerate(chunk_lines_for_embed(lines, max_len=1000), start=1):
        embed.add_field(name="The Lists" if i == 1 else f"The Lists ({i})", value=chunk, inline=False)

    return embed


def get_event_fresh(conn, event_id: int):
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM tourney.events WHERE id = %s LIMIT 1", (event_id,))
        return cur.fetchone()

def get_head_to_head_contested_state(conn, guild_id: int, event_id: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE match_type <> 'bye') AS contested_total,
                COUNT(*) FILTER (WHERE match_type <> 'bye' AND status = 'completed') AS contested_completed,
                COUNT(*) FILTER (WHERE match_type <> 'bye' AND status IN ('pending','active')) AS contested_open,
                COALESCE(MAX(round_number), 0) AS max_round
            FROM tourney.matches
            WHERE guild_id = %s AND event_id = %s
            """,
            (guild_id, event_id),
        )
        state = cur.fetchone()
        cur.execute(
            """
            SELECT m.*
            FROM tourney.matches m
            WHERE m.guild_id = %s
              AND m.event_id = %s
              AND m.match_type <> 'bye'
              AND m.status = 'completed'
            ORDER BY m.round_number DESC, m.match_order DESC, m.id DESC
            LIMIT 1
            """,
            (guild_id, event_id),
        )
        latest_completed = cur.fetchone()
    return state, latest_completed




def build_public_results_embed(tournament_name: str, event_name: str, champion_name: str, runner_up_name: str | None):
    embed = discord.Embed(title=f"🏆 {clean_display_name(event_name)} — Final Results", color=discord.Color.gold())
    embed.description = f"**Tournament:** {clean_display_name(tournament_name)}\n*The contest is concluded and the honors are declared.*"
    embed.add_field(name="Champion", value=f"**{clean_display_name(champion_name)}**", inline=True)
    embed.add_field(name="Runner-up", value=f"**{clean_display_name(runner_up_name)}**" if runner_up_name else "—", inline=True)
    embed.add_field(name="Status", value="Completed", inline=True)
    return embed

def build_event_recap_intro(event_name: str, event_type: str, champion_name: str, runner_up_name: str | None):
    champion_name = clean_display_name(champion_name)
    runner_up_name = clean_display_name(runner_up_name) if runner_up_name else None
    if event_type == "joust":
        return [
            f"The lists are closed, the shattered lances gathered, and **{champion_name}** is acclaimed victor of **{clean_display_name(event_name)}**.",
            f"{runner_up_name or 'A worthy rival'} carried the challenge deep into the contest, and every rider who entered the tilt added glory to the day.",
        ]
    if event_type == "duel":
        return [
            f"The blades are lowered and the judges proclaim **{champion_name}** champion of **{clean_display_name(event_name)}**.",
            f"{runner_up_name or 'The final challenger'} stood in the last measure with honor, and the full field gave the crowd a contest to remember.",
        ]
    if event_type == "archery":
        return [
            f"The targets are drawn and the butts-master hails **{champion_name}** as champion of **{clean_display_name(event_name)}**.",
            f"{runner_up_name or 'The final rival'} pressed the last round fiercely, while every archer in the field earned the applause of the range.",
        ]
    if event_type == "horse_race":
        return [
            f"The course falls quiet at last, and **{champion_name}** is crowned victor of **{clean_display_name(event_name)}**.",
            f"{runner_up_name or 'The final rider'} drove hard to the final stretch, and every racer who took the course gave the crowd thunder to remember.",
        ]
    if event_type == "hunt":
        return [
            f"The horns are lowered and **{champion_name}** is hailed master of **{clean_display_name(event_name)}**.",
            f"{runner_up_name or 'The last rival'} endured to the final cast, and every hunter in the field added skill, nerve, and worthy sport to the day.",
        ]
    return [
        f"The melee ground stands churned and battered, and **{champion_name}** is hailed champion of **{clean_display_name(event_name)}**.",
        f"{runner_up_name or 'The last rival'} endured to the final clash, and every combatant who entered the ring gave the crowd a brutal spectacle worth singing of.",
    ]

def build_event_recap_embeds(tournament_name: str, event_row: dict, participant_rows: list[dict], champion_name: str, runner_up_name: str | None):
    event_name = clean_display_name(event_row["name"])
    event_type = event_row["event_type"]
    intro_lines = build_event_recap_intro(event_name, event_type, champion_name, runner_up_name)

    lead = discord.Embed(title=f"{event_name} — Event Recap", color=discord.Color.gold())
    lead.description = f"**Tournament:** {clean_display_name(tournament_name)}\n**Status:** Completed"
    for i, chunk in enumerate(chunk_lines_for_embed(intro_lines, max_len=1000), start=1):
        lead.add_field(name="Herald's Recap" if i == 1 else f"Herald's Recap ({i})", value=chunk, inline=False)
    lead.add_field(name="Champion", value=clean_display_name(champion_name), inline=True)
    lead.add_field(name="Runner-up", value=clean_display_name(runner_up_name) if runner_up_name else "—", inline=True)
    embeds = [lead]

    honor_lines = []
    for row in participant_rows:
        status = row["registration_status"]
        display_status = (
            "Champion" if status == "champion" else
            "Runner-up" if status == "runner_up" else
            "Competitor"
        )
        honor_lines.append(f"• **{clean_display_name(row['name'])}** — {display_status}")

    for i, chunk in enumerate(chunk_lines_for_embed(honor_lines, max_len=1000), start=1):
        honor = discord.Embed(title=f"{event_name} — Roll of Honor", color=discord.Color.gold())
        honor.description = f"**Tournament:** {clean_display_name(tournament_name)}"
        honor.add_field(name="Participants" if i == 1 else f"Participants ({i})", value=chunk, inline=False)
        embeds.append(honor)

    return embeds[:10]

def get_event_participant_rows(conn, guild_id: int, event_id: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.name, en.character_id, en.registration_status
            FROM tourney.entries en
            JOIN characters c ON en.character_id = c.character_id
            WHERE en.guild_id = %s
              AND en.event_id = %s
              AND en.registration_status <> 'withdrawn'
            ORDER BY
              CASE
                WHEN en.registration_status = 'champion' THEN 0
                WHEN en.registration_status = 'runner_up' THEN 1
                ELSE 2
              END,
              COALESCE(en.seed, 999999),
              c.name
            """,
            (guild_id, event_id),
        )
        return cur.fetchall()

def chunk_lines_for_embed(lines: list[str], max_len: int = 1000) -> list[str]:
    def split_line(text: str, limit: int) -> list[str]:
        text = clean_display_name(text) or "—"
        if len(text) <= limit:
            return [text]
        parts = []
        remaining = text
        while len(remaining) > limit:
            split_at = remaining.rfind(" ", 0, limit)
            if split_at < max(1, limit // 2):
                split_at = limit
            parts.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        if remaining:
            parts.append(remaining)
        return parts

    normalized = []
    for line in lines:
        normalized.extend(split_line(line, max_len - 20))

    chunks = []
    current = ""
    for line in normalized:
        candidate = line if not current else current + "\n" + line
        if len(candidate) > max_len:
            if current:
                chunks.append(current)
                current = line
            else:
                chunks.append(line[:max_len])
                current = ""
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def estimate_embed_size(embed: discord.Embed) -> int:
    size = 0
    if embed.title:
        size += len(embed.title)
    if embed.description:
        size += len(embed.description)
    if embed.footer and embed.footer.text:
        size += len(embed.footer.text)
    if embed.author and embed.author.name:
        size += len(embed.author.name)
    for field in embed.fields:
        size += len(field.name) + len(field.value)
    return size

def clone_embed_shell(embed: discord.Embed) -> discord.Embed:
    new = discord.Embed(title=embed.title, description=embed.description, color=embed.color)
    if embed.footer and embed.footer.text:
        new.set_footer(text=embed.footer.text, icon_url=embed.footer.icon_url)
    if embed.author and embed.author.name:
        new.set_author(name=embed.author.name, url=embed.author.url, icon_url=embed.author.icon_url)
    return new

def sanitize_embeds_for_send(embeds: list[discord.Embed], max_total: int = 5800, max_fields: int = 20) -> list[discord.Embed]:
    sanitized: list[discord.Embed] = []
    for embed in embeds:
        if estimate_embed_size(embed) <= max_total and len(embed.fields) <= max_fields:
            sanitized.append(embed)
            continue

        current = clone_embed_shell(embed)
        # If description itself is too long, trim it conservatively
        if current.description and len(current.description) > 1200:
            current.description = current.description[:1197] + "..."

        for field in embed.fields:
            field_name = field.name if len(field.name) <= 256 else field.name[:253] + "..."
            value_chunks = chunk_lines_for_embed(field.value.split("\n"), max_len=1000) if len(field.value) > 1000 else [field.value]
            for idx_chunk, chunk in enumerate(value_chunks, start=1):
                this_name = field_name if idx_chunk == 1 else f"{field_name} ({idx_chunk})"
                projected = estimate_embed_size(current) + len(this_name) + len(chunk)
                if len(current.fields) >= max_fields or projected > max_total:
                    sanitized.append(current)
                    current = clone_embed_shell(embed)
                    if current.description and len(current.description) > 1200:
                        current.description = current.description[:1197] + "..."
                current.add_field(name=this_name, value=chunk[:1024], inline=field.inline)
        sanitized.append(current)

    return sanitized[:10]






async def send_followup_embeds_chunked(interaction: discord.Interaction, embeds: list[discord.Embed], *, ephemeral: bool = False):
    sanitized = sanitize_embeds_for_send(embeds)
    if not sanitized:
        return

    batches: list[list[discord.Embed]] = []
    current_batch: list[discord.Embed] = []
    current_size = 0

    for embed in sanitized:
        embed_size = estimate_embed_size(embed)
        # Keep each outbound payload well under Discord aggregate embed limits.
        if current_batch and (len(current_batch) >= 10 or current_size + embed_size > 5600):
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(embed)
        current_size += embed_size

    if current_batch:
        batches.append(current_batch)

    for batch in batches:
        await interaction.followup.send(embeds=batch, ephemeral=ephemeral)




async def send_public_round_embeds(interaction: discord.Interaction, embeds: list[discord.Embed], event_name: str):
    try:
        await send_followup_embeds_chunked(interaction, embeds, ephemeral=False)
    except discord.HTTPException as exc:
        log.exception("Public round embed send failed for event=%s: %s", event_name, exc)
        await interaction.followup.send(
            f"The public result payload for **{clean_display_name(event_name)}** was too large or invalid to send cleanly. "
            "The round resolved successfully, but the narration/results post failed.",
            ephemeral=True,
        )
        raise

def fetch_bracket_rows(conn, guild_id: int, event_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                m.id AS match_id,
                m.round_number,
                m.match_order,
                m.status,
                m.winner_character_id,
                MAX(CASE WHEN mp.slot_number = 1 THEN c.name END) AS slot_1_name,
                MAX(CASE WHEN mp.slot_number = 2 THEN c.name END) AS slot_2_name,
                MAX(CASE WHEN c.character_id = m.winner_character_id THEN c.name END) AS winner_name
            FROM tourney.matches m
            JOIN tourney.match_participants mp ON m.id = mp.match_id
            JOIN characters c ON mp.character_id = c.character_id
            WHERE m.guild_id = %s AND m.event_id = %s
            GROUP BY m.id, m.round_number, m.match_order, m.status, m.winner_character_id
            ORDER BY m.round_number, m.match_order, m.id
        """, (guild_id, event_id))
        return cur.fetchall()


def fetch_registered_entrants(conn, guild_id: int, event_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.name, en.seed, en.registration_status, en.character_id, en.user_id
            FROM tourney.entries en
            JOIN characters c ON en.character_id = c.character_id
            WHERE en.guild_id = %s
              AND en.event_id = %s
              AND en.registration_status <> 'withdrawn'
            ORDER BY COALESCE(en.seed, 999999), c.name
        """, (guild_id, event_id))
        return cur.fetchall()

def fetch_latest_scored_round_rows(conn, guild_id: int, event_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COALESCE(MAX(round_number), 0) AS round_no
            FROM tourney.scored_round_results
            WHERE guild_id = %s AND event_id = %s
        """, (guild_id, event_id))
        round_no = cur.fetchone()["round_no"]
        if round_no == 0:
            return 0, []
        cur.execute("""
            SELECT r.round_number, r.character_id, c.name, r.total_points, r.best_arrow_points,
                   r.advanced, r.rank_in_round
            FROM tourney.scored_round_results r
            JOIN characters c ON r.character_id = c.character_id
            WHERE r.guild_id = %s
              AND r.event_id = %s
              AND r.round_number = %s
            ORDER BY r.rank_in_round, c.name
        """, (guild_id, event_id, round_no))
        return round_no, cur.fetchall()



def build_completed_event_standings_embed(tournament_name: str, event_row: dict, participant_rows: list[dict], champion_name: str, runner_up_name: str | None):
    event_name = clean_display_name(event_row["name"])
    event_type = EVENT_LABELS.get(event_row["event_type"], clean_display_name(event_row["event_type"]))
    embed = discord.Embed(title=f"🏆 {event_name} — Final Standings", color=discord.Color.gold())
    embed.description = (
        f"**Tournament:** {clean_display_name(tournament_name)}\n"
        f"**Event Type:** {event_type}\n"
        f"**Status:** Completed"
    )
    embed.add_field(name="Champion", value=f"**{clean_display_name(champion_name)}**", inline=True)
    embed.add_field(name="Runner-up", value=f"**{clean_display_name(runner_up_name)}**" if runner_up_name else "—", inline=True)
    embed.add_field(name="Honor", value="The field has been settled.", inline=True)

    participant_lines = []
    for idx, row in enumerate(participant_rows, start=1):
        status = row["registration_status"]
        label = (
            "👑 Champion" if status == "champion" else
            "🥈 Runner-up" if status == "runner_up" else
            "Competitor"
        )
        participant_lines.append(f"{idx}. **{clean_display_name(row['name'])}** — {label}")

    for i, chunk in enumerate(chunk_lines_for_embed(participant_lines if participant_lines else ["No participants recorded."], max_len=1000), start=1):
        embed.add_field(name="Roll of Honor" if i == 1 else f"Roll of Honor ({i})", value=chunk, inline=False)
    return embed


async def upsert_public_standings_post(interaction: discord.Interaction, tournament_name: str, event_row: dict):
    settings = get_event_settings(event_row.get("settings_json"))
    channel_id, message_id = get_public_board_ids(settings)
    if not channel_id:
        return False

    channel = interaction.guild.get_channel(int(channel_id))
    if channel is None:
        try:
            channel = await bot.fetch_channel(int(channel_id))
        except Exception:
            return False

    with get_db() as board_conn:
        fresh_event = get_event_fresh(board_conn, event_row["id"]) or event_row
        entrants = fetch_registered_entrants(board_conn, interaction.guild.id, fresh_event["id"])
        entrant_names = [row["name"] for row in entrants]

        if fresh_event["status"] == "completed":
            participant_rows = get_event_participant_rows(board_conn, interaction.guild.id, fresh_event["id"])
            champion_name = None
            runner_up_name = None
            for row in participant_rows:
                if row["registration_status"] == "champion":
                    champion_name = clean_display_name(row["name"])
                elif row["registration_status"] == "runner_up":
                    runner_up_name = clean_display_name(row["name"])
            if not champion_name and participant_rows:
                champion_name = clean_display_name(participant_rows[0]["name"])
            embed = build_completed_event_standings_embed(tournament_name, fresh_event, participant_rows, champion_name or "—", runner_up_name)
        elif fresh_event["format_type"] == "head_to_head":
            rows = fetch_bracket_rows(board_conn, interaction.guild.id, fresh_event["id"])
            embed = build_public_bracket_embed(tournament_name, fresh_event["name"], rows) if rows else build_no_results_standings_embed(tournament_name, fresh_event["name"], fresh_event["format_type"], entrant_names)
        elif fresh_event["event_type"] in ("archery", "horse_race", "hunt", "grand_melee"):
            latest = get_latest_scored_round_match(board_conn, interaction.guild.id, fresh_event["id"])
            if latest:
                rankings = [dict(r) for r in fetch_scored_round_rankings(board_conn, latest["id"])]
                champion_name = clean_display_name(rankings[0]["name"]) if fresh_event["status"] in ("ready_to_finalize", "completed") and rankings else None
                embed = build_scored_round_embeds(tournament_name, fresh_event["name"], fresh_event["event_type"], latest["round_number"], rankings, champion_name, public=True)[0]
            else:
                embed = build_no_results_standings_embed(tournament_name, fresh_event["name"], fresh_event["format_type"], entrant_names)
        else:
            embed = build_no_results_standings_embed(tournament_name, fresh_event["name"], fresh_event["format_type"], entrant_names)

    edited = False
    if message_id:
        try:
            msg = await channel.fetch_message(int(message_id))
            await msg.edit(embed=embed)
            edited = True
        except Exception:
            edited = False

    if not edited:
        msg = await channel.send(embed=embed)
        settings = set_public_board_ids(settings, channel.id, msg.id)
        with get_db() as conn:
            with conn.cursor() as cur:
                set_event_settings(cur, fresh_event["id"], settings)
            conn.commit()
    return True


def event_has_completed_contested_matches(conn, guild_id: int, event_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM tourney.matches WHERE guild_id = %s AND event_id = %s AND status = 'completed' AND match_type <> 'bye'",
            (guild_id, event_id)
        )
        return cur.fetchone()["cnt"] > 0

async def refresh_public_standings_if_safe(interaction: discord.Interaction, tournament_row: dict, event_row: dict):
    settings = get_event_settings(event_row.get("settings_json"))
    board_channel_id, board_message_id = get_public_board_ids(settings)
    if not board_channel_id or not board_message_id:
        return False, ""
    if event_registration_locked(event_row, interaction.guild.id):
        return False, ""
    if event_row["format_type"] != "head_to_head":
        return False, ""

    with get_db() as conn:
        if event_has_completed_contested_matches(conn, interaction.guild.id, event_row["id"]):
            return False, "Standings not refreshed because contested matches have already been completed."

        with conn.cursor() as cur:
            auto_seed_event(cur, conn, interaction.guild.id, event_row)

            cur.execute("""
                UPDATE tourney.entries
                SET registration_status = 'registered'
                WHERE guild_id = %s
                  AND event_id = %s
                  AND registration_status IN ('advanced', 'pending')
            """, (interaction.guild.id, event_row["id"]))

            cur.execute("SELECT id FROM tourney.matches WHERE guild_id = %s AND event_id = %s", (interaction.guild.id, event_row["id"]))
            existing_match_ids = [row["id"] for row in cur.fetchall()]
            if existing_match_ids:
                cur.execute("DELETE FROM tourney.match_rolls WHERE match_id = ANY(%s)", (existing_match_ids,))
                cur.execute("DELETE FROM tourney.match_participants WHERE match_id = ANY(%s)", (existing_match_ids,))
                cur.execute("DELETE FROM tourney.matches WHERE id = ANY(%s)", (existing_match_ids,))

            cur.execute("""
                SELECT id AS entry_id, character_id, seed
                FROM tourney.entries
                WHERE guild_id = %s
                  AND event_id = %s
                  AND registration_status = 'registered'
                ORDER BY seed ASC NULLS LAST, id ASC
            """, (interaction.guild.id, event_row["id"]))
            seeded_entries = cur.fetchall()

            if len(seeded_entries) >= 2:
                pairings, bye_entry = build_first_round_pairings(cur, interaction.guild.id, event_row["id"])
                match_order = 1
                for top, bottom in pairings:
                    cur.execute("""
                        INSERT INTO tourney.matches
                        (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, created_at)
                        VALUES (%s, %s, %s, 1, %s, 'pending', 'head_to_head', NOW())
                        RETURNING id
                    """, (event_row["id"], tournament_row["id"], interaction.guild.id, match_order))
                    match_id = cur.fetchone()["id"]
                    cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 1, FALSE)", (match_id, top["entry_id"], top["character_id"]))
                    cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 2, FALSE)", (match_id, bottom["entry_id"], bottom["character_id"]))
                    match_order += 1

                if bye_entry:
                    cur.execute("""
                        INSERT INTO tourney.matches
                        (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, narrative_summary, created_at)
                        VALUES (%s, %s, %s, 1, %s, 'pending', 'bye', %s, NOW())
                        RETURNING id
                    """, (event_row["id"], tournament_row["id"], interaction.guild.id, match_order, "Pending automatic bye advancement."))
                    bye_match_id = cur.fetchone()["id"]
                    cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 1, FALSE)", (bye_match_id, bye_entry["entry_id"], bye_entry["character_id"]))

                cur.execute("UPDATE tourney.events SET status = 'active', round_number = 1, updated_at = NOW() WHERE id = %s", (event_row["id"],))
                conn.commit()
            else:
                cur.execute("UPDATE tourney.events SET status = 'draft', round_number = 0, updated_at = NOW() WHERE id = %s", (event_row["id"],))
                conn.commit()

    await upsert_public_standings_post(interaction, tournament_row["name"], event_row)
    if len(seeded_entries) >= 2:
        return True, "Public bracket refreshed from current entrants."
    return True, "Public bracket refreshed, but there are fewer than two entrants so no active bracket exists."

def round_is_complete(conn, guild_id: int, event_id: int, round_number: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FILTER (WHERE match_type <> 'bye') AS total_contested,
                   COUNT(*) FILTER (WHERE match_type <> 'bye' AND status = 'completed') AS completed_contested
            FROM tourney.matches
            WHERE guild_id = %s AND event_id = %s AND round_number = %s
        """, (guild_id, event_id, round_number))
        row = cur.fetchone()
    return row["total_contested"] > 0 and row["total_contested"] == row["completed_contested"]

def finalize_pending_byes_in_round(conn, guild_id: int, event_id: int, round_number: int):
    finalized_names = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.id AS match_id, mp.entry_id, mp.character_id, c.name
            FROM tourney.matches m
            JOIN tourney.match_participants mp ON m.id = mp.match_id
            JOIN characters c ON mp.character_id = c.character_id
            WHERE m.guild_id = %s
              AND m.event_id = %s
              AND m.round_number = %s
              AND m.match_type = 'bye'
              AND m.status <> 'completed'
            ORDER BY m.match_order, m.id
        """, (guild_id, event_id, round_number))
        rows = cur.fetchall()
        for row in rows:
            cur.execute("""
                UPDATE tourney.matches
                SET status = 'completed',
                    winner_character_id = %s,
                    narrative_summary = %s,
                    completed_at = NOW()
                WHERE id = %s
            """, (row["character_id"], "Automatic bye advancement.", row["match_id"]))
            cur.execute("""
                UPDATE tourney.match_participants
                SET eliminated = FALSE,
                    final_position = 1
                WHERE match_id = %s
            """, (row["match_id"],))
            cur.execute("""
                UPDATE tourney.entries
                SET registration_status = 'advanced'
                WHERE id = %s
            """, (row["entry_id"],))
            finalized_names.append(clean_display_name(row["name"]))
    return finalized_names

def auto_advance_event_if_ready(conn, guild_id: int, tournament_row: dict, event_row: dict):
    notes = []
    if event_row["format_type"] != "head_to_head":
        return notes

    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(round_number), 0) AS current_round FROM tourney.matches WHERE guild_id = %s AND event_id = %s", (guild_id, event_row["id"]))
            current_round = cur.fetchone()["current_round"]
            if current_round == 0:
                return notes
            if not round_is_complete(conn, guild_id, event_row["id"], current_round):
                return notes

            finalized_byes = finalize_pending_byes_in_round(conn, guild_id, event_row["id"], current_round)
            if finalized_byes:
                conn.commit()
                notes.append("Automatic bye finalized: " + ", ".join(finalized_byes))

            cur.execute("""
                SELECT winner_character_id
                FROM tourney.matches
                WHERE guild_id = %s
                  AND event_id = %s
                  AND round_number = %s
                  AND status = 'completed'
                ORDER BY match_order, id
            """, (guild_id, event_row["id"], current_round))
            winners = [r["winner_character_id"] for r in cur.fetchall() if r["winner_character_id"]]

            if len(winners) <= 1:
                return notes

            next_round = current_round + 1
            cur.execute("SELECT COUNT(*) AS cnt FROM tourney.matches WHERE guild_id = %s AND event_id = %s AND round_number = %s", (guild_id, event_row["id"], next_round))
            if cur.fetchone()["cnt"] > 0:
                return notes

            winner_entries = []
            for character_id in winners:
                cur.execute("SELECT id AS entry_id FROM tourney.entries WHERE guild_id = %s AND event_id = %s AND character_id = %s LIMIT 1", (guild_id, event_row["id"], character_id))
                entry = cur.fetchone()
                if entry:
                    winner_entries.append({"character_id": character_id, "entry_id": entry["entry_id"]})

            if len(winner_entries) <= 1:
                return notes

            bye_name = None
            i, j, match_order = 0, len(winner_entries) - 1, 1
            while i < j:
                top = winner_entries[i]; bottom = winner_entries[j]
                cur.execute("""
                    INSERT INTO tourney.matches
                    (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, created_at)
                    VALUES (%s, %s, %s, %s, %s, 'pending', 'head_to_head', NOW())
                    RETURNING id
                """, (event_row["id"], tournament_row["id"], guild_id, next_round, match_order))
                match_id = cur.fetchone()["id"]
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 1, FALSE)", (match_id, top["entry_id"], top["character_id"]))
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 2, FALSE)", (match_id, bottom["entry_id"], bottom["character_id"]))
                match_order += 1
                i += 1
                j -= 1

            if i == j:
                bye_entry = winner_entries[i]
                cur.execute("""
                    INSERT INTO tourney.matches
                    (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, narrative_summary, created_at)
                    VALUES (%s, %s, %s, %s, %s, 'pending', 'bye', %s, NOW())
                    RETURNING id
                """, (
                    event_row["id"], tournament_row["id"], guild_id, next_round, match_order,
                    "Pending automatic bye advancement."
                ))
                bye_match_id = cur.fetchone()["id"]
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 1, FALSE)", (bye_match_id, bye_entry["entry_id"], bye_entry["character_id"]))
                cur.execute("SELECT name FROM characters WHERE character_id = %s LIMIT 1", (bye_entry["character_id"],))
                row = cur.fetchone()
                bye_name = clean_display_name(row["name"]) if row else None

            cur.execute("UPDATE tourney.events SET round_number = %s, updated_at = NOW() WHERE id = %s", (next_round, event_row["id"]))
            conn.commit()

            note = f"Round {next_round} was generated automatically."
            if bye_name:
                note += f" Automatic bye: {bye_name} advances."
            notes.append(note)

async def character_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM characters WHERE guild_id = %s AND archived = FALSE AND name ILIKE %s ORDER BY name LIMIT 20", (interaction.guild.id, f"%{current}%"))
            rows = cur.fetchall()
    return [app_commands.Choice(name=clean_display_name(row["name"]), value=clean_display_name(row["name"])) for row in rows]

async def tournament_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM tourney.tournaments WHERE guild_id = %s AND status <> 'completed' AND name ILIKE %s ORDER BY id DESC LIMIT 20", (interaction.guild.id, f"%{current}%"))
            rows = cur.fetchall()
    return [app_commands.Choice(name=clean_display_name(row["name"]), value=clean_display_name(row["name"])) for row in rows]

async def event_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    tournament_name = getattr(interaction.namespace, "tournament", None)
    with get_db() as conn:
        tournament = get_tournament_by_name(conn, interaction.guild.id, tournament_name) if tournament_name else None
        with conn.cursor() as cur:
            if tournament:
                cur.execute("SELECT name FROM tourney.events WHERE guild_id = %s AND tournament_id = %s AND name ILIKE %s ORDER BY id DESC LIMIT 20", (interaction.guild.id, tournament["id"], f"%{current}%"))
            else:
                cur.execute("SELECT name FROM tourney.events WHERE guild_id = %s AND name ILIKE %s ORDER BY id DESC LIMIT 20", (interaction.guild.id, f"%{current}%"))
            rows = cur.fetchall()
    return [app_commands.Choice(name=clean_display_name(row["name"]), value=clean_display_name(row["name"])) for row in rows]

async def pending_match_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[int]]:
    tournament_name = getattr(interaction.namespace, "tournament", None)
    event_name = getattr(interaction.namespace, "event", None)
    if not tournament_name or not event_name:
        return []

    with get_db() as conn:
        tournament = get_tournament_by_name(conn, interaction.guild.id, tournament_name)
        if not tournament:
            return []
        event = get_event_by_name(conn, interaction.guild.id, tournament["id"], event_name)
        if not event:
            return []

        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    m.id AS match_id,
                    m.round_number,
                    m.status,
                    MAX(CASE WHEN mp.slot_number = 1 THEN c.name END) AS slot_1_name,
                    MAX(CASE WHEN mp.slot_number = 2 THEN c.name END) AS slot_2_name
                FROM tourney.matches m
                JOIN tourney.match_participants mp ON m.id = mp.match_id
                JOIN characters c ON mp.character_id = c.character_id
                WHERE m.guild_id = %s
                  AND m.event_id = %s
                  AND m.status IN ('pending', 'active')
                  AND m.match_type <> 'bye'
                GROUP BY m.id, m.round_number, m.status
                ORDER BY m.round_number, m.match_order, m.id
                LIMIT 25
            """, (interaction.guild.id, event["id"]))
            rows = cur.fetchall()

    current_lower = current.lower()
    choices = []
    for row in rows:
        label = f"{row['match_id']} - {clean_display_name(row['slot_1_name'])} vs {clean_display_name(row['slot_2_name'])} (R{row['round_number']}, {row['status']})"
        if current_lower and current_lower not in label.lower():
            continue
        if len(label) > 100:
            label = label[:97] + "..."
        choices.append(app_commands.Choice(name=label, value=row["match_id"]))
    return choices[:25]

async def all_match_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[int]]:
    tournament_name = getattr(interaction.namespace, "tournament", None)
    event_name = getattr(interaction.namespace, "event", None)
    if not tournament_name or not event_name:
        return []

    with get_db() as conn:
        tournament = get_tournament_by_name(conn, interaction.guild.id, tournament_name)
        if not tournament:
            return []
        event = get_event_by_name(conn, interaction.guild.id, tournament["id"], event_name)
        if not event:
            return []

        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    m.id AS match_id,
                    m.round_number,
                    m.status,
                    MAX(CASE WHEN mp.slot_number = 1 THEN c.name END) AS slot_1_name,
                    MAX(CASE WHEN mp.slot_number = 2 THEN c.name END) AS slot_2_name
                FROM tourney.matches m
                JOIN tourney.match_participants mp ON m.id = mp.match_id
                JOIN characters c ON mp.character_id = c.character_id
                WHERE m.guild_id = %s
                  AND m.event_id = %s
                GROUP BY m.id, m.round_number, m.status
                ORDER BY m.round_number, m.match_order, m.id
                LIMIT 25
            """, (interaction.guild.id, event["id"]))
            rows = cur.fetchall()

    current_lower = current.lower()
    choices = []
    for row in rows:
        slot_2 = clean_display_name(row["slot_2_name"]) if row["slot_2_name"] else "BYE"
        label = f"{row['match_id']} - {clean_display_name(row['slot_1_name'])} vs {slot_2} (R{row['round_number']}, {row['status']})"
        if current_lower and current_lower not in label.lower():
            continue
        if len(label) > 100:
            label = label[:97] + "..."
        choices.append(app_commands.Choice(name=label, value=row["match_id"]))
    return choices[:25]

WIPE_MODES = [
    app_commands.Choice(name="Full Reset (wipe everything)", value="full_reset"),
    app_commands.Choice(name="Keep Profiles (wipe tourneys, xp, awards, records)", value="keep_profiles"),
]
EVENT_TYPES = [
    app_commands.Choice(name="Joust", value="joust"),
    app_commands.Choice(name="Archery", value="archery"),
    app_commands.Choice(name="Grand Melee", value="grand_melee"),
    app_commands.Choice(name="Duel", value="duel"),
    app_commands.Choice(name="Horse Race", value="horse_race"),
    app_commands.Choice(name="Hunt", value="hunt"),
]
FORMAT_TYPES = [
    app_commands.Choice(name="Head to Head", value="head_to_head"),
    app_commands.Choice(name="Free for All", value="free_for_all"),
    app_commands.Choice(name="Scored Round", value="scored_round"),
]



CORE_PROFILE_FIELDS = ["strength", "agility", "precision", "endurance", "presence"]


COMMAND_REFERENCE = [
    {"name": "tourney-commands", "audience": "all", "summary": "Show the command list you can use right now."},
    {"name": "tourney-profile-create", "audience": "all", "summary": "Open the interactive profile builder for one character you own, or any character if you are staff."},
    {"name": "tourney-profile-view", "audience": "all", "summary": "View any tournament profile, including skill tracks and laurels."},
    {"name": "tourney-profile-update", "audience": "staff", "summary": "Edit the five core tournament stats on a profile."},
    {"name": "tourney-profile-lock", "audience": "staff", "summary": "Lock a profile so it can no longer be edited."},
    {"name": "tourney-profile-retire", "audience": "staff", "summary": "Retire a tournament profile from active use."},
    {"name": "tourney-create", "audience": "staff", "summary": "Create a new tournament shell."},
    {"name": "tourney-list", "audience": "staff", "summary": "List tournaments in this server."},
    {"name": "tourney-view", "audience": "staff", "summary": "View one tournament's core details."},
    {"name": "tourney-status", "audience": "staff", "summary": "Show the staff status dashboard for a tournament."},
    {"name": "tourney-event-status", "audience": "staff", "summary": "Show the operational status of one event."},
    {"name": "tourney-post-announcement", "audience": "staff", "summary": "Post the public tournament announcement in the current channel."},
    {"name": "tourney-event-add", "audience": "staff", "summary": "Add a new event to a tournament."},
    {"name": "tourney-event-entrants", "audience": "staff", "summary": "List the current entrants for one event."},
    {"name": "tourney-register-multi", "audience": "all", "summary": "Open the multi-event registration picker for one character you own, or any character if you are staff."},
    {"name": "tourney-register-all", "audience": "all", "summary": "Register one character you own for every open event in a tournament, or any character if you are staff."},
    {"name": "tourney-event-withdraw", "audience": "all", "summary": "Withdraw one owned character from an event before registration closes, or withdraw any character if you are staff."},
    {"name": "tourney-event-generate-bracket", "audience": "staff", "summary": "Seed entrants and build or rewrite a first-round bracket."},
    {"name": "tourney-rebuild-standings", "audience": "staff", "summary": "Rebuild or relink the public standings post from DB truth."},
    {"name": "tourney-diagnose-event", "audience": "staff", "summary": "Run a deeper diagnostic report on one event."},
    {"name": "event-description", "audience": "staff", "summary": "Post the public in-world description for one event."},
    {"name": "tourney-post-standings", "audience": "staff", "summary": "Post the opening call and linked public standings board."},
    {"name": "tourney-event-run-round", "audience": "staff", "summary": "Run the next round ephemerally for staff review."},
    {"name": "tourney-event-run-round-public", "audience": "staff", "summary": "Run the next round publicly with call-up text and results."},
    {"name": "tourney-event-advance-round", "audience": "staff", "summary": "Advance a head-to-head event from completed matches."},
    {"name": "tourney-event-finalize", "audience": "staff", "summary": "Finalize one event, apply XP, and mark champion and runner-up."},
    {"name": "tourney-finalize", "audience": "staff", "summary": "Finalize the whole tournament and issue payout and recap posts."},
    {"name": "tourney-admin-wipe", "audience": "staff", "summary": "Dangerous reset command that wipes tournament data for testing."},
]

def build_command_reference_embeds(is_admin: bool) -> list[discord.Embed]:
    user_commands = [row for row in COMMAND_REFERENCE if row["audience"] == "all"]
    staff_commands = [row for row in COMMAND_REFERENCE if row["audience"] == "staff"]

    embeds: list[discord.Embed] = []
    lead = discord.Embed(
        title="Tournament Bot — Command Reference",
        color=discord.Color.blurple(),
        description=(
            "This list is generated from the current bot code and grouped by who can run each command. "
            "Summaries describe what each command actually does in this build."
        ),
    )

    def add_command_fields(target_embed: discord.Embed, section_name: str, rows: list[dict]):
        lines = [f"• **/{row['name']}** — {row['summary']}" for row in rows]
        for idx_chunk, chunk in enumerate(chunk_lines_for_embed(lines, max_len=950), start=1):
            if len(target_embed.fields) >= 5:
                target_embed = discord.Embed(
                    title="Tournament Bot — Command Reference",
                    description="Command list continued.",
                    color=discord.Color.blurple(),
                )
                embeds.append(target_embed)
            target_embed.add_field(
                name=section_name if idx_chunk == 1 else f"{section_name} ({idx_chunk})",
                value=chunk,
                inline=False,
            )
        return target_embed

    if is_admin:
        lead.add_field(
            name="Who can see what",
            value="You are recognized as staff, so this list includes both user-accessible and staff-only commands.",
            inline=False,
        )
        current = add_command_fields(lead, "Available to All Users", user_commands)
        if current is not embeds[-1] if embeds else False:
            pass
        current = add_command_fields(current, "Staff-Only Commands", staff_commands)
    else:
        lead.add_field(
            name="Who can see what",
            value="You are not recognized as staff, so this list only shows commands available to you in the current build.",
            inline=False,
        )
        current = add_command_fields(lead, "Available to You", user_commands)

    if not embeds:
        embeds.append(lead)
    elif embeds[0] is not lead:
        embeds.insert(0, lead)

    return sanitize_embeds_for_send(embeds)


CORE_SKILL_EXPLANATIONS = {
    "strength": "Power, force, impact, and the ability to drive through resistance.",
    "agility": "Speed, balance, coordination, and control of body and weapon.",
    "precision": "Accuracy, timing, aim, and exact execution under pressure.",
    "endurance": "Stamina, resilience, grit, and the ability to keep performing deep into a contest.",
    "presence": "Composure, confidence, poise, and command of self beneath watchful eyes.",
}

def fetch_character_for_profile_creation(conn, guild_id: int, character_name: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.character_id, c.user_id, c.name
            FROM characters c
            WHERE c.guild_id = %s
              AND c.archived = FALSE
              AND c.name = %s
            LIMIT 1
            """,
            (guild_id, character_name),
        )
        return cur.fetchone()

def profile_already_exists(conn, guild_id: int, character_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM tourney.character_profiles WHERE guild_id = %s AND character_id = %s LIMIT 1",
            (guild_id, character_id),
        )
        return cur.fetchone() is not None



def validate_core_profile_stats(stats: dict) -> tuple[bool, str]:
    if not isinstance(stats, dict):
        return False, "Profile stats were not provided in a valid format."

    total = 0
    for field in CORE_PROFILE_FIELDS:
        value = stats.get(field)
        if not isinstance(value, int):
            return False, f"{field.title()} must be selected before the profile can be created."
        if value < 0 or value > 5:
            return False, f"{field.title()} must be between 0 and 5."
        total += value

    if total != 14:
        return False, "Core attributes must total exactly 14 points."

    return True, ""

def create_tournament_profile_record(conn, guild_id: int, character_row: dict, actor_user_id: int, stats: dict):
    valid, error_text = validate_core_profile_stats(stats)
    if not valid:
        raise ValueError(error_text)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tourney.character_profiles
            (guild_id, character_id, user_id, strength, agility, precision, endurance, presence, status, created_by_user_id, updated_by_user_id, approved_by_user_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'approved', %s, %s, %s, NOW(), NOW())
            """,
            (
                guild_id,
                character_row["character_id"],
                character_row["user_id"],
                stats["strength"],
                stats["agility"],
                stats["precision"],
                stats["endurance"],
                stats["presence"],
                actor_user_id,
                actor_user_id,
                actor_user_id,
            ),
        )
        for skill_code in ["riding", "weapon", "archery", "duel", "stamina", "composure"]:
            cur.execute(
                """
                INSERT INTO tourney.character_skill_xp
                (guild_id, character_id, skill_code, xp_total, rank_bonus, updated_at)
                VALUES (%s, %s, %s, 0, 0, NOW())
                ON CONFLICT DO NOTHING
                """,
                (guild_id, character_row["character_id"], skill_code),
            )
    conn.commit()

def build_profile_creation_embed(character_name: str, stats: dict[str, int | None], status_text: str | None = None) -> discord.Embed:
    clean_name = clean_display_name(character_name)
    total = sum(v for v in stats.values() if isinstance(v, int))
    all_selected = all(isinstance(stats[field], int) for field in CORE_PROFILE_FIELDS)
    remaining = 14 - total

    embed = discord.Embed(
        title=f"{clean_name} — Tournament Profile Registration",
        color=discord.Color.gold(),
        description=(
            "**You have 14 points to distribute across five core attributes.**\n"
            "Use the **stat buttons** below to choose an attribute, then use the **value selector** to set it from **0–5**. Your total must equal **14** exactly before the profile can be finalized.\n\n"
            "**Derived tournament skills**\n"
            "• Riding = Agility + Endurance\n"
            "• Weapon = Strength + Agility\n"
            "• Archery = Precision + Agility\n"
            "• Duel = Strength + Agility + Precision\n"
            "• Stamina = Endurance + Strength\n"
            "• Composure = Presence + Endurance"
        ),
    )

    stat_lines = []
    for field in CORE_PROFILE_FIELDS:
        value = stats[field]
        shown = str(value) if isinstance(value, int) else "—"
        stat_lines.append(f"**{field.title()}**: {shown} — {CORE_SKILL_EXPLANATIONS[field]}")
    for idx_chunk, chunk in enumerate(chunk_lines_for_embed(stat_lines, max_len=1000), start=1):
        embed.add_field(name="Core Attributes" if idx_chunk == 1 else f"Core Attributes ({idx_chunk})", value=chunk, inline=False)

    if all_selected:
        derived = {
            "Riding": stats["agility"] + stats["endurance"],
            "Weapon": stats["strength"] + stats["agility"],
            "Archery": stats["precision"] + stats["agility"],
            "Duel": stats["strength"] + stats["agility"] + stats["precision"],
            "Stamina": stats["endurance"] + stats["strength"],
            "Composure": stats["presence"] + stats["endurance"],
        }
        derived_lines = [f"**{name}**: {value}" for name, value in derived.items()]
        embed.add_field(name="Derived Tournament Skills", value="\n".join(derived_lines), inline=False)
    else:
        embed.add_field(name="Derived Tournament Skills", value="Select all five attributes to preview the derived tournament skills.", inline=False)

    if all_selected and total == 14:
        summary = "**Total:** 14 / 14\n**State:** Ready — review your final selections, then press **Confirm Profile** to create it."
    elif all_selected:
        summary = f"**Total:** {total} / 14\n**Remaining:** {remaining:+d}\n**State:** Not ready — adjust the values until the total is exactly 14."
    else:
        summary = f"**Total:** {total} / 14\n**Remaining:** {remaining:+d}\n**State:** Continue selecting values."
    embed.add_field(name="Point Summary", value=summary, inline=False)

    if status_text:
        embed.add_field(name="Status", value=status_text, inline=False)

    return embed


class TournamentProfileStatSelect(discord.ui.Select):
    def __init__(self, field_name: str, current_value: int | None = None):
        options = [
            discord.SelectOption(label=str(i), value=str(i), default=(current_value == i))
            for i in range(0, 6)
        ]
        super().__init__(
            placeholder=f"Set {field_name.title()}",
            min_values=1,
            max_values=1,
            options=options,
            row=1,
        )
        self.field_name = field_name

    async def callback(self, interaction: discord.Interaction):
        view = self.view
        if not isinstance(view, TournamentProfileCreateView):
            await interaction.response.send_message("Profile builder state was invalid.", ephemeral=True)
            return
        await view.apply_selection(interaction, self.field_name, int(self.values[0]))


class TournamentProfileCreateView(discord.ui.View):
    def __init__(self, owner_id: int, guild_id: int, character_row: dict):
        super().__init__(timeout=900)
        self.owner_id = owner_id
        self.guild_id = guild_id
        self.character_row = character_row
        self.stats = {field: None for field in CORE_PROFILE_FIELDS}
        self.profile_created = False
        self.cancelled = False
        self.active_field = CORE_PROFILE_FIELDS[0]
        self.refresh_components()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the user who opened this profile builder can use it.", ephemeral=True)
            return False
        return True

    def is_ready_to_confirm(self) -> bool:
        all_selected = all(isinstance(self.stats[field], int) for field in CORE_PROFILE_FIELDS)
        total = sum(v for v in self.stats.values() if isinstance(v, int))
        if not all_selected or total != 14:
            return False
        valid, _ = validate_core_profile_stats(self.stats)
        return valid

    def refresh_components(self):
        self.clear_items()

        for field in CORE_PROFILE_FIELDS:
            value = self.stats[field]
            shown = str(value) if isinstance(value, int) else "—"
            style = discord.ButtonStyle.primary if field == self.active_field else discord.ButtonStyle.secondary
            button = discord.ui.Button(
                label=f"{field.title()}: {shown}",
                style=style,
                row=0,
            )

            async def stat_callback(interaction: discord.Interaction, chosen_field=field):
                if self.profile_created or self.cancelled:
                    await interaction.response.send_message("This profile builder is no longer active.", ephemeral=True)
                    return
                self.active_field = chosen_field
                self.refresh_components()
                embed = build_profile_creation_embed(
                    self.character_row["name"],
                    self.stats,
                    status_text=f"Now editing **{chosen_field.title()}**. Choose a value from the selector below.",
                )
                await interaction.response.edit_message(embed=embed, view=self)

            button.callback = stat_callback
            self.add_item(button)

        self.add_item(TournamentProfileStatSelect(self.active_field, self.stats[self.active_field]))

        confirm_button = discord.ui.Button(
            label="Confirm Profile",
            style=discord.ButtonStyle.success,
            row=2,
            disabled=(self.profile_created or self.cancelled or not self.is_ready_to_confirm()),
        )

        cancel_button = discord.ui.Button(
            label="Cancel",
            style=discord.ButtonStyle.danger,
            row=2,
            disabled=self.profile_created or self.cancelled,
        )

        async def confirm_callback(interaction: discord.Interaction):
            if self.profile_created:
                await interaction.response.send_message("This profile has already been created.", ephemeral=True)
                return
            if self.cancelled:
                await interaction.response.send_message("This profile builder was cancelled.", ephemeral=True)
                return

            valid, error_text = validate_core_profile_stats(self.stats)
            if not valid:
                embed = build_profile_creation_embed(
                    self.character_row["name"],
                    self.stats,
                    status_text=error_text,
                )
                self.refresh_components()
                await interaction.response.edit_message(embed=embed, view=self)
                return

            with get_db() as conn:
                if profile_already_exists(conn, self.guild_id, self.character_row["character_id"]):
                    self.profile_created = True
                    self.refresh_components()
                    for child in self.children:
                        child.disabled = True
                    embed = build_profile_creation_embed(
                        self.character_row["name"],
                        self.stats,
                        status_text="A tournament profile already exists for this character. No new profile was created.",
                    )
                    await interaction.response.edit_message(embed=embed, view=self)
                    self.stop()
                    return

                try:
                    create_tournament_profile_record(conn, self.guild_id, self.character_row, interaction.user.id, self.stats)
                except ValueError as exc:
                    embed = build_profile_creation_embed(
                        self.character_row["name"],
                        self.stats,
                        status_text=str(exc),
                    )
                    self.refresh_components()
                    await interaction.response.edit_message(embed=embed, view=self)
                    return

            self.profile_created = True
            self.refresh_components()
            for child in self.children:
                child.disabled = True

            embed = build_profile_creation_embed(
                self.character_row["name"],
                self.stats,
                status_text="Tournament profile created successfully.",
            )
            await interaction.response.edit_message(embed=embed, view=self)
            self.stop()

        async def cancel_callback(interaction: discord.Interaction):
            if self.profile_created:
                await interaction.response.send_message("This profile has already been created.", ephemeral=True)
                return
            self.cancelled = True
            self.refresh_components()
            for child in self.children:
                child.disabled = True
            embed = build_profile_creation_embed(
                self.character_row["name"],
                self.stats,
                status_text="Tournament profile creation cancelled.",
            )
            await interaction.response.edit_message(embed=embed, view=self)
            self.stop()

        confirm_button.callback = confirm_callback
        cancel_button.callback = cancel_callback
        self.add_item(confirm_button)
        self.add_item(cancel_button)

    async def apply_selection(self, interaction: discord.Interaction, field_name: str, value: int):
        if self.profile_created:
            await interaction.response.send_message("This profile has already been created.", ephemeral=True)
            return
        if self.cancelled:
            await interaction.response.send_message("This profile builder was cancelled.", ephemeral=True)
            return

        self.stats[field_name] = value
        self.active_field = field_name
        self.refresh_components()

        total = sum(v for v in self.stats.values() if isinstance(v, int))
        all_selected = all(isinstance(self.stats[field], int) for field in CORE_PROFILE_FIELDS)

        if not all_selected:
            embed = build_profile_creation_embed(
                self.character_row["name"],
                self.stats,
                status_text=f"Saved **{field_name.title()} = {value}**. Continue assigning the remaining attributes.",
            )
            await interaction.response.edit_message(embed=embed, view=self)
            return

        if total != 14:
            embed = build_profile_creation_embed(
                self.character_row["name"],
                self.stats,
                status_text="Your selections are saved in the builder, but the total must equal **14** before the profile can be confirmed.",
            )
            await interaction.response.edit_message(embed=embed, view=self)
            return

        valid, error_text = validate_core_profile_stats(self.stats)
        if not valid:
            embed = build_profile_creation_embed(
                self.character_row["name"],
                self.stats,
                status_text=error_text,
            )
            await interaction.response.edit_message(embed=embed, view=self)
            return

        embed = build_profile_creation_embed(
            self.character_row["name"],
            self.stats,
            status_text="Your selections are valid. Review them, then press **Confirm Profile** to finalize this tournament profile.",
        )
        await interaction.response.edit_message(embed=embed, view=self)


@tree.command(name="tourney-profile-create", description="Open the interactive tournament profile builder for a character.")
@app_commands.autocomplete(character=character_autocomplete)
async def create_profile(interaction: discord.Interaction, character: str):
    with get_db() as conn:
        character_row = fetch_character_for_profile_creation(conn, interaction.guild.id, character)
        if not character_row:
            await interaction.response.send_message("Character not found.", ephemeral=True); return

        actor_is_admin = user_is_admin(interaction)
        if not actor_is_admin and character_row["user_id"] != interaction.user.id:
            await interaction.response.send_message("You may only create a tournament profile for a character you own.", ephemeral=True); return

        if profile_already_exists(conn, interaction.guild.id, character_row["character_id"]):
            await interaction.response.send_message("A tournament profile already exists for that character.", ephemeral=True); return

    view = TournamentProfileCreateView(interaction.user.id, interaction.guild.id, character_row)
    embed = build_profile_creation_embed(character_row["name"], view.stats)
    await interaction.response.send_message(embed=embed, view=view, ephemeral=True)

@tree.command(name="tourney-profile-view", description="View a tournament profile.")
@app_commands.autocomplete(character=character_autocomplete)
async def view_profile(interaction: discord.Interaction, character: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT p.*, c.name FROM tourney.character_profiles p JOIN characters c ON p.character_id = c.character_id WHERE p.guild_id = %s AND c.name = %s LIMIT 1",
                (interaction.guild.id, character),
            )
            p = cur.fetchone()
            if not p:
                await interaction.response.send_message("No profile exists.", ephemeral=True); return

            cur.execute(
                "SELECT skill_code, xp_total, rank_bonus FROM tourney.character_skill_xp WHERE guild_id = %s AND character_id = %s ORDER BY skill_code",
                (interaction.guild.id, p["character_id"]),
            )
            skills = cur.fetchall()

            cur.execute(
                """
                SELECT
                    a.award_name,
                    a.award_code,
                    e.event_type,
                    e.name AS event_name,
                    t.name AS tournament_name,
                    a.awarded_at
                FROM tourney.awards a
                LEFT JOIN tourney.events e ON a.event_id = e.id
                LEFT JOIN tourney.tournaments t ON a.tournament_id = t.id
                WHERE a.guild_id = %s
                  AND a.character_id = %s
                  AND a.award_code = 'champion'
                ORDER BY a.awarded_at DESC, a.id DESC
                """,
                (interaction.guild.id, p["character_id"]),
            )
            laurels = cur.fetchall()

    embed = discord.Embed(title=f"{clean_display_name(character)} — Tournament Profile", color=discord.Color.gold())
    for field in ["strength", "agility", "precision", "endurance", "presence"]:
        embed.add_field(name=field.title(), value=p[field])

    embed.add_field(name="Status", value=p["status"], inline=False)

    if skills:
        embed.add_field(
            name="Skill Tracks",
            value="\n".join([f"**{r['skill_code'].title()}** — XP {r['xp_total']} | Bonus +{r['rank_bonus']}" for r in skills]),
            inline=False,
        )

    if laurels:
        laurel_lines = []
        for row in laurels:
            event_label = EVENT_LABELS.get(row["event_type"], clean_display_name(row["event_name"] or "Unknown Event"))
            tournament_name = clean_display_name(row["tournament_name"] or "Unknown Tournament")
            laurel_lines.append(f"Champion of the {event_label} at {tournament_name}")

        for idx_chunk, chunk in enumerate(chunk_lines_for_embed(laurel_lines, max_len=1000), start=1):
            embed.add_field(name="Laurels" if idx_chunk == 1 else f"Laurels ({idx_chunk})", value=chunk, inline=False)

    await interaction.response.send_message(embed=embed, ephemeral=True)

@tree.command(name="tourney-profile-update", description="Update a tournament profile.")
@app_commands.autocomplete(character=character_autocomplete)
async def update_profile(interaction: discord.Interaction, character: str, strength: int, agility: int, precision: int, endurance: int, presence: int):
    if await deny_if_not_admin(interaction): return
    proposed_stats = {
        "strength": strength,
        "agility": agility,
        "precision": precision,
        "endurance": endurance,
        "presence": presence,
    }
    valid, error_text = validate_core_profile_stats(proposed_stats)
    if not valid:
        await interaction.response.send_message(error_text, ephemeral=True); return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE tourney.character_profiles p SET strength = %s, agility = %s, precision = %s, endurance = %s, presence = %s, updated_by_user_id = %s, updated_at = NOW() FROM characters c WHERE p.guild_id = %s AND c.guild_id = %s AND c.name = %s AND p.character_id = c.character_id", (strength, agility, precision, endurance, presence, interaction.user.id, interaction.guild.id, interaction.guild.id, character))
            updated = cur.rowcount
        conn.commit()
    if updated == 0: await interaction.response.send_message("No profile found to update.", ephemeral=True); return
    await interaction.response.send_message(f"Profile updated for **{clean_display_name(character)}**.", ephemeral=True)

@tree.command(name="tourney-profile-lock", description="Lock a tournament profile.")
@app_commands.autocomplete(character=character_autocomplete)
async def lock_profile(interaction: discord.Interaction, character: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE tourney.character_profiles p SET status = 'locked', locked_at = NOW(), updated_by_user_id = %s, updated_at = NOW() FROM characters c WHERE p.guild_id = %s AND c.guild_id = %s AND c.name = %s AND p.character_id = c.character_id", (interaction.user.id, interaction.guild.id, interaction.guild.id, character))
            updated = cur.rowcount
        conn.commit()
    if updated == 0: await interaction.response.send_message("No profile found to lock.", ephemeral=True); return
    await interaction.response.send_message(f"Profile locked for **{clean_display_name(character)}**.", ephemeral=True)

@tree.command(name="tourney-profile-retire", description="Retire a tournament profile.")
@app_commands.autocomplete(character=character_autocomplete)
async def retire_profile(interaction: discord.Interaction, character: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE tourney.character_profiles p SET status = 'retired', retired_at = NOW(), updated_by_user_id = %s, updated_at = NOW() FROM characters c WHERE p.guild_id = %s AND c.guild_id = %s AND c.name = %s AND p.character_id = c.character_id", (interaction.user.id, interaction.guild.id, interaction.guild.id, character))
            updated = cur.rowcount
        conn.commit()
    if updated == 0: await interaction.response.send_message("No profile found to retire.", ephemeral=True); return
    await interaction.response.send_message(f"Profile retired for **{clean_display_name(character)}**.", ephemeral=True)

@tree.command(name="tourney-create", description="Create a tournament.")
@app_commands.choices(host_location=KINGDOM_CHOICES, season_label=SEASON_CHOICES)
async def create_tournament(interaction: discord.Interaction, name: str, host_location: app_commands.Choice[str], season_label: app_commands.Choice[str], division: Optional[str] = "open", notes: Optional[str] = None):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO tourney.tournaments (guild_id, name, host_location, season_label, status, division, created_by_user_id, notes, created_at, updated_at) VALUES (%s, %s, %s, %s, 'draft', %s, %s, %s, NOW(), NOW()) RETURNING id", (interaction.guild.id, name, host_location.value, season_label.value, division or 'open', interaction.user.id, notes))
            row = cur.fetchone()
        conn.commit()
    await interaction.response.send_message(f"Tournament created: **{clean_display_name(name)}** (ID {row['id']}).", ephemeral=True)

@tree.command(name="tourney-list", description="List tournaments.")
async def list_tournaments(interaction: discord.Interaction):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, status, division, season_label FROM tourney.tournaments WHERE guild_id = %s ORDER BY id DESC LIMIT 25", (interaction.guild.id,))
            rows = cur.fetchall()
    if not rows: await interaction.response.send_message("No tournaments found.", ephemeral=True); return
    await interaction.response.send_message("\n".join([f"**{clean_display_name(r['name'])}** | ID {r['id']} | Status: `{r['status']}` | Division: `{r['division']}` | Season: `{r['season_label'] or '—'}`" for r in rows]), ephemeral=True)

@tree.command(name="tourney-view", description="View one tournament.")
@app_commands.autocomplete(name=tournament_autocomplete)
async def view_tournament(interaction: discord.Interaction, name: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        row = get_tournament_by_name(conn, interaction.guild.id, name)
    if not row: await interaction.response.send_message("Tournament not found.", ephemeral=True); return
    embed = discord.Embed(title=f"{clean_display_name(row['name'])} — Tournament", color=discord.Color.blue())
    embed.add_field(name="Status", value=row["status"])
    embed.add_field(name="Division", value=row["division"] or "open")
    embed.add_field(name="Season", value=row["season_label"] or "—")
    embed.add_field(name="Host Location", value=row["host_location"] or "—", inline=False)
    embed.add_field(name="Notes", value=row["notes"] or "—", inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)



@tree.command(name="tourney-status", description="Show a staff summary of a tournament's live status.")
@app_commands.autocomplete(name=tournament_autocomplete)
async def tournament_status(interaction: discord.Interaction, name: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, name)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM tourney.events WHERE guild_id = %s AND tournament_id = %s ORDER BY id ASC", (interaction.guild.id, t["id"]))
            event_rows = cur.fetchall()
        metrics_by_event = {row["id"]: get_event_live_metrics(conn, interaction.guild.id, row) for row in event_rows}
    embed = build_tournament_status_embed(t, event_rows, metrics_by_event)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@tree.command(name="tourney-event-status", description="Show a staff operational status view for one event.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def tournament_event_status(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e:
            await interaction.response.send_message("Event not found.", ephemeral=True); return
        metrics = get_event_live_metrics(conn, interaction.guild.id, e)
    embed = build_event_status_embed(t, e, metrics)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@tree.command(name="tourney-post-announcement", description="Post a public tournament announcement in the current channel.")
@app_commands.autocomplete(tournament=tournament_autocomplete)
async def post_announcement(interaction: discord.Interaction, tournament: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t: await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        with conn.cursor() as cur:
            cur.execute("SELECT name, event_type, status FROM tourney.events WHERE guild_id = %s AND tournament_id = %s ORDER BY id ASC", (interaction.guild.id, t["id"]))
            events = cur.fetchall()
    await interaction.response.send_message(embed=build_public_announcement_embed(t, events), ephemeral=False)


@tree.command(name="tourney-event-add", description="Add an event to a tournament.")
@app_commands.autocomplete(tournament=tournament_autocomplete)
@app_commands.choices(event_type=EVENT_TYPES)
async def add_event(interaction: discord.Interaction, tournament: str, name: str, event_type: app_commands.Choice[str], max_entrants: Optional[int] = None, min_entrants: Optional[int] = 2):
    if await deny_if_not_admin(interaction): return
    format_type = EVENT_FORMAT_MAP[event_type.value]
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t: await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        with conn.cursor() as cur:
            cur.execute("INSERT INTO tourney.events (tournament_id, guild_id, event_type, name, format_type, status, max_entrants, min_entrants, round_number, settings_json, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, 'draft', %s, %s, 0, NULL, NOW(), NOW()) RETURNING id", (t['id'], interaction.guild.id, event_type.value, name, format_type, max_entrants, min_entrants or 2))
            row = cur.fetchone()
        conn.commit()
    await interaction.response.send_message(f"Event created: **{clean_display_name(name)}** (ID {row['id']}) in **{clean_display_name(tournament)}** as `{format_type}`.", ephemeral=True)

@tree.command(name="tourney-event-entrants", description="List entrants for an event.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def event_entrants(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t: await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e: await interaction.response.send_message("Event not found.", ephemeral=True); return
        with conn.cursor() as cur:
            cur.execute("SELECT c.name, en.registration_status, en.seed FROM tourney.entries en JOIN characters c ON en.character_id = c.character_id WHERE en.guild_id = %s AND en.event_id = %s ORDER BY COALESCE(en.seed, 999999), c.name", (interaction.guild.id, e["id"]))
            rows = cur.fetchall()
    if not rows: await interaction.response.send_message("No entrants found for that event.", ephemeral=True); return
    await interaction.response.send_message("\n".join([f"**{clean_display_name(r['name'])}** | Status: `{r['registration_status']}` | Seed: `{r['seed'] if r['seed'] is not None else '—'}`" for r in rows]), ephemeral=True)




def fetch_character_registration_profile(conn, guild_id: int, character_name: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.character_id, c.user_id, c.name
            FROM characters c
            JOIN tourney.character_profiles p ON c.character_id = p.character_id
            WHERE c.guild_id = %s
              AND c.archived = FALSE
              AND c.name = %s
              AND p.guild_id = %s
              AND p.status IN ('approved', 'locked')
            LIMIT 1
            """,
            (guild_id, character_name, guild_id),
        )
        return cur.fetchone()



def event_has_completed_contested_matches_by_event_id(guild_id: int, event_id: int) -> bool:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM tourney.matches
                WHERE guild_id = %s
                  AND event_id = %s
                  AND match_type <> 'bye'
                  AND status = 'completed'
                """,
                (guild_id, event_id),
            )
            return cur.fetchone()["cnt"] > 0

def event_registration_locked(event_row: dict, guild_id: int | None = None) -> bool:
    if event_row.get("status") in ("ready_to_finalize", "completed"):
        return True

    if event_row.get("format_type") == "head_to_head":
        if guild_id is None:
            return False
        return event_has_completed_contested_matches_by_event_id(guild_id, event_row["id"])

    return (event_row.get("round_number") or 0) > 0

def register_character_for_event_core(cur, conn, guild_id: int, tournament_row: dict, event_row: dict, character_row: dict):
    if event_registration_locked(event_row, guild_id):
        return False, "registration_closed"
    if event_row["status"] == "completed":
        return False, "completed"

    cur.execute(
        "SELECT COUNT(*) AS cnt FROM tourney.entries WHERE guild_id = %s AND event_id = %s AND registration_status <> 'withdrawn'",
        (guild_id, event_row["id"]),
    )
    current_count = cur.fetchone()["cnt"]
    if event_row.get("max_entrants") and current_count >= event_row["max_entrants"]:
        return False, "full"

    cur.execute(
        "SELECT 1 FROM tourney.entries WHERE guild_id = %s AND event_id = %s AND character_id = %s",
        (guild_id, event_row["id"], character_row["character_id"]),
    )
    if cur.fetchone():
        return False, "already_registered"

    cur.execute(
        """
        INSERT INTO tourney.entries
        (event_id, tournament_id, guild_id, character_id, user_id, registration_status, fatigue_points, check_in_confirmed, entered_at)
        VALUES (%s, %s, %s, %s, %s, 'registered', 0, FALSE, NOW())
        """,
        (event_row["id"], tournament_row["id"], guild_id, character_row["character_id"], character_row["user_id"]),
    )

    if event_row["format_type"] == "head_to_head":
        auto_seed_event(cur, conn, guild_id, event_row)

    return True, "registered"

async def refresh_event_after_registration(interaction: discord.Interaction, tournament_row: dict, event_row: dict):
    refresh_note = ""
    _, msg = await refresh_public_standings_if_safe(interaction, tournament_row, event_row)
    if msg:
        refresh_note = msg

    settings = get_event_settings(event_row.get("settings_json"))
    board_channel_id, board_message_id = get_public_board_ids(settings)
    if board_channel_id and board_message_id:
        await upsert_public_standings_post(interaction, tournament_row["name"], event_row)
    return refresh_note


def fetch_open_events_for_tournament(conn, guild_id: int, tournament_id: int):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT *
            FROM tourney.events
            WHERE guild_id = %s
              AND tournament_id = %s
              AND status <> 'completed'
            ORDER BY id ASC
            """,
            (guild_id, tournament_id),
        )
        return cur.fetchall()

class TourneyRegisterMultiSelect(discord.ui.Select):
    def __init__(self, tournament_name: str, character_name: str, event_names: list[str]):
        options = [
            discord.SelectOption(label=clean_display_name(name), value=clean_display_name(name))
            for name in event_names[:25]
        ]
        super().__init__(
            placeholder="Choose one or more events",
            min_values=1,
            max_values=max(1, min(len(options), 25)),
            options=options,
        )
        self.tournament_name = tournament_name
        self.character_name = character_name

    async def callback(self, interaction: discord.Interaction):
        view = self.view
        if not isinstance(view, TourneyRegisterMultiView):
            await interaction.response.send_message("Registration view state was invalid.", ephemeral=True)
            return
        await view.process_selection(interaction, list(self.values))

class TourneyRegisterMultiView(discord.ui.View):
    def __init__(self, owner_id: int, tournament_name: str, character_name: str, event_names: list[str]):
        super().__init__(timeout=600)
        self.owner_id = owner_id
        self.tournament_name = tournament_name
        self.character_name = character_name
        self.event_names = event_names
        self.add_item(TourneyRegisterMultiSelect(tournament_name, character_name, event_names))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("Only the staff member who opened this menu can use it.", ephemeral=True)
            return False
        return True

    async def process_selection(self, interaction: discord.Interaction, selected_event_names: list[str]):
        with get_db() as conn:
            t = get_tournament_by_name(conn, interaction.guild.id, self.tournament_name)
            if not t:
                await interaction.response.send_message("Tournament not found.", ephemeral=True); return
            c = fetch_character_registration_profile(conn, interaction.guild.id, self.character_name)
            if not c:
                await interaction.response.send_message("Character not found or no approved tournament profile exists.", ephemeral=True); return

            results = []
            touched_events = []
            with conn.cursor() as cur:
                for event_name in selected_event_names:
                    e = get_event_by_name(conn, interaction.guild.id, t["id"], event_name)
                    if not e:
                        results.append((event_name, "not_found"))
                        continue
                    ok, status = register_character_for_event_core(cur, conn, interaction.guild.id, t, e, c)
                    results.append((e["name"], status))
                    if ok:
                        touched_events.append(e)
            conn.commit()

        lines = [f"Registration results for **{clean_display_name(self.character_name)}** in **{clean_display_name(self.tournament_name)}**:"]
        for event_name, status in results:
            lines.append(
                f"• **{clean_display_name(event_name)}** — "
                + (
                    "registered" if status == "registered" else
                    "registration closed after round one" if status == "registration_closed" else
                    "already registered" if status == "already_registered" else
                    "event is full" if status == "full" else
                    "event already completed" if status == "completed" else
                    "event not found"
                )
            )

        for child in self.children:
            child.disabled = True

        await interaction.response.edit_message(content="\n".join(lines), view=self)

        followups = []
        for e in touched_events:
            note = await refresh_event_after_registration(interaction, t, e)
            if note:
                followups.append(f"{clean_display_name(e['name'])}: {note}")
        if followups:
            await interaction.followup.send("\n".join(followups), ephemeral=True)

    async def on_timeout(self):
        for child in self.children:
            child.disabled = True




@tree.command(name="tourney-register-multi", description="Open an interactive event picker to register a character in multiple events.")
@app_commands.autocomplete(tournament=tournament_autocomplete, character=character_autocomplete)
async def register_character_multi(
    interaction: discord.Interaction,
    tournament: str,
    character: str,
):
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return

        c = fetch_character_registration_profile(conn, interaction.guild.id, character)
        if not c:
            await interaction.response.send_message("Character not found or no approved tournament profile exists.", ephemeral=True); return

        if not actor_can_use_character(interaction, c):
            await interaction.response.send_message("You may only register a character you own.", ephemeral=True); return

        open_events = fetch_open_events_for_tournament(conn, interaction.guild.id, t["id"])

    if not open_events:
        await interaction.response.send_message("That tournament has no open events.", ephemeral=True); return

    event_names = [row["name"] for row in open_events]
    view = TourneyRegisterMultiView(interaction.user.id, t["name"], c["name"], event_names)
    await interaction.response.send_message(
        "\n".join([
            f"Select events for **{clean_display_name(c['name'])}** in **{clean_display_name(t['name'])}**.",
            "This menu behaves like a multi-select picker. Choose one or more events, then submit through the menu itself.",
        ]),
        view=view,
        ephemeral=True,
    )

@tree.command(name="tourney-register-all", description="Register a character for every open event in a tournament.")
@app_commands.autocomplete(tournament=tournament_autocomplete, character=character_autocomplete)
async def register_character_all(interaction: discord.Interaction, tournament: str, character: str):
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return

        c = fetch_character_registration_profile(conn, interaction.guild.id, character)
        if not c:
            await interaction.response.send_message("Character not found or no approved tournament profile exists.", ephemeral=True); return

        if not actor_can_use_character(interaction, c):
            await interaction.response.send_message("You may only register a character you own.", ephemeral=True); return

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM tourney.events
                WHERE guild_id = %s
                  AND tournament_id = %s
                  AND status <> 'completed'
                ORDER BY id ASC
                """,
                (interaction.guild.id, t["id"]),
            )
            events = cur.fetchall()

            if not events:
                await interaction.response.send_message("That tournament has no open events.", ephemeral=True); return

            results = []
            touched_events = []
            for e in events:
                ok, status = register_character_for_event_core(cur, conn, interaction.guild.id, t, e, c)
                results.append((e["name"], status))
                if ok:
                    touched_events.append(e)
        conn.commit()

    await interaction.response.send_message(
        "\n".join(
            [f"Register-all results for **{clean_display_name(character)}** in **{clean_display_name(tournament)}**:"]
            + [
                f"• **{clean_display_name(event_name)}** — "
                + (
                    "registered" if status == "registered" else
                    "registration closed after round one" if status == "registration_closed" else
                    "already registered" if status == "already_registered" else
                    "event is full" if status == "full" else
                    "event already completed" if status == "completed" else
                    "event not found"
                )
                for event_name, status in results
            ]
        ),
        ephemeral=True
    )

    followups = []
    for e in touched_events:
        note = await refresh_event_after_registration(interaction, t, e)
        if note:
            followups.append(f"{clean_display_name(e['name'])}: {note}")
    if followups:
        await interaction.followup.send("\n".join(followups), ephemeral=True)

@tree.command(name="tourney-event-withdraw", description="Withdraw a character from an event.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete, character=character_autocomplete)
async def withdraw_character_from_event(interaction: discord.Interaction, tournament: str, event: str, character: str):
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t: await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e: await interaction.response.send_message("Event not found.", ephemeral=True); return

        character_row = fetch_character_for_profile_creation(conn, interaction.guild.id, character)
        if not character_row:
            await interaction.response.send_message("Character not found.", ephemeral=True); return
        if not actor_can_use_character(interaction, character_row):
            await interaction.response.send_message("You may only withdraw a character you own.", ephemeral=True); return

        if event_registration_locked(e, interaction.guild.id):
            await interaction.response.send_message("Registration for this event is closed because the first round has already been run.", ephemeral=True); return
        with conn.cursor() as cur:
            cur.execute("UPDATE tourney.entries en SET registration_status = 'withdrawn' FROM characters c WHERE en.guild_id = %s AND en.event_id = %s AND c.guild_id = %s AND c.name = %s AND en.character_id = c.character_id", (interaction.guild.id, e["id"], interaction.guild.id, character))
            updated = cur.rowcount
            if e["format_type"] == "head_to_head":
                auto_seed_event(cur, conn, interaction.guild.id, e)
        conn.commit()
    if updated == 0: await interaction.response.send_message("No matching event registration found.", ephemeral=True); return
    await interaction.response.send_message(f"**{clean_display_name(character)}** withdrawn from **{clean_display_name(event)}** in **{clean_display_name(tournament)}**.", ephemeral=True)
    _, msg = await refresh_public_standings_if_safe(interaction, t, e)
    if msg:
        await interaction.followup.send(msg, ephemeral=True)
    settings = get_event_settings(e.get("settings_json"))
    board_channel_id, board_message_id = get_public_board_ids(settings)
    if board_channel_id and board_message_id:
        await upsert_public_standings_post(interaction, t["name"], e)

@tree.command(name="tourney-event-generate-bracket", description="Auto-seed entrants and generate or rewrite a first-round bracket for an event.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def generate_bracket(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t: await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e: await interaction.response.send_message("Event not found.", ephemeral=True); return
        if e["format_type"] != "head_to_head":
            await interaction.response.send_message(f"{EVENT_LABELS.get(e['event_type'], e['event_type'])} events do not use brackets. Use `/tourney-event-run-round` for scored events or the appropriate event runner instead.", ephemeral=True); return
        if e["status"] == "completed":
            await interaction.response.send_message("This event is already completed. Its bracket cannot be regenerated.", ephemeral=True); return

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM tourney.matches WHERE guild_id = %s AND event_id = %s AND status = 'completed' AND match_type <> 'bye'", (interaction.guild.id, e["id"]))
            completed_match_count = cur.fetchone()["cnt"]
            if completed_match_count > 0:
                await interaction.response.send_message("This event already has completed matches. To protect recorded results, the bracket cannot be rewritten now.", ephemeral=True); return

            cur.execute("""
                UPDATE tourney.entries
                SET registration_status = 'registered'
                WHERE guild_id = %s
                  AND event_id = %s
                  AND registration_status IN ('advanced', 'pending')
            """, (interaction.guild.id, e["id"]))

            cur.execute("SELECT id FROM tourney.matches WHERE guild_id = %s AND event_id = %s", (interaction.guild.id, e["id"]))
            existing_match_ids = [row["id"] for row in cur.fetchall()]
            if existing_match_ids:
                cur.execute("DELETE FROM tourney.match_rolls WHERE match_id = ANY(%s)", (existing_match_ids,))
                cur.execute("DELETE FROM tourney.match_participants WHERE match_id = ANY(%s)", (existing_match_ids,))
                cur.execute("DELETE FROM tourney.matches WHERE id = ANY(%s)", (existing_match_ids,))

            scored = auto_seed_event(cur, conn, interaction.guild.id, e)
            cur.execute("""
                SELECT id AS entry_id, character_id, seed
                FROM tourney.entries
                WHERE guild_id = %s
                  AND event_id = %s
                  AND registration_status = 'registered'
                ORDER BY seed ASC NULLS LAST, id ASC
            """, (interaction.guild.id, e["id"]))
            seeded_entries = cur.fetchall()
            if len(seeded_entries) < 2:
                await interaction.response.send_message("At least two registered entrants are required to generate a bracket.", ephemeral=True); return

            created_match_ids = []
            bye_name = None
            pairings, bye_entry = build_first_round_pairings(cur, interaction.guild.id, e["id"])
            match_order = 1
            for top, bottom in pairings:
                cur.execute("""
                    INSERT INTO tourney.matches
                    (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, created_at)
                    VALUES (%s, %s, %s, 1, %s, 'pending', 'head_to_head', NOW())
                    RETURNING id
                """, (e["id"], t["id"], interaction.guild.id, match_order))
                match_id = cur.fetchone()["id"]
                created_match_ids.append(match_id)
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 1, FALSE)", (match_id, top["entry_id"], top["character_id"]))
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 2, FALSE)", (match_id, bottom["entry_id"], bottom["character_id"]))
                match_order += 1

            if bye_entry:
                cur.execute("""
                    INSERT INTO tourney.matches
                    (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, narrative_summary, created_at)
                    VALUES (%s, %s, %s, 1, %s, 'pending', 'bye', %s, NOW())
                    RETURNING id
                """, (
                    e["id"], t["id"], interaction.guild.id, match_order,
                    "Pending automatic bye advancement."
                ))
                bye_match_id = cur.fetchone()["id"]
                created_match_ids.append(bye_match_id)
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 1, FALSE)", (bye_match_id, bye_entry["entry_id"], bye_entry["character_id"]))
                cur.execute("SELECT name FROM characters WHERE character_id = %s LIMIT 1", (bye_entry["character_id"],))
                bye_name = clean_display_name(cur.fetchone()["name"])

            cur.execute("UPDATE tourney.events SET status = 'active', round_number = 1, updated_at = NOW() WHERE id = %s", (e["id"],))
        conn.commit()

    action_word = "rewritten" if existing_match_ids else "generated"
    lines = [f"Bracket {action_word} for **{clean_display_name(event)}** in **{clean_display_name(tournament)}**.", f"Matches created: `{len(created_match_ids)}`", "", "Auto-seeding:"] + [f"Seed {idx}: **{entrant['name']}** (`{entrant['score']}`)" for idx, entrant in enumerate(scored, start=1)]
    if bye_name:
        lines += ["", f"Automatic bye: **{bye_name}** advances from round 1."]
    await interaction.response.send_message("\n".join(lines), ephemeral=True)




def get_event_live_metrics(conn, guild_id: int, event_row: dict) -> dict:
    metrics = {
        "entrant_count": 0,
        "registration_locked": False,
        "pending_contested": 0,
        "completed_contested": 0,
        "bye_count": 0,
        "latest_scored_round": 0,
        "board_linked": False,
        "ready_to_finalize": False,
        "remaining_display": "—",
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FILTER (WHERE registration_status <> 'withdrawn') AS cnt
            FROM tourney.entries
            WHERE guild_id = %s AND event_id = %s
            """,
            (guild_id, event_row["id"]),
        )
        metrics["entrant_count"] = cur.fetchone()["cnt"] or 0

        settings = get_event_settings(event_row.get("settings_json"))
        board_channel_id, board_message_id = get_public_board_ids(settings)
        metrics["board_linked"] = bool(board_channel_id and board_message_id)

        metrics["registration_locked"] = event_registration_locked(event_row, guild_id)
        metrics["ready_to_finalize"] = event_row["status"] == "ready_to_finalize"

        if event_row["format_type"] == "head_to_head":
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE match_type <> 'bye' AND status IN ('pending','active')) AS pending_contested,
                    COUNT(*) FILTER (WHERE match_type <> 'bye' AND status = 'completed') AS completed_contested,
                    COUNT(*) FILTER (WHERE match_type = 'bye') AS bye_count
                FROM tourney.matches
                WHERE guild_id = %s AND event_id = %s
                """,
                (guild_id, event_row["id"]),
            )
            row = cur.fetchone()
            metrics["pending_contested"] = row["pending_contested"] or 0
            metrics["completed_contested"] = row["completed_contested"] or 0
            metrics["bye_count"] = row["bye_count"] or 0

            cur.execute(
                """
                SELECT COUNT(DISTINCT winner_character_id) AS remaining
                FROM tourney.matches
                WHERE guild_id = %s AND event_id = %s AND status = 'completed' AND winner_character_id IS NOT NULL
                  AND round_number = (SELECT COALESCE(MAX(round_number), 0) FROM tourney.matches WHERE guild_id = %s AND event_id = %s)
                """,
                (guild_id, event_row["id"], guild_id, event_row["id"]),
            )
            rem = cur.fetchone()["remaining"]
            metrics["remaining_display"] = str(rem) if rem else "—"
        else:
            latest = get_latest_scored_round_match(conn, guild_id, event_row["id"])
            if latest:
                metrics["latest_scored_round"] = latest["round_number"]
                rankings = [dict(r) for r in fetch_scored_round_rankings(conn, latest["id"])]
                metrics["remaining_display"] = str(len(rankings)) if rankings else "—"
                metrics["ready_to_finalize"] = metrics["ready_to_finalize"] or (len(rankings) >= 2 and rankings[0].get("round_status") == "Champion")
            else:
                metrics["remaining_display"] = str(metrics["entrant_count"]) if metrics["entrant_count"] else "0"

    return metrics

def build_tournament_status_embed(tournament_row: dict, event_rows: list[dict], metrics_by_event: dict[int, dict]) -> discord.Embed:
    embed = discord.Embed(
        title=f"{clean_display_name(tournament_row['name'])} — Tournament Status",
        color=discord.Color.blue(),
        description=(
            f"**Division:** {clean_display_name(tournament_row['division'] or 'open')}\n"
            f"**Season:** {clean_display_name(tournament_row['season_label'] or '—')}\n"
            f"**Host Location:** {clean_display_name(tournament_row['host_location'] or '—')}\n"
            f"**Status:** {STATUS_LABELS.get(tournament_row['status'], clean_display_name(tournament_row['status']))}"
        ),
    )

    counts = {"draft": 0, "active": 0, "ready_to_finalize": 0, "completed": 0}
    for row in event_rows:
        status = row["status"]
        if status in counts:
            counts[status] += 1

    embed.add_field(
        name="Event Summary",
        value=(
            f"**Total Events:** {len(event_rows)}\n"
            f"**Draft:** {counts['draft']}\n"
            f"**Active:** {counts['active']}\n"
            f"**Ready to Finalize:** {counts['ready_to_finalize']}\n"
            f"**Completed:** {counts['completed']}"
        ),
        inline=False,
    )

    lines = []
    for row in event_rows:
        m = metrics_by_event.get(row["id"], {})
        lines.append(
            f"• **{clean_display_name(row['name'])}** — "
            f"{STATUS_LABELS.get(row['status'], clean_display_name(row['status']))} | "
            f"Round {row.get('round_number') or 0} | "
            f"Entrants {m.get('entrant_count', 0)} | "
            f"Remaining {m.get('remaining_display', '—')}"
        )

    for i, chunk in enumerate(chunk_lines_for_embed(lines if lines else ["No events found."], max_len=1000), start=1):
        embed.add_field(name="Events" if i == 1 else f"Events ({i})", value=chunk, inline=False)

    return embed

def build_event_status_embed(tournament_row: dict, event_row: dict, metrics: dict) -> discord.Embed:
    embed = discord.Embed(
        title=f"{clean_display_name(event_row['name'])} — Event Status",
        color=discord.Color.dark_teal(),
        description=(
            f"**Tournament:** {clean_display_name(tournament_row['name'])}\n"
            f"**Type:** {EVENT_LABELS.get(event_row['event_type'], clean_display_name(event_row['event_type']))}\n"
            f"**Format:** {clean_display_name(event_row['format_type'])}\n"
            f"**Status:** {STATUS_LABELS.get(event_row['status'], clean_display_name(event_row['status']))}"
        ),
    )

    embed.add_field(
        name="Operational State",
        value=(
            f"**Current Round:** {event_row.get('round_number') or 0}\n"
            f"**Entrants:** {metrics['entrant_count']}\n"
            f"**Remaining:** {metrics['remaining_display']}\n"
            f"**Registration Locked:** {'Yes' if metrics['registration_locked'] else 'No'}\n"
            f"**Standings Linked:** {'Yes' if metrics['board_linked'] else 'No'}\n"
            f"**Ready to Finalize:** {'Yes' if metrics['ready_to_finalize'] else 'No'}"
        ),
        inline=False,
    )

    if event_row["format_type"] == "head_to_head":
        embed.add_field(
            name="Head-to-Head Progress",
            value=(
                f"**Pending Contested Matches:** {metrics['pending_contested']}\n"
                f"**Completed Contested Matches:** {metrics['completed_contested']}\n"
                f"**Bye Matches:** {metrics['bye_count']}"
            ),
            inline=False,
        )
    else:
        embed.add_field(
            name="Round Progress",
            value=(
                f"**Latest Scored Round:** {metrics['latest_scored_round']}\n"
                f"**Next Round Available:** {'No' if metrics['ready_to_finalize'] else 'Yes'}"
            ),
            inline=False,
        )

    return embed


def diagnose_event_state(conn, guild_id: int, tournament_row: dict, event_row: dict) -> dict:
    metrics = get_event_live_metrics(conn, guild_id, event_row)
    info = {
        "event_status": event_row["status"],
        "round_number": event_row.get("round_number") or 0,
        "format_type": event_row["format_type"],
        "event_type": event_row["event_type"],
        "entrant_count": metrics["entrant_count"],
        "registration_locked": metrics["registration_locked"],
        "board_linked": metrics["board_linked"],
        "ready_to_finalize": metrics["ready_to_finalize"],
        "pending_contested": metrics["pending_contested"],
        "completed_contested": metrics["completed_contested"],
        "bye_count": metrics["bye_count"],
        "latest_scored_round": metrics["latest_scored_round"],
        "remaining_display": metrics["remaining_display"],
        "issues": [],
    }

    settings = get_event_settings(event_row.get("settings_json"))
    board_channel_id, board_message_id = get_public_board_ids(settings)
    info["board_channel_id"] = board_channel_id
    info["board_message_id"] = board_message_id

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM tourney.matches
            WHERE guild_id = %s AND event_id = %s
            """,
            (guild_id, event_row["id"]),
        )
        info["total_match_rows"] = cur.fetchone()["cnt"] or 0

        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM tourney.matches
            WHERE guild_id = %s AND event_id = %s AND status IN ('pending','active')
            """,
            (guild_id, event_row["id"]),
        )
        info["open_match_rows"] = cur.fetchone()["cnt"] or 0

    if not info["board_linked"]:
        info["issues"].append("No linked public standings post is stored for this event.")

    if event_row["format_type"] == "head_to_head":
        if info["total_match_rows"] == 0 and info["entrant_count"] >= 2:
            info["issues"].append("This event has entrants but no bracket or match rows yet.")
        if info["open_match_rows"] > 0:
            info["issues"].append("This event still has unresolved match rows.")
        if event_row["status"] == "ready_to_finalize" and info["pending_contested"] > 0:
            info["issues"].append("Event is marked ready_to_finalize but contested matches still appear pending.")
    else:
        if info["latest_scored_round"] == 0 and info["entrant_count"] >= 2 and event_row["status"] not in ("draft", "completed"):
            info["issues"].append("This scored event has entrants but no scored round has been recorded yet.")
        if event_row["status"] == "ready_to_finalize" and not info["ready_to_finalize"]:
            info["issues"].append("Event status says ready_to_finalize but the latest round does not clearly show a final state.")

    return info

def build_event_diagnose_embed(tournament_row: dict, event_row: dict, info: dict) -> discord.Embed:
    embed = discord.Embed(
        title=f"{clean_display_name(event_row['name'])} — Event Diagnostics",
        color=discord.Color.orange(),
        description=(
            f"**Tournament:** {clean_display_name(tournament_row['name'])}\n"
            f"**Type:** {EVENT_LABELS.get(event_row['event_type'], clean_display_name(event_row['event_type']))}\n"
            f"**Format:** {clean_display_name(event_row['format_type'])}"
        ),
    )
    embed.add_field(
        name="Core State",
        value=(
            f"**Status:** {STATUS_LABELS.get(info['event_status'], clean_display_name(info['event_status']))}\n"
            f"**Round Number:** {info['round_number']}\n"
            f"**Entrants:** {info['entrant_count']}\n"
            f"**Remaining:** {info['remaining_display']}\n"
            f"**Registration Locked:** {'Yes' if info['registration_locked'] else 'No'}\n"
            f"**Ready to Finalize:** {'Yes' if info['ready_to_finalize'] else 'No'}"
        ),
        inline=False,
    )

    if info["format_type"] == "head_to_head":
        embed.add_field(
            name="Head-to-Head Match State",
            value=(
                f"**Pending Contested:** {info['pending_contested']}\n"
                f"**Completed Contested:** {info['completed_contested']}\n"
                f"**Bye Rows:** {info['bye_count']}\n"
                f"**Total Match Rows:** {info['total_match_rows']}\n"
                f"**Open Match Rows:** {info['open_match_rows']}"
            ),
            inline=False,
        )
    else:
        embed.add_field(
            name="Scored Event State",
            value=(
                f"**Latest Scored Round:** {info['latest_scored_round']}\n"
                f"**Total Match Rows:** {info['total_match_rows']}\n"
                f"**Open Match Rows:** {info['open_match_rows']}"
            ),
            inline=False,
        )

    board_value = (
        f"**Linked:** {'Yes' if info['board_linked'] else 'No'}\n"
        f"**Channel ID Stored:** {info['board_channel_id'] or '—'}\n"
        f"**Message ID Stored:** {info['board_message_id'] or '—'}"
    )
    embed.add_field(name="Standings Linkage", value=board_value, inline=False)

    issue_lines = info["issues"] if info["issues"] else ["No obvious integrity issues were detected."]
    for idx_chunk, chunk in enumerate(chunk_lines_for_embed(issue_lines, max_len=1000), start=1):
        embed.add_field(name="Detected Issues" if idx_chunk == 1 else f"Detected Issues ({idx_chunk})", value=chunk, inline=False)

    return embed




@tree.command(name="tourney-rebuild-standings", description="Rebuild or relink the public standings post for an event from database truth.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def rebuild_standings(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e:
            await interaction.response.send_message("Event not found.", ephemeral=True); return

        settings = get_event_settings(e.get("settings_json"))
        settings = set_public_board_ids(settings, interaction.channel.id, settings.get("public_standings_message_id") or settings.get("public_bracket_message_id"))
        with conn.cursor() as cur:
            set_event_settings(cur, e["id"], settings)
        conn.commit()

    await upsert_public_standings_post(interaction, t["name"], e)
    await interaction.response.send_message(
        f"Public standings for **{clean_display_name(event)}** in **{clean_display_name(tournament)}** were rebuilt from current database state.",
        ephemeral=True,
    )

@tree.command(name="tourney-diagnose-event", description="Show a deeper staff diagnostic report for one event.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def diagnose_event(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e:
            await interaction.response.send_message("Event not found.", ephemeral=True); return
        info = diagnose_event_state(conn, interaction.guild.id, t, e)

    embed = build_event_diagnose_embed(t, e, info)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@tree.command(name="event-description", description="Post a flavorful public description of an event and its in-world rules.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def event_description(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e:
            await interaction.response.send_message("Event not found.", ephemeral=True); return

    embed = build_event_description_embed(t["name"], e)
    await interaction.response.send_message(embed=embed, ephemeral=False)


@tree.command(name="tourney-post-standings", description="Post the current public standings board in the current channel.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def post_standings(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e:
            await interaction.response.send_message("Event not found.", ephemeral=True); return

        settings = get_event_settings(e.get("settings_json"))
        settings = set_public_board_ids(settings, interaction.channel.id, settings.get("public_standings_message_id") or settings.get("public_bracket_message_id"))
        with conn.cursor() as cur:
            set_event_settings(cur, e["id"], settings)
        conn.commit()

    await upsert_public_standings_post(interaction, t["name"], e)
    await interaction.response.send_message("Public standings posted and linked for automatic updates.", ephemeral=True)
    await interaction.channel.send(embed=build_event_intro_embed(t["name"], e["name"], e["event_type"]))

def get_next_pending_head_to_head_match(conn, guild_id: int, event_id: int):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.id
            FROM tourney.matches m
            WHERE m.guild_id = %s
              AND m.event_id = %s
              AND m.status IN ('pending', 'active')
              AND m.match_type <> 'bye'
            ORDER BY m.round_number, m.match_order, m.id
            LIMIT 1
        """, (guild_id, event_id))
        row = cur.fetchone()
    return row["id"] if row else None



@tree.command(name="tourney-event-run-round", description="Run the next round of an event ephemerally.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def run_event_round(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    await interaction.response.defer(ephemeral=True)

    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.followup.send("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e:
            await interaction.followup.send("Event not found.", ephemeral=True); return

        if e["format_type"] == "head_to_head":
            match_id = get_next_pending_head_to_head_match(conn, interaction.guild.id, e["id"])
            if not match_id:
                await interaction.followup.send("No pending contested match remains for that event. It may be ready to finalize.", ephemeral=True); return
            ctx = load_match_context(conn, interaction.guild.id, match_id)
            match_row = ctx["match_row"]
            result = resolve_match_and_store(conn, interaction.guild.id, match_id)
            if not result["ok"]:
                await interaction.followup.send(result["error"], ephemeral=True); return
            auto_notes = auto_advance_event_if_ready(conn, interaction.guild.id, t, e)
            embeds = sanitize_embeds_for_send(build_match_result_embeds(match_row, match_id, result["winner_name"], result["loser_name"], result["summary_lines"], public=False))
            await interaction.followup.send(embeds=embeds, ephemeral=True)
            settings = get_event_settings(e.get("settings_json"))
            board_channel_id, board_message_id = get_public_board_ids(settings)
            if board_channel_id and board_message_id:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM tourney.events WHERE id = %s LIMIT 1", (e["id"],))
                    refreshed_event = cur.fetchone() or e
                await upsert_public_standings_post(interaction, t["name"], refreshed_event)
            if auto_notes:
                await interaction.followup.send("\n".join(auto_notes), ephemeral=True)
            return

        result = run_scored_round_and_store(conn, interaction.guild.id, t, e)
        if not result["ok"]:
            await interaction.followup.send(result["error"], ephemeral=True); return
        embeds = build_scored_round_embeds(
            t["name"],
            e["name"],
            result["event_type"],
            result["round_no"],
            result["rankings"],
            result["champion_name"],
            public=True,
            summary_lines=result["summary_lines"],
        )
        await send_followup_embeds_chunked(interaction, embeds, ephemeral=False)
        settings = get_event_settings(e.get("settings_json"))
        board_channel_id, board_message_id = get_public_board_ids(settings)
        if board_channel_id and board_message_id:
            await upsert_public_standings_post(interaction, t["name"], e)




def event_public_call_title(event_type: str) -> str:
    return {
        "joust": "🏇 Match Called to the Lists!",
        "duel": "⚔️ Duelists Called to the Ring!",
        "archery": "🏹 Archers Called to the Butts!",
        "grand_melee": "🛡️ Warriors Called to the Melee!",
        "horse_race": "🏇 Riders Called to the Course!",
        "hunt": "🦌 Hunters Called to the Field!",
    }.get(event_type, "📣 Competitors Called to the Field!")

def format_participant_call_line(character_name: str, user_id) -> str:
    clean_name = clean_display_name(character_name)
    if user_id:
        return f"{clean_name} | (||<@{user_id}>||)"
    return clean_name

def build_public_call_message(event_type: str, participant_rows: list[dict]) -> str:
    seen_user_ids = set()
    lines = [event_public_call_title(event_type), ""]
    for row in participant_rows:
        user_id = row.get("user_id")
        dedupe_key = (row.get("character_id"), user_id, clean_display_name(row.get("name")))
        if dedupe_key in seen_user_ids:
            continue
        seen_user_ids.add(dedupe_key)
        lines.append(format_participant_call_line(row.get("name"), user_id))
    return "\n".join(lines)

def get_active_public_round_participants(conn, guild_id: int, event_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.name, en.character_id, en.user_id, en.registration_status
            FROM tourney.entries en
            JOIN characters c ON en.character_id = c.character_id
            WHERE en.guild_id = %s
              AND en.event_id = %s
              AND en.registration_status IN ('registered', 'advanced')
            ORDER BY COALESCE(en.seed, 999999), c.name
            """,
            (guild_id, event_id),
        )
        return cur.fetchall()


@tree.command(name="tourney-event-run-round-public", description="Run the next round of an event publicly in the current channel.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def run_event_round_public(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    await interaction.response.defer()

    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.followup.send("Tournament not found."); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e:
            await interaction.followup.send("Event not found."); return

        if e["format_type"] == "head_to_head":
            match_id = get_next_pending_head_to_head_match(conn, interaction.guild.id, e["id"])
            if not match_id:
                await interaction.followup.send("No pending contested match remains for that event. It may be ready to finalize."); return
            ctx = load_match_context(conn, interaction.guild.id, match_id)
            match_row = ctx["match_row"]
            participants = ctx["participants"]

            call_message = build_public_call_message(e["event_type"], participants)
            await interaction.followup.send(content=call_message)

            result = resolve_match_and_store(conn, interaction.guild.id, match_id)
            if not result["ok"]:
                await interaction.followup.send(result["error"]); return
            auto_notes = auto_advance_event_if_ready(conn, interaction.guild.id, t, e)
            embeds = sanitize_embeds_for_send(
                build_match_result_embeds(
                    match_row,
                    match_id,
                    result["winner_name"],
                    result["loser_name"],
                    result["summary_lines"],
                    public=True,
                )
            )
            await send_public_round_embeds(interaction, embeds, match_row["event_name"])
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM tourney.events WHERE id = %s LIMIT 1", (e["id"],))
                refreshed_event = cur.fetchone() or e
            settings = get_event_settings(refreshed_event.get("settings_json"))
            board_channel_id, board_message_id = get_public_board_ids(settings)
            if board_channel_id and board_message_id:
                await upsert_public_standings_post(interaction, t["name"], refreshed_event)
            if auto_notes:
                await interaction.followup.send("\n".join(auto_notes))
            return

        participants = get_active_public_round_participants(conn, interaction.guild.id, e["id"])
        call_message = build_public_call_message(e["event_type"], participants)
        await interaction.followup.send(content=call_message)

        result = run_scored_round_and_store(conn, interaction.guild.id, t, e)
        if not result["ok"]:
            await interaction.followup.send(result["error"]); return
        embeds = sanitize_embeds_for_send(
            build_scored_round_embeds(
                t["name"],
                e["name"],
                result["event_type"],
                result["round_no"],
                result["rankings"],
                result["champion_name"],
                public=True,
                summary_lines=result["summary_lines"],
            )
        )
        await interaction.followup.send(embeds=embeds)
        settings = get_event_settings(e.get("settings_json"))
        board_channel_id, board_message_id = get_public_board_ids(settings)
        if board_channel_id and board_message_id:
            await upsert_public_standings_post(interaction, t["name"], e)

@tree.command(name="tourney-event-advance-round", description="Advance a head-to-head event to the next round from completed matches.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def advance_round(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t: await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e: await interaction.response.send_message("Event not found.", ephemeral=True); return
        if e["format_type"] != "head_to_head":
            await interaction.response.send_message("Round advancement currently supports only head-to-head events.", ephemeral=True); return

        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(round_number), 0) AS current_round FROM tourney.matches WHERE guild_id = %s AND event_id = %s", (interaction.guild.id, e["id"]))
            current_round = cur.fetchone()["current_round"]
            if current_round == 0:
                await interaction.response.send_message("No bracket exists for that event yet.", ephemeral=True); return

            if not round_is_complete(conn, interaction.guild.id, e["id"], current_round):
                await interaction.response.send_message("All contested matches in the current round must be completed before advancing.", ephemeral=True); return

            finalized_byes = finalize_pending_byes_in_round(conn, interaction.guild.id, e["id"], current_round)
            if finalized_byes:
                conn.commit()

            cur.execute("""
                SELECT winner_character_id
                FROM tourney.matches
                WHERE guild_id = %s
                  AND event_id = %s
                  AND round_number = %s
                  AND status = 'completed'
                ORDER BY match_order, id
            """, (interaction.guild.id, e["id"], current_round))
            winners = [r["winner_character_id"] for r in cur.fetchall() if r["winner_character_id"]]

            if len(winners) == 1:
                await interaction.response.send_message(f"Only one winner remains in **{clean_display_name(event)}**. The event is ready to finalize.", ephemeral=True); return

            next_round = current_round + 1
            cur.execute("SELECT COUNT(*) AS cnt FROM tourney.matches WHERE guild_id = %s AND event_id = %s AND round_number = %s", (interaction.guild.id, e["id"], next_round))
            if cur.fetchone()["cnt"] > 0:
                await interaction.response.send_message("The next round has already been generated.", ephemeral=True); return

            winner_entries = []
            for character_id in winners:
                cur.execute("SELECT id AS entry_id FROM tourney.entries WHERE guild_id = %s AND event_id = %s AND character_id = %s LIMIT 1", (interaction.guild.id, e["id"], character_id))
                entry = cur.fetchone()
                if entry:
                    winner_entries.append({"character_id": character_id, "entry_id": entry["entry_id"]})

            if len(winner_entries) < 2:
                await interaction.response.send_message(f"Only one winner remains in **{clean_display_name(event)}**. The event is ready to finalize.", ephemeral=True); return

            created_match_ids = []
            bye_name = None
            i, j, match_order = 0, len(winner_entries) - 1, 1
            while i < j:
                top = winner_entries[i]; bottom = winner_entries[j]
                cur.execute("""
                    INSERT INTO tourney.matches
                    (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, created_at)
                    VALUES (%s, %s, %s, %s, %s, 'pending', 'head_to_head', NOW())
                    RETURNING id
                """, (e["id"], t["id"], interaction.guild.id, next_round, match_order))
                match_id = cur.fetchone()["id"]
                created_match_ids.append(match_id)
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 1, FALSE)", (match_id, top["entry_id"], top["character_id"]))
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 2, FALSE)", (match_id, bottom["entry_id"], bottom["character_id"]))
                match_order += 1
                i += 1
                j -= 1

            if i == j:
                bye_entry = winner_entries[i]
                cur.execute("""
                    INSERT INTO tourney.matches
                    (event_id, tournament_id, guild_id, round_number, match_order, status, match_type, narrative_summary, created_at)
                    VALUES (%s, %s, %s, %s, %s, 'pending', 'bye', %s, NOW())
                    RETURNING id
                """, (
                    e["id"], t["id"], interaction.guild.id, next_round, match_order,
                    "Pending automatic bye advancement."
                ))
                bye_match_id = cur.fetchone()["id"]
                created_match_ids.append(bye_match_id)
                cur.execute("INSERT INTO tourney.match_participants (match_id, entry_id, character_id, slot_number, eliminated) VALUES (%s, %s, %s, 1, FALSE)", (bye_match_id, bye_entry["entry_id"], bye_entry["character_id"]))
                cur.execute("SELECT name FROM characters WHERE character_id = %s LIMIT 1", (bye_entry["character_id"],))
                bye_name = clean_display_name(cur.fetchone()["name"])

            cur.execute("UPDATE tourney.events SET round_number = %s, updated_at = NOW() WHERE id = %s", (next_round, e["id"]))
        conn.commit()

    lines = [f"Advanced **{clean_display_name(event)}** in **{clean_display_name(tournament)}** to round `{next_round}`.", f"New matches created: `{len(created_match_ids)}`"]
    if finalized_byes:
        lines += ["", f"Finalized pending bye(s): **{', '.join(finalized_byes)}**"]
    if bye_name:
        lines += ["", f"Automatic bye this round: **{bye_name}** advances."]
    await interaction.response.send_message("\n".join(lines), ephemeral=True)



@tree.command(name="tourney-event-finalize", description="Finalize an event, assign XP, update records, and mark a champion.")
@app_commands.autocomplete(tournament=tournament_autocomplete, event=event_autocomplete)
async def finalize_event(interaction: discord.Interaction, tournament: str, event: str):
    if await deny_if_not_admin(interaction): return
    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, tournament)
        if not t:
            await interaction.response.send_message("Tournament not found.", ephemeral=True); return
        e = get_event_by_name(conn, interaction.guild.id, t["id"], event)
        if not e:
            await interaction.response.send_message("Event not found.", ephemeral=True); return
        if e["status"] == "completed":
            await interaction.response.send_message("That event is already completed.", ephemeral=True); return

        champion_character_id = None
        champion_name = None
        runner_up_character_id = None
        runner_up_name = None

        with conn.cursor() as cur:
            cur.execute(
                "SELECT character_id FROM tourney.entries WHERE guild_id = %s AND event_id = %s AND registration_status <> 'withdrawn'",
                (interaction.guild.id, e["id"])
            )
            entrant_ids = [row["character_id"] for row in cur.fetchall()]
            if not entrant_ids:
                await interaction.response.send_message("No entrants were found for that event.", ephemeral=True); return

            if e["format_type"] == "head_to_head":
                cur.execute(
                    "SELECT DISTINCT round_number FROM tourney.matches WHERE guild_id = %s AND event_id = %s ORDER BY round_number",
                    (interaction.guild.id, e["id"])
                )
                for row in cur.fetchall():
                    if round_is_complete(conn, interaction.guild.id, e["id"], row["round_number"]):
                        finalized_byes = finalize_pending_byes_in_round(conn, interaction.guild.id, e["id"], row["round_number"])
                        if finalized_byes:
                            conn.commit()

                state, final_match = get_head_to_head_contested_state(conn, interaction.guild.id, e["id"])
                if state["contested_total"] == 0:
                    await interaction.response.send_message("No contested matches exist for that event.", ephemeral=True); return
                if state["contested_open"] > 0:
                    await interaction.response.send_message("A contested match still remains unresolved. Run the remaining round before finalizing the event.", ephemeral=True); return
                if state["contested_total"] != state["contested_completed"]:
                    await interaction.response.send_message("All contested matches and byes must be fully resolved before the event can be finalized.", ephemeral=True); return
                if not final_match or not final_match["winner_character_id"]:
                    await interaction.response.send_message("Could not determine the event champion from the true final round.", ephemeral=True); return

                champion_character_id = final_match["winner_character_id"]
                cur.execute("SELECT name FROM characters WHERE character_id = %s LIMIT 1", (champion_character_id,))
                champion_name = clean_display_name(cur.fetchone()["name"])

                cur.execute(
                    "SELECT mp.character_id, c.name FROM tourney.match_participants mp JOIN characters c ON mp.character_id = c.character_id WHERE mp.match_id = %s AND mp.character_id <> %s LIMIT 1",
                    (final_match["id"], champion_character_id)
                )
                runner = cur.fetchone()
                if runner:
                    runner_up_character_id = runner["character_id"]
                    runner_up_name = clean_display_name(runner["name"])

                cur.execute(
                    """
                    SELECT DISTINCT m.id AS match_id, mp.character_id
                    FROM tourney.matches m
                    JOIN tourney.match_participants mp ON m.id = mp.match_id
                    WHERE m.guild_id = %s AND m.event_id = %s AND m.status = 'completed'
                    ORDER BY m.id, mp.character_id
                    """,
                    (interaction.guild.id, e["id"])
                )
                for row in cur.fetchall():
                    award_xp_packet(
                        cur,
                        interaction.guild.id,
                        row["character_id"],
                        EVENT_XP_RULES[e["event_type"]]["round_participation"],
                        "round_participation",
                        t["id"],
                        e["id"],
                        source_match_id=row["match_id"],
                        notes=f"Round participation XP for {e['name']}"
                    )

                for cid in entrant_ids:
                    update_record_counter(cur, interaction.guild.id, cid, e["event_type"], "entries_count", 1)

                cur.execute(
                    """
                    SELECT m.winner_character_id, MAX(CASE WHEN mp.character_id <> m.winner_character_id THEN mp.character_id END) AS loser_character_id
                    FROM tourney.matches m
                    JOIN tourney.match_participants mp ON m.id = mp.match_id
                    WHERE m.guild_id = %s AND m.event_id = %s AND m.status = 'completed'
                    GROUP BY m.id, m.winner_character_id
                    """,
                    (interaction.guild.id, e["id"])
                )
                for row in cur.fetchall():
                    if row["winner_character_id"]:
                        update_record_counter(cur, interaction.guild.id, row["winner_character_id"], e["event_type"], "wins_count", 1)
                    if row["loser_character_id"]:
                        update_record_counter(cur, interaction.guild.id, row["loser_character_id"], e["event_type"], "losses_count", 1)

            else:
                latest = get_latest_scored_round_match(conn, interaction.guild.id, e["id"])
                if not latest:
                    await interaction.response.send_message("No scored rounds exist for that event yet.", ephemeral=True); return

                latest_rankings = [dict(r) for r in fetch_scored_round_rankings(conn, latest["id"])]
                if len(latest_rankings) < 2:
                    await interaction.response.send_message("The event needs a final two before it can be finalized.", ephemeral=True); return

                is_event_ready = (
                    e["status"] == "ready_to_finalize"
                    or latest["status"] == "completed"
                    or (
                        latest_rankings[0].get("round_status") in ("Champion", "Advances")
                        and latest_rankings[1].get("round_status") == "Runner-up"
                    )
                )
                if not is_event_ready:
                    await interaction.response.send_message("The event is not yet ready to finalize.", ephemeral=True); return

                champion_character_id = latest_rankings[0]["character_id"]
                champion_name = clean_display_name(latest_rankings[0]["name"])
                runner_up_character_id = latest_rankings[1]["character_id"]
                runner_up_name = clean_display_name(latest_rankings[1]["name"])

                # Reconcile stale in-progress rows without blocking finalization.
                cur.execute(
                    """
                    UPDATE tourney.matches
                    SET status = 'completed',
                        narrative_summary = COALESCE(narrative_summary, 'Administrative closure during event finalization.'),
                        completed_at = COALESCE(completed_at, NOW())
                    WHERE guild_id = %s
                      AND event_id = %s
                      AND match_type = 'scored_round'
                      AND status <> 'completed'
                    """,
                    (interaction.guild.id, e["id"])
                )

                cur.execute(
                    """
                    SELECT DISTINCT m.id AS match_id, mp.character_id
                    FROM tourney.matches m
                    JOIN tourney.match_participants mp ON m.id = mp.match_id
                    WHERE m.guild_id = %s AND m.event_id = %s AND m.status = 'completed' AND m.match_type = 'scored_round'
                    ORDER BY m.id, mp.character_id
                    """,
                    (interaction.guild.id, e["id"])
                )
                for row in cur.fetchall():
                    award_xp_packet(
                        cur,
                        interaction.guild.id,
                        row["character_id"],
                        EVENT_XP_RULES[e["event_type"]]["round_participation"],
                        "round_participation",
                        t["id"],
                        e["id"],
                        source_match_id=row["match_id"],
                        notes=f"Round participation XP for {e['name']}"
                    )

                for cid in entrant_ids:
                    update_record_counter(cur, interaction.guild.id, cid, e["event_type"], "entries_count", 1)

                cur.execute(
                    "SELECT id FROM tourney.matches WHERE guild_id = %s AND event_id = %s AND status = 'completed' AND match_type = 'scored_round' ORDER BY round_number, id",
                    (interaction.guild.id, e["id"])
                )
                for match_row in cur.fetchall():
                    for ranked in fetch_scored_round_rankings(conn, match_row["id"]):
                        if ranked["round_status"] in ("Champion", "Advances"):
                            update_record_counter(cur, interaction.guild.id, ranked["character_id"], e["event_type"], "wins_count", 1)
                        else:
                            update_record_counter(cur, interaction.guild.id, ranked["character_id"], e["event_type"], "losses_count", 1)

            if runner_up_character_id:
                award_xp_packet(
                    cur, interaction.guild.id, runner_up_character_id,
                    EVENT_XP_RULES[e["event_type"]]["runner_up"],
                    "runner_up", t["id"], e["id"],
                    notes=f"Runner-up XP for {e['name']}"
                )
                update_record_counter(cur, interaction.guild.id, runner_up_character_id, e["event_type"], "runner_up_count", 1)

            award_xp_packet(
                cur, interaction.guild.id, champion_character_id,
                EVENT_XP_RULES[e["event_type"]]["champion"],
                "event_champion", t["id"], e["id"],
                notes=f"Champion XP for {e['name']}"
            )
            update_record_counter(cur, interaction.guild.id, champion_character_id, e["event_type"], "championships_count", 1)

            cur.execute(
                "INSERT INTO tourney.awards (guild_id, tournament_id, event_id, character_id, award_code, award_name, notes, awarded_at) VALUES (%s, %s, %s, %s, 'champion', %s, %s, NOW())",
                (interaction.guild.id, t["id"], e["id"], champion_character_id, f"{e['name']} Champion", f"Champion of {e['name']} in {t['name']}")
            )
            cur.execute(
                "UPDATE tourney.entries SET registration_status = CASE WHEN character_id = %s THEN 'champion' WHEN character_id = %s THEN 'runner_up' WHEN registration_status <> 'withdrawn' THEN 'eliminated' ELSE registration_status END WHERE guild_id = %s AND event_id = %s",
                (champion_character_id, runner_up_character_id or -1, interaction.guild.id, e["id"])
            )
            cur.execute("UPDATE tourney.events SET status = 'completed', updated_at = NOW() WHERE id = %s", (e["id"],))
            cur.execute(
                "SELECT COUNT(*) FILTER (WHERE status <> 'completed') AS remaining FROM tourney.events WHERE guild_id = %s AND tournament_id = %s",
                (interaction.guild.id, t["id"])
            )
            if cur.fetchone()["remaining"] == 0:
                cur.execute("UPDATE tourney.tournaments SET status = 'ready_to_finalize', updated_at = NOW() WHERE id = %s", (t["id"],))
        conn.commit()

        participant_rows = get_event_participant_rows(conn, interaction.guild.id, e["id"])

    await interaction.response.send_message(
        "\n".join(
            [f"**{clean_display_name(event)}** in **{clean_display_name(tournament)}** has been finalized.",
             f"Champion: **{champion_name}**"] +
            ([f"Runner-up: **{runner_up_name}**"] if runner_up_name else []) +
            ["Tiny round participation XP and finisher bonuses have been applied.",
             "A full event recap has been posted publicly."]
        ),
        ephemeral=True
    )

    settings = get_event_settings(e.get("settings_json"))
    target_channel = interaction.channel
    channel_id = settings.get("public_standings_channel_id") or settings.get("public_bracket_channel_id")
    if channel_id:
        try:
            target_channel = interaction.guild.get_channel(int(channel_id)) or await bot.fetch_channel(int(channel_id))
        except Exception:
            target_channel = interaction.channel

    recap_embeds = build_event_recap_embeds(t["name"], e, participant_rows, champion_name, runner_up_name)
    for embed in recap_embeds:
        await target_channel.send(embed=embed)

    await post_event_champion_summary(interaction, t, e, champion_name, runner_up_name)

    with get_db() as board_conn:
        completed_event = get_event_fresh(board_conn, e["id"]) or dict(e)
    completed_event = dict(completed_event)
    completed_event["status"] = "completed"
    await upsert_public_standings_post(interaction, t["name"], completed_event)



HALL_OF_CHAMPIONS_CHANNEL_ID = 1481432487614742560

async def post_event_champion_summary(
    interaction: discord.Interaction,
    tournament_row: dict,
    event_row: dict,
    champion_name: str,
    runner_up_name: str | None = None,
) -> bool:
    try:
        hall_channel = interaction.guild.get_channel(HALL_OF_CHAMPIONS_CHANNEL_ID) or await bot.fetch_channel(HALL_OF_CHAMPIONS_CHANNEL_ID)
    except Exception as exc:
        log.exception("Could not resolve Hall of Champions channel for event=%s: %s", event_row.get("name"), exc)
        return False

    event_label = EVENT_LABELS.get(event_row.get("event_type"), clean_display_name(event_row.get("name")))
    embed = discord.Embed(
        title=f"🏆 {event_label} — Champion Crowned",
        color=discord.Color.gold(),
        description=(
            f"**Tournament:** {clean_display_name(tournament_row.get('name'))}\n"
            f"**Event:** {clean_display_name(event_row.get('name'))}\n\n"
            "Before the gathered realm, the final judgment is given and the victor's name is entered among the honored."
        ),
    )
    embed.add_field(name="Champion", value=f"**{clean_display_name(champion_name)}**", inline=False)
    if runner_up_name:
        embed.add_field(name="Runner-up", value=f"**{clean_display_name(runner_up_name)}**", inline=False)
    embed.set_footer(text="Hall of Champions Record")
    try:
        await hall_channel.send(embed=embed)
        return True
    except Exception as exc:
        log.exception("Failed posting Hall of Champions summary for event=%s: %s", event_row.get("name"), exc)
        return False

FINALIZE_PAYOUT_MARKER = "[TOURNEY_PAYOUTS_APPLIED_V1]"

def tournament_payouts_already_applied(tournament_row: dict) -> bool:
    notes = tournament_row.get("notes") or ""
    return FINALIZE_PAYOUT_MARKER in str(notes)

def mark_tournament_payouts_applied(cur, tournament_row: dict):
    notes = tournament_row.get("notes") or ""
    if FINALIZE_PAYOUT_MARKER not in notes:
        notes = (notes + "\n" if notes else "") + FINALIZE_PAYOUT_MARKER
    cur.execute(
        "UPDATE tourney.tournaments SET notes = %s, updated_at = NOW() WHERE id = %s",
        (notes, tournament_row["id"]),
    )

def format_val_amount(value: int) -> str:
    remaining = max(0, int(value))
    units = [
        ("Mythic Crystal Novir", 10000),
        ("Platinum Oril", 1000),
        ("Gold Elsh", 100),
        ("Silver Arce", 10),
        ("Copper Cinth", 1),
    ]
    plural_map = {
        "Mythic Crystal Novir": "Mythic Crystal Novirs",
        "Platinum Oril": "Platinum Orils",
        "Gold Elsh": "Gold Elsh",
        "Silver Arce": "Silver Arces",
        "Copper Cinth": "Copper Cinths",
    }
    parts = []
    for label, denom in units:
        count, remaining = divmod(remaining, denom)
        if count:
            parts.append(f"{count} {label if count == 1 else plural_map[label]}")
    return ", ".join(parts) if parts else "0 Copper Cinths"

def collect_tournament_event_results(conn, guild_id: int, tournament_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, event_type, format_type, status, settings_json
            FROM tourney.events
            WHERE guild_id = %s
              AND tournament_id = %s
            ORDER BY id ASC
            """,
            (guild_id, tournament_id),
        )
        events = cur.fetchall()

    results: list[dict] = []
    for event_row in events:
        participant_rows = get_event_participant_rows(conn, guild_id, event_row["id"])
        champion_id = None
        champion_name = None
        runner_up_id = None
        runner_up_name = None
        for row in participant_rows:
            if row["registration_status"] == "champion":
                champion_id = row["character_id"]
                champion_name = clean_display_name(row["name"])
            elif row["registration_status"] == "runner_up":
                runner_up_id = row["character_id"]
                runner_up_name = clean_display_name(row["name"])
        if not champion_name and participant_rows:
            champion_id = participant_rows[0]["character_id"]
            champion_name = clean_display_name(participant_rows[0]["name"])
        results.append({
            "event_id": event_row["id"],
            "event_name": clean_display_name(event_row["name"]),
            "event_type": event_row["event_type"],
            "format_type": event_row["format_type"],
            "status": event_row["status"],
            "settings_json": event_row.get("settings_json"),
            "champion_character_id": champion_id,
            "champion_name": champion_name,
            "runner_up_character_id": runner_up_id,
            "runner_up_name": runner_up_name,
            "participant_rows": participant_rows,
        })
    return results

def build_payout_map(event_results: list[dict]) -> tuple[dict[int, dict], list[dict]]:
    payout_map: dict[int, dict] = {}
    event_payouts: list[dict] = []
    for result in event_results:
        event_total = 0
        champion_id = result.get("champion_character_id")
        runner_id = result.get("runner_up_character_id")

        for row in result["participant_rows"]:
            cid = row["character_id"]
            cname = clean_display_name(row["name"])
            bucket = payout_map.setdefault(cid, {"character_id": cid, "character_name": cname, "val": 0, "event_wins": 0})
            bucket["val"] += 5
            event_total += 5

        if champion_id and champion_id in payout_map:
            payout_map[champion_id]["val"] += 100
            payout_map[champion_id]["event_wins"] += 1
            event_total += 100

        if runner_id and runner_id in payout_map:
            payout_map[runner_id]["val"] += 50
            event_total += 50

        event_payouts.append({
            "event_id": result["event_id"],
            "event_name": result["event_name"],
            "event_total_val": event_total,
            "kingdom_bonus_val": round(event_total * 0.10),
            "champion_name": result.get("champion_name"),
            "runner_up_name": result.get("runner_up_name"),
        })

    return payout_map, event_payouts

def determine_tournament_champions_from_payouts(payout_map: dict[int, dict]) -> list[dict]:
    if not payout_map:
        return []
    highest = max((row.get("event_wins", 0) for row in payout_map.values()), default=0)
    if highest < 2:
        return []
    winners = [row for row in payout_map.values() if row.get("event_wins", 0) == highest]
    winners.sort(key=lambda r: clean_display_name(r["character_name"]))
    return winners

def apply_tournament_payouts(cur, guild_id: int, payout_map: dict[int, dict]) -> int:
    total_val = 0
    for row in payout_map.values():
        amount = int(row.get("val", 0) or 0)
        if amount <= 0:
            continue
        cur.execute(
            """
            SELECT 1
            FROM public.characters
            WHERE guild_id = %s
              AND name = %s
            LIMIT 1
            """,
            (guild_id, row["character_name"]),
        )
        if not cur.fetchone():
            continue
        cur.execute(
            """
            INSERT INTO public.econ_balances (guild_id, character_name, balance_val)
            VALUES (%s, %s, %s)
            ON CONFLICT (guild_id, character_name)
            DO UPDATE SET balance_val = public.econ_balances.balance_val + EXCLUDED.balance_val
            """,
            (guild_id, row["character_name"], amount),
        )
        total_val += amount
    return total_val

def apply_kingdom_hosting_bonus(cur, guild_id: int, host_kingdom: str, event_payouts: list[dict]) -> int:
    total_bonus = sum(int(row.get("kingdom_bonus_val", 0) or 0) for row in event_payouts)
    if total_bonus <= 0 or not (host_kingdom or "").strip():
        return 0

    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'econ_kingdoms'
        """
    )
    econ_kingdom_columns = {row["column_name"] for row in cur.fetchall()}

    treasury_column = "treasury" if "treasury" in econ_kingdom_columns else "treasury_val" if "treasury_val" in econ_kingdom_columns else None
    if not treasury_column:
        raise RuntimeError("public.econ_kingdoms is missing both treasury and treasury_val columns.")

    has_guild_id = "guild_id" in econ_kingdom_columns
    has_tax_rate_bp = "tax_rate_bp" in econ_kingdom_columns

    where_parts = []
    params: list[object] = []
    if has_guild_id:
        where_parts.append("guild_id = %s")
        params.append(guild_id)
    where_parts.append("kingdom = %s")
    params.append(host_kingdom)
    where_sql = " AND ".join(where_parts)

    cur.execute(
        f"SELECT {treasury_column} FROM public.econ_kingdoms WHERE {where_sql} LIMIT 1",
        tuple(params),
    )
    existing = cur.fetchone()

    if existing:
        cur.execute(
            f"UPDATE public.econ_kingdoms SET {treasury_column} = COALESCE({treasury_column}, 0) + %s WHERE {where_sql}",
            (total_bonus, *params),
        )
        return total_bonus

    insert_columns = []
    insert_values = []
    if has_guild_id:
        insert_columns.append("guild_id")
        insert_values.append(guild_id)
    insert_columns.append("kingdom")
    insert_values.append(host_kingdom)
    if has_tax_rate_bp:
        insert_columns.append("tax_rate_bp")
        insert_values.append(0)
    insert_columns.append(treasury_column)
    insert_values.append(total_bonus)

    placeholders = ", ".join(["%s"] * len(insert_columns))
    cur.execute(
        f"INSERT INTO public.econ_kingdoms ({', '.join(insert_columns)}) VALUES ({placeholders})",
        tuple(insert_values),
    )
    return total_bonus

def ensure_tournament_champion_awards(cur, guild_id: int, tournament_id: int, tournament_name: str, champions: list[dict]):
    if not champions:
        return
    award_code = "tournament_champion" if len(champions) == 1 else "tournament_co_champion"
    award_name = "Tournament Champion" if len(champions) == 1 else "Tournament Co-Champion"
    for row in champions:
        cur.execute(
            """
            SELECT 1
            FROM tourney.awards
            WHERE guild_id = %s
              AND tournament_id = %s
              AND character_id = %s
              AND award_code = %s
            LIMIT 1
            """,
            (guild_id, tournament_id, row["character_id"], award_code),
        )
        if cur.fetchone():
            continue
        cur.execute(
            """
            INSERT INTO tourney.awards
            (guild_id, tournament_id, event_id, character_id, award_code, award_name, notes, awarded_at)
            VALUES (%s, %s, NULL, %s, %s, %s, %s, NOW())
            """,
            (guild_id, tournament_id, row["character_id"], award_code, award_name, f"{award_name} of {tournament_name}"),
        )

def build_tournament_recap_embeds(tournament_row: dict, event_results: list[dict], champions: list[dict], payout_total: int, kingdom_bonus_total: int) -> list[discord.Embed]:
    tournament_name = clean_display_name(tournament_row["name"])
    host_location = clean_display_name(tournament_row.get("host_location") or "Unknown")
    season = clean_display_name(tournament_row.get("season_label") or "—")
    embed = discord.Embed(
        title=f"🏰 {tournament_name} — Tourney Recap",
        color=discord.Color.dark_gold(),
        description=(
            f"**Host Kingdom:** {host_location}\n"
            f"**Season:** {season}\n\n"
            "The final horns have faded, the grounds lie quiet, and before crown and court the honors of the tournament are proclaimed."
        ),
    )
    honor_lines = []
    for result in event_results:
        event_label = EVENT_LABELS.get(result["event_type"], result["event_name"])
        champion_name = clean_display_name(result.get("champion_name") or "—")
        line = f"**{event_label}** — {champion_name}"
        if result.get("runner_up_name"):
            line += f" *(Runner-up: {clean_display_name(result['runner_up_name'])})*"
        honor_lines.append(line)
    if not honor_lines:
        honor_lines = ["No completed event honors could be assembled."]
    for i, chunk in enumerate(chunk_lines_for_embed(honor_lines, max_len=1000), start=1):
        embed.add_field(name="Roll of Honors" if i == 1 else f"Roll of Honors ({i})", value=chunk, inline=False)

    if len(champions) == 1:
        champion = champions[0]
        embed.add_field(
            name="👑 Tournament Champion",
            value=f"**{clean_display_name(champion['character_name'])}**\nBy repeated victory and highest honors won, this competitor is proclaimed Tournament Champion.",
            inline=False,
        )
    elif len(champions) > 1:
        champ_lines = [f"**{clean_display_name(row['character_name'])}** — {row['event_wins']} event victories" for row in champions]
        for i, chunk in enumerate(chunk_lines_for_embed(champ_lines, max_len=1000), start=1):
            embed.add_field(name="👑 Tournament Co-Champions" if i == 1 else f"👑 Tournament Co-Champions ({i})", value=chunk, inline=False)
        embed.add_field(name="Proclamation", value="By equal claim of repeated victory, the court names them Tournament Co-Champions of these games.", inline=False)
    else:
        embed.add_field(name="No Tournament Champion Declared", value="Though many took honors, no single competitor won more events than the rest.", inline=False)

    embed.add_field(
        name="Rewards Distributed",
        value=f"**Characters Paid:** {format_val_amount(payout_total)}\n**Host Kingdom Bonus:** {format_val_amount(kingdom_bonus_total)}",
        inline=False,
    )
    embed.set_footer(text="The tournament stands concluded, and its honors are entered into the record.")
    return sanitize_embeds_for_send([embed])

def build_tournament_payout_audit_embed(tournament_row: dict, event_payouts: list[dict], payout_map: dict[int, dict], kingdom_bonus_total: int) -> discord.Embed:
    embed = discord.Embed(
        title="🏆 Tournament Rewards Distributed",
        color=discord.Color.blue(),
        description=(
            f"**Tournament:** {clean_display_name(tournament_row['name'])}\n"
            f"**Host Kingdom:** {clean_display_name(tournament_row.get('host_location') or 'Unknown')}"
        ),
    )
    total_val = sum(int(row.get("val", 0) or 0) for row in payout_map.values())
    embed.add_field(name="Totals", value=f"**Character Payouts:** {total_val} Val\n**Kingdom Bonus:** {kingdom_bonus_total} Val", inline=False)

    event_lines = [f"**{clean_display_name(row['event_name'])}** — Paid {row['event_total_val']} Val | Host Bonus {row['kingdom_bonus_val']} Val" for row in event_payouts]
    for i, chunk in enumerate(chunk_lines_for_embed(event_lines, max_len=1000), start=1):
        embed.add_field(name="By Event" if i == 1 else f"By Event ({i})", value=chunk, inline=False)

    payout_lines = [f"**{clean_display_name(row['character_name'])}** +{int(row.get('val',0))} Val" for row in sorted(payout_map.values(), key=lambda r: (-int(r.get('val',0)), clean_display_name(r['character_name'])))]
    for i, chunk in enumerate(chunk_lines_for_embed(payout_lines, max_len=1000), start=1):
        embed.add_field(name="By Character" if i == 1 else f"By Character ({i})", value=chunk, inline=False)
    return embed

async def post_finalize_outputs(interaction: discord.Interaction, tournament_row: dict, recap_embeds: list[discord.Embed], audit_embed: discord.Embed | None = None) -> tuple[bool, list[str]]:
    notes: list[str] = []
    me = interaction.guild.get_member(bot.user.id) if interaction.guild and bot.user else None
    perms = interaction.channel.permissions_for(me) if me else None
    if perms and (not perms.send_messages or not perms.embed_links):
        notes.append("Tournament data finalized, but I do not have permission to post embeds in this channel.")
        log.warning("Finalize recap post blocked by channel permissions for tournament=%s channel=%s", tournament_row["name"], getattr(interaction.channel, "id", None))
    else:
        try:
            for embed in recap_embeds:
                await interaction.channel.send(embed=embed)
            if audit_embed:
                await interaction.channel.send(embed=audit_embed)
            notes.append("Public tournament recap posted in this channel.")
        except discord.HTTPException as exc:
            log.exception("Finalize channel posts failed for tournament=%s: %s", tournament_row["name"], exc)
            notes.append(f"Tournament data finalized, but public recap posting failed here ({exc.__class__.__name__}).")

    econ_log_channel_id = os.getenv("ECON_LOG_CHANNEL_ID")
    if audit_embed and econ_log_channel_id:
        try:
            ch = interaction.guild.get_channel(int(econ_log_channel_id)) or await bot.fetch_channel(int(econ_log_channel_id))
            await ch.send(embed=audit_embed)
            notes.append("Economy audit log posted.")
        except Exception as exc:
            log.exception("Finalize econ audit post failed for tournament=%s: %s", tournament_row["name"], exc)
            notes.append(f"Economy audit post failed ({exc.__class__.__name__}).")

    return True, notes


@tree.command(name="tourney-finalize", description="Finalize a tournament after all events are complete.")
@app_commands.autocomplete(name=tournament_autocomplete)
async def finalize_tournament(interaction: discord.Interaction, name: str):
    if await deny_if_not_admin(interaction):
        return

    await interaction.response.defer(ephemeral=True)
    log.info("Finalize start requested by user=%s guild=%s tournament=%s", interaction.user.id, interaction.guild.id, name)

    with get_db() as conn:
        t = get_tournament_by_name(conn, interaction.guild.id, name)
        if not t:
            await interaction.followup.send("Tournament not found.", ephemeral=True)
            return

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FILTER (WHERE status <> 'completed') AS remaining, COUNT(*) AS total_events FROM tourney.events WHERE guild_id = %s AND tournament_id = %s",
                (interaction.guild.id, t["id"]),
            )
            counts = cur.fetchone()
            if counts["total_events"] == 0:
                await interaction.followup.send("That tournament has no events to finalize.", ephemeral=True)
                return
            if counts["remaining"] > 0:
                await interaction.followup.send("All events must be completed before the tournament can be finalized.", ephemeral=True)
                return

        if tournament_payouts_already_applied(t):
            await interaction.followup.send("That tournament has already been finalized and paid out.", ephemeral=True)
            return

        event_results = collect_tournament_event_results(conn, interaction.guild.id, t["id"])
        payout_map, event_payouts = build_payout_map(event_results)
        champions = determine_tournament_champions_from_payouts(payout_map)
        host_kingdom = clean_display_name(t.get("host_location") or "Unknown")

        try:
            with conn.cursor() as cur:
                payout_total = apply_tournament_payouts(cur, interaction.guild.id, payout_map)
                kingdom_bonus_total = apply_kingdom_hosting_bonus(cur, guild_id, host_kingdom, event_payouts)
                ensure_tournament_champion_awards(cur, interaction.guild.id, t["id"], t["name"], champions)
                cur.execute("UPDATE tourney.tournaments SET status = 'completed', updated_at = NOW() WHERE id = %s", (t["id"],))
                mark_tournament_payouts_applied(cur, t)
            conn.commit()
            log.info(
                "Finalize DB commit complete tournament=%s payout_total=%s kingdom_bonus_total=%s champions=%s",
                t["name"], payout_total, kingdom_bonus_total, [row["character_name"] for row in champions]
            )
        except Exception as exc:
            conn.rollback()
            log.exception("Finalize DB stage failed for tournament=%s: %s", t["name"], exc)
            await interaction.followup.send(
                f"Tournament finalize failed before public posting. Database changes were rolled back. ({exc.__class__.__name__})",
                ephemeral=True,
            )
            return

    recap_embeds = build_tournament_recap_embeds(t, event_results, champions, payout_total, kingdom_bonus_total)
    audit_embed = build_tournament_payout_audit_embed(t, event_payouts, payout_map, kingdom_bonus_total)
    _, notes = await post_finalize_outputs(interaction, t, recap_embeds, audit_embed=audit_embed)

    lines = [f"**{clean_display_name(name)}** has been finalized and marked `completed`."]
    lines.append(f"Character payouts applied: **{format_val_amount(payout_total)}**")
    lines.append(f"Host kingdom bonus applied: **{format_val_amount(kingdom_bonus_total)}**")
    if len(champions) == 1:
        lines.append(f"Tournament Champion: **{clean_display_name(champions[0]['character_name'])}**")
    elif len(champions) > 1:
        lines.append("Tournament Co-Champions: " + ", ".join(f"**{clean_display_name(row['character_name'])}**" for row in champions))
    else:
        lines.append("No Tournament Champion was declared.")
    lines.extend(notes)
    await interaction.followup.send("\n".join(lines), ephemeral=True)

@tree.command(name="tourney-admin-wipe", description="Nuclear wipe of tournament data for testing.")
@app_commands.choices(mode=WIPE_MODES)
async def admin_wipe(interaction: discord.Interaction, mode: app_commands.Choice[str], confirm_text: str):
    if await deny_if_not_admin(interaction): return
    if confirm_text.strip() != "WIPE": await interaction.response.send_message("Refusing wipe. To confirm, set confirm_text to exactly: WIPE", ephemeral=True); return
    with get_db() as conn:
        with conn.cursor() as cur:
            if mode.value == "keep_profiles":
                cur.execute("TRUNCATE TABLE tourney.match_rolls, tourney.match_participants, tourney.matches, tourney.entries, tourney.events, tourney.tournaments, tourney.character_skill_xp_log, tourney.records, tourney.awards RESTART IDENTITY CASCADE")
                cur.execute("UPDATE tourney.character_skill_xp SET xp_total = 0, rank_bonus = 0, updated_at = NOW()")
            else:
                cur.execute("TRUNCATE TABLE tourney.match_rolls, tourney.match_participants, tourney.matches, tourney.entries, tourney.events, tourney.tournaments, tourney.character_skill_xp_log, tourney.character_skill_xp, tourney.character_profiles, tourney.records, tourney.awards RESTART IDENTITY CASCADE")
        conn.commit()
    msg = "Tournament data wiped. Profiles were kept, and all skill XP was reset to zero." if mode.value == "keep_profiles" else "Tournament data wiped completely. Profiles, XP, records, awards, events, and matches are all gone."
    await interaction.response.send_message(msg, ephemeral=True)

@bot.event
async def setup_hook():
    guild = discord.Object(id=BOT_STATE["guild"])
    tree.copy_global_to(guild=guild)
    synced = await tree.sync(guild=guild)
    log.info("Slash commands synced to guild %s: %s", BOT_STATE["guild"], [cmd.name for cmd in synced])

@bot.event
async def on_ready():
    log.info("Connected to Discord as %s (%s).", bot.user, bot.user.id if bot.user else "unknown")

async def main():
    retry_delay = 30
    max_retry_delay = 300

    while True:
        try:
            await bot.start(BOT_STATE["token"])
            break
        except discord.HTTPException as exc:
            status = getattr(exc, "status", None)
            if status == 429:
                log.warning("Discord login/start was rate limited (HTTP 429). Sleeping for %s seconds before retry.", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
                continue
            raise
        finally:
            if not bot.is_closed():
                await bot.close()

if __name__ == "__main__":
    asyncio.run(main())
