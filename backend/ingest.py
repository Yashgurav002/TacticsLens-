# ingest.py
import json
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBED_MODEL_NAME,
    DATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ------------------------------------------------
# STEP 1 — Load match metadata
# ------------------------------------------------
def load_matches(competition_id: int, season_id: int) -> list[dict]:
    path = Path(DATA_PATH) / "data" / "matches" / str(competition_id) / f"{season_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        matches = json.load(f)
    print(f"Found {len(matches)} matches for competition {competition_id}, season {season_id}")
    return matches


# ------------------------------------------------
# STEP 2 — Load events for one match
# ------------------------------------------------
def load_events(match_id: int) -> list[dict]:
    path = Path(DATA_PATH) / "data" / "events" / f"{match_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f)
    return events


# ------------------------------------------------
# STEP 3 — Load lineups for one match
# Returns {team_name: [player_name, ...]}
# First 11 in the list = starting 11
# ------------------------------------------------
def load_lineups(match_id: int) -> dict:
    path = Path(DATA_PATH) / "data" / "lineups" / f"{match_id}.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            lineups = json.load(f)
        result = {}
        for team in lineups:
            team_name = team["team_name"]
            players = [p["player_name"] for p in team["lineup"]]
            result[team_name] = players
        return result
    except Exception:
        return {}


# ------------------------------------------------
# STEP 4 — Create comprehensive match summary
# ------------------------------------------------
def create_match_summary(match_info: dict, events: list[dict]) -> Document:
    home_team = match_info["home_team"]["home_team_name"]
    away_team = match_info["away_team"]["away_team_name"]
    home_score = match_info["home_score"]
    away_score = match_info["away_score"]
    match_date = match_info["match_date"]
    match_id = match_info["match_id"]
    competition = match_info["competition"]["competition_name"]
    season = match_info["season"]["season_name"]

    # Load starting lineups
    lineups = load_lineups(match_id)
    home_lineup = lineups.get(home_team, [])
    away_lineup = lineups.get(away_team, [])
    home_starting_11 = home_lineup[:11]
    away_starting_11 = away_lineup[:11]

    # ---- build event_id -> player map for assist resolution ----
    event_id_to_player = {}
    for e in events:
        eid = e.get("id")
        pname = e.get("player", {}).get("name", "Unknown")
        if eid:
            event_id_to_player[eid] = pname

    # ---- stat buckets ----
    goals = []
    shots = {home_team: 0, away_team: 0}
    shots_on_target = {home_team: 0, away_team: 0}
    shots_off_target = {home_team: 0, away_team: 0}
    blocked_shots = {home_team: 0, away_team: 0}
    passes = {}
    passes_attempted = {}
    key_passes = {}
    crosses = {}
    long_balls = {}
    dribbles_completed = {}
    dribbles_attempted = {}
    tackles_won = {}
    interceptions = {}
    clearances = {}
    ball_recoveries = {}
    fouls_committed = {}
    fouls_won = {}
    cards = []
    saves = {}
    offsides = {home_team: 0, away_team: 0}
    corners = {home_team: 0, away_team: 0}
    pressures = {}
    aerial_won = {}
    team_passes = {home_team: 0, away_team: 0}

    for event in events:
        etype = event.get("type", {}).get("name", "")
        player = event.get("player", {}).get("name", "Unknown")
        team = event.get("team", {}).get("name", "Unknown")
        minute = event.get("minute", 0)

        if etype == "Shot":
            if team in shots:
                shots[team] += 1
            outcome = event.get("shot", {}).get("outcome", {}).get("name", "")
            technique = event.get("shot", {}).get("technique", {}).get("name", "")
            body_part = event.get("shot", {}).get("body_part", {}).get("name", "")
            xg = event.get("shot", {}).get("statsbomb_xg", 0)
            key_pass_id = event.get("shot", {}).get("key_pass_id")
            open_goal = event.get("shot", {}).get("open_goal", False)
            one_on_one = event.get("shot", {}).get("one_on_one", False)

            if outcome == "Goal":
                # Resolve assist
                if key_pass_id:
                    assist = event_id_to_player.get(key_pass_id, "Unknown")
                elif open_goal:
                    assist = "Solo effort (open goal)"
                else:
                    assist = "Solo effort"

                goals.append({
                    "player": player,
                    "team": team,
                    "minute": minute,
                    "xg": round(xg, 3) if xg else 0,
                    "technique": technique,
                    "body_part": body_part,
                    "assist": assist,
                    "open_goal": open_goal,
                    "one_on_one": one_on_one,
                })
                if team in shots_on_target:
                    shots_on_target[team] += 1

            elif outcome == "Saved":
                if team in shots_on_target:
                    shots_on_target[team] += 1
                gk = event.get("shot", {}).get("saved_by", {})
                if gk:
                    gk_name = gk.get("name", "Unknown")
                    saves[gk_name] = saves.get(gk_name, 0) + 1

            elif outcome in ["Off T", "Wayward"]:
                if team in shots_off_target:
                    shots_off_target[team] += 1

            elif outcome == "Blocked":
                if team in blocked_shots:
                    blocked_shots[team] += 1

        elif etype == "Pass":
            passes_attempted[player] = passes_attempted.get(player, 0) + 1
            outcome = event.get("pass", {}).get("outcome", {}).get("name", "")
            length = event.get("pass", {}).get("length", 0)
            technique = event.get("pass", {}).get("technique", {}).get("name", "")
            is_key = event.get("pass", {}).get("shot_assist") or event.get("pass", {}).get("goal_assist")
            is_cross = event.get("pass", {}).get("cross", False)
            is_long = length > 32 if length else False
            pass_type = event.get("pass", {}).get("type", {}).get("name", "")

            # Successful pass = no outcome or outcome not a failure
            failed_outcomes = {"Incomplete", "Out", "Pass Offside", "Unknown"}
            if not outcome or outcome not in failed_outcomes:
                passes[player] = passes.get(player, 0) + 1
                if team in team_passes:
                    team_passes[team] += 1

            if is_key:
                key_passes[player] = key_passes.get(player, 0) + 1
            if is_cross:
                crosses[player] = crosses.get(player, 0) + 1
            if is_long:
                long_balls[player] = long_balls.get(player, 0) + 1
            if pass_type == "Corner":
                if team in corners:
                    corners[team] += 1

        elif etype == "Dribble":
            dribbles_attempted[player] = dribbles_attempted.get(player, 0) + 1
            outcome = event.get("dribble", {}).get("outcome", {}).get("name", "")
            if outcome == "Complete":
                dribbles_completed[player] = dribbles_completed.get(player, 0) + 1

        elif etype == "Tackle":
            outcome = event.get("tackle", {}).get("outcome", {}).get("name", "")
            if outcome in ["Won", "Success", "Success In Play", "Success Out"]:
                tackles_won[player] = tackles_won.get(player, 0) + 1

        elif etype == "Interception":
            outcome = event.get("interception", {}).get("outcome", {}).get("name", "")
            if outcome not in ["Lost", "Lost In Play", "Lost Out"]:
                interceptions[player] = interceptions.get(player, 0) + 1

        elif etype == "Clearance":
            clearances[player] = clearances.get(player, 0) + 1

        elif etype == "Ball Recovery":
            ball_recoveries[player] = ball_recoveries.get(player, 0) + 1

        elif etype == "Foul Committed":
            fouls_committed[player] = fouls_committed.get(player, 0) + 1
            card = event.get("foul_committed", {}).get("card", {}).get("name", "")
            if card and card != "No Card":
                cards.append(f"{player} ({team}) — {card} at minute {minute}")

        elif etype == "Foul Won":
            fouls_won[player] = fouls_won.get(player, 0) + 1

        elif etype == "Bad Behaviour":
            card = event.get("bad_behaviour", {}).get("card", {}).get("name", "")
            if card and card != "No Card":
                cards.append(f"{player} ({team}) — {card} at minute {minute}")

        elif etype == "Offside":
            if team in offsides:
                offsides[team] += 1

        elif etype == "Pressure":
            pressures[player] = pressures.get(player, 0) + 1

        elif etype == "Duel":
            duel_type = event.get("duel", {}).get("type", {}).get("name", "")
            outcome = event.get("duel", {}).get("outcome", {}).get("name", "")
            if "Aerial" in (duel_type or "") and outcome in ["Won", "Success", "Success In Play", "Success Out"]:
                aerial_won[player] = aerial_won.get(player, 0) + 1

    # ---- helper: format top N as bullet list ----
    def top(d, n=5, suffix=""):
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]
        return "\n".join([f"  - {k}: {v}{suffix}" for k, v in items]) or "  - None"

    # ---- goals text ----
    if goals:
        goals_text = "\n".join([
            f"  - {g['player']} ({g['team']}) at minute {g['minute']} "
            f"| Assist: {g['assist']} "
            f"| xG: {g['xg']} "
            f"| Body part: {g['body_part']}"
            + (" | Open goal" if g['open_goal'] else "")
            + (" | One on one" if g['one_on_one'] else "")
            for g in goals
        ])
    else:
        goals_text = "  - No goals scored"

    # ---- pass accuracy top 5 ----
    pass_acc_lines = []
    for p in sorted(passes.keys(), key=lambda x: passes.get(x, 0), reverse=True)[:5]:
        completed = passes.get(p, 0)
        attempted = passes_attempted.get(p, 0)
        acc = round((completed / attempted) * 100, 1) if attempted > 0 else 0
        pass_acc_lines.append(f"  - {p}: {completed}/{attempted} ({acc}%)")
    pass_acc_text = "\n".join(pass_acc_lines) or "  - None"

    # ---- cards ----
    cards_text = "\n".join([f"  - {c}" for c in cards]) if cards else "  - No cards"

    # ---- lineups ----
    home_lineup_text = ", ".join(home_starting_11) if home_starting_11 else "Not available"
    away_lineup_text = ", ".join(away_starting_11) if away_starting_11 else "Not available"

    # ---- winner ----
    if home_score > away_score:
        winner = home_team
    elif away_score > home_score:
        winner = away_team
    else:
        winner = "Draw"

    content = f"""Match Summary: Goals scored, shots, passes, cards, tackles, dribbles, lineups and full statistics for {home_team} vs {away_team}.
Who scored goals in this match. Who assisted the goals. Final score and scorers. Starting 11 lineup for both teams. Top performers and player statistics.

MATCH INFO:
Competition: {competition} {season}
Date: {match_date}
Home Team: {home_team}
Away Team: {away_team}
Final Score: {home_team} {home_score} - {away_score} {away_team}
Winner: {winner}

STARTING LINEUPS:
  {home_team}: {home_lineup_text}
  {away_team}: {away_lineup_text}

GOALS ({home_score + away_score} total):
{goals_text}

SHOTS:
  - {home_team}: {shots.get(home_team, 0)} total | {shots_on_target.get(home_team, 0)} on target | {shots_off_target.get(home_team, 0)} off target | {blocked_shots.get(home_team, 0)} blocked
  - {away_team}: {shots.get(away_team, 0)} total | {shots_on_target.get(away_team, 0)} on target | {shots_off_target.get(away_team, 0)} off target | {blocked_shots.get(away_team, 0)} blocked

PASSES (completed/attempted):
  Team totals:
  - {home_team}: {team_passes.get(home_team, 0)} completed
  - {away_team}: {team_passes.get(away_team, 0)} completed
  Top individual passers:
{pass_acc_text}

KEY PASSES (created shot opportunities):
{top(key_passes)}

CROSSES:
{top(crosses)}

LONG BALLS:
{top(long_balls)}

CORNERS:
  - {home_team}: {corners.get(home_team, 0)}
  - {away_team}: {corners.get(away_team, 0)}

DRIBBLES COMPLETED:
{top(dribbles_completed)}

TACKLES WON:
{top(tackles_won)}

INTERCEPTIONS:
{top(interceptions)}

CLEARANCES:
{top(clearances)}

BALL RECOVERIES:
{top(ball_recoveries)}

PRESSURES APPLIED:
{top(pressures)}

FOULS COMMITTED:
{top(fouls_committed)}

FOULS WON:
{top(fouls_won)}

AERIAL DUELS WON:
{top(aerial_won)}

GOALKEEPER SAVES:
{top(saves)}

OFFSIDES:
  - {home_team}: {offsides.get(home_team, 0)}
  - {away_team}: {offsides.get(away_team, 0)}

CARDS:
{cards_text}
"""

    return Document(
        page_content=content,
        metadata={
            "match_id": str(match_id),
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "competition": competition,
            "season": season,
            "event_type": "MATCH_SUMMARY",
            "player": "N/A",
            "team": "N/A",
            "count": home_score + away_score,
        },
    )


# ------------------------------------------------
# STEP 5 — Player event chunks
# ------------------------------------------------
def events_to_documents(match_info: dict, events: list[dict]) -> list[Document]:
    home_team = match_info["home_team"]["home_team_name"]
    away_team = match_info["away_team"]["away_team_name"]
    home_score = match_info["home_score"]
    away_score = match_info["away_score"]
    match_date = match_info["match_date"]
    match_id = match_info["match_id"]
    competition = match_info["competition"]["competition_name"]
    season = match_info["season"]["season_name"]

    match_header = (
        f"Match: {home_team} {home_score} - {away_score} {away_team} | "
        f"Date: {match_date} | Competition: {competition} {season}"
    )

    player_events: dict = {}

    for event in events:
        event_type = event.get("type", {}).get("name", "Unknown")
        player = event.get("player", {}).get("name", "Unknown")
        team = event.get("team", {}).get("name", "Unknown")
        minute = event.get("minute", 0)

        if player == "Unknown":
            continue

        key = f"{player}||{event_type}"
        if key not in player_events:
            player_events[key] = {
                "player": player,
                "team": team,
                "event_type": event_type,
                "events": [],
            }

        description = f"Minute {minute}: {event_type} by {player}"

        if event_type == "Shot":
            outcome = event.get("shot", {}).get("outcome", {}).get("name", "")
            technique = event.get("shot", {}).get("technique", {}).get("name", "")
            body_part = event.get("shot", {}).get("body_part", {}).get("name", "")
            xg = event.get("shot", {}).get("statsbomb_xg", "")
            description += f" | Outcome: {outcome} | Technique: {technique} | Body part: {body_part} | xG: {xg}"

        elif event_type == "Pass":
            outcome = event.get("pass", {}).get("outcome", {}).get("name", "Successful")
            length = event.get("pass", {}).get("length", "")
            is_cross = event.get("pass", {}).get("cross", False)
            description += f" | Outcome: {outcome}"
            if length:
                description += f" | Length: {length:.1f}m"
            if is_cross:
                description += " | Cross"

        elif event_type == "Dribble":
            outcome = event.get("dribble", {}).get("outcome", {}).get("name", "")
            description += f" | Outcome: {outcome}"

        elif event_type == "Tackle":
            outcome = event.get("tackle", {}).get("outcome", {}).get("name", "")
            description += f" | Outcome: {outcome}"

        elif event_type == "Interception":
            outcome = event.get("interception", {}).get("outcome", {}).get("name", "")
            description += f" | Outcome: {outcome}"

        player_events[key]["events"].append(description)

    documents = []
    for key, data in player_events.items():
        player = data["player"]
        team = data["team"]
        event_type = data["event_type"]
        all_events = "\n".join(data["events"])
        count = len(data["events"])

        content = (
            f"{match_header}\n"
            f"Player: {player} | Team: {team}\n"
            f"Event Type: {event_type} | Total Count: {count}\n"
            f"Details:\n{all_events}"
        )

        doc = Document(
            page_content=content,
            metadata={
                "match_id": str(match_id),
                "player": player,
                "team": team,
                "event_type": event_type,
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "competition": competition,
                "season": season,
                "count": count,
            },
        )
        documents.append(doc)

    return documents


# ------------------------------------------------
# STEP 6 — Split documents
# ------------------------------------------------
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


# ------------------------------------------------
# STEP 7 — Embed and store in ChromaDB
# ------------------------------------------------
def store_in_chroma(chunks: list[Document]):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")
    return vectorstore


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def run_ingestion():
    COMPETITION_ID = 11
    SEASON_ID = 42

    all_matches = load_matches(COMPETITION_ID, SEASON_ID)
    selected_matches = all_matches[:3]
    all_chunks = []

    for match in selected_matches:
        match_id = match["match_id"]
        home = match["home_team"]["home_team_name"]
        away = match["away_team"]["away_team_name"]
        print(f"\nProcessing: {home} vs {away} (match_id: {match_id})")

        events = load_events(match_id)
        print(f"  Loaded {len(events)} events")

        summary = create_match_summary(match, events)
        all_chunks.append(summary)
        print(f"  Created match summary chunk")

        documents = events_to_documents(match, events)
        print(f"  Created {len(documents)} player documents")

        chunks = split_documents(documents)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all matches: {len(all_chunks)}")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared old ChromaDB")

    store_in_chroma(all_chunks)
    print("\nIngestion complete!")


if __name__ == "__main__":
    run_ingestion()