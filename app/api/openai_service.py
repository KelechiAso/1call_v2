# /app/api/openai_service.py

import json
import openai
import os
import traceback
import re
from dotenv import load_dotenv
from typing import Dict, List, Any

from openai import AsyncOpenAI, Timeout

# --- Setup ---
print("--- openai_service.py: TOP OF FILE (Single-Call Architecture) ---")
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("CRITICAL: OPENENAI_API_KEY is not set in openai_service.py.")

client = AsyncOpenAI(
    api_key=openai_api_key,
    timeout=Timeout(180.0, connect=15.0) # Increased timeout for potentially longer calls
)
print("--- openai_service.py: AsyncOpenAI client INITIALIZED ---")


# --- Schemas & Tools (Re-used from original file) ---
# These definitions are crucial for the model to structure data reliably for the UI.
SCHEMA_DATA_H2H = {
    "type": "object", "title": "H2HData", "description": "Data for head-to-head comparisons.",
    "properties": {
        "h2h_summary": {"type": "object", "properties": {
                "team1": {"type": "object", "properties": {"name": {"type": "string"}, "wins": {"type": ["integer", "null"]}, "draws": {"type": ["integer", "null"]}, "losses": {"type": ["integer", "null"]}, "goals_for": {"type": ["integer", "null"]}, "goals_against": {"type": ["integer", "null"]}}, "required": ["name"]},
                "team2": {"type": "object", "properties": {"name": {"type": "string"}, "wins": {"type": ["integer", "null"]}, "draws": {"type": ["integer", "null"]}, "losses": {"type": ["integer", "null"]}, "goals_for": {"type": ["integer", "null"]}, "goals_against": {"type": ["integer", "null"]}}, "required": ["name"]},
                "total_matches": {"type": ["integer", "null"]}}, "required": ["team1", "team2"]},
        "recent_meetings": {"type": "array", "items": {"type": "object", "properties": {"date": {"type": "string", "format": "date"}, "score": {"type": "string"}, "competition": {"type": "string"}}, "required": ["date", "score"]}}},
    "required": ["h2h_summary"]
}
SCHEMA_DATA_MATCH_SCHEDULE_TABLE = {
    "type": "object", "title": "MatchScheduleTableData", "description": "Data for a table of upcoming matches.",
    "properties": {"title": {"type": "string"}, "headers": {"type": "array", "items": {"type": "string"}},
                   "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                   "sort_info": {"type": ["string", "null"]}},
    "required": ["headers", "rows"]
}
SCHEMA_DATA_STANDINGS_TABLE = {
    "type": "object", "title": "StandingsTableData", "description": "Data for a league standings table.",
    "properties": {"league_name": {"type": "string"}, "season": {"type": ["string", "null"]},
                   "standings": {"type": "array", "items": {"type": "object", "properties": {
                       "rank": {"type": ["integer", "string"]}, "team_name": {"type": "string"}, "logo_url": {"type": ["string", "null"], "format": "uri"},
                       "played": {"type": "integer"}, "wins": {"type": "integer"}, "draws": {"type": "integer"}, "losses": {"type": "integer"},
                       "goals_for": {"type": "integer"}, "goals_against": {"type": "integer"}, "goal_difference": {"type": "integer"}, "points": {"type": "integer"},
                       "form": {"type": ["string", "null"]}}, "required": ["rank", "team_name", "played", "points"]}}},
    "required": ["league_name", "standings"]
}
SCHEMA_DATA_PLAYER_PROFILE = {
    "type": "object", "title": "PlayerProfileData", "description": "Detailed profile information for a specific player.",
    "properties": {"full_name": {"type": "string"}, "common_name": {"type": ["string", "null"]}, "nationality": {"type": "string"}, "date_of_birth": {"type": "string", "format": "date"}, "age": {"type": "integer"}, "primary_position": {"type": "string"}, "secondary_positions": {"type": "array", "items": {"type": "string"}}, "current_club_name": {"type": ["string", "null"]}, "jersey_number": {"type": ["integer", "string", "null"]}, "height_cm": {"type": ["integer", "null"]}, "weight_kg": {"type": ["integer", "null"]}, "preferred_foot": {"type": ["string", "null"], "enum": [None, "Right", "Left", "Both"]},
                   "career_summary_stats": {"type": "object", "properties": {"appearances": {"type": ["integer", "null"]}, "goals": {"type": ["integer", "null"]}, "assists": {"type": ["integer", "null"]}}},
                   "market_value": {"type": ["string", "null"]}},
    "required": ["full_name", "nationality", "date_of_birth", "primary_position"]
}
SCHEMA_DATA_TEAM_NEWS = {
    "type": "object", "title": "TeamNewsData", "description": "Latest news articles or summaries for a specific team.",
    "properties": {"team_name": {"type": "string"},
                   "news_articles": {"type": "array", "items": {"type": "object", "properties": {"title": {"type": "string"}, "source_name": {"type": ["string", "null"]}, "published_date": {"type": ["string", "null"], "format": "date-time"}, "url": {"type": ["string", "null"], "format": "uri"}, "summary": {"type": "string"}}, "required": ["title", "summary"]}}},
    "required": ["team_name", "news_articles"]
}
# (Add other schemas if they exist in your original full file)
SCHEMA_DATA_TEAM_STATS = {
    "type": "object", "title": "TeamStatsData", "description": "Comprehensive statistics for a specific sports team, possibly broken into sections (e.g., offense, defense).",
    "properties": {
        "title": {"type": "string", "description": "A descriptive title for the statistics, e.g., 'Manchester United 2023-2024 Season Stats'."},
        "stats_type": {"type": "string", "description": "The type of statistics, e.g., 'overall', 'home', 'away', 'attack', 'defense'."},
        "narrative_summary": {"type": ["string", "null"], "description": "A brief narrative summary or overview of the team's performance based on the stats."},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_title": {"type": "string", "description": "Title of the statistical section (e.g., 'General Stats', 'Attacking', 'Defensive')."},
                    "key_value_pairs": {
                        "type": "object",
                        "description": "Key-value pairs of statistics (e.g., {'Matches Played': 38, 'Goals Scored': 75}).",
                        "additionalProperties": {"type": ["string", "number", "boolean", "null"]}
                    }
                },
                "required": ["section_title", "key_value_pairs"]
            }
        },
        "disclaimer": {"type": ["string", "null"], "description": "Any relevant disclaimers about the data accuracy or source."}
    },
    "required": ["title", "stats_type"]
}

SCHEMA_DATA_RESULTS_LIST = {
    "type": "object", "title": "MatchResultsList", "description": "A list of past match results.",
    "properties": {
        "title": {"type": "string", "description": "A title for the results list, e.g., 'Recent Premier League Results'."},
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "format": "date", "description": "Date of the match."},
                    "time": {"type": ["string", "null"], "description": "Time of the match (e.g., '15:00')."},
                    "home_team_name": {"type": "string"},
                    "away_team_name": {"type": "string"},
                    "score": {
                        "type": "object",
                        "properties": {
                            "fulltime": {"type": "object", "properties": {"home": {"type": "integer"}, "away": {"type": "integer"}}},
                            "halftime": {"type": ["object", "null"], "properties": {"home": {"type": "integer"}, "away": {"type": "integer"}}}
                        },
                        "description": "Scores at fulltime and optionally halftime."
                    },
                    "competition": {"type": ["string", "null"], "description": "The league or competition."},
                    "round": {"type": ["string", "null"], "description": "The match round (e.g., 'Matchday 1')."},
                    "status": {"type": "string", "description": "Match status (e.g., 'Finished')."}
                },
                "required": ["date", "home_team_name", "away_team_name", "score", "status"]
            }
        }
    },
    "required": ["matches"]
}

SCHEMA_DATA_LIVE_MATCH_FEED = {
    "type": "object", "title": "LiveMatchFeedData", "description": "Real-time updates for a live sports match.",
    "properties": {
        "match_id": {"type": "string", "description": "Unique identifier for the match."},
        "home_team_name": {"type": "string"},
        "away_team_name": {"type": "string"},
        "home_team_score": {"type": "integer"},
        "away_team_score": {"type": "integer"},
        "current_minute": {"type": ["integer", "null"], "description": "Current minute of the match."},
        "status_description": {"type": "string", "description": "e.g., 'Half Time', 'Live', 'Full Time'."},
        "competition": {"type": ["string", "null"]},
        "venue": {"type": ["string", "null"]},
        "key_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "minute": {"type": "integer"},
                    "type": {"type": "string", "enum": ["goal", "yellow card", "red card", "substitution", "penalty"]},
                    "player_name": {"type": "string"},
                    "team_name": {"type": "string"},
                    "detail": {"type": ["string", "null"], "description": "e.g., 'Penalty Goal', 'Own Goal'."}
                },
                "required": ["minute", "type", "player_name", "team_name"]
            },
            "description": "Chronological list of key events."
        },
        "live_stats": {
            "type": "object",
            "description": "Key live statistics (e.g., 'Possession', 'Shots on Target').",
            "additionalProperties": {"type": ["string", "number", "null"]}
        },
        "narrative_updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short textual updates about the match flow."
        }
    },
    "required": ["match_id", "home_team_name", "away_team_name", "home_team_score", "away_team_score", "status_description"]
}

TOOLS_AVAILABLE = [
    {"type": "function", "function": {"name": "present_h2h_comparison", "description": "Presents a head-to-head comparison between two teams. Use this when the user asks for historical match-ups, rivalry statistics, or past results between two specific teams.", "parameters": SCHEMA_DATA_H2H}},
    {"type": "function", "function": {"name": "display_standings_table", "description": "Displays a league standings table. Use this when the user asks for league tables, positions, or current rankings in a competition.", "parameters": SCHEMA_DATA_STANDINGS_TABLE}},
    {"type": "function", "function": {"name": "show_match_schedule", "description": "Shows a schedule of upcoming matches for a specific day or period. Use this when the user asks for 'what matches are on today', 'upcoming games', or a schedule for a team/league.", "parameters": SCHEMA_DATA_MATCH_SCHEDULE_TABLE}},
    {"type": "function", "function": {"name": "get_player_profile", "description": "Retrieves detailed information about a sports player. Use this when the user asks for a player's bio, stats, team, nationality, or career details.", "parameters": SCHEMA_DATA_PLAYER_PROFILE}},
    {"type": "function", "function": {"name": "get_team_news", "description": "Fetches latest news articles for a specific sports team. Use this when the user asks for recent news, updates, or headlines about a team.", "parameters": SCHEMA_DATA_TEAM_NEWS}},
    {"type": "function", "function": {"name": "display_team_stats", "description": "Displays comprehensive statistics for a specific sports team. Use this when the user asks for team performance, attacking/defensive stats, or general statistics for a club.", "parameters": SCHEMA_DATA_TEAM_STATS}},
    {"type": "function", "function": {"name": "show_match_results", "description": "Shows a list of past match results. Use this when the user asks for recent scores, results of completed games, or a specific match outcome.", "parameters": SCHEMA_DATA_RESULTS_LIST}},
    {"type": "function", "function": {"name": "display_live_match_feed", "description": "Provides real-time updates and key events for a live sports match. Use this when the user asks for live scores, current match status, or real-time commentary.", "parameters": SCHEMA_DATA_LIVE_MATCH_FEED}},
    # Added example comments to tool descriptions for clarity
]

TOOL_NAME_TO_COMPONENT_TYPE = {
    "present_h2h_comparison": "h2h_comparison_table",
    "display_standings_table": "standings_table",
    "show_match_schedule": "match_schedule_table",
    "get_player_profile": "player_profile_card",
    "get_team_news": "news_article_list",
    "display_team_stats": "team_stats", # Added mapping for new tool
    "show_match_results": "results_list", # Added mapping for new tool
    "display_live_match_feed": "live_match_feed", # Added mapping for new tool
    # (The full mapping from your original file with additional tools)
}
print("--- openai_service.py: Schemas and Tools DEFINED ---")


async def process_user_query(user_query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Single Call: Uses gpt-4o-search-preview to both gather data and generate a
    friendly reply and structured UI data using tools, removing external links.
    """
    print(f"--- Processing query with SINGLE CALL: '{user_query[:60]}...' ---")

    # Define a default response in case of failure or if no specific tool is called
    final_response = {
        "reply": "I'm sorry, I couldn't find the information you were looking for or process your request.",
        "ui_data": {"component_type": "generic_text", "data": {}}
    }

    system_prompt = """
    You are GameNerd, an expert sports AI assistant. Your goal is to be helpful, informative, and concise.
    You have integrated search capabilities to find real-time, comprehensive, and up-to-date sports information.

    Your tasks are:
    1.  **Understand the User's Intent:** Analyze the user's query and the conversation history to determine what sports information they are seeking.
    2.  **Gather Data (Implicit Search):** Use your integrated search to find all necessary factual information (statistics, schedules, player details, news, live scores, etc.).
    3.  **Generate a Friendly Reply:** Formulate a concise and helpful text `reply` based on the gathered information that directly answers the user's query.
        * **CRITICAL:** Your text `reply` MUST NOT contain any markdown links, URLs, or explicit references to sources (e.g., "According to Wikipedia", "from ESPN.com"). Just present the information naturally.
    4.  **Select and Populate Tool (if applicable):** If the query is data-related and the gathered information can be structured for a richer UI experience, select the SINGLE most appropriate `tool` from the available list.
        * Populate ALL required and relevant optional arguments for your chosen tool completely and accurately using the gathered data. Ensure the data matches the schema precisely.
        * If no tool is suitable (e.g., a conversational query like "hello", "who are you?", or an out-of-scope question), do NOT call a tool; simply provide a conversational text `reply`.

    Conversation Examples & Guidelines:
    - If a user asks "Who are you?", introduce yourself as GameNerd, a sports and gaming AI.
    - If a user asks a non-sports question, politely state you only handle sports and gaming.
    - If information is not found, state that clearly in your `reply` and do not call a tool.
    - Prioritize using tools for structured data (tables, lists, profiles) over just text if the information clearly fits a tool's purpose.
    - When providing a text `reply` alongside a tool, ensure the `reply` summarizes or introduces the structured data.
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-6:]) # Use last 3 turns for context
    messages.append({"role": "user", "content": user_query})

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-search-preview", # Use this model for both search and tool calling
            messages=messages,
            tools=TOOLS_AVAILABLE, # Always provide tools, let the model decide
            tool_choice="auto",    # Model decides if it should use a tool or not
            temperature=0.2,       # Keep temperature low for factual consistency and structured output
        )

        response_message = response.choices[0].message

        # 1. Set the text reply
        if response_message.content:
            # Safely strip any lingering markdown links or URLs from the content
            # This regex will remove [link text](url) and bare http/https URLs
            cleaned_reply = re.sub(r'\[(.*?)\]\(http[s]?://.*?\)|http[s]?://[^\s]+', r'\1', response_message.content)
            final_response["reply"] = cleaned_reply.strip()
        
        # 2. Process tool call for structured UI data
        if response_message.tool_calls:
            # Assume only one tool call for simplicity based on prompt's "SINGLE most appropriate tool"
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"!!! ERROR: Could not parse function arguments for tool '{function_name}': {e}")
                print(f"Raw arguments: {tool_call.function.arguments}")
                # Fallback if args are malformed, but try to still provide a text reply if available
                final_response["reply"] = final_response["reply"] or "I tried to get some structured data, but there was an issue formatting it."
                final_response["ui_data"] = {"component_type": "generic_text", "data": {"error": f"Failed to parse UI data for {function_name}."}}
                return final_response # Exit early if args are unusable

            component_type = TOOL_NAME_TO_COMPONENT_TYPE.get(function_name, "generic_text")
            final_response["ui_data"] = {
                "component_type": component_type,
                "data": function_args
            }
            # If the LLM made a tool call but didn't provide a text reply, create a default one
            if not final_response["reply"]:
                final_response["reply"] = "Certainly! Here is the information you requested."
            print(f">>> UI component generated: '{component_type}'")
        else:
            print(">>> No tool call was made. Response is text-only or conversational.")
            # If no tool was called and no text was generated (e.g., initial parsing error), use a fallback.
            if not final_response["reply"]:
                final_response["reply"] = "I've processed your request, but I don't have specific data to show right now."

        return final_response

    except Exception as e:
        print(f"!!! UNEXPECTED ERROR in single-call process_user_query: {e}")
        traceback.print_exc()
        # Return a more specific error for the user
        return {
            "reply": f"I encountered a problem processing your request. Please try again. ({type(e).__name__})",
            "ui_data": {"component_type": "generic_text", "data": {"error": f"Internal server error: {type(e).__name__} - {str(e)}"}}
        }

print("--- openai_service.py: All functions DEFINED. Ready. ---")