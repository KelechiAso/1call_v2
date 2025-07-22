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
print("--- openai_service.py: TOP OF FILE (Text-Only Architecture) ---")
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("CRITICAL: OPENAI_API_KEY is not set in openai_service.py.")

client = AsyncOpenAI(
    
    api_key=openai_api_key,
    timeout=Timeout(180.0, connect=15.0) # Increased timeout for potentially longer calls
)
print("--- openai_service.py: AsyncOpenAI client INITIALIZED ---")

# --- NO SCHEMAS & TOOLS ARE NEEDED FOR TEXT-ONLY RESPONSES ---
# We are explicitly moving away from structured tool calls.
# The following variables will be removed or not used:
# SCHEMA_DATA_H2H, SCHEMA_DATA_MATCH_SCHEDULE_TABLE, SCHEMA_DATA_STANDINGS_TABLE,
# SCHEMA_DATA_PLAYER_PROFILE, SCHEMA_DATA_TEAM_NEWS, SCHEMA_DATA_TEAM_STATS,
# SCHEMA_DATA_RESULTS_LIST, SCHEMA_DATA_LIVE_MATCH_FEED
# TOOLS_AVAILABLE
# TOOL_NAME_TO_COMPONENT_TYPE

# You can either comment them out or delete them entirely if they are no longer used anywhere else.
# For clarity in this explanation, I'll remove them.
print("--- openai_service.py: Schemas and Tools REMOVED (for text-only output) ---")


async def process_user_query(user_query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Single Call: Uses gpt-4o-search-preview to gather data and generate a
    friendly, pure-text reply, ensuring no external links or source references.
    """
    print(f"--- Processing query with TEXT-ONLY SINGLE CALL: '{user_query[:60]}...' ---")

    # Define a default response in case of failure
    final_response = {
        "reply": "I'm sorry, I couldn't find the information you were looking for or process your request. Please try again.",
        "ui_data": {"component_type": "generic_text", "data": {}} # ui_data will always be generic_text
    }

    system_prompt = """
    You are GameNerd, an expert sports AI assistant. Your goal is to be helpful, informative, concise, and easy to read.
    You have integrated search capabilities to find real-time, comprehensive, and up-to-date sports information.

    Your tasks are:
    1.  **Understand the User's Intent:** Analyze the user's query and the conversation history to determine what sports information they are seeking.
    2.  **Gather Data (Implicit Search):** Use your integrated search to find all necessary factual information (statistics, schedules, player details, news, live scores, etc.).
    3.  **Generate a Friendly and STRUCTURED Reply:** Formulate a concise and helpful text `reply` based on the gathered information.
        * **CRITICAL for formatting:** Organize the information in a clear, modular way using headings, bullet points, and consistent text formatting (like `**bold**`).
        * For schedules, group by sport, then by league/competition if possible. Use clear labels.
        * Separate different categories of information with blank lines or text dividers.
        * **DO NOT** use actual HTML tables or complex structures. Stick to text-based formatting.
        * Your text `reply` MUST NOT contain any markdown links, URLs, or explicit references to sources (e.g., "According to Wikipedia", "from ESPN.com", "Source: BBC"). Just present the information naturally and concisely.
        * Do NOT suggest visiting external websites or providing URLs.
    4.  **Handle Conversational/Out-of-Scope:** If the query is conversational (e.g., "hello", "who are you?", "thanks") or clearly out-of-scope (e.g., "what is the capital of France?"), simply provide a direct conversational text `reply` without attempting to find sports data.
    5.  **Information Not Found:** If you cannot find relevant information for a sports-related query, clearly state that the information is not available in your `reply`.

    Conversation Examples & Guidelines:
    - If a user asks "Who are you?", introduce yourself as GameNerd, a sports and gaming AI.
    - If a user asks a non-sports question, politely state you only handle sports and gaming topics.
    - For schedules, use format like:
        **Sport Name**
        - League/Competition: Event/Match (Time/Date)
        - League/Competition: Event/Match (Time/Date)
    """

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-6:]) # Use last 3 turns for context
    messages.append({"role": "user", "content": user_query})

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-search-preview", # Use this model for its search capabilities
            messages=messages,
            # Removed tools and tool_choice as we no longer want structured output via function calling
            #temperature=0.2, # Keep temperature low for factual consistency
        )

        response_message = response.choices[0].message

        # Set the text reply
        if response_message.content:
            # Robustly strip any lingering markdown links or bare URLs from the content
            # This regex removes [link text](url) and bare http/https URLs
            cleaned_reply = re.sub(
                r'\[(.*?)\]\(http[s]?://.*?\)|'  # Markdown links
                r'http[s]?://[^\s]+|'           # Bare URLs
                r'\(\s*(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\s*\)|\(sportshistori\.com\)', # Parenthetical domains like (fourfourtwo.com) or (whattowatch.com) or (sportshistori.com)
                r'\1', # For Markdown links, keep the captured group 1 (the link text). For other patterns, it effectively removes them.
                response_message.content,
                flags=re.IGNORECASE # Ignore case for domain names
            )
            # A final clean-up to remove any extra spaces that might result from removals
            cleaned_reply = re.sub(r'\s{2,}', ' ', cleaned_reply).strip()
            final_response["reply"] = cleaned_reply.strip()
        else:
            # Fallback if the model somehow returns empty content
            final_response["reply"] = "I've processed your request, but I don't have a specific text response right now."

        # ui_data will always be generic_text, so no tool_calls processing is needed
        final_response["ui_data"] = {"component_type": "generic_text", "data": {}} # Reset to ensure it's generic

        print(f">>> Response generated (text-only). Snippet: '{final_response['reply'][:100]}...'")
        return final_response

    except Exception as e:
        print(f"!!! UNEXPECTED ERROR in text-only process_user_query: {e}")
        traceback.print_exc()
        # Return a more specific error for the user
        return {
            "reply": f"I encountered a problem processing your request. Please try again. ({type(e).__name__})",
            "ui_data": {"component_type": "generic_text", "data": {"error": f"Internal server error: {type(e).__name__} - {str(e)}"}}
        }

print("--- openai_service.py: All functions DEFINED. Ready. ---")