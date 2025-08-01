# filename: openai_service_stream.py
# This service file demonstrates using the OpenAI API for streaming responses.

import os
import openai
import json
from dotenv import load_dotenv
from typing import AsyncGenerator

from openai import AsyncOpenAI

# --- Setup ---
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("CRITICAL: OPENAI_API_KEY is not set.")

client = AsyncOpenAI(api_key=openai_api_key)

print("--- openai_service_stream.py: AsyncOpenAI client INITIALIZED for streaming ---")

async def stream_llm_response(user_query: str) -> AsyncGenerator[str, None]:
    """
    Calls the OpenAI API with stream=True and yields the content of each chunk.
    This function is an asynchronous generator.
    """
    # System prompt to guide the model's behavior and formatting.
    system_prompt = """
    You are a helpful and informative AI assistant.
    Respond to the user's query by providing a response that is clearly formatted
    using Markdown. Use headings, bold text, and bullet points to make the
    information easy to read and understand.
    """

    try:
        # Use `client.chat.completions.create` with `stream=True`
        stream = await client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            stream=True
        )

        async for chunk in stream:
            # Check if there is content in the chunk
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                # Yield the content from the chunk
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"!!! UNEXPECTED ERROR during streaming: {e}")
        traceback.print_exc()
        # Yield a formatted error message
        yield f"An error occurred while streaming the response: {e}"

print("--- openai_service_stream.py: Streaming function DEFINED. ---")