# filename: main.py
# This is the main FastAPI application file for a streaming LLM backend.

import os
import traceback
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, AsyncGenerator

# --- Service Import ---
# We will use a separate service file for clarity
from .api.openai_service_stream import stream_llm_response

# --- App Setup ---
app = FastAPI(
    title="Streaming Chatbot Microservice",
    description="Provides streaming responses using an OpenAI LLM."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Models ---
class ChatRequest(BaseModel):
    user_id: str = "default_user"
    query: str

# --- Routes ---

# Route to serve the HTML file (unchanged from your original)
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def read_index():
    html_file_path = "app/static/htmlsim.html"
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="Index HTML not found.")
    return FileResponse(html_file_path)

# New streaming chat route
@app.post("/chat-stream")
async def handle_chat_stream(request: ChatRequest):
    """
    Handles a streaming chat request. The response is sent token by token.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    print(f"--- /chat-stream CALLED by user: {request.user_id}, query: '{request.query[:60]}...' ---")
    
    try:
        # We wrap the service function in a StreamingResponse
        return StreamingResponse(
            stream_llm_response(request.query),
            media_type="text/event-stream"
        )
    except Exception as e:
        error_type = type(e).__name__
        print(f"!!! UNHANDLED EXCEPTION in /chat-stream endpoint: {error_type} - {e}")
        traceback.print_exc()
        # Return a non-streaming error response if the stream fails to start
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': True, 'message': f'Server error: {error_type}'})}\n\n"]),
            media_type="text/event-stream"
        )