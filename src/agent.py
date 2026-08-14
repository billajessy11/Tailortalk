"""
TailorTalk agent: a thin tool-calling loop over the Google Gemini API
(via the current `google-genai` SDK).

Uses Gemini instead of a paid API because Google AI Studio's free tier
(https://aistudio.google.com/app/apikey) requires no billing/credit card —
this keeps the whole project runnable at zero cost.

Design choice on the tool schema (see README "Agent design"):
The LLM does NOT receive raw image bytes as a tool argument. Instead, the
Streamlit app stages the user's uploaded/linked image server-side; the LLM
only decides *whether* and *how* (top_k) to search. This keeps the tool call
cheap/reliable (no base64 image round-tripping through the LLM, no risk of
the model inventing a file path) while still making the search a genuine
function call the model chooses to invoke based on intent.
"""
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional

from google import genai
from google.genai import types
from google.genai.errors import ServerError

import config
from src import search as search_mod

SYSTEM_PROMPT = """You are TailorTalk, a warm and knowledgeable shopping assistant for a saree \
catalogue. You chat naturally about sarees — fabric, weave, colour, occasion, styling — the \
way a helpful boutique assistant would.

The user can attach a photo (upload or link) of a saree they like. When they do, and they ask \
you to find similar items, matches, or "something like this", call the `search_similar_sarees` \
function. Only call it when an image is actually attached in this turn (you'll be told whether \
one is attached) — if they ask to search but haven't attached anything yet, ask them to upload \
a photo or paste a link first, don't call the function.

After the function returns, don't just dump the raw data back — briefly describe in your own \
words what stands out about the top matches (colour, border/pallu work, fabric feel) based on \
the score breakdown and names provided, in a couple of sentences. The actual result cards with \
images are rendered separately by the app, so you don't need to list every field yourself.

Keep replies concise and conversational."""

SEARCH_FUNCTION = types.FunctionDeclaration(
    name="search_similar_sarees",
    description=(
        "Search the saree catalogue's vector index for items visually similar to the "
        "image the user has attached in this conversation. Returns ranked matches with "
        "similarity scores and a per-signal score breakdown (overall look, pallu/top "
        "region, border/bottom region, fabric texture, colour). Only call this when an "
        "image has actually been attached."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "top_k": types.Schema(
                type="INTEGER",
                description=f"How many matches to return (default {config.DEFAULT_TOP_K}, max {config.MAX_TOP_K}).",
            ),
            "notes": types.Schema(
                type="STRING",
                description="Brief restatement of what the user is looking for, if they said anything beyond 'find similar' (e.g. 'wants brighter colours', 'cares most about the border').",
            ),
        },
    ),
)

SEARCH_TOOL = types.Tool(function_declarations=[SEARCH_FUNCTION])


class TailorTalkAgent:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.chat_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
        )

    def _send_with_retry(self, chat, message):
        """Send a message to Gemini, retrying on transient 503s (server overload)."""
        for attempt in range(3):
            try:
                return chat.send_message(message)
            except ServerError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)  # 1s, 2s, 4s

    def run_turn(
        self,
        history: list,
        user_text: str,
        pending_image=None,  # PIL.Image or None
    ) -> tuple[str, list, Optional[list]]:
        """
        Run one conversational turn.

        Returns:
            assistant_text: final natural-language reply
            new_history: updated Gemini chat history to persist across turns
            tool_results: list[MatchResult] if the search tool was called, else None
        """
        image_note = (
            "\n\n[An image is attached to this message.]" if pending_image is not None
            else "\n\n[No image is attached.]"
        )

        chat = self.client.chats.create(
            model=config.GEMINI_MODEL, config=self.chat_config, history=history
        )
        response = self._send_with_retry(chat, user_text + image_note)

        fn_call = None
        if response.function_calls:
            fn_call = response.function_calls[0]

        tool_results = None

        if fn_call is not None:
            if pending_image is None:
                tool_output = {"error": "No image attached — ask the user to upload or link one."}
            else:
                args = fn_call.args or {}
                top_k = int(args.get("top_k", config.DEFAULT_TOP_K))
                results = search_mod.search(pending_image, top_k=top_k)
                tool_results = results
                tool_output = {"matches": search_mod.results_to_json(results)}

            response = self._send_with_retry(
                chat,
                types.Part.from_function_response(
                    name="search_similar_sarees",
                    response=tool_output,
                ),
            )

        assistant_text = response.text
        return assistant_text, chat.get_history(), tool_results
