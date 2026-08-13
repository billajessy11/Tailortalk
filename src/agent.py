"""
TailorTalk agent: a thin tool-calling loop over the Anthropic Messages API.

Design choice on the tool schema (see README "Agent design"):
The LLM does NOT receive raw image bytes as a tool argument. Instead, the
Streamlit app stages the user's uploaded/linked image server-side; the LLM
only decides *whether* and *how* (top_k, colour hint) to search. This keeps
the tool call cheap/reliable (no base64 image round-tripping through the
LLM, no risk of the model inventing a file path) while still making the
search a genuine function call the model chooses to invoke based on intent.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from typing import Optional

import anthropic

import config
from src import search as search_mod

SYSTEM_PROMPT = """You are TailorTalk, a warm and knowledgeable shopping assistant for a saree \
catalogue. You chat naturally about sarees — fabric, weave, colour, occasion, styling — the \
way a helpful boutique assistant would.

The user can attach a photo (upload or link) of a saree they like. When they do, and they ask \
you to find similar items, matches, or "something like this", call the `search_similar_sarees` \
tool. Only call it when an image is actually attached in this turn (you'll be told whether one \
is attached) — if they ask to search but haven't attached anything yet, ask them to upload a \
photo or paste a link first, don't call the tool.

After the tool returns, don't just dump the JSON — briefly describe in your own words what \
stands out about the top matches (colour, border/pallu work, fabric feel) based on the score \
breakdown and names provided, in a couple of sentences. The actual result cards with images are \
rendered separately by the app, so you don't need to list every field yourself.

Keep replies concise and conversational."""

TOOLS = [
    {
        "name": "search_similar_sarees",
        "description": (
            "Search the saree catalogue's vector index for items visually similar to the "
            "image the user has attached in this conversation. Returns ranked matches with "
            "similarity scores and a per-signal score breakdown (overall look, pallu/top "
            "region, border/bottom region, fabric texture, colour). Only call this when an "
            "image has actually been attached."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": f"How many matches to return (default {config.DEFAULT_TOP_K}, max {config.MAX_TOP_K}).",
                    "default": config.DEFAULT_TOP_K,
                },
                "notes": {
                    "type": "string",
                    "description": "Brief restatement of what the user is looking for, if they said anything beyond 'find similar' (e.g. 'wants brighter colours', 'cares most about the border').",
                },
            },
            "required": [],
        },
    }
]


class TailorTalkAgent:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def run_turn(
        self,
        history: list[dict],
        user_text: str,
        pending_image=None,  # PIL.Image or None
    ) -> tuple[str, list[dict], Optional[list]]:
        """
        Run one conversational turn.

        Returns:
            assistant_text: final natural-language reply
            new_history: updated message history (Anthropic format) to persist
            tool_results: list[MatchResult] if the search tool was called, else None
        """
        image_note = (
            "\n\n[An image is attached to this message.]" if pending_image is not None
            else "\n\n[No image is attached.]"
        )
        messages = history + [{"role": "user", "content": user_text + image_note}]

        tool_results = None

        resp = self.client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Handle a tool call if the model made one
        tool_use_block = next((b for b in resp.content if b.type == "tool_use"), None)

        if tool_use_block is not None:
            if pending_image is None:
                tool_output = {"error": "No image attached — ask the user to upload or link one."}
            else:
                top_k = tool_use_block.input.get("top_k", config.DEFAULT_TOP_K)
                results = search_mod.search(pending_image, top_k=top_k)
                tool_results = results
                tool_output = search_mod.results_to_json(results)

            messages_with_tool = messages + [
                {"role": "assistant", "content": resp.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": json.dumps(tool_output, default=str),
                        }
                    ],
                },
            ]

            final_resp = self.client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages_with_tool,
            )
            assistant_text = "".join(
                b.text for b in final_resp.content if b.type == "text"
            )
            new_history = messages_with_tool + [
                {"role": "assistant", "content": final_resp.content}
            ]
            return assistant_text, new_history, tool_results

        # No tool call — plain conversational reply
        assistant_text = "".join(b.text for b in resp.content if b.type == "text")
        new_history = messages + [{"role": "assistant", "content": resp.content}]
        return assistant_text, new_history, None
