import io
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import requests
import streamlit as st
from PIL import Image

import config
from src.agent import TailorTalkAgent
from src import search as search_mod

st.set_page_config(page_title="TailorTalk — Saree Similarity Search", page_icon="🧵", layout="wide")

st.markdown("""
<style>
:root { --maroon: #6b1e2b; --gold: #b8860b; }
.stApp { background: #fbf7f2; }
h1, h2, h3 { color: var(--maroon); }
.match-card {
    border: 1px solid #e6ddd0; border-radius: 10px; padding: 10px;
    background: white; height: 100%;
}
.match-score { color: var(--gold); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🧵 TailorTalk")
st.caption("Chat about sarees, attach a photo, and I'll find the closest visual matches in the catalogue.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []          # Anthropic message-format history
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # what we render in the chat UI
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "pending_image_thumb" not in st.session_state:
    st.session_state.pending_image_thumb = None

# ---------------------------------------------------------------------------
# Sidebar: API key + image intake + dataset status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Setup")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.text_input(
        "Anthropic API key", type="password",
        help="Set ANTHROPIC_API_KEY as a deployment secret to skip this.",
    )

    st.divider()
    st.subheader("Attach an image")
    uploaded = st.file_uploader("Upload a saree photo", type=["jpg", "jpeg", "png", "webp"])
    url_input = st.text_input("...or paste an image URL")

    if uploaded is not None:
        st.session_state.pending_image = Image.open(uploaded).convert("RGB")
        st.session_state.pending_image_thumb = uploaded
    elif url_input:
        try:
            resp = requests.get(url_input, timeout=10)
            resp.raise_for_status()
            st.session_state.pending_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            st.session_state.pending_image_thumb = st.session_state.pending_image
        except Exception as e:
            st.error(f"Couldn't load that image: {e}")

    if st.session_state.pending_image is not None:
        st.image(st.session_state.pending_image, caption="Attached", use_container_width=True)
        if st.button("Clear image"):
            st.session_state.pending_image = None
            st.session_state.pending_image_thumb = None
            st.rerun()

    st.divider()
    st.subheader("Dataset")
    try:
        search_mod.load_resources()
        st.success(f"Index ready — {search_mod._index.ntotal} sarees indexed.")
    except FileNotFoundError as e:
        st.warning(str(e))

# ---------------------------------------------------------------------------
# Chat history render
# ---------------------------------------------------------------------------
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("results"):
            cols = st.columns(min(len(msg["results"]), 5))
            for i, r in enumerate(msg["results"]):
                with cols[i % len(cols)]:
                    st.markdown('<div class="match-card">', unsafe_allow_html=True)
                    st.image(r.image_url, use_container_width=True)
                    st.markdown(f"**{r.name[:45]}**")
                    st.markdown(f'<span class="match-score">score {r.score:.3f}</span>', unsafe_allow_html=True)
                    if r.discounted_price:
                        st.caption(f"₹{r.discounted_price:,.0f}")
                    with st.expander("breakdown"):
                        for k, v in r.score_breakdown.items():
                            st.caption(f"{k}: {v:.3f}")
                    st.markdown(f"[View product]({r.product_url})")
                    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_text = st.chat_input("Ask about sarees, or say 'find similar' after attaching a photo...")

if user_text:
    if not api_key:
        st.error("Add your Anthropic API key in the sidebar first.")
        st.stop()

    st.session_state.display_messages.append({"role": "user", "text": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    agent = TailorTalkAgent(api_key=api_key)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, new_history, results = agent.run_turn(
                    st.session_state.history, user_text, st.session_state.pending_image
                )
            except FileNotFoundError as e:
                reply, results = str(e), None
                new_history = st.session_state.history

        st.markdown(reply)
        if results:
            cols = st.columns(min(len(results), 5))
            for i, r in enumerate(results):
                with cols[i % len(cols)]:
                    st.markdown('<div class="match-card">', unsafe_allow_html=True)
                    st.image(r.image_url, use_container_width=True)
                    st.markdown(f"**{r.name[:45]}**")
                    st.markdown(f'<span class="match-score">score {r.score:.3f}</span>', unsafe_allow_html=True)
                    if r.discounted_price:
                        st.caption(f"₹{r.discounted_price:,.0f}")
                    with st.expander("breakdown"):
                        for k, v in r.score_breakdown.items():
                            st.caption(f"{k}: {v:.3f}")
                    st.markdown(f"[View product]({r.product_url})")
                    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.history = new_history
    st.session_state.display_messages.append({"role": "assistant", "text": reply, "results": results})
