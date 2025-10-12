import os
import json
import re
import math
from typing import Tuple, Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load .env so OPENAI_API_KEY is available via environment
load_dotenv()

# =====================================
# Common Helpers
# =====================================

def _coerce_to_json(s: str) -> Dict[str, Any]:
    """Try to parse strict JSON. If it fails, attempt to extract the first JSON object via regex."""
    if not s:
        raise ValueError("Empty response from model.")

    try:
        return json.loads(s)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {"error": "Model did not return valid JSON.", "raw": s}


def _responses_call(system: str, user: str, model: str = "gpt-5-mini") -> Dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    text = getattr(resp, "output_text", "") or ""
    if not text and getattr(resp, "output", None):
        parts = []
        for item in resp.output:  # type: ignore[attr-defined]
            if isinstance(item, dict) and item.get("type") == "output_text":
                parts.append(item.get("text", ""))
        text = "".join(parts).strip()

    if not text:
        raise ValueError("Empty response from model.")

    return _coerce_to_json(text)

# =====================================
# Module 1: Generate Tweet
# =====================================

def build_prompts_generate_tweet(
    niche: str,
    topic: str,
    purpose: str,
    instructions: str,
    samples: str,
    n: int,
) -> Tuple[str, str]:
    system = f"""
ROLE
You are a senior growth copywriter for X (Twitter).

OBJECTIVE
Deeply analyze the provided sample tweets to extract a STYLE PLAYBOOK, then write {n} new tweets for the user's topic/purpose that maximize engagement (scroll-stopping hooks, clarity, skimmability, and replies/likes/saves) while staying aligned with the playbook.

FACT POLICY
Do not fabricate precise facts (numbers, dates, rankings, quotes, named studies). If a precise claim is not present in the samples or user input, keep it qualitative.

CONTENT SAFETY & TONE FLEX
Be direct; when it truly fits the voice and audience, you may use strong/informal language or occasional curse words. Avoid slurs, harassment, or targeted abuse. Keep it on-brand and purposeful—never gratuitous.

WORKFLOW
1) ANALYZE SAMPLES → Build a STYLE PLAYBOOK:
   A) Content & Hook: opening devices (question, contrarian, bold claim, curiosity, list, mini-story), common angles.
   B) Tone & Style: voice traits (e.g., direct, witty, authoritative), sentence cadence, punctuation habits.
   C) Structure: line breaks, list usage, rhythm, parallelism, emphasis (caps/quotes), pacing.
   D) Psychological Triggers: curiosity gap, novelty, authority, social proof, tension & release, specificity (without fabricated stats).
   E) Engagement Engineering: CTA style (question, soft prompt, reply bait), conversation starters, pattern interrupts.
   F) Visual & Add-ons: emoji style/density, bullets/dividers/arrows/checkmarks, any visual flourishes.
   G) Platform Dynamics: single vs thread preference, hook placement, typical length range observed.

2) GENERATE TWEETS → Apply the playbook to NICHE/TOPIC/PURPOSE with the sole goal of higher engagement.
   - Use the INSTRUCTIONS exactly as constraints on length, tone, formatting, emoji/hashtag usage, formality, etc.
   - No forced length limit unless specified in INSTRUCTIONS; write the length that best fits the playbook and engagement goal.
   - Use hashtags/links/emojis only if they improve engagement and are consistent with the playbook and INSTRUCTIONS.

3) DIVERSITY → Use a different hook/opening device for each of the {n} tweets (e.g., bold claim, curiosity question, mini-story).

OUTPUT (STRICT JSON ONLY)
{{
  "analysis": {{
    "content_hook": "...",
    "tone_style": "...",
    "structure": "...",
    "psych_triggers": "...",
    "engagement": "...",
    "visual_addons": "...",
    "platform_dynamics": "..."
  }},
  "tweets": ["...", "...", "..."]
}}
""".strip()

    user = f"""
NICHE: {niche}
TOPIC: {topic}
PURPOSE: {purpose}
INSTRUCTIONS: {instructions}
COUNT: {n}

SAMPLES (study carefully; emulate techniques, not content):
---
{samples}
---

TASK
- Produce exactly {n} tweets that apply the STYLE PLAYBOOK derived from SAMPLES.
- Optimize for engagement (hook strength, clarity, cadence, skimmability, replyability), not rigid rules.
- Each tweet must use a different opening device.
- Return STRICT JSON exactly matching the schema in SYSTEM. No preamble, no code fences.
""".strip()

    return system, user


def call_generate_tweet(system: str, user: str, model: str = "gpt-5-mini") -> Dict[str, Any]:
    return _responses_call(system, user, model)

# =====================================
# Module 2: Reply Generator
# =====================================

def _compute_distribution(total_replies: int, tones: int = 10) -> List[int]:
    """Distribute total_replies across tones, roughly equally, preserving total."""
    per = math.ceil(total_replies / tones)
    counts = [per] * tones
    # Trim extra if we overshoot
    extra = tones * per - total_replies
    for i in range(extra):
        idx = tones - 1 - i
        if idx >= 0:
            counts[idx] -= 1
    return [c for c in counts if c > 0]


def build_prompts_reply_generator(
    tweet: str,
    goal: str,
    persona: str,
    opinion_mix: str,
    instructions: str,
    total_replies: int,
) -> Tuple[str, str]:
    # We keep 10 tones max, distribute replies across them
    counts = _compute_distribution(total_replies, tones=10)
    system = f"""
ROLE
You are a senior social copy strategist specializing in high-engagement Twitter replies.

OBJECTIVE
Generate natural, varied, conversation-starting replies under the provided tweet. Replies must feel like they come from a real person (not a brand), match the given persona, and align with the requested opinion mix.

FACT POLICY
Do not fabricate precise facts (numbers, dates, rankings, quotes, named studies). Keep claims qualitative unless present in the source tweet or user input.

STYLE & SAFETY GUARDRAILS
- No emojis or hashtags.
- Informal, direct, opinionated.
- You may use strong/informal language when it fits the persona and context, but avoid slurs, harassment, or targeted abuse.
- Vary length (some short, some mid). Avoid uniformity.

OUTPUT FORMAT (STRICT JSON ONLY)
{{
  "tones": [
    {{"name": "Tone Name", "replies": ["reply 1", "reply 2", "..."]}},
    ... (one object per tone)
  ]
}}
""".strip()

    # Build tone counts string for clarity in the prompt
    tone_counts_desc = ", ".join([f"Tone {i+1}: {c} replies" for i, c in enumerate(counts)])

    user = f"""
TWEET (to reply to):
---
{tweet}
---

GOAL: {goal}
PERSONA: {persona}
OPINION MIX: {opinion_mix}
INSTRUCTIONS: {instructions}

REPLY COUNT (total): {total_replies}
DISTRIBUTION: {tone_counts_desc}

TASK
- Produce replies grouped by labeled tones. Use as many tones (up to 10) as needed to cover the distribution above.
- Each tone should explore a distinct emotional/stylistic range (e.g., Curious, Sarcastic, Skeptical, Supportive, Cocky, Analytical, Dismissive, Humorous, Reflective, Provocative — or others fitting the context).
- Each reply must:
  • Sound like it's from a real personal account — informal, opinionated, natural, and direct.
  • Express a unique POV that could spark conversation, debate, or engagement.
  • Vary length across replies.
  • Use no emojis or hashtags.
  • Match the persona's voice and expertise.
  • Reflect the chosen opinion mix across the set.

CONSTRAINTS
- Return STRICT JSON matching the schema in SYSTEM. No preamble, no code fences.
- The total number of replies across all tones must equal {total_replies}.
- For each tone i, produce exactly the number of replies specified by the DISTRIBUTION.
""".strip()

    return system, user


def call_reply_generator(system: str, user: str, model: str = "gpt-5-mini") -> Dict[str, Any]:
    return _responses_call(system, user, model)

# =====================================
# Streamlit App (Two Modules via Sidebar)
# =====================================

st.set_page_config(page_title="TweetGen • Tweets & Replies", page_icon="🐦", layout="wide")

with st.sidebar:
    st.header("TweetGen")
    module = st.radio("Module", ["Generate Tweet", "Reply Generator"], index=0)
    model = st.selectbox("Model", ["gpt-5-mini"], index=0)
    st.text("API key loaded: ✅" if os.getenv("OPENAI_API_KEY") else "API key missing ❌")
    st.caption("Keys are read from your .env via python-dotenv.")

if module == "Generate Tweet":
    st.title("🐦 Generate Tweets")

    col1, col2 = st.columns(2)
    with col1:
        niche = st.text_input("Niche", placeholder="e.g., SaaS growth, AI in education")
        topic = st.text_input("Topic", placeholder="e.g., cold email frameworks for founders")
        purpose = st.text_area("Purpose", placeholder="e.g., promote Thursday webinar with soft CTA; highlight 2 pains + 1 payoff.")
        n = st.number_input("Number of tweets", min_value=1, max_value=10, value=3, step=1)
    with col2:
        instructions = st.text_area(
            "Instructions (length, tone, etc.)",
            placeholder=(
                "e.g., 1-2 lines each; punchy; no hashtags; informal; allow mild profanity; "
                "one rhetorical question per tweet; end with a soft CTA in first tweet only."
            ),
            height=140,
        )
        samples = st.text_area(
            "Sample Tweets (as many as you want)",
            placeholder=(
                "Paste example tweets here.\n\n"
                "Separate with blank lines; these will be analyzed to build the style playbook."
            ),
            height=220,
        )

    btn = st.button("✨ Generate Tweets", type="primary")

    if btn:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not set. Create a .env with OPENAI_API_KEY=... and restart.")
        else:
            with st.spinner("Calling the model…"):
                system, user = build_prompts_generate_tweet(niche, topic, purpose, instructions, samples, int(n))
                try:
                    result = call_generate_tweet(system, user, model=model)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")
                    result = None

            if result is None:
                pass
            elif "error" in result:
                st.error(result.get("error"))
                with st.expander("Raw Model Output"):
                    st.code(result.get("raw", ""), language="json")
            else:
                # Show analysis
                analysis = result.get("analysis", {})
                with st.expander("Style Playbook — Analysis", expanded=True):
                    st.markdown("**Content & Hook**\n\n" + analysis.get("content_hook", "-"))
                    st.markdown("**Tone & Style**\n\n" + analysis.get("tone_style", "-"))
                    st.markdown("**Structure**\n\n" + analysis.get("structure", "-"))
                    st.markdown("**Psychological Triggers**\n\n" + analysis.get("psych_triggers", "-"))
                    st.markdown("**Engagement Engineering**\n\n" + analysis.get("engagement", "-"))
                    st.markdown("**Visual & Add-ons**\n\n" + analysis.get("visual_addons", "-"))
                    st.markdown("**Platform Dynamics**\n\n" + analysis.get("platform_dynamics", "-"))

                # Show tweets
                tweets = result.get("tweets", [])
                st.subheader("Tweets")
                if not tweets:
                    st.info("No tweets returned.")
                for i, tw in enumerate(tweets, start=1):
                    st.code(tw, language="text")

                # Download full JSON
                st.download_button(
                    "Download JSON",
                    data=json.dumps(result, ensure_ascii=False, indent=2),
                    file_name="tweetgen_result.json",
                    mime="application/json",
                )

else:  # Reply Generator
    st.title("💬 Reply Generator")

    tweet = st.text_area("Tweet (paste the tweet to reply to)", height=160, placeholder="Paste the original tweet text here…")
    goal = st.text_input("Goal", placeholder="e.g., start debate, add humor, challenge idea, share insight")
    persona = st.text_input("Persona", placeholder="e.g., sarcastic marketer, chill founder, investor")
    opinion_mix = st.text_input("Opinion Mix", placeholder="e.g., mostly supportive, mixed, mostly contrarian")
    rg_instructions = st.text_area(
        "Instructions",
        placeholder=(
            "Any constraints on length, tone extremes, taboo topics to avoid, formatting quirks, etc.\n"
            "No emojis/hashtags will be used."
        ),
        height=120,
    )
    total_replies = st.number_input("Number of replies (total)", min_value=1, max_value=100, value=5, step=1)

    btn2 = st.button("⚡ Generate Replies", type="primary")

    if btn2:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not set. Create a .env with OPENAI_API_KEY=... and restart.")
        else:
            with st.spinner("Calling the model…"):
                system, user = build_prompts_reply_generator(tweet, goal, persona, opinion_mix, rg_instructions, int(total_replies))
                try:
                    result = call_reply_generator(system, user, model=model)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")
                    result = None

            if result is None:
                pass
            elif "error" in result:
                st.error(result.get("error"))
                with st.expander("Raw Model Output"):
                    st.code(result.get("raw", ""), language="json")
            else:
                tones = result.get("tones", [])
                if not tones:
                    st.info("No replies returned.")
                else:
                    st.subheader("Tones & Replies")
                    for idx, tone in enumerate(tones, start=1):
                        name = tone.get("name", f"Tone {idx}")
                        replies: List[str] = tone.get("replies", [])
                        with st.expander(f"{idx}. {name} ({len(replies)})", expanded=False):
                            for j, r in enumerate(replies, start=1):
                                st.write(f"{j}. {r}")

                    # Validate total count
                    total = sum(len(t.get("replies", [])) for t in tones)
                    st.caption(f"Total replies generated: {total}")

                    st.download_button(
                        "Download JSON",
                        data=json.dumps(result, ensure_ascii=False, indent=2),
                        file_name="replygen_result.json",
                        mime="application/json",
                    )
