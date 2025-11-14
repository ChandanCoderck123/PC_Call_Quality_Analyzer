# 1) Ensure/create the scores table (with TEXT mat_sco_og_pc_script)
#    python realtime_oi_pc_script_match.py reset_db
#
# 2) Process all pending rows (status='pending' AND transcript present) once
#    python realtime_oi_pc_script_match.py match --limit 10 --model gpt-4o
#
# 3) Watch mode (polling): repeatedly scan every N seconds
#    python realtime_oi_pc_script_match.py watch --interval 30 --model gpt-4o
#
# 4) Listen/Notify mode (event-driven): react immediately to INSERTs
#    (Create DB trigger first; see SQL in comments below)
#    python realtime_oi_pc_script_match.py listen --model gpt-4o --sweep 300

"""
PC CALL ANALYZER - PENDING MATCHER (STORE BUCKET LABELS ONLY)
-------------------------------------------------------------

This script:
- Reads canonical og_pc_script from public.pc_ref_table (first row).
- Reads pending pc_recordings (status='pending' AND pc_en_transcribe IS NOT NULL).
- Matches full transcript ↔ full canonical script (GPT-based 0..100).
- Quantizes the score into one of these labels:
    "1-4","5-8","9-12","13-16","17-20","21-24","25-28","29-32","33-36",
    "37-40","41-44","45-48","49-52","53-56","57-60","61-64","65-68",
    "69-72","73-76","77-80","81-84","85-88","89-92","93-96","97-100"
- **Stores ONLY the label** (e.g., "49-52") in pc_match_scores.mat_sco_og_pc_script (TEXT).
- Also stores feedback_full_json (GPT), feedback_auto_json (rule-based),
  gpt_model_used, total_tokens_used, estimated_cost, created_at.
- Marks processed recording to status='successful'.

Modes:
- match  : one-shot batch
- watch  : polling loop
- listen : Postgres LISTEN/NOTIFY (event-driven) with periodic sweep fallback

Environment:
  DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
  OPENAI_API_KEY=sk-...
  GPT_MODEL=gpt-4o  # default, can be overridden
"""

# -------------------- Standard Library Imports --------------------
import os                              # Env vars / OS tools
import time                            # Optional timing/logging
import json                            # JSON encoding/decoding
import argparse                        # CLI parsing
from typing import Optional, Dict      # Type hints

# -------------------- Optional .env Loader ------------------------
try:
    from dotenv import load_dotenv     # Load environment variables from .env
    load_dotenv()                      # Attempt to load .env if present
    print(" Environment variables loaded from .env file")  # Info log
except Exception:
    print(" Using system environment variables")           # Fallback log

# -------------------- SQLAlchemy (DB) Imports ---------------------
from sqlalchemy import create_engine, text        # DB engine + SQL text wrapper
from sqlalchemy.engine import Engine              # Engine type hint

# -------------------- OpenAI (Async) Imports ----------------------
try:
    import openai                                # OpenAI SDK presence
    from openai import AsyncOpenAI               # Async client for Chat Completions
    OPENAI_AVAILABLE = True                      # Flag used to validate capability
except ImportError:
    OPENAI_AVAILABLE = False                     # No OpenAI SDK found
    print(" OpenAI package not installed. Run: pip install openai")

# -------------------- Token Counting (Optional) -------------------
try:
    import tiktoken                              # Tokenizer for cost estimates
    TIKTOKEN_AVAILABLE = True                    # Use when available
except ImportError:
    TIKTOKEN_AVAILABLE = False                   # Optional; proceed without it
    print(" tiktoken not installed (optional). Run: pip install tiktoken")

# ==================== Global Configuration =======================
DATABASE_URL = os.getenv(                        # Read DB DSN from env
    "DATABASE_URL",
    # NOTE: fallback kept to mirror your earlier scripts; remove in prod
    "postgresql+psycopg2://qispineadmin:TrOpSnl1H1QdKAFsAWnY@qispine-db.cqjl02ffrczp.ap-south-1.rds.amazonaws.com:5432/qed_prod"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") # OpenAI API Key from env
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")     # Default model (can override via CLI)

# Table names (consistent with your schema)
REFERENCE_TABLE  = "public.pc_ref_table"         # Holds og_pc_script (reference)
RECORDINGS_TABLE = "public.pc_recordings"        # Holds transcripts + status
SCORES_TABLE     = "public.pc_match_scores"      # Destination scoring table

# In-memory aggregator for this run’s token/cost usage
cost_tracker = {                                 # Initialize counters
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "estimated_cost": 0.0,
    "api_calls": 0
}

# ==================== Small Utility Helpers ======================
def _as_text(x) -> str:
    """Return a string for any input; None becomes empty string."""
    if isinstance(x, str):       # If already str, return unchanged
        return x
    if x is None:                # Map None → ''
        return ""
    return str(x)                # Generic cast to string

def _json_dumps_safe(obj: dict) -> str:
    """Safely dump dict to JSON; '{}' on failure."""
    try:
        return json.dumps(obj, ensure_ascii=False)  # Keep Unicode intact
    except Exception:
        return "{}"                                  # Minimal fallback

# ==================== 4-Point Bucket Quantization =================
import math                                         # For floor operations

BUCKET_STEP = 4                                     # Each bucket spans 4 values

def _clamp_0_100(v: float) -> float:
    """Clamp numeric to 0..100 range."""
    try:
        return max(0.0, min(100.0, float(v)))       # Constrain range
    except Exception:
        return 0.0                                   # Fallback to 0

def quantize_score_to_label(score: float) -> str:
    """
    Convert a raw 0..100 score into **label** string only:
    '1-4','5-8',...,'97-100'. (No numeric returned.)
    """
    s = _clamp_0_100(score)                          # Normalize input score
    if s <= 1.0:                                     # Edge case for ~0/1
        return "1-4"                                 # First bucket label
    # Compute bucket index so that 1..4 is index 0, 5..8 is index 1, etc.
    bucket_index = int(math.floor((s - 1.0000001) / BUCKET_STEP))  # Epsilon pushes 5.0 into next bucket
    low = 1 + bucket_index * BUCKET_STEP             # Lower bound of bucket
    high = low + (BUCKET_STEP - 1)                   # Upper bound of bucket
    if high > 100:                                   # Cap to 100
        high = 100
    if low > 97:                                     # Final bucket is 97–100
        low = 97
    return f"{low}-{high}"                           # Return label, e.g., "49-52"

def label_to_upper_bound(label: str) -> float:
    """
    Utility: Convert '49-52' → 52. This is only for **internal** rules,
    not for DB storage. If parse fails, return 0.0.
    """
    try:
        parts = label.split("-")                     # Split at hyphen
        return float(parts[-1])                      # Use upper bound
    except Exception:
        return 0.0                                   # Fallback to 0

# ==================== Database Utilities =========================
def get_engine() -> Engine:
    """Create a SQLAlchemy engine with pooling."""
    return create_engine(                            # Build engine
        DATABASE_URL,
        pool_pre_ping=True,                          # Validate connections
        pool_size=5,                                 # Base pool size
        max_overflow=5                               # Allow bursts
    )

def fetch_all(engine: Engine, sql: str, params: Optional[dict] = None) -> list:
    """Run SELECT and return rows as list[dict]."""
    with engine.begin() as conn:                     # Transactional context
        rs = conn.execute(text(sql), params or {})   # Execute parameterized SQL
        return [dict(r._mapping) for r in rs.fetchall()]  # Rows to dicts

def run_sql(engine: Engine, sql: str, params: Optional[dict] = None) -> None:
    """Run INSERT/UPDATE/DELETE without returning rows."""
    with engine.begin() as conn:                     # Transactional context
        conn.execute(text(sql), params or {})        # Execute statement

# ==================== Schema Management ==========================
def ensure_scores_table_exists() -> None:
    """Create scores table if missing (mat_sco_og_pc_script as TEXT)."""
    engine = get_engine()                            # Acquire engine
    exists = fetch_all(engine, """                   -- Check existence in information_schema
        SELECT EXISTS (
          SELECT FROM information_schema.tables
          WHERE table_schema='public' AND table_name='pc_match_scores'
        ) AS e;
    """)[0]["e"]
    if exists:                                       # If table present, nothing to do
        return
    ddl = f"""                                       
    CREATE TABLE {SCORES_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        ref_script_code TEXT,
        pc_raw_recordings_s3_file_name TEXT UNIQUE NOT NULL,
        pc_en_transcribe TEXT,
        og_pc_script TEXT,
        mat_sco_og_pc_script TEXT,                   
        feedback_full_json TEXT,
        feedback_auto_json TEXT,
        gpt_model_used TEXT,
        total_tokens_used INTEGER,
        estimated_cost NUMERIC,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    run_sql(engine, ddl)                             # Create table
    print(" Created scores table public.pc_match_scores")  # Log creation

# ==================== Reference / Recording Access ==============
def get_reference_row() -> dict:
    """Fetch first canonical reference row (ordered by script_code)."""
    engine = get_engine()                             # Engine
    rows = fetch_all(engine,                          # Get one reference row
        f"SELECT * FROM {REFERENCE_TABLE} ORDER BY script_code LIMIT 1"
    )
    if not rows:                                      # If empty, abort with guidance
        raise RuntimeError(" No reference data in pc_ref_table. Load Script 1 first.")
    return rows[0]                                    # Return the single dict row

def get_pending_recordings(limit: Optional[int] = None) -> list:
    """Fetch rows with status='pending' and non-null transcript."""
    engine = get_engine()                             # Engine
    base_sql = f"""
        SELECT id, pc_raw_recordings_s3_file_name, pc_en_transcribe
        FROM {RECORDINGS_TABLE}
        WHERE status='pending' AND pc_en_transcribe IS NOT NULL
        ORDER BY id ASC
    """                                               # Base query
    if limit:                                         # Optional limit
        return fetch_all(engine, base_sql + " LIMIT :lim", {"lim": limit})
    return fetch_all(engine, base_sql)                # Return all

def mark_successful(recording_id: int) -> None:
    """Set status='successful' for a processed recording."""
    engine = get_engine()                             # Engine
    run_sql(engine,
        f"UPDATE {RECORDINGS_TABLE} SET status='successful' WHERE id=:i",
        {"i": recording_id}
    )                                                 # Update status

# ==================== GPT Matching & Feedback ===================
class Matcher:
    """GPT wrapper: similarity scoring, feedback generation, usage tracking."""

    def __init__(self, model: str):
        if not OPENAI_AVAILABLE:                      # Ensure SDK available
            raise RuntimeError("OpenAI package not installed. Run: pip install openai")
        if not OPENAI_API_KEY:                        # Ensure API key present
            raise RuntimeError("OPENAI_API_KEY is required.")
        self.model = model                            # Store model
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Build async client

        # Tokenizer setup for estimates (optional)
        if TIKTOKEN_AVAILABLE:                        # Use tiktoken if available
            try:
                self.encoder = tiktoken.encoding_for_model(model)  # Model-specific encoding
            except Exception:
                self.encoder = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
        else:
            self.encoder = None                       # No token counting

        # Nominal USD pricing per 1M tokens (adjust if needed)
        self.pricing_usd_per_mtok = {
            "gpt-4o": {"input": 2.50, "output": 10.00},          # Example pricing
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},    # Example pricing
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},    # Example pricing
        }
        print(f" Matcher initialized with model: {model}")        # Log init

    def _track_usage(self, usage) -> None:
        """Accumulate token usage and estimated cost into cost_tracker."""
        if not usage:                                 # If no server usage field, skip
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0  # Prompt tokens
        ct = getattr(usage, "completion_tokens", 0) or 0  # Completion tokens
        tt = getattr(usage, "total_tokens", 0) or (pt + ct)  # Total tokens

        # Update counters
        cost_tracker["prompt_tokens"] += pt
        cost_tracker["completion_tokens"] += ct
        cost_tracker["total_tokens"] += tt
        cost_tracker["api_calls"] = cost_tracker.get("api_calls", 0) + 1

        # Cost estimate if pricing exists
        price = self.pricing_usd_per_mtok.get(self.model)         # Per-model price row
        if price:
            in_cost = (pt / 1_000_000.0) * price["input"]         # Input cost portion
            out_cost = (ct / 1_000_000.0) * price["output"]       # Output cost portion
            cost_tracker["estimated_cost"] += round(in_cost + out_cost, 6)  # Accumulate rounded cost

    async def score_similarity(self, a: str, b: str) -> float:
        """Return a strict numeric similarity 0..100 via GPT; 0.0 on failure."""
        a = _as_text(a)                                 # Ensure str
        b = _as_text(b)                                 # Ensure str
        if not a.strip() or not b.strip():              # Guard for empty inputs
            return 0.0

        # Strict instruction to return **only a number**
        prompt = f"""
You are a strict grader. Score SEMANTIC similarity (0-100) of two passages.

- 90-100: nearly identical meaning
- 70-89: very similar intent
- 50-69: related but distinct
- 30-49: loosely related
- 10-29: minimal overlap
- 0-9: unrelated

TEXT A: {a}
TEXT B: {b}

Reply with only the number (no words).
""".strip()                                           # Clean final prompt

        try:
            # Chat completion call to get a tiny numeric answer
            resp = await self.client.chat.completions.create(
                model=self.model,                      # Model
                messages=[{"role": "user", "content": prompt}],  # Single message
                max_tokens=10,                         # We expect just a number
                temperature=0.1,                       # Low randomness for stability
                timeout=30                             # Timeout seconds
            )
            # Extract and parse numeric content
            score_text = _as_text(resp.choices[0].message.content).strip()
            score = max(0.0, min(100.0, float(score_text)))      # Clamp to [0,100]
            # Track token usage for costing
            self._track_usage(getattr(resp, "usage", None))
            return score                               # Return numeric similarity
        except Exception as e:
            print(f" GPT similarity error: {e}")       # Log error
            return 0.0                                 # Fallback 0

    async def make_structured_feedback(self, transcript: str, canonical: str) -> dict:
        """Ask GPT for strict-JSON feedback; return minimal dict on error."""
        transcript = _as_text(transcript)              # Ensure str
        canonical = _as_text(canonical)                # Ensure str
        if not transcript.strip() or not canonical.strip():   # Empty guard
            return {"summary": "Insufficient text to analyze.", "categories": []}

        # System message to enforce JSON-only response with the schema
        system_msg = (
            "You are a QA auditor for patient counsellor calls. "
            "Compare the COUNSELLOR'S transcript to the canonical script and return STRICT JSON only."
        )

        # JSON schema instructions with both texts
        user_msg = f"""
Return ONLY JSON per this schema (no prose outside JSON):

{{
  "summary": "one-paragraph summary of deviations and overall quality",
  "overall_rating": "Excellent|Good|Fair|Poor",
  "categories": [
    {{
      "name": "Communication clarity",
      "score": 0-100,
      "strengths": ["point", "..."],
      "improvements": ["point", "..."],
      "examples_from_transcript": ["short quote or paraphrase", "..."]
    }},
    {{
      "name": "Tone of voice and sentiment",
      "score": 0-100,
      "strengths": [],
      "improvements": [],
      "examples_from_transcript": []
    }},
    {{
      "name": "Conversation flow and adherence to script",
      "score": 0-100,
      "strengths": [],
      "improvements": [],
      "examples_from_transcript": []
    }},
    {{
      "name": "Empathy and professionalism",
      "score": 0-100,
      "strengths": [],
      "improvements": [],
      "examples_from_transcript": []
    }},
    {{
      "name": "Call outcome alignment",
      "score": 0-100,
      "strengths": [],
      "improvements": [],
      "examples_from_transcript": []
    }}
  ],
  "top_3_actions": ["action 1", "action 2", "action 3"]
}}

Canonical script:
{canonical}

Counsellor transcript:
{transcript}
""".strip()                                           # Clean final prompt

        try:
            # Query GPT for JSON feedback
            resp = await self.client.chat.completions.create(
                model=self.model,                      # Model
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                max_tokens=800,                        # Room for structured JSON
                temperature=0.2,                       # Low randomness
                timeout=60                             # Timeout seconds
            )
            # Extract content
            content = _as_text(resp.choices[0].message.content).strip()
            # Track usage for cost estimates
            self._track_usage(getattr(resp, "usage", None))
            # Strip code fences if present
            if content.startswith("```"):
                content = content.strip("`")
                if "\n" in content:
                    content = content.split("\n", 1)[1]
            # Parse JSON
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"summary": "Unexpected feedback format.", "raw": content}
        except Exception as e:
            print(f" GPT feedback error: {e}")         # Log error
            return {"summary": "Feedback generation failed.", "error": str(e)}  # Minimal fallback

# ==================== Auto Feedback Builder =====================
from typing import Optional

def build_auto_feedback(bucket_label: Optional[str]) -> dict:
    """
    Rule-based auto feedback using **label** only for storage.
    Accepts None or invalid label and returns a clear payload.
    """
    # If no valid label provided, return a 'no_score' payload
    if not bucket_label or not isinstance(bucket_label, str) or "-" not in bucket_label:
        return {
            "full_script_bucket_label": None,
            "full_script_bucket_band": "unknown",
            "hints": ["No similarity score computed — only structured feedback available."]
        }

    ub = label_to_upper_bound(bucket_label)           # Convert "49-52" → 52.0

    def band(s: float) -> str:                        # Qualitative band by upper bound
        if s >= 85: return "excellent"
        if s >= 70: return "good"
        if s >= 55: return "fair"
        return "poor"

    payload = {                                       # Build JSON payload
        "full_script_bucket_label": bucket_label,     # Exact stored label
        "full_script_bucket_band": band(ub),          # Derived qualitative band
        "hints": []
    }
    if ub < 85:
        payload["hints"].append("Strengthen adherence to canonical phrasing for key transitions.")
    if ub < 70:
        payload["hints"].append("Clarify explanations (diagnosis, stages, and benefits) in simpler terms.")
    if ub < 55:
        payload["hints"].append("Rehearse opening/closing flows to tighten structure and reduce digressions.")
    return payload

# ==================== Persistence (INSERT) ======================
def insert_match_row(payload: Dict) -> None:
    """Insert one row into public.pc_match_scores (delete+insert for idempotency)."""
    engine = get_engine()                            # Get engine
    ensure_scores_table_exists()                     # Ensure table exists
    run_sql(engine,                                  # Delete prior score for same file (idempotent)
        f"DELETE FROM {SCORES_TABLE} WHERE pc_raw_recordings_s3_file_name = :k",
        {"k": payload["pc_raw_recordings_s3_file_name"]}
    )
    insert_sql = f"""                                -- Insert row with label + feedbacks
    INSERT INTO {SCORES_TABLE} (
        ref_script_code,
        pc_raw_recordings_s3_file_name,
        pc_en_transcribe,
        og_pc_script,
        mat_sco_og_pc_script,
        feedback_full_json,
        feedback_auto_json,
        gpt_model_used,
        total_tokens_used,
        estimated_cost
    ) VALUES (
        :ref_script_code,
        :pc_raw_recordings_s3_file_name,
        :pc_en_transcribe,
        :og_pc_script,
        :mat_sco_og_pc_script,
        :feedback_full_json,
        :feedback_auto_json,
        :gpt_model_used,
        :total_tokens_used,
        :estimated_cost
    )
    """
    run_sql(engine, insert_sql, payload)            # Execute insert

# ==================== Main Processing (batch) ===================
import asyncio                                       # Async runtime for GPT calls

async def run_matching(limit: Optional[int] = None, model: str = GPT_MODEL) -> None:
    """Process pending rows: score, bucket-to-label, store, and mark successful."""
    global cost_tracker                               # Use module-scope counters
    cost_tracker = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "estimated_cost": 0.0, "api_calls": 0}  # Reset counters

    print(f" START: Pending-only matcher (labels) using model={model}")  # Banner
    matcher = Matcher(model=model)                    # GPT wrapper

    ref = get_reference_row()                         # Load canonical reference
    ref_code = _as_text(ref.get("script_code"))       # Script code
    og_script = _as_text(ref.get("og_pc_script"))     # Canonical script

    recs = get_pending_recordings(limit=limit)        # Fetch pending rows
    print(f" Found {len(recs)} pending recording(s) to process")  # Log count

    for i, row in enumerate(recs, start=1):           # Iterate candidates
        rec_id = row["id"]                            # Primary key
        s3_key = _as_text(row["pc_raw_recordings_s3_file_name"])  # S3 key
        transcript = _as_text(row["pc_en_transcribe"])            # Transcript text

        print(f"\n [{i}/{len(recs)}] Matching {s3_key}")          # Progress log
        # raw_score = await matcher.score_similarity(transcript, og_script)  # GPT call #1
        # print(f"  Raw similarity score: {raw_score:.2f}")                  # Raw number
        # label = quantize_score_to_label(raw_score)                         # Bucket label
        # print(f"  Bucket label: {label}")                                  # Label
        # SKIP GPT similarity scoring to avoid extra API call
        # label can be None meaning "no similarity score computed"
        label = None

        # Call GPT only once for structured feedback (keep this)
        fb_full = await matcher.make_structured_feedback(transcript, og_script)  # GPT call #1 (only call now)

        # Build auto feedback; build_auto_feedback should handle None (see recommended change below)
        fb_auto = build_auto_feedback(label)                                # Rule-based feedback

        payload = {                                                         # Persist payload
            "ref_script_code": ref_code,
            "pc_raw_recordings_s3_file_name": s3_key,
            "pc_en_transcribe": transcript,
            "og_pc_script": og_script,
            "mat_sco_og_pc_script": label,
            "feedback_full_json": _json_dumps_safe(fb_full),
            "feedback_auto_json": _json_dumps_safe(fb_auto),
            "gpt_model_used": model,
            "total_tokens_used": cost_tracker.get("total_tokens", 0),
            "estimated_cost": cost_tracker.get("estimated_cost", 0.0),
        }
        insert_match_row(payload)                                            # Insert row
        print("  Saved match row → public.pc_match_scores")                  # Log success
        mark_successful(rec_id)                                              # Update status
        print("  Marked recording as successful")                            # Log status change

    print("\n==== GPT USAGE SUMMARY ====")                                    # Footer
    print(f" Total API calls: {cost_tracker.get('api_calls', 0)}")            # Calls
    print(f" Total tokens:   {cost_tracker.get('total_tokens', 0)}")          # Tokens
    print(f" Est. cost:      ${cost_tracker.get('estimated_cost', 0.0):.6f}") # Cost

# ==================== Watch Mode (polling) ======================
async def watch_polling(interval_seconds: int = 30, model: str = GPT_MODEL):
    """Continuously poll for pending rows every interval and process them."""
    print(f" WATCH: polling every {interval_seconds}s for new pending rows...")  # Banner
    while True:                                           # Endless loop
        try:
            await run_matching(limit=None, model=model)   # Process all pending
        except Exception as e:
            print(f" WATCH error (continuing): {e}")      # Log and continue
        await asyncio.sleep(interval_seconds)             # Sleep until next poll

# ================== Listen/Notify Mode (event-driven) ===========
# -- SQL we must run once on our Postgres (RDS) to enable NOTIFY:
#
# CREATE OR REPLACE FUNCTION public.notify_pc_recordings_pending()
# RETURNS trigger
# LANGUAGE plpgsql
# AS $$
# BEGIN
#   IF NEW.status = 'pending' AND NEW.pc_en_transcribe IS NOT NULL THEN
#     PERFORM pg_notify(
#       'pc_recordings_pending',
#       json_build_object(
#         'id', NEW.id,
#         'pc_raw_recordings_s3_file_name', NEW.pc_raw_recordings_s3_file_name
#       )::text
#     );
#   END IF;
#   RETURN NEW;
# END;
# $$;
#
# DROP TRIGGER IF EXISTS trg_pc_recordings_pending ON public.pc_recordings;
# CREATE TRIGGER trg_pc_recordings_pending
# AFTER INSERT ON public.pc_recordings
# FOR EACH ROW
# EXECUTE FUNCTION public.notify_pc_recordings_pending();

import select                                   # For waiting on notifications
import json as _json                            # For decoding NOTIFY payloads

async def process_one_by_id(rec_id: int, model: str = GPT_MODEL):
    """Process exactly one recording ID if still pending & transcript present."""
    engine = get_engine()                           # DB engine
    # Fetch the specific row by ID ensuring it's still pending and has transcript
    row = fetch_all(engine, f"""
        SELECT id, pc_raw_recordings_s3_file_name, pc_en_transcribe
        FROM {RECORDINGS_TABLE}
        WHERE id = :i AND status='pending' AND pc_en_transcribe IS NOT NULL
        LIMIT 1
    """, {"i": rec_id})
    if not row:                                     # Nothing to process
        return

    ref = get_reference_row()                       # Reference row
    ref_code = _as_text(ref.get("script_code"))     # Script code
    og_script = _as_text(ref.get("og_pc_script"))   # Canonical script

    matcher = Matcher(model=model)                  # GPT wrapper

    s3_key = _as_text(row[0]["pc_raw_recordings_s3_file_name"])  # S3 key
    transcript = _as_text(row[0]["pc_en_transcribe"])            # Transcript

    # raw_score = await matcher.score_similarity(transcript, og_script)  # GPT call #1
    # label = quantize_score_to_label(raw_score)                         # Label
    # fb_full = await matcher.make_structured_feedback(transcript, og_script)  # GPT call #2
    # fb_auto = build_auto_feedback(label)                                # Rule feedback
    # SKIP GPT similarity scoring to avoid extra API call
    label = None

    # Single GPT call for structured feedback only
    fb_full = await matcher.make_structured_feedback(transcript, og_script)  # GPT call (only call now)

    # Rule-based feedback must handle None
    fb_auto = build_auto_feedback(label)

    payload = {                                                         # Persist payload
        "ref_script_code": ref_code,
        "pc_raw_recordings_s3_file_name": s3_key,
        "pc_en_transcribe": transcript,
        "og_pc_script": og_script,
        "mat_sco_og_pc_script": label,
        "feedback_full_json": _json_dumps_safe(fb_full),
        "feedback_auto_json": _json_dumps_safe(fb_auto),
        "gpt_model_used": model,
        "total_tokens_used": cost_tracker.get("total_tokens", 0),
        "estimated_cost": cost_tracker.get("estimated_cost", 0.0),
    }
    insert_match_row(payload)                                            # Insert row
    mark_successful(rec_id)                                              # Update source
    print(f" Event-processed recording id={rec_id} ({s3_key})")          # Log

async def watch_listen_notify(model: str = GPT_MODEL, sweep_every_seconds: int = 300):
    """
    LISTEN on 'pc_recordings_pending' and process incoming IDs.
    Also do an occasional full sweep to avoid missing anything.
    """
    print(" WATCH: LISTEN/NOTIFY mode on channel 'pc_recordings_pending'")  # Banner
    engine = get_engine()                           # SQLAlchemy engine
    conn = engine.raw_connection()                  # psycopg2 connection
    conn.set_isolation_level(0)                     # autocommit for LISTEN
    curs = conn.cursor()                            # Cursor
    curs.execute("LISTEN pc_recordings_pending;")   # Start listening
    conn.commit()                                   # Commit LISTEN

    last_sweep = time.time()                        # Track last sweep time

    try:
        while True:                                 # Loop forever
            # Wait up to 5 seconds for notifications (non-blocking check)
            if select.select([conn], [], [], 5) == ([], [], []):
                pass  # timeout; no notifications
            conn.poll()                             # Poll connection for notices
            while conn.notifies:                    # Drain notifications
                note = conn.notifies.pop(0)         # Pop one notification
                try:
                    payload = _json.loads(note.payload)   # Parse JSON payload
                    rec_id = int(payload.get("id"))       # Extract ID
                    await process_one_by_id(rec_id, model=model)  # Process that row
                except Exception as e:
                    print(f" WATCH notify error: {e}")     # Log and continue

            # Periodic safety sweep for missed rows
            if time.time() - last_sweep > sweep_every_seconds:
                print(" WATCH: periodic sweep for any missed pending rows...")
                await run_matching(limit=None, model=model)  # Process any leftovers
                last_sweep = time.time()                     # Reset sweep timer
    finally:
        try:
            curs.close()                      # Close cursor on exit
            conn.close()                      # Close connection on exit
        except Exception:
            pass

# ==================== Command Line Interface ====================
def main() -> None:
    """CLI with 'reset_db', 'match', 'watch', and 'listen' commands."""
    parser = argparse.ArgumentParser(                                         # Create parser
        description="PC Call Analyzer - Pending Matcher (stores bucket LABELS only)"
    )
    sub = parser.add_subparsers(dest="command", help="Commands")             # Subcommands

    sub.add_parser("reset_db", help="Create scores table if missing")        # Reset/create cmd

    p_match = sub.add_parser("match", help="Match pending recordings (one-shot)")  # Match cmd
    p_match.add_argument("--limit", type=int, default=None, help="Limit number of rows")  # Limit
    p_match.add_argument("--model", default=GPT_MODEL,                       # Model override
                         choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                         help="OpenAI model to use")

    p_watch = sub.add_parser("watch", help="Continuously watch (poll) for new pending rows")  # Watch cmd
    p_watch.add_argument("--interval", type=int, default=30, help="Polling interval seconds") # Interval
    p_watch.add_argument("--model", default=GPT_MODEL,
                         choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                         help="OpenAI model to use")

    p_listen = sub.add_parser("listen", help="Event-driven mode via LISTEN/NOTIFY + trigger") # Listen cmd
    p_listen.add_argument("--model", default=GPT_MODEL,
                          choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                          help="OpenAI model to use")
    p_listen.add_argument("--sweep", type=int, default=300,
                          help="Seconds between safety sweeps for missed rows")

    args = parser.parse_args()                                               # Parse args

    if args.command == "reset_db":                                           # reset_db branch
        ensure_scores_table_exists()                                         # Ensure table
        print(" Scores table is ready.")                                     # Log OK
        return                                                               # Exit

    elif args.command == "match":                                            # match branch
        asyncio.run(run_matching(limit=args.limit, model=args.model))        # Run async loop
        return                                                               # Exit

    elif args.command == "watch":                                            # watch branch
        asyncio.run(watch_polling(interval_seconds=args.interval, model=args.model))  # Start poller
        return

    elif args.command == "listen":                                           # listen branch
        asyncio.run(watch_listen_notify(model=args.model, sweep_every_seconds=args.sweep))  # Start listener
        return

    else:                                                                    # No command
        parser.print_help()                                                  # Show usage

# ==================== Entry Point ===============================
if __name__ == "__main__":                                                   # Direct execution
    try:
        main()                                                               # Run CLI
    except KeyboardInterrupt:
        print("\n Operation cancelled by user")                               # Graceful cancel
        raise SystemExit(1)                                                  # Exit non-zero
    except Exception as e:
        print(f" FATAL: {e}")                                                # Log error
        raise SystemExit(1)                                                  # Exit non-zero

