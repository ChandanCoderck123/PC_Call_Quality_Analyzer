#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PC CALL ANALYZER - PENDING MATCHER (STORE BUCKET LABELS ONLY) - FULLY COMMENTED
# This file matches a chosen canonical script from pc_ref_table (default PC_Script002)
# against pending rows in pc_recordings, produces structured JSON feedback (one GPT call),
# optionally derives a numeric bucket label from the feedback, stores only that label,
# stores rule-based hints, and marks recordings successful.
# Each line is commented to explain purpose and flow.
# Usage examples:
#   python realtime_oi_pc_script_match.py reset_db
#   python realtime_oi_pc_script_match.py match --limit 2 --model gpt-4o --script-code PC_Script002
#   python realtime_oi_pc_script_match.py watch --interval 60 --model gpt-4o
#   python realtime_oi_pc_script_match.py listen --sweep 300

# -------------------- Standard library imports --------------------
import os                                      # operating system helpers (env vars, file paths)
import time                                    # time utilities (sleep, time tracking)
import json                                    # JSON encoding/decoding for DB storage
import argparse                                # CLI argument parsing for script modes
import math                                    # math helpers (floor) used for quantization
from typing import Optional, Dict             # type hints for functions and payloads
import asyncio                                 # async runtime for GPT API calls
import select                                  # used for DB notifications in listen mode

# -------------------- Optional .env loader ------------------------
try:
    from dotenv import load_dotenv             # attempt to import load_dotenv to load .env file
    load_dotenv()                              # if present, read .env into environment variables
    print(" Environment variables loaded from .env file")  # informational log when .env loaded
except Exception:
    print(" Using system environment variables")           # fallback message when dotenv not available

# -------------------- SQLAlchemy (DB) imports ---------------------
from sqlalchemy import create_engine, text     # SQLAlchemy engine builder and safe text wrapper
from sqlalchemy.engine import Engine           # Engine type hint for functions that return an engine

# -------------------- OpenAI (async) imports ----------------------
try:
    import openai                              # import OpenAI SDK (modern namespace)
    from openai import AsyncOpenAI             # import AsyncOpenAI client class for async calls
    OPENAI_AVAILABLE = True                    # flag to indicate OpenAI SDK present
except ImportError:
    OPENAI_AVAILABLE = False                   # mark false if import fails
    print(" OpenAI package not installed. Run: pip install openai")  # instruct user to install SDK

# -------------------- Token counting (optional) -------------------
try:
    import tiktoken                             # optional tokenizer for more accurate token counts
    TIKTOKEN_AVAILABLE = True                   # mark availability
except ImportError:
    TIKTOKEN_AVAILABLE = False                  # mark not available
    print(" tiktoken not installed (optional). Run: pip install tiktoken")  # user hint

# ==================== Global configuration =======================
DATABASE_URL = os.getenv(                         # read DB connection string from environment
    "DATABASE_URL",
    "postgresql+psycopg2://qispineadmin:TrOpSnl1H1QdKAFsAWnY@qispine-db.cqjl02ffrczp.ap-south-1.rds.amazonaws.com:5432/qed_prod"
)                                                # fallback DSN (keep secrets out of code in prod)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # read OpenAI API key from env (required for GPT calls)
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")      # default model, can be overridden by CLI or env
PC_SCRIPT_CODE = os.getenv("PC_SCRIPT_CODE", "PC_Script002")  # default script code to match against

REFERENCE_TABLE = "public.pc_ref_table"           # canonical scripts table name
RECORDINGS_TABLE = "public.pc_recordings"         # raw transcripts / statuses table name
SCORES_TABLE = "public.pc_match_scores"           # destination table for labels + feedback
API_CALLS_TABLE = "public.pc_api_calls"           # individual API calls tracking table

# ==================== Small utility helpers ======================
def _as_text(x) -> str:
    """Return a safe string for any input value; None -> '' to avoid 'None' in prompts."""
    if isinstance(x, str):     # if already string, return as-is
        return x
    if x is None:              # map None to empty string to avoid sending 'None' to GPT
        return ""
    return str(x)              # otherwise cast to str

def _json_dumps_safe(obj: dict) -> str:
    """Safely JSON-dump a dict for DB storage; on error return '{}' as a fallback."""
    try:
        return json.dumps(obj, ensure_ascii=False)  # preserve unicode with ensure_ascii=False
    except Exception:
        return "{}"                                 # minimal fallback to keep DB column valid

# ==================== Quantization helpers =======================
BUCKET_STEP = 4  # bucket width of 4 points for labels like "1-4","5-8",...,"97-100"

def _clamp_0_100(v: float) -> float:
    """Clamp float to inclusive 0..100 range, defensively handling bad input."""
    try:
        return max(0.0, min(100.0, float(v)))  # ensure value is between 0 and 100
    except Exception:
        return 0.0                              # return 0 if conversion fails

def quantize_score_to_label(score: float) -> str:
    """Map numeric 0..100 score to textual 4-point bucket label (e.g., 49.2 -> '49-52')."""
    s = _clamp_0_100(score)              # clamp first to ensure valid range
    if s <= 1.0:                         # treat very small values as first bucket
        return "1-4"
    bucket_index = int(math.floor((s - 1.0000001) / BUCKET_STEP))  # compute zero-based bucket index
    low = 1 + bucket_index * BUCKET_STEP  # lower bound of bucket
    high = low + (BUCKET_STEP - 1)        # upper bound (inclusive)
    if high > 100:                        # cap upper bound to 100
        high = 100
    if low > 97:                          # final clamp: final bucket should start at 97
        low = 97
    return f"{low}-{high}"                # return label string like '49-52'

def label_to_upper_bound(label: str) -> float:
    """Parse a label like '49-52' and return its numeric upper bound (52.0)."""
    try:
        parts = label.split("-")           # split label by hyphen
        return float(parts[-1])            # return last part as float (upper bound)
    except Exception:
        return 0.0                         # return 0 if parsing fails

# ==================== Database utilities =========================
def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine configured with DATABASE_URL."""
    return create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)  # create engine with connection pooling

def fetch_all(engine: Engine, sql: str, params: Optional[dict] = None) -> list:
    """Run a SELECT query and return list[dict] rows using engine.begin() transaction context."""
    with engine.begin() as conn:            # create transaction context
        rs = conn.execute(text(sql), params or {})  # execute SQL with parameters
        return [dict(r._mapping) for r in rs.fetchall()]  # convert rows to dictionaries

def run_sql(engine: Engine, sql: str, params: Optional[dict] = None) -> None:
    """Run a non-select SQL statement (INSERT/UPDATE/DELETE) inside a transactional block."""
    with engine.begin() as conn:            # create transaction context
        conn.execute(text(sql), params or {})  # execute SQL with parameters

# ==================== Schema management ==========================
def ensure_scores_table_exists() -> None:
    """Create the pc_match_scores table if it doesn't exist (idempotent DDL)."""
    engine = get_engine()                                 # get DB engine
    exists = fetch_all(engine, """
        SELECT EXISTS (
          SELECT FROM information_schema.tables
          WHERE table_schema='public' AND table_name='pc_match_scores'
        ) AS e;
    """)[0]["e"]                                         # query existence boolean
    if exists:                                           # if table exists, nothing to do
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
    """                                                  # DDL to create table
    run_sql(engine, ddl)                                  # execute DDL to create table
    print(" Created scores table public.pc_match_scores")  # log creation

def ensure_api_calls_table_exists() -> None:
    """Create the pc_api_calls table if it doesn't exist (idempotent DDL)."""
    engine = get_engine()                                 # get DB engine
    exists = fetch_all(engine, """
        SELECT EXISTS (
          SELECT FROM information_schema.tables
          WHERE table_schema='public' AND table_name='pc_api_calls'
        ) AS e;
    """)[0]["e"]                                         # query existence boolean
    if exists:                                           # if table exists, nothing to do
        return
    ddl = f"""
    CREATE TABLE {API_CALLS_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        pc_raw_recordings_s3_file_name TEXT NOT NULL,
        call_type TEXT NOT NULL,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        estimated_cost NUMERIC DEFAULT 0.0,
        gpt_model_used TEXT,
        call_timestamp TIMESTAMP DEFAULT NOW(),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """                                                  # DDL to create table
    run_sql(engine, ddl)                                  # execute DDL to create table
    print(" Created API calls table public.pc_api_calls")  # log creation

def ensure_all_tables_exist() -> None:
    """Ensure both scores and API calls tables exist."""
    ensure_scores_table_exists()                         # ensure main scores table exists
    ensure_api_calls_table_exists()                      # ensure API calls tracking table exists

# ==================== Reference / recording access ==============
def get_reference_row(script_code: str = PC_SCRIPT_CODE) -> dict:
    """Fetch the canonical reference row for a given script_code from pc_ref_table."""
    engine = get_engine()                                # get DB engine
    rows = fetch_all(engine, f"SELECT * FROM {REFERENCE_TABLE} WHERE script_code = :code LIMIT 1", {"code": script_code})  # query reference data
    if not rows:                                          # if not found, raise a clear error
        raise RuntimeError(f" No reference data in pc_ref_table for script_code={script_code}. Load Script 1 first.")
    return rows[0]                                        # return single reference row dict

def get_pending_recordings(limit: Optional[int] = None) -> list:
    """Fetch pending pc_recordings rows that have a non-null transcript (pc_en_transcribe)."""
    engine = get_engine()                                # get DB engine
    base_sql = f"""
        SELECT id, pc_raw_recordings_s3_file_name, pc_en_transcribe
        FROM {RECORDINGS_TABLE}
        WHERE status='pending' AND pc_en_transcribe IS NOT NULL
        ORDER BY id ASC
    """                                                  # base SQL for pending recordings
    if limit:                                             # if limit provided, append LIMIT clause
        return fetch_all(engine, base_sql + " LIMIT :lim", {"lim": limit})  # return limited results
    return fetch_all(engine, base_sql)                     # otherwise return all pending rows

def mark_successful(recording_id: int) -> None:
    """Update a recording's status to 'successful' after processing."""
    engine = get_engine()                                # get DB engine
    run_sql(engine, f"UPDATE {RECORDINGS_TABLE} SET status='successful' WHERE id=:i", {"i": recording_id})  # update status

# ==================== GPT matching & feedback ===================
class Matcher:
    """OpenAI Async wrapper that provides:
       - score_similarity(a,b): numeric 0..100 similarity (kept available but optional)
       - make_structured_feedback(transcript, canonical): returns strict JSON feedback
       The wrapper tracks token usage to a provided cost_tracker dict.
    """
    def __init__(self, model: str, cost_tracker: Optional[dict] = None):
        if not OPENAI_AVAILABLE:                         # ensure SDK present
            raise RuntimeError("OpenAI package not installed. Run: pip install openai")
        if not OPENAI_API_KEY:                           # require API key
            raise RuntimeError("OPENAI_API_KEY is required.")
        self.model = model                               # store model name
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # instantiate async OpenAI client
        self.cost_tracker = cost_tracker or {}           # use provided tracker or empty dict
        if TIKTOKEN_AVAILABLE:                           # optional encoder for token counting
            try:
                self.encoder = tiktoken.encoding_for_model(model)  # get encoder for specific model
            except Exception:
                self.encoder = tiktoken.get_encoding("cl100k_base")  # fallback to default encoder
        else:
            self.encoder = None                          # no encoder available
        # conservative nominal pricing (USD per 1M tokens) used to estimate run cost
        self.pricing_usd_per_mtok = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }
        print(f" Matcher initialized with model: {model}")  # log initialization

    def _track_usage(self, usage, call_type: str = "feedback", s3_file_name: str = "") -> None:
        """Update cost_tracker using the OpenAI SDK's usage object (if present).
        Now tracks individual call usage in addition to cumulative totals.
        """
        if not usage:                                    # guard if usage missing
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0     # prompt tokens used
        ct = getattr(usage, "completion_tokens", 0) or 0 # completion tokens used
        tt = getattr(usage, "total_tokens", 0) or (pt + ct)  # total tokens
        
        # accumulate tokens to the instance's cost_tracker for cumulative totals
        self.cost_tracker["prompt_tokens"] = self.cost_tracker.get("prompt_tokens", 0) + pt
        self.cost_tracker["completion_tokens"] = self.cost_tracker.get("completion_tokens", 0) + ct
        self.cost_tracker["total_tokens"] = self.cost_tracker.get("total_tokens", 0) + tt
        self.cost_tracker["api_calls"] = self.cost_tracker.get("api_calls", 0) + 1
        
        # calculate cost for this individual call
        price = self.pricing_usd_per_mtok.get(self.model)  # lookup pricing for model
        call_cost = 0.0
        if price:
            in_cost = (pt / 1_000_000.0) * price["input"]   # input portion cost
            out_cost = (ct / 1_000_000.0) * price["output"]# output portion cost
            call_cost = round(in_cost + out_cost, 6)        # cost for this individual call
        
        # store individual call details
        call_details = {
            "call_type": call_type,                        # type of call (feedback or similarity)
            "prompt_tokens": pt,                          # tokens used in prompt for this call
            "completion_tokens": ct,                      # tokens used in completion for this call
            "total_tokens": tt,                           # total tokens for this call
            "estimated_cost": call_cost,                  # estimated cost for this individual call
            "timestamp": time.time(),                     # when this call was made
            "s3_file_name": s3_file_name,                 # associated S3 file name for this call
            "gpt_model_used": self.model                  # GPT model used for this call
        }
        
        # initialize individual_calls list if it doesn't exist
        if "individual_calls" not in self.cost_tracker:
            self.cost_tracker["individual_calls"] = []
        
        # append this call's details to the individual calls list
        self.cost_tracker["individual_calls"].append(call_details)
        
        # update cumulative estimated cost
        self.cost_tracker["estimated_cost"] = self.cost_tracker.get("estimated_cost", 0.0) + call_cost

    async def score_similarity(self, a: str, b: str, s3_file_name: str = "") -> float:
        """Return numeric similarity 0..100 using GPT; kept available but not used in single-call mode.
        This function instructs the model to reply with ONLY a number for determinism.
        """
        a = _as_text(a)                                   # normalize input A
        b = _as_text(b)                                   # normalize input B
        if not a.strip() or not b.strip():                # if either empty, return 0
            return 0.0
        # concise rubric prompt for numeric scorer (kept in file for possible future re-enable)
        prompt = (
            "You are an objective grader. Given two passages (A and B), produce a single "
            "numeric similarity score between 0 and 100. Higher means more similar in intent, "
            "sequence, and phrasing. Use the following rubric sentences as guidance, but reply "
            "with ONLY the number and nothing else:\n"
            "- 90-100: passages are nearly identical in meaning and structure (Excellent).\n"
            "- 75-89: strong similarity (Good).\n"
            "- 60-74: moderate similarity (Average).\n"
            "- 45-59: weak similarity (Fair).\n"
            "- 0-44: minimal or no similarity (Poor).\n\n"
            f"PASSAGE A:\n{a}\n\nPASSAGE B:\n{b}\n\nReply with only the numeric score (e.g., 72.5)."
        )
        try:
            resp = await self.client.chat.completions.create(  # call OpenAI (numeric scorer)
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,        # expect just a number
                temperature=0.0,      # deterministic output
                timeout=30,
            )
            score_text = _as_text(resp.choices[0].message.content).strip()  # extract content
            try:
                score = float(score_text)                    # try parse float directly
            except Exception:
                token = score_text.split()[0] if score_text.split() else "0"  # fallback token
                try:
                    score = float(token)
                except Exception:
                    score = 0.0
            score = max(0.0, min(100.0, score))              # clamp to [0,100]
            self._track_usage(getattr(resp, "usage", None), "similarity", s3_file_name)  # track token usage for similarity call
            return score                                     # return numeric score
        except Exception as e:
            print(f" GPT similarity error: {e}")              # log error
            return 0.0                                       # conservative fallback

    async def make_structured_feedback(self, transcript: str, canonical: str, s3_file_name: str = "") -> dict:
        """Request strict JSON feedback from GPT comparing transcript vs canonical.
        This is the single API call used in the 'single-call' flow; the returned JSON
        should ideally include 'overall_score' (0..100) or 'overall_rating' (text).
        """
        transcript = _as_text(transcript)                   # normalize transcript
        canonical = _as_text(canonical)                     # normalize canonical script
        if not transcript.strip() or not canonical.strip(): # if either empty, return minimal payload
            return {"summary": "Insufficient text to analyze.", "categories": []}
        # system message enforces role and that only valid JSON must be returned
        system_msg = (
            "You are a highly experienced quality-assurance auditor for healthcare patient counsellor calls. "
            "Your role is to provide detailed, actionable feedback that helps counsellors improve their communication skills."
            "You MUST return ONLY valid JSON - no additional text, explanations, or markdown formatting."
        )
        # user message provides explicit JSON schema and an explicit scoring rubric
        user_msg = (
            "Analyze the counsellor's transcript against the canonical script and provide comprehensive feedback. "
            "Return ONLY a single valid JSON object with the following exact structure (no other text):\n"
            "{\n"
            "  \"summary\": \"concise_one_line_summary_here\",\n"
            "  \"overall_rating\": \"Excellent|Good|Average|Fair|Poor\",\n"
            "  \"overall_score\": 0-100,\n"
            "  \"key_strengths\": [\"strength1\", \"strength2\", \"strength3\"],\n"
            "  \"critical_improvements\": [\"improvement1\", \"improvement2\", \"improvement3\"],\n"
            "  \"categories\": [\n"
            "    {\n"
            "      \"name\": \"Communication Clarity\",\n"
            "      \"score\": 0-100,\n"
            "      \"strengths\": [\"specific_strength\"],\n"
            "      \"improvements\": [\"specific_improvement\"],\n"
            "      \"examples\": [\"exact_transcript_snippet\", \"corresponding_canonical_snippet\"]\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"Tone and Empathy\",\n"
            "      \"score\": 0-100,\n"
            "      \"strengths\": [\"specific_strength\"],\n"
            "      \"improvements\": [\"specific_improvement\"],\n"
            "      \"examples\": [\"exact_transcript_snippet\", \"corresponding_canonical_snippet\"]\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"Script Adherence and Flow\",\n"
            "      \"score\": 0-100,\n"
            "      \"strengths\": [\"specific_strength\"],\n"
            "      \"improvements\": [\"specific_improvement\"],\n"
            "      \"examples\": [\"exact_transcript_snippet\", \"corresponding_canonical_snippet\"]\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"Medical Accuracy\",\n"
            "      \"score\": 0-100,\n"
            "      \"strengths\": [\"specific_strength\"],\n"
            "      \"improvements\": [\"specific_improvement\"],\n"
            "      \"examples\": [\"exact_transcript_snippet\", \"corresponding_canonical_snippet\"]\n"
            "    },\n"
            "    {\n"
            "      \"name\": \"Patient Engagement\",\n"
            "      \"score\": 0-100,\n"
            "      \"strengths\": [\"specific_strength\"],\n"
            "      \"improvements\": [\"specific_improvement\"],\n"
            "      \"examples\": [\"exact_transcript_snippet\", \"corresponding_canonical_snippet\"]\n"
            "    }\n"
            "  ],\n"
            "  \"missed_key_points\": [\"key_point1_from_canonical\", \"key_point2_from_canonical\"],\n"
            "  \"excellent_improvisations\": [\"positive_deviation1\", \"positive_deviation2\"],\n"
            "  \"top_3_actions\": [\"actionable_item1\", \"actionable_item2\", \"actionable_item3\"],\n"
            "  \"coaching_priority\": \"High|Medium|Low\"\n"
            "}\n\n"
            "GUIDELINES FOR ANALYSIS:\n"
            "1. Be specific and evidence-based - always reference exact phrases from the transcript\n"
            "2. Focus on healthcare communication best practices\n"
            "3. Identify both adherence to script AND positive improvisations\n"
            "4. Note any medical inaccuracies or potentially confusing explanations\n"
            "5. Assess empathy, active listening, and patient-centered approach\n"
            "6. Evaluate clarity of medical explanations for patient understanding\n"
            "7. Check for proper call structure (opening, information delivery, closing)\n\n"
            "SCORING RUBRIC:\n"
            "- 90-100: Excellent - Exceeds expectations, demonstrates mastery\n"
            "- 75-89: Good - Meets expectations with minor areas for improvement\n"
            "- 60-74: Average - Acceptable performance but room for clear improvements\n"
            "- 45-59: Fair - Below expectations; needs focused coaching\n"
            "- Below 45: Poor - Requires substantial coaching and retraining\n\n"
            f"CANONICAL SCRIPT:\n{canonical}\n\n"
            f"COUNSELLOR TRANSCRIPT:\n{transcript}"
        )
        try:
            resp = await self.client.chat.completions.create(  # single API call for structured feedback
                model=self.model,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                max_tokens=1000,     # generous token budget to allow structured JSON
                temperature=0.2,     # slight creativity but generally stable
                timeout=60,
            )
            content = _as_text(resp.choices[0].message.content).strip()  # get response text
            self._track_usage(getattr(resp, "usage", None), "feedback", s3_file_name)  # track tokens/cost for feedback call
            # robust code-fence removal: handle ```json and surrounding triple backticks
            if content.startswith("```"):                                # if starts with triple backticks
                lines = content.splitlines()                             # split into lines
                if lines and lines[0].startswith("```"):                 # remove first line if it's a fence
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):                # remove trailing fence if present
                    lines = lines[:-1]
                content = "\n".join(lines).strip()                       # rejoin and strip whitespace
            # try parse JSON safely
            try:
                parsed = json.loads(content)                             # parse JSON content
                return parsed if isinstance(parsed, dict) else {"summary": "Unexpected feedback format.", "raw": content}
            except Exception:
                return {"summary": "Unable to parse JSON from model output.", "raw": content}  # fallback with raw text
        except Exception as e:
            print(f" GPT feedback error: {e}")                            # log and return minimal failure payload
            return {"summary": "Feedback generation failed.", "error": str(e)}

# ==================== Auto feedback builder =====================
def build_auto_feedback(bucket_label: Optional[str]) -> dict:
    """Create rule-based summary feedback and band from the stored label (or None)."""
    if not bucket_label or not isinstance(bucket_label, str) or "-" not in bucket_label:
        # if no label available, return unknown band and a hint
        return {"full_script_bucket_label": None, "full_script_bucket_band": "unknown", "hints": ["No similarity score computed — only structured feedback available."]}
    ub = label_to_upper_bound(bucket_label)   # get numeric upper bound from label
    def band(s: float) -> str:
        # map numeric upper bound to qualitative band matching the grading rubric
        if s >= 90:
            return "excellent"
        if s >= 75:
            return "good"
        if s >= 60:
            return "average"
        if s >= 45:
            return "fair"
        return "poor"
    payload = {"full_script_bucket_label": bucket_label, "full_script_bucket_band": band(ub), "hints": []}
    # additive hints based on numeric thresholds
    if ub < 85:
        payload["hints"].append("Strengthen adherence to canonical phrasing for key transitions.")
    if ub < 70:
        payload["hints"].append("Clarify explanations (diagnosis, stages, and benefits) in simpler terms.")
    if ub < 55:
        payload["hints"].append("Rehearse opening/closing flows to tighten structure and reduce digressions.")
    return payload

# ==================== Persistence (insert) ======================
def insert_match_row(payload: Dict) -> None:
    """Insert or upsert one row into public.pc_match_scores; delete existing row for same file for idempotency."""
    engine = get_engine()                       # create DB engine
    ensure_all_tables_exist()                   # ensure both tables exist
    # delete any existing row for same s3 filename so insert is idempotent
    run_sql(engine, f"DELETE FROM {SCORES_TABLE} WHERE pc_raw_recordings_s3_file_name = :k", {"k": payload["pc_raw_recordings_s3_file_name"]})
    # prepare insert SQL to store label + feedback JSONs and usage metadata
    insert_sql = f"""
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
        :ref_script_code, :pc_raw_recordings_s3_file_name, :pc_en_transcribe, :og_pc_script,
        :mat_sco_og_pc_script, :feedback_full_json, :feedback_auto_json, :gpt_model_used,
        :total_tokens_used, :estimated_cost
    )
    """
    run_sql(engine, insert_sql, payload)        # execute insert

def insert_api_call_details(call_details: Dict) -> None:
    """Insert individual API call details into public.pc_api_calls table."""
    engine = get_engine()                       # create DB engine
    ensure_all_tables_exist()                   # ensure both tables exist
    # prepare insert SQL for individual API call tracking
    insert_sql = f"""
    INSERT INTO {API_CALLS_TABLE} (
        pc_raw_recordings_s3_file_name,
        call_type,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        estimated_cost,
        gpt_model_used,
        call_timestamp
    ) VALUES (
        :s3_file_name, :call_type, :prompt_tokens, :completion_tokens, 
        :total_tokens, :estimated_cost, :gpt_model_used, 
        to_timestamp(:timestamp)
    )
    """
    # prepare parameters for the insert
    params = {
        "s3_file_name": call_details.get("s3_file_name", ""),
        "call_type": call_details.get("call_type", "unknown"),
        "prompt_tokens": call_details.get("prompt_tokens", 0),
        "completion_tokens": call_details.get("completion_tokens", 0),
        "total_tokens": call_details.get("total_tokens", 0),
        "estimated_cost": call_details.get("estimated_cost", 0.0),
        "gpt_model_used": call_details.get("gpt_model_used", ""),
        "timestamp": call_details.get("timestamp", time.time())
    }
    run_sql(engine, insert_sql, params)         # execute insert for individual API call

def store_individual_api_calls(individual_calls: list) -> None:
    """Store all individual API calls from cost_tracker to the database."""
    for call in individual_calls:                # iterate through each individual call
        insert_api_call_details(call)            # store each call in the database
    print(f"  Stored {len(individual_calls)} individual API call(s) → public.pc_api_calls")  # log storage

# ==================== Main processing (batch) ===================
async def run_matching(limit: Optional[int] = None, model: str = GPT_MODEL, script_code: str = PC_SCRIPT_CODE) -> None:
    """Main batch runner: for each pending recording call make_structured_feedback once,
    optionally derive label from overall_score or overall_rating, build rule-based hints,
    insert into db and mark source row successful.
    """
    # use local cost tracker for this batch run to ensure accurate per-run accounting
    cost_tracker = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "estimated_cost": 0.0, "api_calls": 0, "individual_calls": []}
    print(f" START: Pending-only matcher using model={model} and script_code={script_code}")  # banner log
    matcher = Matcher(model=model, cost_tracker=cost_tracker)  # instantiate matcher with local tracker
    ref = get_reference_row(script_code=script_code)  # load canonical reference row by script_code
    ref_code = _as_text(ref.get("script_code"))   # normalize script_code value
    og_script = _as_text(ref.get("og_pc_script")) # normalize canonical script content
    recs = get_pending_recordings(limit=limit)    # fetch pending recordings
    print(f" Found {len(recs)} pending recording(s) to process")  # log how many to process
    # iterate sequentially over fetched records
    for i, row in enumerate(recs, start=1):
        rec_id = row["id"]                         # recording primary key id
        s3_key = _as_text(row["pc_raw_recordings_s3_file_name"])  # s3 key / filename
        transcript = _as_text(row["pc_en_transcribe"])            # transcript text
        print(f"\n [{i}/{len(recs)}] Matching {s3_key}")          # log current processing item

        # SINGLE API CALL MODE:
        # call structured feedback only (this is the one API call per recording)
        fb_full = await matcher.make_structured_feedback(transcript, og_script, s3_key)

        # normalize overall_rating if present so stored data is consistent
        overall_rating_norm = None
        if isinstance(fb_full, dict):
            oraw = fb_full.get("overall_rating")  # attempt to read overall_rating if model provided it
            if isinstance(oraw, str) and oraw.strip():
                overall_rating_norm = oraw.strip().lower()  # normalize to lowercase for internal consistency
                fb_full["overall_rating_normalized"] = overall_rating_norm  # store normalized rating into feedback JSON

        # attempt to derive numeric overall_score or derive label from overall_rating if only rating present
        label = None
        try:
            overall = None
            if isinstance(fb_full, dict):
                overall = fb_full.get("overall_score")  # try numeric overall_score first
            if overall is None:
                # fallback: derive from overall_rating if present
                orating = None
                if isinstance(fb_full, dict):
                    orating = fb_full.get("overall_rating") or fb_full.get("overall_rating_normalized")
                if isinstance(orating, str):
                    # map textual rating to a representative numeric center for quantization
                    r = orating.strip().lower()
                    mapping = {"excellent": 95.0, "good": 82.0, "average": 67.0, "fair": 52.0, "poor": 30.0}
                    if r in mapping:
                        overall_num = mapping[r]         # numeric surrogate for each rating
                        label = quantize_score_to_label(overall_num)  # quantize to textual bucket
                        print(f"  Derived label from overall_rating '{orating}': {label}")  # log derivation
                # if still no numeric value, leave label None
            else:
                # if model returned numeric overall_score, parse and quantize
                overall_num = float(overall)
                label = quantize_score_to_label(overall_num)
                print(f"  Derived overall_score from feedback: {overall_num:.2f}")
                print(f"  Bucket label: {label}")
        except Exception as e:
            # defensive handling when parsing fails; keep label None
            print(f"  Warning: couldn't derive numeric overall_score from structured feedback: {e}")
            label = None

        # build rule-based auto-feedback using the derived label (or None)
        fb_auto = build_auto_feedback(label)
        # if we have normalized rating, add it to auto feedback for easy machine use
        if overall_rating_norm:
            fb_auto["overall_rating"] = overall_rating_norm

        # Find the individual API call data for this specific recording
        individual_calls_for_recording = [call for call in cost_tracker.get("individual_calls", []) 
                                        if call.get("s3_file_name") == s3_key]  # filter calls by S3 filename
        
        # Calculate individual totals for this recording (not cumulative)
        recording_total_tokens = 0
        recording_estimated_cost = 0.0
        
        if individual_calls_for_recording:
            # Sum up tokens and costs for all calls related to this specific recording
            for call in individual_calls_for_recording:
                recording_total_tokens += call.get("total_tokens", 0)  # sum tokens for this recording
                recording_estimated_cost += call.get("estimated_cost", 0.0)  # sum costs for this recording

        # prepare DB payload with INDIVIDUAL recording totals (not cumulative)
        payload = {
            "ref_script_code": ref_code,
            "pc_raw_recordings_s3_file_name": s3_key,
            "pc_en_transcribe": transcript,
            "og_pc_script": og_script,
            "mat_sco_og_pc_script": label,
            "feedback_full_json": _json_dumps_safe(fb_full),
            "feedback_auto_json": _json_dumps_safe(fb_auto),
            "gpt_model_used": model,
            "total_tokens_used": recording_total_tokens,  # INDIVIDUAL recording tokens, not cumulative
            "estimated_cost": recording_estimated_cost,   # INDIVIDUAL recording cost, not cumulative
        }

        # persist results and mark original recording successful to avoid reprocessing
        insert_match_row(payload)
        print("  Saved match row → public.pc_match_scores")
        
        # store individual API calls for this recording
        if individual_calls_for_recording:
            store_individual_api_calls(individual_calls_for_recording)
        
        mark_successful(rec_id)
        print("  Marked recording as successful")

    # after processing all rows, print detailed summary of token usage and estimated cost
    print("\n==== GPT USAGE SUMMARY ====")
    print(f" Total API calls: {cost_tracker.get('api_calls', 0)}")  # total calls in this run
    print(f" Total tokens:   {cost_tracker.get('total_tokens', 0)}")  # total tokens in this run
    print(f" Est. cost:      ${cost_tracker.get('estimated_cost', 0.0):.6f}")  # total cost in this run
    
    # print individual call details if available
    individual_calls = cost_tracker.get("individual_calls", [])
    if individual_calls:
        print("\n==== INDIVIDUAL CALL BREAKDOWN ====")
        for i, call in enumerate(individual_calls, 1):
            call_type = call.get("call_type", "unknown")
            prompt_tokens = call.get("prompt_tokens", 0)
            completion_tokens = call.get("completion_tokens", 0)
            total_tokens = call.get("total_tokens", 0)
            cost = call.get("estimated_cost", 0.0)
            timestamp = call.get("timestamp", 0)
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            s3_file = call.get("s3_file_name", "unknown")
            
            print(f" Call {i} ({call_type}) for {s3_file} at {time_str}:")
            print(f"   Prompt tokens: {prompt_tokens}")
            print(f"   Completion tokens: {completion_tokens}")
            print(f"   Total tokens: {total_tokens}")
            print(f"   Estimated cost: ${cost:.6f}")
            print()

# ==================== Watch mode (polling) ======================
async def watch_polling(interval_seconds: int = 60, model: str = GPT_MODEL, script_code: str = PC_SCRIPT_CODE):
    """Polling loop: periodically run the batch matcher to pick up new pending rows."""
    print(f" WATCH: polling every {interval_seconds}s for new pending rows using script_code={script_code}...")
    while True:
        try:
            await run_matching(limit=None, model=model, script_code=script_code)  # process all pending
        except Exception as e:
            print(f" WATCH error (continuing): {e}")  # log and continue on exceptions
        await asyncio.sleep(interval_seconds)         # sleep until next poll

# ================== Listen/Notify mode (event-driven) ===========
# SQL trigger example (run once in DB) is provided in comments below to enable immediate notifications:
# CREATE OR REPLACE FUNCTION public.notify_pc_recordings_pending() RETURNS trigger AS $$
# BEGIN
#   IF NEW.status = 'pending' AND NEW.pc_en_transcribe IS NOT NULL THEN
#     PERFORM pg_notify('pc_recordings_pending', json_build_object('id', NEW.id, 'pc_raw_recordings_s3_file_name', NEW.pc_raw_recordings_s3_file_name)::text);
#   END IF;
#   RETURN NEW;
# END;
# $$ LANGUAGE plpgsql;
# DROP TRIGGER IF EXISTS trg_pc_recordings_pending ON public.pc_recordings;
# CREATE TRIGGER trg_pc_recordings_pending AFTER INSERT ON public.pc_recordings FOR EACH ROW EXECUTE FUNCTION public.notify_pc_recordings_pending();

async def process_one_by_id(rec_id: int, model: str = GPT_MODEL, script_code: str = PC_SCRIPT_CODE):
    """Process a single recording ID if still pending and with transcript present (event-driven path)."""
    # use local cost tracker for this single recording to ensure accurate per-recording accounting
    local_tracker = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "estimated_cost": 0.0, "api_calls": 0, "individual_calls": []}
    engine = get_engine()                           # get DB engine
    row = fetch_all(engine, f"""
        SELECT id, pc_raw_recordings_s3_file_name, pc_en_transcribe
        FROM {RECORDINGS_TABLE}
        WHERE id = :i AND status='pending' AND pc_en_transcribe IS NOT NULL
        LIMIT 1
    """, {"i": rec_id})                            # fetch the specific pending row by id
    if not row:
        return                                     # nothing to process (maybe already handled)
    ref = get_reference_row(script_code=script_code)  # fetch canonical reference
    ref_code = _as_text(ref.get("script_code"))      # normalize script code
    og_script = _as_text(ref.get("og_pc_script"))    # normalize canonical script text
    matcher = Matcher(model=model, cost_tracker=local_tracker)  # create matcher with local tracker
    s3_key = _as_text(row[0]["pc_raw_recordings_s3_file_name"])  # s3 file name for current row
    transcript = _as_text(row[0]["pc_en_transcribe"])            # normalize transcript

    # SINGLE API CALL: structured feedback only (skip numeric scorer to avoid double call)
    fb_full = await matcher.make_structured_feedback(transcript, og_script, s3_key)

    # normalize any textual overall_rating for consistency
    overall_rating_norm = None
    if isinstance(fb_full, dict):
        oraw = fb_full.get("overall_rating")
        if isinstance(oraw, str) and oraw.strip():
            overall_rating_norm = oraw.strip().lower()
            fb_full["overall_rating_normalized"] = overall_rating_norm

    # attempt to derive label from overall_score or overall_rating fallback
    label = None
    try:
        if isinstance(fb_full, dict) and fb_full.get("overall_score") is not None:
            overall_num = float(fb_full.get("overall_score"))
            label = quantize_score_to_label(overall_num)          # quantize numeric score
            print(f"  Derived overall_score from feedback: {overall_num:.2f}")
            print(f"  Bucket label: {label}")
        else:
            # fallback to textual rating -> numeric mapping
            orating = None
            if isinstance(fb_full, dict):
                orating = fb_full.get("overall_rating") or fb_full.get("overall_rating_normalized")
            if isinstance(orating, str):
                r = orating.strip().lower()
                mapping = {"excellent": 95.0, "good": 82.0, "average": 67.0, "fair": 52.0, "poor": 30.0}
                if r in mapping:
                    overall_num = mapping[r]
                    label = quantize_score_to_label(overall_num)
                    print(f"  Derived label from overall_rating '{orating}': {label}")
    except Exception as e:
        print(f"  Warning: couldn't derive numeric overall_score from structured feedback: {e}")
        label = None

    fb_auto = build_auto_feedback(label)               # generate rule-based hints from label
    if overall_rating_norm:
        fb_auto["overall_rating"] = overall_rating_norm  # attach normalized rating for consistency

    # Calculate individual totals for this recording
    individual_calls = local_tracker.get("individual_calls", [])
    recording_total_tokens = 0
    recording_estimated_cost = 0.0
    
    for call in individual_calls:
        recording_total_tokens += call.get("total_tokens", 0)  # sum tokens for this recording
        recording_estimated_cost += call.get("estimated_cost", 0.0)  # sum costs for this recording

    payload = {                                         # build payload to persist into scores table
        "ref_script_code": ref_code,
        "pc_raw_recordings_s3_file_name": s3_key,
        "pc_en_transcribe": transcript,
        "og_pc_script": og_script,
        "mat_sco_og_pc_script": label,
        "feedback_full_json": _json_dumps_safe(fb_full),
        "feedback_auto_json": _json_dumps_safe(fb_auto),
        "gpt_model_used": model,
        "total_tokens_used": recording_total_tokens,    # INDIVIDUAL recording tokens
        "estimated_cost": recording_estimated_cost,     # INDIVIDUAL recording cost
    }
    insert_match_row(payload)                          # insert into scores table (idempotent delete+insert)
    
    # store individual API calls for this recording
    if individual_calls:
        store_individual_api_calls(individual_calls)
    
    mark_successful(rec_id)                            # mark original recording as successful
    print(f" Event-processed recording id={rec_id} ({s3_key})")  # log event-processed message
    
    # print individual call details for this single processing
    if individual_calls:
        print("\n==== INDIVIDUAL CALL BREAKDOWN ====")
        for i, call in enumerate(individual_calls, 1):
            call_type = call.get("call_type", "unknown")
            prompt_tokens = call.get("prompt_tokens", 0)
            completion_tokens = call.get("completion_tokens", 0)
            total_tokens = call.get("total_tokens", 0)
            cost = call.get("estimated_cost", 0.0)
            timestamp = call.get("timestamp", 0)
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            print(f" Call {i} ({call_type}) at {time_str}:")
            print(f"   Prompt tokens: {prompt_tokens}")
            print(f"   Completion tokens: {completion_tokens}")
            print(f"   Total tokens: {total_tokens}")
            print(f"   Estimated cost: ${cost:.6f}")
            print()

async def watch_listen_notify(model: str = GPT_MODEL, sweep_every_seconds: int = 300, script_code: str = PC_SCRIPT_CODE):
    """Listen for Postgres NOTIFY events and process incoming IDs; periodically sweep as fallback."""
    print(" WATCH: LISTEN/NOTIFY mode on channel 'pc_recordings_pending'")  # banner message
    engine = get_engine()                              # SQLAlchemy engine
    conn = engine.raw_connection()                     # get raw psycopg2 connection for LISTEN/NOTIFY
    conn.set_isolation_level(0)                        # autocommit required for LISTEN
    curs = conn.cursor()                               # cursor for receiving notifications
    curs.execute("LISTEN pc_recordings_pending;")      # subscribe to channel 'pc_recordings_pending'
    conn.commit()                                      # ensure LISTEN is registered
    last_sweep = time.time()                           # track last sweep time
    try:
        while True:
            # wait up to 5 seconds for notifications (non-blocking)
            if select.select([conn], [], [], 5) == ([], [], []):
                pass
            conn.poll()                                # poll connection for notifications
            while conn.notifies:                       # drain all available notifications
                note = conn.notifies.pop(0)            # pop first notification
                try:
                    payload = json.loads(note.payload) # parse JSON payload sent by trigger
                    rec_id = int(payload.get("id"))    # extract id field
                    await process_one_by_id(rec_id, model=model, script_code=script_code)  # process that id
                except Exception as e:
                    print(f" WATCH notify error: {e}")  # log and continue if any processing error
            # periodic safety sweep to catch anything missed by notify
            if time.time() - last_sweep > sweep_every_seconds:
                print(" WATCH: periodic sweep for any missed pending rows...")
                await run_matching(limit=None, model=model, script_code=script_code)  # sweep all pending
                last_sweep = time.time()
    finally:
        try:
            curs.close()                                # close cursor on exit
            conn.close()                                # close connection on exit
        except Exception:
            pass

# ==================== Command line interface ====================
def main() -> None:
    """Parse CLI args and dispatch to reset_db, match (one-shot), watch (polling), or listen (notify)."""
    parser = argparse.ArgumentParser(description="PC Call Analyzer - Pending Matcher (labels)")
    sub = parser.add_subparsers(dest="command", help="Commands")
    sub.add_parser("reset_db", help="Create scores table if missing")  # reset/create table

    # match: one-shot processing (batch)
    p_match = sub.add_parser("match", help="Match pending recordings (one-shot)")
    p_match.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    p_match.add_argument("--model", default=GPT_MODEL, choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], help="OpenAI model to use")
    p_match.add_argument("--script-code", dest="script_code", default=PC_SCRIPT_CODE, help="Script code to use for matching (default: PC_Script002)")

    # watch: polling mode
    p_watch = sub.add_parser("watch", help="Continuously watch (poll) for new pending rows")
    p_watch.add_argument("--interval", type=int, default=30, help="Polling interval seconds")
    p_watch.add_argument("--model", default=GPT_MODEL, choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], help="OpenAI model to use")
    p_watch.add_argument("--script-code", dest="script_code", default=PC_SCRIPT_CODE)

    # listen: event-driven LISTEN/NOTIFY mode
    p_listen = sub.add_parser("listen", help="Event-driven mode via LISTEN/NOTIFY + trigger")
    p_listen.add_argument("--model", default=GPT_MODEL, choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], help="OpenAI model to use")
    p_listen.add_argument("--sweep", type=int, default=300, help="Seconds between safety sweeps for missed rows")
    p_listen.add_argument("--script-code", dest="script_code", default=PC_SCRIPT_CODE)

    args = parser.parse_args()                          # parse CLI arguments

    # dispatch based on selected command
    if args.command == "reset_db":
        ensure_all_tables_exist()                       # ensure both tables exist
        print(" Scores and API calls tables are ready.") # confirmation
        return
    elif args.command == "match":
        asyncio.run(run_matching(limit=args.limit, model=args.model, script_code=args.script_code))  # run one-shot pass
        return
    elif args.command == "watch":
        asyncio.run(watch_polling(interval_seconds=args.interval, model=args.model, script_code=args.script_code))  # start polling loop
        return
    elif args.command == "listen":
        asyncio.run(watch_listen_notify(model=args.model, sweep_every_seconds=args.sweep, script_code=args.script_code))  # start listener
        return
    else:
        parser.print_help()                             # if no command provided, show help

# ==================== Entry point ===============================
if __name__ == "__main__":
    try:
        main()                                          # run CLI entry point
    except KeyboardInterrupt:
        print("\n Operation cancelled by user")         # graceful message for Ctrl+C
        raise SystemExit(1)                             # exit non-zero to indicate cancellation
    except Exception as e:
        print(f" FATAL: {e}")                           # fatal error log
        raise SystemExit(1)                             # exit non-zero for failure
