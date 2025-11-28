"""
realtime_specific_pc_script3_ingest.py
--------------------
PC CALL ANALYZER - AUDIO INGESTION AND TRANSCRIPTION MODULE (Realtime-ready)

What's new (this edited copy):
- After fetching a recording candidate from S3, verify it exists in the
  appointment_recording table with recording_type = 'TO Closure' before
  transcribing/upserting into public.pc_recordings.
- Verification attempts exact URL with region, exact URL without region,
  and a fallback suffix match (recording_link LIKE '%<key>') to be robust.
- All original logic preserved; added comments on nearly every operation.

Usage:
  # 1) Init DB
  python realtime_specific_pc_script3_ingest.py init_db

  # 2a) Polling mode (checks every 60s; change with --interval)
  python realtime_specific_pc_script3_ingest.py watch_s3 --bucket qispine-documents --prefix "PC_TO_Conversation/" --interval 60 --batch-size 10

  # 2b) Event-driven mode (requires S3->SQS notification already configured)
  python realtime_specific_pc_script3_ingest.py consume_sqs --queue-url https://sqs.ap-south-1.amazonaws.com/ACCOUNT/queue-name --batch-size 10

  # Existing:
  python realtime_specific_pc_script3_ingest.py ingest_s3 --bucket qispine-documents --prefix "PC_TO_Conversation/" --max-files 01
  python realtime_specific_pc_script3_ingest.py ingest_file --s3-uri "s3://qispine-documents/PC_TO_Conversation/file.m4a"
"""

# ------------------------ Standard Library Imports ------------------------
import os                                      # environment access
import io                                      # in-memory bytes buffer
import json                                    # json decode/encode
import time                                    # sleep for watcher
import argparse                                # CLI argument parsing
import tempfile                                # temporary files
from typing import Optional, List, Tuple      # type hints
from datetime import datetime, date, timezone # date/time utilities
from zoneinfo import ZoneInfo                 # timezone handling

# ------------------------ Environment Configuration ------------------------
try:
    from dotenv import load_dotenv             # optional .env loader
    load_dotenv()                              # load env vars from .env if present
    print(" Environment variables loaded from .env file")
except Exception:
    print(" No .env file found, using system environment variables")

# ------------------------ Database (SQLAlchemy) Imports ------------------------
from sqlalchemy import create_engine, text     # SQL engine and text queries
from sqlalchemy.engine import Engine           # engine typing
from sqlalchemy.exc import SQLAlchemyError     # catch DB errors

# ------------------------ AWS (Boto3) Imports ------------------------
import boto3                                   # AWS SDK for python
from botocore.exceptions import BotoCoreError, ClientError  # AWS errors

# ------------------------ OpenAI SDK (New + Legacy) ------------------------
try:
    from openai import OpenAI                   # new SDK import
    _HAS_NEW_OPENAI = True
    print(" Using OpenAI SDK v1.0+")
except ImportError:
    _HAS_NEW_OPENAI = False
    import openai  # type: ignore                    # legacy SDK fallback
    print(" Using Legacy OpenAI SDK")

# ========================== Global Configuration ==========================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    # fallback value if not set in env (you probably override in production)
    "postgresql+psycopg2://qispineadmin:TrOpSnl1H1QdKAFsAWnY@qispine-db.cqjl02ffrczp.ap-south-1.rds.amazonaws.com:5432/qed_prod"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # openai api key from env
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")  # aws region fallback
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-1")  # default ASR model

# Optional envs for realtime modes
DEFAULT_WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL_SECONDS", "60"))  # polling seconds
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))                 # batch size

# allowed audio file extensions
ALLOWED_EXTS: List[str] = [
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".webm", ".mp4", ".aac", ".wma"
]

# -------------------------- Cutoff Date Configuration --------------------------
CUTOFF_TZ = ZoneInfo("Asia/Kolkata")            # cutoff timezone (IST)
# NOTE: change this date if you want; current is 2025-11-16 as provided
CUTOFF_LOCAL_DATE = date(2025, 11, 28)          # local cutoff date (IST)

# ========================== DB Utils ==========================
def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine using DATABASE_URL."""
    return create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)

def run_sql(engine: Engine, sql: str, params: Optional[dict] = None) -> None:
    """Execute a SQL statement (no returned rows)."""
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def fetch_all(engine: Engine, sql: str, params: Optional[dict] = None) -> list:
    """Execute a SELECT-style query and return list of row-mappings."""
    with engine.begin() as conn:
        result = conn.execute(text(sql), params or {})
        return [dict(row._mapping) for row in result.fetchall()]

# ========================== Schema ==========================
DDL_PC_RECORDINGS = """
CREATE TABLE IF NOT EXISTS public.pc_recordings (
    id BIGSERIAL PRIMARY KEY,
    pc_raw_recordings_s3_file_name TEXT UNIQUE NOT NULL,
    pc_en_transcribe TEXT,
    status TEXT DEFAULT 'pending',
    s3_last_modified_utc TIMESTAMPTZ,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pc_recordings_status ON public.pc_recordings (status);",
    "CREATE INDEX IF NOT EXISTS idx_pc_recordings_filename ON public.pc_recordings (pc_raw_recordings_s3_file_name);",
    "CREATE INDEX IF NOT EXISTS idx_pc_recordings_s3_last_modified ON public.pc_recordings (s3_last_modified_utc);"
]

DDL_ALTER_COLUMNS = [
    "ALTER TABLE public.pc_recordings ADD COLUMN IF NOT EXISTS s3_last_modified_utc TIMESTAMPTZ;"
]

def init_db():
    """Ensure pc_recordings table and indexes exist in the DB."""
    engine = get_engine()
    try:
        run_sql(engine, DDL_PC_RECORDINGS)         # create table if not exists
        for stmt in DDL_ALTER_COLUMNS:
            run_sql(engine, stmt)                  # ensure column exists
        for stmt in DDL_INDEXES:
            run_sql(engine, stmt)                  # create indexes
        print("Table ensured: public.pc_recordings with indexes and s3_last_modified_utc column")
    except SQLAlchemyError as e:
        print(f"Database initialization failed: {e}")
        raise

# ====================== OpenAI ======================
def _get_openai_client():
    """Return an OpenAI client object (new SDK) or raise if API key missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment variables")
    if _HAS_NEW_OPENAI:
        return OpenAI(api_key=OPENAI_API_KEY)     # new SDK uses an instance
    else:
        openai.api_key = OPENAI_API_KEY           # legacy sets global
        return None

def transcribe_to_english(local_file_path: str, prefer_translate: bool = True) -> str:
    """
    Transcribe (or translate->English) an audio file on disk using the OpenAI client.
    prefer_translate=True will call translations API (useful for Hindi->English).
    """
    client = _get_openai_client()                 # get client or raise
    try:
        if _HAS_NEW_OPENAI:
            with open(local_file_path, "rb") as f:
                if prefer_translate:
                    resp = client.audio.translations.create(
                        model=ASR_MODEL,
                        file=f,
                        response_format="text",
                        temperature=0.0,
                    )
                else:
                    resp = client.audio.transcriptions.create(
                        model=ASR_MODEL,
                        file=f,
                        response_format="text",
                        temperature=0.0,
                    )
            return resp.strip() if isinstance(resp, str) else str(resp).strip()
        else:
            with open(local_file_path, "rb") as f:
                if prefer_translate:
                    resp = openai.Audio.translations.create(  # type: ignore
                        model=ASR_MODEL,
                        file=f,
                        response_format="text",
                        temperature=0.0,
                    )
                else:
                    resp = openai.Audio.transcriptions.create(  # type: ignore
                        model=ASR_MODEL,
                        file=f,
                        response_format="text",
                        temperature=0.0,
                    )
            return resp.strip() if isinstance(resp, str) else str(resp).strip()
    except Exception as e:
        print(f"[ASR] API error: {e}")
        raise

# ====================== Helpers ======================
def is_audio_file(s3_key: str) -> bool:
    """Return True if the S3 object key ends with an allowed audio extension."""
    key_lower = s3_key.lower()
    return any(key_lower.endswith(ext) for ext in ALLOWED_EXTS)

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key) or raise ValueError."""
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI. Expected format: s3://<bucket>/<key>")
    without = s3_uri[len("s3://"):]
    first_slash = without.find("/")
    if first_slash <= 0:
        raise ValueError("Invalid S3 URI. Must include bucket and key.")
    return without[:first_slash], without[first_slash+1:]

def is_after_cutoff(last_modified_utc: datetime) -> bool:
    """
    Return True if the last_modified_utc, converted to IST, has date strictly greater
    than CUTOFF_LOCAL_DATE. Ensures enforcement in IST regardless of S3 object's UTC.
    """
    if last_modified_utc.tzinfo is None:
        last_modified_utc = last_modified_utc.replace(tzinfo=timezone.utc)
    last_modified_ist = last_modified_utc.astimezone(CUTOFF_TZ)
    return last_modified_ist.date() > CUTOFF_LOCAL_DATE

def upsert_transcript(engine: Engine, object_key: str, transcript: str, s3_last_modified_utc: datetime) -> None:
    """Insert or update a transcription row in public.pc_recordings (idempotent)."""
    upsert_query = """
    INSERT INTO public.pc_recordings
        (pc_raw_recordings_s3_file_name, pc_en_transcribe, status, s3_last_modified_utc)
    VALUES
        (:object_key, :transcript, 'pending', :s3_last_modified_utc)
    ON CONFLICT (pc_raw_recordings_s3_file_name)
    DO UPDATE SET
        pc_en_transcribe = EXCLUDED.pc_en_transcribe,
        status = 'pending',
        s3_last_modified_utc = EXCLUDED.s3_last_modified_utc
    """
    run_sql(engine, upsert_query, {
        "object_key": object_key,
        "transcript": transcript,
        "s3_last_modified_utc": s3_last_modified_utc,
    })

def already_transcribed(engine: Engine, object_key: str) -> bool:
    """Return True if a transcription already exists for object_key in pc_recordings."""
    rows = fetch_all(
        engine,
        "SELECT pc_en_transcribe FROM public.pc_recordings WHERE pc_raw_recordings_s3_file_name = :k",
        {"k": object_key},
    )
    return bool(rows and rows[0].get("pc_en_transcribe"))

def max_seen_s3_last_modified(engine: Engine) -> Optional[datetime]:
    """Return the maximum s3_last_modified_utc value from pc_recordings (or None)."""
    rows = fetch_all(engine, "SELECT max(s3_last_modified_utc) AS mx FROM public.pc_recordings")
    mx = rows[0]["mx"] if rows else None
    return mx

# ====================== New: Appointment Recording Verification ======================
def verify_recording_in_appointment(engine: Engine, bucket: str, key: str) -> bool:
    """
    Verify that the recording (identified by bucket/key) exists in appointment_recording
    with recording_type = 'TO Closure'.
    This function attempts:
      1) exact URL with region (https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{key})
      2) exact URL without region (https://{bucket}.s3.amazonaws.com/{key})
      3) suffix match on key (recording_link LIKE '%' || :key)
    Returns True if any match found; False otherwise.
    """
    # build candidate URLs to check against appointment_recording.recording_link
    url_with_region = f"https://{bucket}.s3.{AWS_REGION}.amazonaws.com/{key}"  # region-specific form
    url_no_region = f"https://{bucket}.s3.amazonaws.com/{key}"                 # generic S3 form

    # SQL query tries exact matches and suffix match; uses parameterization to avoid injection
    sql = """
    SELECT 1
    FROM appointment_recording ar
    WHERE ar.recording_type = 'TO Closure'
      AND (
            ar.recording_link = :u1
         OR ar.recording_link = :u2
         OR ar.recording_link LIKE '%' || :k
      )
    LIMIT 1
    """
    try:
        # execute the query and fetch results
        rows = fetch_all(engine, sql, {"u1": url_with_region, "u2": url_no_region, "k": key})
        # return True if any row present
        return bool(rows)
    except SQLAlchemyError as e:
        # log DB errors and conservatively return False to avoid transcribing non-matching files
        print(f"[VERIFY] DB error while verifying appointment_recording: {e}")
        return False

# ========================== Core Ingestion (Batch) ==========================
def ingest_s3(bucket_name: str, prefix: str, max_files: Optional[int] = None) -> None:
    """
    Batch ingest from an S3 prefix:
      - list objects under prefix
      - for each audio file newer than cutoff and not already transcribed:
          - verify presence in appointment_recording with recording_type='TO Closure'
          - if verified, download, transcribe, upsert into pc_recordings
          - else skip
    """
    s3_client = boto3.client("s3", region_name=AWS_REGION)  # S3 client
    db_engine = get_engine()                                 # DB engine for queries

    paginator = s3_client.get_paginator("list_objects_v2")   # paginator for large listings
    pagination_config = {"Bucket": bucket_name, "Prefix": prefix}

    files_inserted = 0
    files_processed = 0
    files_skipped = 0
    files_failed = 0

    print(f" Starting S3 ingestion from: s3://{bucket_name}/{prefix}")
    if max_files:
        print(f" Maximum files to process: {max_files}")

    # paginate through objects
    for page_number, page in enumerate(paginator.paginate(**pagination_config)):
        print(f"Processing S3 page {page_number + 1}...")
        if "Contents" not in page:
            print(" No objects found in this page, continuing...")
            continue

        for s3_object in page["Contents"]:
            object_key = s3_object.get("Key", "").strip()   # object key
            if not object_key or object_key.endswith("/"):  # skip prefixes/empty keys
                continue

            # skip non-audio files by extension
            if not is_audio_file(object_key):
                print(f" Skipping non-audio file: {object_key}")
                files_skipped += 1
                continue

            last_modified_utc: datetime = s3_object.get("LastModified")  # S3 LastModified datetime
            if not last_modified_utc:
                print(f" Skipping (missing LastModified): {object_key}")
                files_skipped += 1
                continue

            # enforce IST cutoff (strict)
            if not is_after_cutoff(last_modified_utc):
                print(f" Skipping (before/at cutoff {CUTOFF_LOCAL_DATE.isoformat()} IST): {object_key}")
                files_skipped += 1
                continue

            # skip if already transcribed into pc_recordings
            if already_transcribed(db_engine, object_key):
                print(f"  Skipping (already transcribed): {object_key}")
                files_skipped += 1
                continue

            # NEW: Verify that this recording exists in appointment_recording with recording_type='TO Closure'
            verified = verify_recording_in_appointment(db_engine, bucket_name, object_key)
            if not verified:
                # if not verified, skip and do not transcribe
                print(f"  Skipping (not a TO Closure appointment_recording): {object_key}")
                files_skipped += 1
                continue

            # proceed to download the verified audio object
            print(f" Downloading: s3://{bucket_name}/{object_key}")
            file_buffer = io.BytesIO()
            try:
                s3_client.download_fileobj(bucket_name, object_key, file_buffer)  # download into buffer
                file_buffer.seek(0)  # rewind buffer
            except (BotoCoreError, ClientError) as download_error:
                print(f" S3 download failed for {object_key}: {download_error}")
                files_failed += 1
                continue

            file_extension = os.path.splitext(object_key)[1] or ".mp3"  # ensure suffix exists
            temporary_path: Optional[str] = None

            try:
                # create a temp file and write downloaded bytes
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, prefix="pc_audio_") as temp_file:
                    temporary_path = temp_file.name
                    temp_file.write(file_buffer.getvalue())

                # transcribe (translate to English if prefer_translate True)
                print(f" Transcribing to English: {object_key}")
                is_hindi_audio = True
                english_transcript = transcribe_to_english(temporary_path, prefer_translate=is_hindi_audio)
                print(f" Transcription complete: {len(english_transcript)} characters")

                # upsert transcript into pc_recordings
                upsert_transcript(db_engine, object_key, english_transcript, last_modified_utc)
                files_inserted += 1
                files_processed += 1
                print(f" Saved to database: {object_key}")

            except Exception as processing_error:
                print(f" Processing failed for {object_key}: {processing_error}")
                files_failed += 1
            finally:
                # cleanup temp file
                if temporary_path and os.path.exists(temporary_path):
                    try:
                        os.remove(temporary_path)
                    except OSError as cleanup_error:
                        print(f" Temporary file cleanup failed: {cleanup_error}")

            # respect max_files limit if provided
            if max_files and files_processed >= max_files:
                print(f" Reached maximum file limit ({max_files}), stopping ingestion.")
                break

        if max_files and files_processed >= max_files:
            break

    # summary report
    print("\n" + "="*50)
    print(" INGESTION SUMMARY REPORT")
    print("="*50)
    print(f" Files successfully processed: {files_processed}")
    print(f" New database entries: {files_inserted}")
    print(f" Files skipped (duplicates/non-audio/older/not-verified): {files_skipped}")
    print(f" Files failed: {files_failed}")
    print(f" Total objects examined: {files_processed + files_skipped + files_failed}")
    print("="*50)

# ====================== Precise Single-File Ingestion ======================
def ingest_exact_file(bucket: str, key: str) -> None:
    """
    Ingest and transcribe exactly one S3 object (bucket + key).
    This also verifies the appointment_recording table first.
    """
    if not key or key.endswith("/"):
        raise ValueError("Provided key looks like a folder. Please pass an exact object key to a file.")
    if not is_audio_file(key):
        raise ValueError(f"Key is not a supported audio file: {key}")

    s3_client = boto3.client("s3", region_name=AWS_REGION)  # s3 client
    db_engine = get_engine()                                # db engine

    try:
        # fetch object head to obtain LastModified
        head = s3_client.head_object(Bucket=bucket, Key=key)
        last_modified_utc: datetime = head["LastModified"]
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to read metadata for s3://{bucket}/{key}: {e}")

    # cutoff enforcement
    if not is_after_cutoff(last_modified_utc):
        print(f" Skipping (before/at cutoff {CUTOFF_LOCAL_DATE.isoformat()} IST): s3://{bucket}/{key}")
        return

    # already transcribed check
    if already_transcribed(db_engine, key):
        print(f" Skipping (already transcribed): s3://{bucket}/{key}")
        return

    # NEW: Verify existence in appointment_recording
    verified = verify_recording_in_appointment(db_engine, bucket, key)
    if not verified:
        print(f" Skipping (not a TO Closure appointment_recording): s3://{bucket}/{key}")
        return

    # download the file if verified
    print(f" Downloading EXACT file: s3://{bucket}/{key}")
    file_buffer = io.BytesIO()
    try:
        s3_client.download_fileobj(bucket, key, file_buffer)
        file_buffer.seek(0)
    except (BotoCoreError, ClientError) as download_error:
        raise RuntimeError(f"S3 download failed for {key}: {download_error}")

    file_extension = os.path.splitext(key)[1] or ".mp3"
    temporary_path: Optional[str] = None

    try:
        # write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, prefix="pc_audio_") as temp_file:
            temporary_path = temp_file.name
            temp_file.write(file_buffer.getvalue())

        # transcribe
        print(f" Transcribing to English: {key}")
        is_hindi_audio = True
        english_transcript = transcribe_to_english(temporary_path, prefer_translate=is_hindi_audio)
        print(f" Transcription complete: {len(english_transcript)} characters")

        # upsert
        upsert_transcript(db_engine, key, english_transcript, last_modified_utc)
        print(f" Saved to database: s3://{bucket}/{key}")

    finally:
        # cleanup temp file
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError as cleanup_error:
                print(f" Temporary file cleanup failed: {cleanup_error}")

# ====================== Realtime: Polling Watcher ======================
def watch_s3(bucket: str, prefix: str, interval: int, batch_size: int) -> None:
    """
    Polling/daemon mode:
    - Finds the high-water-mark: max(s3_last_modified_utc) from DB OR cutoff date.
    - Every `interval` seconds, lists the prefix and processes up to `batch_size`
      new objects with LastModified > high-water-mark and > cutoff.
    - Each candidate is verified in appointment_recording before transcription.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)  # s3 client
    engine = get_engine()                             # db engine

    # High-water-mark: the later of (DB max) or (cutoff at IST converted to UTC)
    db_max = max_seen_s3_last_modified(engine)
    cutoff_utc = datetime.combine(CUTOFF_LOCAL_DATE, datetime.min.time(), CUTOFF_TZ).astimezone(timezone.utc)
    high_water = max(filter(None, [db_max, cutoff_utc])) if db_max else cutoff_utc

    print(f" Watching s3://{bucket}/{prefix} every {interval}s | starting high-water-mark (UTC): {high_water.isoformat()}")

    while True:
        processed_this_cycle = 0
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                contents = page.get("Contents", [])
                if not contents:
                    continue

                # Select objects strictly newer than high_water
                candidates = [obj for obj in contents if obj.get("LastModified") and obj["LastModified"] > high_water]

                # Sort ascending so we update high_water progressively
                candidates.sort(key=lambda o: o["LastModified"])

                for obj in candidates:
                    if processed_this_cycle >= batch_size:  # do not exceed batch_size
                        break

                    key = obj["Key"]                       # object key
                    if not is_audio_file(key):             # skip non-audio
                        continue

                    lm_utc = obj["LastModified"]           # object's last modified time
                    # Enforce IST strict cutoff again
                    if not is_after_cutoff(lm_utc):
                        continue

                    # Skip if already transcribed
                    if already_transcribed(engine, key):
                        # advance high_water so we don't re-evaluate older objects repeatedly
                        high_water = max(high_water, lm_utc)
                        continue

                    # NEW: verify appointment_recording membership before downloading/transcribing
                    verified = verify_recording_in_appointment(engine, bucket, key)
                    if not verified:
                        print(f" Skipping (not verified TO Closure): {key}")
                        # advance high_water to avoid repeatedly checking the same new but unverified file
                        high_water = max(high_water, lm_utc)
                        continue

                    print(f" New object detected -> downloading: s3://{bucket}/{key}")
                    buf = io.BytesIO()
                    try:
                        s3.download_fileobj(bucket, key, buf)  # download into buffer
                        buf.seek(0)
                    except (BotoCoreError, ClientError) as e:
                        print(f" Download failed for {key}: {e}")
                        # Do not advance high_water; try again next cycle
                        continue

                    ext = os.path.splitext(key)[1] or ".mp3"
                    tmp_path = None
                    try:
                        # write bytes to a temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="pc_audio_") as tmp:
                            tmp_path = tmp.name
                            tmp.write(buf.getvalue())

                        # transcribe
                        print(f" Transcribing to English: {key}")
                        english = transcribe_to_english(tmp_path, prefer_translate=True)
                        print(f" Transcription complete: {len(english)} characters")

                        # upsert transcription
                        upsert_transcript(engine, key, english, lm_utc)
                        processed_this_cycle += 1
                        print(f" Saved to database: {key}")

                        # Advance high-water-mark to this object's LastModified
                        high_water = max(high_water, lm_utc)

                    except Exception as err:
                        print(f" Processing failed for {key}: {err}")
                        # Do not advance high_water so we can retry later
                    finally:
                        # cleanup temp file
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError as ce:
                                print(f" Temp cleanup failed: {ce}")

                if processed_this_cycle >= batch_size:
                    break

        except (BotoCoreError, ClientError) as e:
            print(f" S3 list error: {e} (will retry after sleep)")

        # Sleep before the next polling cycle
        print(f" Sleeping {interval}s... (processed {processed_this_cycle} this cycle)")
        time.sleep(interval)

# ====================== Realtime: SQS Consumer ======================
def consume_sqs(queue_url: str, batch_size: int) -> None:
    """
    Event-driven mode:
    - Long-polls SQS for S3 create events (needs S3 Event Notification -> SQS configured).
    - For each message, extracts bucket/key, enforces cutoff+idempotency, verifies appointment_recording,
      transcribes, upserts, deletes message.
    """
    sqs = boto3.client("sqs", region_name=AWS_REGION)  # sqs client
    s3 = boto3.client("s3", region_name=AWS_REGION)   # s3 client
    engine = get_engine()                             # db engine

    print(f" Consuming SQS queue: {queue_url}")
    while True:
        try:
            # long-poll SQS for messages (up to 10)
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=min(10, batch_size),
                WaitTimeSeconds=20,  # long-poll wait
                MessageAttributeNames=['All'],
            )
            msgs = resp.get("Messages", [])
            if not msgs:
                continue

            for m in msgs:
                receipt = m["ReceiptHandle"]  # receipt handle to delete message later
                try:
                    body = json.loads(m["Body"])
                    # S3 event payloads sometimes wrapped by SNS -> Message
                    if "Records" not in body and "Message" in body:
                        body = json.loads(body["Message"])

                    for rec in body.get("Records", []):
                        if rec.get("eventSource") != "aws:s3":
                            continue
                        bkt = rec["s3"]["bucket"]["name"]
                        key = rec["s3"]["object"]["key"]

                        # URL-encoded keys in events should be decoded
                        key = boto3.utils.unquote_str(key)

                        # Head object to get LastModified
                        try:
                            head = s3.head_object(Bucket=bkt, Key=key)
                            lm_utc: datetime = head["LastModified"]
                        except (BotoCoreError, ClientError) as he:
                            print(f" head_object failed for {bkt}/{key}: {he}")
                            continue

                        # Filters: extension, cutoff, and already transcribed
                        if not is_audio_file(key):
                            continue
                        if not is_after_cutoff(lm_utc):
                            continue
                        if already_transcribed(engine, key):
                            print(f" Already transcribed (skip): {key}")
                            continue

                        # NEW: verify appointment_recording membership before download/transcribe
                        verified = verify_recording_in_appointment(engine, bkt, key)
                        if not verified:
                            print(f" Skipping (not verified TO Closure): {key}")
                            continue

                        # Download and transcribe since verified
                        print(f" Event -> downloading: s3://{bkt}/{key}")
                        buf = io.BytesIO()
                        try:
                            s3.download_fileobj(bkt, key, buf)
                            buf.seek(0)
                        except (BotoCoreError, ClientError) as de:
                            print(f" Download failed for {key}: {de}")
                            continue

                        ext = os.path.splitext(key)[1] or ".mp3"
                        tmp_path = None
                        try:
                            # write to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="pc_audio_") as tmp:
                                tmp_path = tmp.name
                                tmp.write(buf.getvalue())

                            # transcribe to english
                            print(f" Transcribing to English: {key}")
                            english = transcribe_to_english(tmp_path, prefer_translate=True)
                            print(f" Transcription complete: {len(english)} characters")

                            # upsert
                            upsert_transcript(engine, key, english, lm_utc)
                            print(f" Saved to database: {key}")

                        except Exception as err:
                            print(f" Processing failed for {key}: {err}")
                        finally:
                            # cleanup temporary file
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except OSError as ce:
                                    print(f" Temp cleanup failed: {ce}")

                finally:
                    # Always attempt to delete the SQS message to avoid reprocessing storms.
                    # Note: if you require a DLQ for failed items, change this behavior accordingly.
                    try:
                        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                    except (BotoCoreError, ClientError) as de:
                        print(f" Failed to delete SQS message: {de}")

        except (BotoCoreError, ClientError) as e:
            print(f" SQS receive error: {e} (continuing)")

# ============================= CLI =============================
def main():
    """Command-line entrypoint and argument parsing."""
    parser = argparse.ArgumentParser(
        description="PC Call Analyzer - Ingestion (batch + realtime modes) with cutoff and S3 timestamp persistence",
        epilog="""
        Examples:
          python pc_script3_ingest.py init_db
          python pc_script3_ingest.py watch_s3 --bucket qispine-documents --prefix "PC_TO_Conversation/" --interval 60 --batch-size 4
          python pc_script3_ingest.py consume_sqs --queue-url https://sqs.ap-south-1.amazonaws.com/ACCOUNT/queue --batch-size 10
          python pc_script3_ingest.py ingest_s3 --bucket qispine-documents --prefix "PC_TO_Conversation/" --max-files 10
          python pc_script3_ingest.py ingest_file --s3-uri "s3://qispine-documents/PC_TO_Conversation/file.m4a"
        """
    )
    subs = parser.add_subparsers(dest="command", help="Commands")

    subs.add_parser("init_db", help="Initialize database tables and indexes")

    p_watch = subs.add_parser("watch_s3", help="Realtime polling mode (daemon)")
    p_watch.add_argument("--bucket", required=True, help="S3 bucket")
    p_watch.add_argument("--prefix", required=True, help="S3 prefix (folder)")
    p_watch.add_argument("--interval", type=int, default=DEFAULT_WATCH_INTERVAL, help="Polling interval seconds")
    p_watch.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Max files to process per cycle")

    p_sqs = subs.add_parser("consume_sqs", help="Consume S3 create events from SQS")
    p_sqs.add_argument("--queue-url", required=True, help="SQS queue URL to consume")
    p_sqs.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Max messages to process per poll")

    p_ingest = subs.add_parser("ingest_s3", help="Batch ingest and transcribe from S3 prefix")
    p_ingest.add_argument("--bucket", required=True)
    p_ingest.add_argument("--prefix", default="")
    p_ingest.add_argument("--max-files", type=int, default=None)

    p_file = subs.add_parser("ingest_file", help="Ingest and transcribe exactly one S3 object")
    p_file.add_argument("--bucket")
    p_file.add_argument("--key")
    p_file.add_argument("--s3-uri", dest="s3_uri")

    args = parser.parse_args()

    if args.command == "init_db":
        init_db()
        print(" Database initialization completed successfully")
        return

    if args.command == "watch_s3":
        if not OPENAI_API_KEY:
            raise SystemExit(" CRITICAL: OPENAI_API_KEY environment variable is not set")
        print(" Starting polling watcher...")
        watch_s3(args.bucket, args.prefix, args.interval, args.batch_size)
        return

    if args.command == "consume_sqs":
        if not OPENAI_API_KEY:
            raise SystemExit(" CRITICAL: OPENAI_API_KEY environment variable is not set")
        print(" Starting SQS consumer...")
        consume_sqs(args.queue_url, args.batch_size)
        return

    if args.command == "ingest_s3":
        if not OPENAI_API_KEY:
            raise SystemExit(" CRITICAL: OPENAI_API_KEY environment variable is not set")
        if not args.bucket:
            raise SystemExit(" CRITICAL: Bucket name is required for ingestion")
        print(" Starting batch ingestion pipeline...")
        ingest_s3(args.bucket, args.prefix, args.max_files)
        return

    if args.command == "ingest_file":
        if not OPENAI_API_KEY:
            raise SystemExit(" CRITICAL: OPENAI_API_KEY environment variable is not set")
        s3_uri = getattr(args, "s3_uri", None)
        bucket = args.bucket
        key = args.key
        if s3_uri:
            bucket, key = parse_s3_uri(s3_uri)
        elif not (bucket and key):
            raise SystemExit(" Provide either --s3-uri OR both --bucket and --key for ingest_file")
        print(f" Ingesting exact file: s3://{bucket}/{key}")
        ingest_exact_file(bucket, key)
        return

    parser.print_help()
    raise SystemExit(" No valid command specified")

# ============================== Entry ==============================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Operation cancelled by user")
        raise SystemExit(1)
    except Exception as unexpected_error:
        print(f" Unexpected error in main execution: {unexpected_error}")
        raise SystemExit(1)


