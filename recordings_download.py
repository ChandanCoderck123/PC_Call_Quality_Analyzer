#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# s3_download_audio.py
# A small utility to list and download audio files from S3 (batch or single-file).
# Each line below is commented to explain what it does.
# Download all audio files under a prefix into ./downloads (preserves S3 structure)
# python recordings_download.py --bucket qispine-documents --prefix PC_TO_Conversation/ --local-folder ./downloads

# Download only the first N files (useful for testing)
# python recordings_download.py --bucket qispine-documents --prefix PC_TO_Conversation/ --local-folder ./downloads --max-files 5

# Download a specific single S3 object using full S3 URI
# python recordings_download.py --s3-uri s3://qispine-documents/PC_TO_Conversation/pc_voice_recorder_1762491616451_2135065269159966.m4a --local-folder ./downloads

# Download a specific single S3 object using bucket + key
# python recordings_download.py --bucket qispine-documents --key PC_TO_Conversation/pc_voice_recorder_1762572254739_2215703558008611.m4a --local-folder ./downloads

# ------------------------- Standard library imports -------------------------
import os                            # file system operations (mkdir, path join, exists)
import sys                           # for graceful exit and platform checks
import argparse                      # command-line argument parsing
import io                            # in-memory byte streams for S3 download_fileobj
from typing import Optional, List    # type hints for function signatures
from urllib.parse import urlparse    # parse s3:// URIs

# --------------------------- Environment Configuration ---------------------------
try:
    # Try to load environment variables from .env file for local development
    from dotenv import load_dotenv  # Optional dependency - if not available, use system env
    load_dotenv()  # Load environment variables from .env file if it exists
    print(" Environment variables loaded from .env file")  # Inform user about source
except Exception:
    # If python-dotenv is not installed or .env file doesn't exist, continue with system environment
    print(" No .env file found, using system environment variables")  # Inform user

# --------------------------- Third-party imports ---------------------------
# boto3 is the AWS SDK for Python and must be installed in your environment.
# Install with: pip install boto3
try:
    import boto3                      # AWS client to list and download objects from S3
    from botocore.exceptions import BotoCoreError, ClientError  # S3/HTTP error handling
except Exception as e:
    print("Missing dependency: boto3. Install with: pip install boto3")
    raise e

# ------------------------- AWS Configuration ----------------------------
# Get AWS credentials from environment variables with fallbacks
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") 
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")  # Default to Mumbai region

# Validate that required AWS credentials are available
def validate_aws_credentials():
    """Check if AWS credentials are properly configured and return True if valid."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("AWS credentials not found in environment variables!")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        print("You can set them in your .env file or system environment variables")
        return False
    return True

def create_s3_client():
    """
    Create and return a configured S3 client with explicit credentials.
    This fixes the 'Unable to locate credentials' error.
    """
    if not validate_aws_credentials():
        raise RuntimeError("AWS credentials not configured")
    
    print(f"Using AWS Region: {AWS_DEFAULT_REGION}")
    print("AWS credentials validated successfully")
    
    # Create S3 client with explicit credentials from environment variables
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )

# ------------------------- Allowed audio formats ----------------------------
# A simple list of audio file extensions that we treat as valid audio.
ALLOWED_EXTS: List[str] = [
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".oga", ".webm", ".mp4", ".aac", ".wma"
]

# ------------------------- Utility helper functions -------------------------
def is_audio_file(key: str) -> bool:
    """Return True if the S3 object key ends with a supported audio extension."""
    # Normalize to lower-case so extension check is case-insensitive
    k = key.lower()
    # Return True if any allowed extension matches the end of the key
    return any(k.endswith(ext) for ext in ALLOWED_EXTS)

def parse_s3_uri(s3_uri: str) -> (str, str):
    """Parse s3://bucket/key into (bucket, key). Raises ValueError on bad input."""
    # Ensure scheme is present and correct
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")
    # Use urlparse for robustness, then split netloc and path
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    # Trim leading slash from path part to form the object key
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError("S3 URI must include both bucket and key, e.g., s3://my-bucket/path/file.m4a")
    return bucket, key

def ensure_local_dir(path: str) -> None:
    """Create local directory if it does not exist (including parents)."""
    # os.makedirs with exist_ok avoids race conditions and multiple-create errors
    os.makedirs(path, exist_ok=True)

def download_s3_object_to_file(s3_client, bucket: str, key: str, local_path: str) -> None:
    """
    Download a single S3 object to local_path using download_fileobj stream to avoid loading whole file to memory.
    Raises exceptions from boto3 if download fails.
    """
    # Ensure parent directory exists for the destination file
    ensure_local_dir(os.path.dirname(local_path) or ".")
    
    print(f"Downloading: s3://{bucket}/{key}")
    print(f"Saving to: {local_path}")
    
    # Use BytesIO then write to disk (safer for partial downloads) OR directly stream to file
    try:
        # Open a temporary local file for streaming download to avoid holding bytes in RAM
        with open(local_path, "wb") as f:
            # boto3's download_fileobj writes to the provided file-like object
            s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=f)
        print(f"Successfully downloaded: {os.path.basename(local_path)}")
        
    except (BotoCoreError, ClientError) as e:
        # Remove possibly partial file on error to avoid confusion next time
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                print(f"🧹 Cleaned up partial file: {local_path}")
        except Exception:
            pass
        # Re-raise so the caller can decide what to log and how to proceed
        raise

# ------------------------- Core batch download logic ------------------------
def download_from_prefix(bucket: str, prefix: str, local_base: str, max_files: Optional[int] = None) -> None:
    """List objects under prefix and download audio files into local_base preserving S3 key structure."""
    # Build an S3 client with explicit credentials
    s3 = create_s3_client()
    
    # Ensure local base folder exists
    ensure_local_dir(local_base)
    
    print(f"Searching in: s3://{bucket}/{prefix}")
    if max_files:
        print(f"Maximum files to download: {max_files}")
    
    # Paginator for large listings to avoid memory blow-up
    paginator = s3.get_paginator("list_objects_v2")
    # Configure pagination arguments
    pagination_params = {"Bucket": bucket, "Prefix": prefix}
    files_downloaded = 0  # counter to respect max_files
    total_files_found = 0
    
    # Iterate pages of S3 objects
    for page_num, page in enumerate(paginator.paginate(**pagination_params), 1):
        print(f"Processing page {page_num}...")
        
        # If page has no 'Contents', skip it
        if "Contents" not in page:
            print("   No objects found in this page")
            continue
            
        # For each object in the page
        for obj in page["Contents"]:
            key = obj.get("Key", "").strip()
            # Skip if empty key or prefix-like folder ending with '/'
            if not key or key.endswith("/"):
                continue
                
            total_files_found += 1
            
            # Check if object is audio by extension
            if not is_audio_file(key):
                # Skip non-audio files
                print(f"Skipping non-audio file: {key}")
                continue
                
            # Build relative path under local_base that mirrors S3 prefix structure
            rel_path = key
            local_path = os.path.join(local_base, rel_path)
            
            # If file already exists locally, skip download (idempotency)
            if os.path.exists(local_path):
                print(f"Skipping (already exists): {os.path.basename(local_path)}")
                continue
                
            # Attempt download and print progress
            try:
                download_s3_object_to_file(s3, bucket, key, local_path)
                files_downloaded += 1
            except Exception as e:
                print(f"Failed to download s3://{bucket}/{key}: {e}")
                
            # If we reached the requested maximum files, stop
            if max_files and files_downloaded >= max_files:
                print(f"Reached max_files limit: {max_files}")
                break
                
        # Break outer loop if max_files reached
        if max_files and files_downloaded >= max_files:
            break
            
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Total files found in S3: {total_files_found}")
    print(f"Files successfully downloaded: {files_downloaded}")
    print(f"Local destination: {local_base}")
    print("="*50)

# ------------------------ Single-file download helper -----------------------
def download_single_s3_object(s3_uri: Optional[str], bucket: Optional[str], key: Optional[str], local_base: str) -> None:
    """
    Download a single S3 object specified by either s3_uri OR bucket+key into local_base folder.
    Saves file under local_base/<key>.
    """
    # Resolve bucket and key from input parameters
    if s3_uri:
        bucket, key = parse_s3_uri(s3_uri)
    if not bucket or not key:
        raise ValueError("Either --s3-uri or both --bucket and --key must be provided for single-file download.")
    
    # Validate audio extension
    if not is_audio_file(key):
        raise ValueError(f"Target object key does not appear to be an audio file (allowed extensions: {ALLOWED_EXTS}): {key}")
    
    # Build full local path and download
    local_path = os.path.join(local_base, key)
    
    if os.path.exists(local_path):
        print(f"Skipping (already exists): {local_path}")
        file_size = os.path.getsize(local_path)
        print(f"File size: {file_size} bytes")
        return
        
    # Create S3 client with explicit credentials
    s3 = create_s3_client()
    
    # First, let's check if the file exists in S3 and get its info
    try:
        print(f"Checking S3 object existence...")
        head_response = s3.head_object(Bucket=bucket, Key=key)
        file_size = head_response['ContentLength']
        last_modified = head_response['LastModified']
        print(f"S3 Object Info:")
        print(f"Size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"Last Modified: {last_modified}")
        print(f"Key: {key}")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise RuntimeError(f"S3 object not found: s3://{bucket}/{key}")
        elif error_code == '403':
            raise RuntimeError(f"Access denied to S3 object: s3://{bucket}/{key}. Check permissions.")
        else:
            raise RuntimeError(f"Error accessing S3 object: {e}")
    
    # Proceed with download
    download_s3_object_to_file(s3, bucket, key, local_path)

# ------------------------------ Command line UI -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argparse parser for CLI usage."""
    p = argparse.ArgumentParser(
        description="Download audio files from S3 (batch by prefix or single file).",
        epilog="""
Examples:

  # Batch download all audio files from a prefix
  python s3_download_audio.py --bucket qispine-documents --prefix PC_TO_Conversation/ --local-folder ./downloads

  # Batch download with file limit (for testing)
  python s3_download_audio.py --bucket qispine-documents --prefix PC_TO_Conversation/ --local-folder ./downloads --max-files 5

  # Single file download using S3 URI
  python s3_download_audio.py --s3-uri s3://qispine-documents/PC_TO_Conversation/file.m4a --local-folder ./downloads

  # Single file download using bucket + key
  python s3_download_audio.py --bucket qispine-documents --key PC_TO_Conversation/file.m4a --local-folder ./downloads
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Batch mode options
    p.add_argument("--bucket", help="S3 bucket name (required for --prefix batch mode unless using --s3-uri single mode).")
    p.add_argument("--prefix", default="", help="S3 prefix to list and download (e.g., 'PC_TO_Conversation/').")
    # Single-file mode options
    p.add_argument("--s3-uri", help='Full S3 URI for a specific file, e.g., "s3://bucket/path/file.m4a".')
    p.add_argument("--key", help="Exact S3 object key for single-file download (alternate to --s3-uri).")
    # Common options
    p.add_argument("--local-folder", required=True, help="Local base folder where downloaded files will be saved.")
    p.add_argument("--max-files", type=int, default=None, help="Optional limit on number of files to download (batch mode).")
    return p

def main():
    """Main entrypoint; parse args and dispatch to batch or single-file download."""
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Normalize local folder and ensure it exists
    local_folder = os.path.abspath(args.local_folder)
    ensure_local_dir(local_folder)
    
    print("Starting S3 Audio Download Utility")
    print(f"Local folder: {local_folder}")
    
    # If s3-uri or key provided, run single-file download
    if args.s3_uri or args.key:
        try:
            download_single_s3_object(args.s3_uri, args.bucket, args.key, local_folder)
            print("Single file download completed successfully!")
        except Exception as e:
            print(f" Error downloading single file: {e}")
            sys.exit(2)
        return
        
    # Otherwise, require bucket + prefix (prefix can be empty string to download whole bucket - BE CAREFUL)
    if not args.bucket:
        print("Batch mode requires --bucket and optional --prefix. For single file use --s3-uri or --key.")
        parser.print_help()
        sys.exit(2)
        
    try:
        download_from_prefix(args.bucket, args.prefix, local_folder, max_files=args.max_files)
        print("Batch download completed successfully!")
    except Exception as e:
        print(f"Error during batch download: {e}")
        sys.exit(3)

# ------------------------------- Entry guard --------------------------------
if __name__ == "__main__":
    # Run the main CLI if executed directly
    try:
        main()
    except KeyboardInterrupt:
        print("\n Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)