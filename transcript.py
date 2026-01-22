import os
from google import genai
from google.genai import types
import time
import pathlib
import random
import concurrent.futures
import threading

# --- Configuration ---
API_KEY = "YOUR_API_KEY_HERE" # Will be overwritten by the GUI/Main script
AUDIO_DIR = "audio_chunks"
INTERMEDIATE_DIR = "intermediate_transcripts"
MAX_RETRIES = 5
INITIAL_DELAY = 3  # Increased base delay
DEFAULT_MODEL = "gemini-3-pro-preview" # Gemini 3 Pro for accurate transcription (Flash causes hallucination)
DEFAULT_MAX_WORKERS = 2  # Reduced from 5 to avoid overloading the API

# Vertex AI regions for failover - ordered by typical reliability/capacity
VERTEX_REGIONS = [
    "us-central1",
    "us-east4", 
    "us-west1",
    "europe-west1",
    "europe-west4",
    "asia-northeast1",
    "asia-southeast1",
]
# ---------------------

# Global client management
_thread_local = threading.local()
_current_region_index = 0
_region_lock = threading.Lock()

def get_system_instruction(target_language="Simplified Chinese"):
    return f"""You are an expert transcription engine powered by Gemini 3.0.
Task:
1.  **Transcribe** the audio verbatim in its original language.
2.  **Translate** the transcript into {target_language} (if the audio is not already in that language).
3.  **Timestamp** every single sentence precisely.

**CRITICAL FORMATTING RULES:**
* You MUST provide timestamps at the start of every segment in the format **[MM:SS.ms]** (Minutes:Seconds.Milliseconds).
* Example: [00:05.123] This is the first sentence.
* Do NOT use (MM:SS) or [MM:SS]. You MUST include milliseconds.
* Output exactly four sections separated by blank lines.
* Output the transcript and translation as it is
* Extract the text/subtitle that appear on the video as it is. 
* Use strikethrough if you found explicit text, translation accuracy are priority so don't replace or censor it, instead `̶S̶t̶r̶i̶k̶e̶-̶t̶h̶r̶o̶u̶g̶h̶ it`

**Strict Output Template:**
Transcript:
[Full transcript text without timestamps]

Translation:
[Full translation text without timestamps]

Timestamped Transcript:
[00:00.000] First sentence of the transcript.
[00:05.500] Second sentence.

Timestamped Translation:
[00:00.000] First sentence of the translation.
[00:05.500] Second sentence of the translation.
"""

def get_next_region():
    """Get the next region in rotation for failover."""
    global _current_region_index
    with _region_lock:
        region = VERTEX_REGIONS[_current_region_index]
        _current_region_index = (_current_region_index + 1) % len(VERTEX_REGIONS)
        return region

def create_client(api_key=None, project_id=None, use_vertex=False, region=None):
    """Creates a GenAI client with optional Vertex AI configuration."""
    if use_vertex and project_id:
        # Use Vertex AI with region-based routing
        location = region or VERTEX_REGIONS[0]
        print(f"  Using Vertex AI region: {location}")
        return genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
    else:
        # Use standard Gemini API with API key
        return genai.Client(api_key=api_key)

def get_client(api_key=None, project_id=None, use_vertex=False, region=None):
    """Gets or creates a thread-local GenAI client."""
    # Create new client if region changed or doesn't exist
    client_key = f"{use_vertex}_{region}_{project_id}_{api_key[:8] if api_key else 'none'}"
    
    if not hasattr(_thread_local, 'client_key') or _thread_local.client_key != client_key:
        _thread_local.client = create_client(api_key, project_id, use_vertex, region)
        _thread_local.client_key = client_key
    
    return _thread_local.client

def initialize_genai_client(api_key=None, project_id=None, use_vertex=False):
    """Initializes the GenAI client (validates the configuration)."""
    try:
        if use_vertex and project_id:
            client = genai.Client(vertexai=True, project=project_id, location=VERTEX_REGIONS[0])
        else:
            client = genai.Client(api_key=api_key)
        _thread_local.client = client
        return True
    except Exception as e:
        print(f"Error initializing GenAI client: {e}")
        return None

def process_audio_file(filepath, intermediate_dir, system_instruction, model_name, api_key, project_id=None, use_vertex=False):
    """Processes a single audio file: uploads, transcribes, and saves the result."""
    filename = os.path.basename(filepath)
    transcript_filename = pathlib.Path(filename).stem + ".txt"
    intermediate_filepath = os.path.join(intermediate_dir, transcript_filename)

    current_region = None
    
    for attempt in range(MAX_RETRIES):
        uploaded_file = None
        try:
            # On retry after 503/504, try a different region if using Vertex AI
            if use_vertex and attempt > 0:
                current_region = get_next_region()
                print(f"  Switching to region: {current_region}")
            
            # Get client (with potential region change)
            client = get_client(api_key, project_id, use_vertex, current_region)
            
            print(f"  Uploading {filename} (Attempt {attempt + 1}/{MAX_RETRIES})...")
            
            # Upload file using new SDK
            uploaded_file = client.files.upload(file=filepath)
            
            # Wait for processing to complete (important for large files)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                raise ValueError("Audio file upload failed processing.")

            print(f"  Transcribing {filename} with model {model_name}...")
            
            # Use simplified content pattern - just pass file and prompt directly
            # Lower temperature (0.2) for more deterministic, accurate transcription
            response = client.models.generate_content(
                model=model_name,
                contents=["Please process this audio file strictly according to the system instructions.", uploaded_file],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2  # Very low for accurate transcription, reduces hallucination
                )
            )
            
            transcript = response.text

            with open(intermediate_filepath, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            print(f"  Successfully processed and saved: {intermediate_filepath}")
            
            # Clean up uploaded file
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass  # Ignore deletion errors
                
            return transcript

        except Exception as e:
            error_str = str(e)
            print(f"  Error processing {filename} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            
            # Try to clean up on error
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
                    
            if attempt < MAX_RETRIES - 1:
                # Determine if this is a rate limit / overload error
                is_overload = "503" in error_str or "overload" in error_str.lower() or "unavailable" in error_str.lower()
                is_timeout = "504" in error_str or "deadline" in error_str.lower() or "timeout" in error_str.lower()
                
                if is_overload:
                    delay = (INITIAL_DELAY * (3 ** attempt)) + random.uniform(2, 6)
                    print(f"  Model overloaded. Will try different region. Waiting {delay:.2f} seconds...")
                elif is_timeout:
                    delay = (INITIAL_DELAY * (2 ** attempt)) + random.uniform(1, 4)
                    print(f"  Request timed out. Will try different region. Waiting {delay:.2f} seconds...")
                else:
                    delay = (INITIAL_DELAY * (2 ** attempt)) + random.uniform(0, 2)
                    print(f"  Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                error_message = f"Error processing {filename} after {MAX_RETRIES} attempts: {e}\n"
                with open(intermediate_filepath, "w", encoding="utf-8") as f:
                    f.write(error_message)
                print(f"  Failed to process {filename}. Error saved to {intermediate_filepath}")
                return ""
    return ""

def run_transcription(api_key, audio_dir, intermediate_dir, system_instruction, model_name=DEFAULT_MODEL, 
                      progress_queue=None, max_workers=DEFAULT_MAX_WORKERS, skip_existing=True,
                      project_id=None, use_vertex=False):
    """Transcribes all audio files in a directory in parallel."""
    if not initialize_genai_client(api_key, project_id, use_vertex):
        if progress_queue: progress_queue.put("Failed to initialize GenAI client. Check API key or project configuration.")
        return False

    pathlib.Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        audio_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".mp3")])
    except FileNotFoundError:
        if progress_queue: progress_queue.put(f"Error: Audio directory not found at {audio_dir}")
        return False

    if not audio_files:
        if progress_queue: progress_queue.put(f"No .mp3 files found in {audio_dir}")
        return False

    processed_count = 0
    success_count = 0
    skipped_count = 0
    count_lock = threading.Lock()

    def update_progress(message):
        if progress_queue: progress_queue.put(message)
        print(message)

    def is_valid_transcript(filepath):
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().lower()
            return "error" not in content and "timestamped transcript:" in content

    def process_file_wrapper(filepath):
        nonlocal processed_count, success_count, skipped_count
        filename = os.path.basename(filepath)
        intermediate_filepath = os.path.join(intermediate_dir, f"{pathlib.Path(filename).stem}.txt")

        if skip_existing and is_valid_transcript(intermediate_filepath):
            with count_lock:
                processed_count += 1
                skipped_count += 1
                success_count += 1
            update_progress(f"({processed_count}/{len(audio_files)}) Skipping existing valid transcript: {filename}")
            return "SKIPPED"
        
        # Add small delay between requests to avoid rate limits
        time.sleep(random.uniform(0.5, 1.5))
        
        result = process_audio_file(filepath, intermediate_dir, system_instruction, model_name, api_key, project_id, use_vertex)
        
        with count_lock:
            processed_count += 1
            if result and "error" not in result.lower():
                success_count += 1
                status = "Success"
            else:
                status = "Failed"
        update_progress(f"({processed_count}/{len(audio_files)}) {status}: {filename}")
        return result

    actual_workers = min(max_workers, len(audio_files))
    mode_str = f"Vertex AI (regions: {', '.join(VERTEX_REGIONS[:3])}...)" if use_vertex else "Gemini API"
    update_progress(f"Starting transcription of {len(audio_files)} files with {actual_workers} workers using {model_name} via {mode_str}...")

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        executor.map(process_file_wrapper, audio_files)

    total_time = time.time() - start_time
    summary_msg = (f"\nTranscription complete. "
                   f"Success: {success_count}/{len(audio_files)} "
                   f"(Skipped: {skipped_count}, Newly Processed: {success_count - skipped_count}). "
                   f"Total time: {total_time:.2f} seconds.")
    update_progress(summary_msg)
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe audio files.")
    parser.add_argument("--api-key", help="Google AI API key (for Gemini API mode).")
    parser.add_argument("--project-id", help="Google Cloud project ID (for Vertex AI mode).")
    parser.add_argument("--use-vertex", action="store_true", help="Use Vertex AI instead of Gemini API for region failover support.")
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--intermediate-dir", default=INTERMEDIATE_DIR)
    parser.add_argument("--target-language", default="Simplified Chinese")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    if not args.api_key and not args.project_id:
        print("Error: Either --api-key or --project-id (with --use-vertex) is required.")
        exit(1)

    system_instruction = get_system_instruction(args.target_language)
    run_transcription(
        api_key=args.api_key, 
        audio_dir=args.audio_dir, 
        intermediate_dir=args.intermediate_dir, 
        system_instruction=system_instruction, 
        model_name=args.model_name, 
        max_workers=args.max_workers,
        project_id=args.project_id,
        use_vertex=args.use_vertex
    )
