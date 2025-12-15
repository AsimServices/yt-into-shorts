# Viralize/main.py
import os, json, csv, uuid, shutil, time, contextlib
from yt_dlp import YoutubeDL
from moviepy.editor import VideoFileClip
import openai
import zipfile
import subprocess

# Optional: suppress ffmpeg/yt-dlp noisy output by redirecting stdout/stderr where desired

def add_user_scripts_to_path():
    import site
    import sys
    # Add user scripts to PATH (Windows specific mainly)
    user_base = site.getuserbase()
    # On Windows, scripts are in PythonXY/Scripts inside userbase usually, or just Scripts
    # My finding was .../Python312/Scripts. site.getuserbase() returns .../Python312
    scripts_path = os.path.join(user_base, 'Scripts')
    if os.path.exists(scripts_path):
        os.environ["PATH"] += os.pathsep + scripts_path
    
    # Also check sys.prefix/Scripts
    sys_scripts = os.path.join(sys.prefix, 'Scripts')
    if os.path.exists(sys_scripts) and sys_scripts not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + sys_scripts

add_user_scripts_to_path()

def ensure_dirs():
    for d in ['videos', 'shorts', 'tmp', 'static/downloads', 'processing_status']:
        os.makedirs(d, exist_ok=True)
ensure_dirs()

# ---------- status helper ----------
class JobCancelledException(Exception):
    pass

def check_cancellation(request_code):
    if not request_code: return
    cancel_path = os.path.join("processing_status", f"{request_code}.cancel")
    if os.path.exists(cancel_path):
        raise JobCancelledException("Job was cancelled by user.")

def update_status(request_code, **kwargs):
    """
    Update processing_status/<request_code>.json with fields in kwargs (merges).
    Also checks for cancellation.
    """
    # Check cancel *before* write
    check_cancellation(request_code)
    
    status_path = os.path.join("processing_status", f"{request_code}.json")
    state = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception:
            state = {}
    
    # Check if we are already in failed/cancelled state? No, let's just write new state.
    state.update(kwargs)
    # keep progress in 0..100
    if "progress" in state:
        try:
            p = float(state["progress"])
            state["progress"] = max(0, min(100, p))
        except Exception:
            pass
    with open(status_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh)

# ---------- zip helper ----------
def zip_the_files(files, output_zip_name):
    # create zip at temp location and move to static/downloads
    tmp_zip = output_zip_name if os.path.isabs(output_zip_name) else os.path.abspath(output_zip_name)
    with zipfile.ZipFile(tmp_zip, 'w') as f:
        for file in files:
            if os.path.exists(file):
                f.write(file, arcname=os.path.basename(file))
    dest = os.path.join('static', 'downloads', os.path.basename(tmp_zip))
    shutil.move(tmp_zip, dest)
    return "/static/downloads/" + os.path.basename(tmp_zip)

# ---------- downloader with progress hook ----------
def download_youtube_video(video_id_or_url, output_folder='videos', request_code=None):
    """
    Download using yt-dlp. Uses a progress_hook to update processing_status with percent.
    Returns absolute path to downloaded file.
    """
    if request_code: check_cancellation(request_code)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if video_id_or_url.startswith("http://") or video_id_or_url.startswith("https://"):
        youtube_url = video_id_or_url
    else:
        youtube_url = f"https://www.youtube.com/watch?v={video_id_or_url}"

    outtmpl = os.path.join(output_folder, "longvideo-%(id)s.%(ext)s")

    def progress_hook(d):
        # check cancel frequently
        if request_code: check_cancellation(request_code)

        # d is a dict from yt-dlp progress events
        if request_code:
            if d.get('status') == 'downloading':
                # percent or bytes
                percent = d.get('progress') or d.get('percentage') or None
                # yt-dlp sometimes includes 'downloaded_bytes' and 'total_bytes'
                try:
                    if not percent:
                        if d.get('total_bytes') and d.get('downloaded_bytes'):
                            percent = d['downloaded_bytes'] / d['total_bytes'] * 100
                except Exception:
                    percent = None
                if percent is None:
                    # fallback to ETA based rough mapping (not precise)
                    update_status(request_code, download_percent=0, message="Downloading video...", status="downloading", progress=10)
                else:
                    update_status(request_code, download_percent=round(percent, 2), message=f"Downloading video... {round(percent,2)}%", status="downloading", progress=10 + int(percent*0.08))
            elif d.get('status') == 'finished':
                update_status(request_code, download_percent=100, message="Download finished, post-processing...", status="downloaded", progress=20)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': outtmpl,
        'noplaylist': True,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [progress_hook]
    }

    # run extract_info with stdout/stderr suppressed to reduce console noise
    with YoutubeDL(ydl_opts) as ydl:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
        # ensure mp4 extension if produced
        if not filename.lower().endswith('.mp4'):
            base, _ = os.path.splitext(filename)
            mp4_file = base + ".mp4"
            if os.path.exists(mp4_file):
                filename = mp4_file
    update_status(request_code, message="Downloaded video.", status="downloaded", progress=25)
    return os.path.abspath(filename)

# ---------- subtitle extraction ----------
def srt_to_transcript(srt_string):
    entries = srt_string.strip().split("\n\n")
    transcript = []
    for entry in entries:
        lines = entry.split("\n")
        if len(lines) < 3:
            continue
        try:
            times = lines[1].split("-->")
            start = times[0].strip().replace(",", ".")
            end = times[1].strip().replace(",", ".")
            text = " ".join(lines[2:])
            transcript.append({"start": start, "end": end, "text": text})
        except Exception:
            continue
    return transcript

def get_auto_subtitle_path():
    import shutil
    import site
    import sys
    
    # 1. Check PATH first
    path = shutil.which("auto_subtitle")
    if path:
        return path
    
    # 2. Check User Scripts (AppData/Roaming/Python/PythonXY/Scripts)
    # site.getuserbase() returns .../PythonXY usually
    user_base = site.getuserbase()
    candidates = [
        os.path.join(user_base, 'Scripts', 'auto_subtitle.exe'),
        os.path.join(user_base, 'Scripts', 'auto_subtitle'),
        # Sometimes userbase is just AppData/Roaming/Python
        os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Python', f'Python{sys.version_info.major}{sys.version_info.minor}', 'Scripts', 'auto_subtitle.exe'),
    ]
    
    # 3. Check System Scripts (Python install dir)
    candidates.append(os.path.join(sys.prefix, 'Scripts', 'auto_subtitle.exe'))
    
    for c in candidates:
        if os.path.exists(c):
            return c
            
    return "auto_subtitle" # Fallback

def extract_subtitles(input_file):
    """
    Try auto_subtitle if installed; if not, return an empty transcript.
    """
    try:
        auto_subtitle_cmd = get_auto_subtitle_path()
        # Quote the executable path in case of spaces
        cmd = f'"{auto_subtitle_cmd}" "{input_file}" --srt_only True --output_srt True -o tmp/ --model base'
        print(f"Running command: {cmd}") # Debug logging
        subprocess.call(cmd, shell=True)
        srt_filename = f"tmp/{os.path.basename(input_file).split('.')[0]}.srt"
        with open(srt_filename, 'r', encoding='utf-8') as fh:
            srt = fh.read()
        return srt_to_transcript(srt)
    except Exception as e:
        print("Subtitle extraction failed or auto_subtitle not present:", e)
        return []

# ---------- time helpers ----------
def hms_to_seconds(hms):
    parts = [int(p) for p in hms.split(":")]
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    if len(parts) == 2:
        return parts[0]*60 + parts[1]
    return parts[0]

def sec_to_hms(t):
    t = int(max(0, t))
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return f"{h:02}:{m:02}:{s:02}"

def get_entry_duration(entry):
    try:
        start = entry["start"].split(".")[0]
        end = entry["end"].split(".")[0]
        return hms_to_seconds(end) - hms_to_seconds(start)
    except Exception:
        return 0

def divide_transcript_into_chunks(transcript, chunk_size):
    """Divide transcript entries by cumulative seconds into chunk lists."""
    chunks = []
    current = []
    curr_dur = 0
    idx = 0
    while idx < len(transcript):
        entry = transcript[idx]
        dur = get_entry_duration(entry)
        if dur > chunk_size:
            # put large entry on its own
            chunks.append([entry])
            idx += 1
            continue
        if curr_dur + dur <= chunk_size:
            current.append(entry)
            curr_dur += dur
            idx += 1
        else:
            chunks.append(current)
            current = []
            curr_dur = 0
    if current:
        chunks.append(current)
    return chunks

# ---------- OpenAI - chunked analysis ----------
from openai.error import InvalidRequestError

DEFAULT_CHUNK_CHARS_LIMIT = 11000

def chunk_entries_to_text(chunk):
    texts = []
    for entry in chunk:
        t = entry.get("text", "")
        if t:
            texts.append(t.strip())
    return "\n".join(texts)

def generate_viral_for_chunk(chunk_text, chunk_start_seconds, max_segments, openai_api_key, min_segment_seconds=55, max_segment_seconds=110):
    """
    Ask OpenAI to analyze a chunk only. Return parsed JSON (dict).
    """
    openai.api_key = openai_api_key

    json_template = '{ "segments": [ { "start_time": "HH:MM:SS", "end_time": "HH:MM:SS" } ] }'

    system = (
        "You are a Viral Content Expert. Your goal is to identify the most engaging, shareable, and viral-worthy segments from a video transcript. "
        "Look for segments that have:"
        "1. A strong hook or opening line."
        "2. High emotional value (humor, shock, inspiration, controversy)."
        "3. A clear beginning, middle, and end context."
        f"Each segment MUST be between {min_segment_seconds} and {max_segment_seconds} seconds. "
        "Return only valid JSON with a top-level 'segments' array, each item having 'start_time' and 'end_time' in HH:MM:SS format. "
        "Start and end times must be relative to the beginning of this chunk (chunk start = 00:00:00). Do NOT include commentary."
    )

    user = (
        f"Chunk start offset (seconds): {chunk_start_seconds}\n"
        f"Desired number of segments (max): {max_segments}\n\n"
        f"Transcript chunk (only lines, no timestamps):\n{chunk_text}\n\n"
        f"Return JSON strictly following: {json_template}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    max_retries = 5
    base_delay = 20
    
    for attempt in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=3000
            )
            break # success
        except openai.error.RateLimitError:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (attempt + 1)
                print(f"Rate limit hit. Sleeping for {sleep_time} seconds before retry {attempt+1}/{max_retries}...")
                time.sleep(sleep_time)
            else:
                raise # re-raise last error
        except openai.error.ServiceUnavailableError:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (attempt + 1)
                print(f"Service unavailable. Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise
        except Exception as e:
            # Check for rate limit message if it wasn't caught by specific error class (sometime happens in diff versions)
            if "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (attempt + 1)
                    print(f"Rate limit hit (generic catch). Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
            raise e

    content = resp.choices[0].message.content.strip()
    if content.endswith(","):
        content = content[:-1] + "\n    }\n]}"
    # try to extract JSON portion
    import re
    m = re.search(r'(\{.*\}\s*)$', content, flags=re.DOTALL)
    json_text = m.group(1) if m else content
    try:
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        # if parse fails, return empty segments
        return {"segments": []}

def analyze_transcript_in_chunks(transcript, video_duration_seconds, openai_api_key, request_code=None):
    """
    Divide transcript into chunks, analyze each chunk, convert to absolute times, and merge results.
    """
    chunk_seconds = 8 * 60
    chars_limit = DEFAULT_CHUNK_CHARS_LIMIT
    max_segments_per_chunk = 2

    transcript_chunks = divide_transcript_into_chunks(transcript, chunk_seconds)
    all_segments = []

    for idx, chunk_entries in enumerate(transcript_chunks):
        chunk_text = chunk_entries_to_text(chunk_entries)
        if len(chunk_text) > chars_limit:
            parts = chunk_text.splitlines()
            while parts and len("\n".join(parts)) > chars_limit:
                parts.pop(0)
            chunk_text = "\n".join(parts)

        try:
            first = chunk_entries[0]
            chunk_start_seconds = hms_to_seconds(first["start"].split(".")[0])
        except Exception:
            chunk_start_seconds = idx * chunk_seconds

        # retry shrink loop if OpenAI complains
        attempt_limit = 3
        attempt = 0
        parsed = {"segments": []}
        while attempt < attempt_limit:
            attempt += 1
            try:
                parsed = generate_viral_for_chunk(chunk_text, chunk_start_seconds, max_segments_per_chunk, openai_api_key)
                break
            except InvalidRequestError as ire:
                # shrink the chunk_text and retry
                chars_limit = int(chars_limit * 0.6)
                if chars_limit < 2000:
                    parsed = {"segments": []}
                    break
                parts = chunk_text.splitlines()
                total = 0
                new_parts = []
                for line in reversed(parts):
                    total += len(line) + 1
                    if total > chars_limit:
                        break
                    new_parts.append(line)
                chunk_text = "\n".join(reversed(new_parts))
                time.sleep(0.25)
            except Exception as e:
                parsed = {"segments": []}
                break

        segs = parsed.get("segments", [])
        for seg in segs:
            s_rel = seg.get("start_time")
            e_rel = seg.get("end_time")
            if not s_rel or not e_rel:
                continue
            try:
                s_abs = chunk_start_seconds + hms_to_seconds(s_rel)
                e_abs = chunk_start_seconds + hms_to_seconds(e_rel)
            except Exception:
                continue
            duration = e_abs - s_abs
            if duration < 45:
                e_abs = s_abs + 45
            if duration > 110:
                e_abs = s_abs + 110
            all_segments.append({
                "start_time": sec_to_hms(s_abs),
                "end_time": sec_to_hms(e_abs),
                "duration": int(e_abs - s_abs)
            })

    # merge overlapping
    if not all_segments:
        return {"segments": []}
    def seg_key(seg):
        hh,mm,ss = map(int, seg["start_time"].split(":"))
        return hh*3600 + mm*60 + ss
    all_segments_sorted = sorted(all_segments, key=seg_key)
    merged = []
    for seg in all_segments_sorted:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        prev_start = hms_to_seconds(prev["start_time"])
        prev_end = hms_to_seconds(prev["end_time"])
        cur_start = hms_to_seconds(seg["start_time"])
        cur_end = hms_to_seconds(seg["end_time"])
        if cur_start <= prev_end + 5:
            new_end = max(prev_end, cur_end)
            prev["end_time"] = sec_to_hms(new_end)
            prev["duration"] = int(new_end - prev_start)
        else:
            merged.append(seg)

    return {"segments": merged}

# ---------- create shorts ----------
def resize_video_to_720x1280(clip):
    width, height = clip.size
    target_w, target_h = 720, 1280
    scale = min(target_w / max(1, width), target_h / max(1, height))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    resized = clip.resize(newsize=(new_w, new_h))
    try:
        padded = resized.on_color(size=(target_w, target_h), color=(0,0,0), pos=('center','center'))
        return padded
    except Exception:
        return resized

def trim_video_with_moviepy(input_file, output_file, start_sec, end_sec):
    with VideoFileClip(input_file) as video:
        start_sec = max(0, float(start_sec))
        end_sec = min(video.duration, float(end_sec))
        if end_sec <= start_sec:
            raise ValueError("Invalid trim times")
        sub = video.subclip(start_sec, end_sec)
        resized = resize_video_to_720x1280(sub)
        # write file quietly
        resized.write_videofile(output_file, codec='libx264', audio_codec='aac', verbose=False, logger=None, ffmpeg_params=['-pix_fmt', 'yuv420p'])

def create_shorts(video_path, viral_segments, request_code):
    files = []
    output_dir = os.path.join('static', 'shorts')
    os.makedirs(output_dir, exist_ok=True)
    total = len(viral_segments) or 1
    update_status(request_code, status="creating_shorts", message="Creating shorts...", progress=60, total_shorts=total)
    for i, seg in enumerate(viral_segments):
        check_cancellation(request_code)
        start_hms = seg.get("start_time") or seg.get("start")
        end_hms = seg.get("end_time") or seg.get("end")
        if not start_hms or not end_hms:
            continue
        start_sec = hms_to_seconds(start_hms)
        end_sec = hms_to_seconds(end_hms)
        if end_sec - start_sec < 1:
             # Ensure at least some duration
             end_sec = start_sec + 5

        outname = os.path.join(output_dir, f"segment_{uuid.uuid4().hex}.mp4")
        try:
            check_cancellation(request_code)
            update_status(request_code, message=f"Trimming short {i+1} of {total}...", current_short=i+1, progress=60 + int((i/total)*30))
            trim_video_with_moviepy(video_path, outname, start_sec, end_sec)
            
            check_cancellation(request_code)
            # Store absolute path for zipping, but we need relative for frontend display
            files.append(os.path.abspath(outname))
            
            # For the frontend status, we'll send the filename which is now in static/shorts
            # The frontend should know to look in /static/shorts/ or we send the full relative path
            # Let's send basename, frontend validation needs to know path.
            # Actually, update_status sends `generated_files`. Let's allow it to send full relative path?
            # Existing logic: generated_names = [os.path.basename(x) for x in files]
            # We'll stick to basename for list, but maybe we change frontend to use that.
            
            generated_names = [os.path.basename(x) for x in files]
            update_status(request_code, generated_files=generated_names, message=f"Created {len(files)} of {total} shorts", progress=60 + int(((i+1)/total)*30))
        except JobCancelledException:
            raise
        except Exception as e:
            update_status(request_code, message=f"Failed to create short {i+1}: {e}")
            continue
    return files

# ---------- caching helpers ----------
def check_cache(video_identifier):
    try:
        with open("video_analysis_cache.csv", "r", newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) >= 2 and row[0] == video_identifier:
                    return row[1]
    except FileNotFoundError:
        return None
    return None

def update_cache(video_identifier, analysis_result):
    header_needed = not os.path.exists("video_analysis_cache.csv")
    with open("video_analysis_cache.csv", "a", newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        if header_needed:
            writer.writerow(["video_identifier", "analysis_result"])
        writer.writerow([video_identifier, analysis_result])

def format_segments(segments):
    formatted = []
    for seg in segments:
        s = seg["start_time"]
        e = seg["end_time"]
        start_seconds = hms_to_seconds(s)
        end_seconds = hms_to_seconds(e)
        formatted.append({"start_time": s, "end_time": e, "duration": end_seconds - start_seconds})
    return {"segments": formatted}

# ---------- main entrypoint ----------
def ViralEm(video_id=None, video_path=None, cviral_response=None, key=None, request_code='videoOne', custom_ranges=None, num_clips=5):
    if not key and not custom_ranges:
        raise ValueError("OpenAI API key is required.")
    
    if key:
        openai.api_key = key

    # normalize id if full url provided
    if video_id and "youtube.com/watch" in video_id:
        if "v=" in video_id:
            try:
                video_id = video_id.split("v=")[1].split("&")[0]
            except:
                pass
    
    # check cache (only for auto mode)
    cached = None
    if video_id and not custom_ranges:
        cached = check_cache(video_id)

    try:
        update_status(request_code, status="started", message="Starting processing...", progress=5)
        
        # If we have a cache and NOT in custom mode, use it
        if cached is not None and custom_ranges is None:
            update_status(request_code, message="Using cached segments. Downloading video...", progress=20)
            if video_id:
                video_path = download_youtube_video(video_id, request_code=request_code)
            try:
                viral_segments = json.loads(cached).get("segments", [])
            except Exception:
                viral_segments = json.loads(cached)
            
            generated_files = create_shorts(video_path, viral_segments, request_code)
            update_status(request_code, message="Zipping...", progress=90)
            zip_rel = zip_the_files(generated_files, f"{request_code}.zip")
            update_status(request_code, status="completed", message="Completed (cached).", progress=100, zip=zip_rel, generated_files=[os.path.basename(x) for x in generated_files])
            return

        # Normal Flow (Download -> [Analyze] -> Cut)
        if video_id:
            video_path = download_youtube_video(video_id, request_code=request_code)
        if not video_path:
            raise ValueError("No video path provided or download failed.")

        if custom_ranges:
            # Bypass extraction and analysis
            check_cancellation(request_code)
            update_status(request_code, message=f"Custom clip mode. Processing {len(custom_ranges)} clips...", status="processing_custom", progress=40)
            
            # Create segments from ranges
            viral_segments = []
            for start, end in custom_ranges:
                viral_segments.append({
                    "start_time": start,
                    "end_time": end
                })
        else:
            # Auto Analysis
            # check before heavy subtitle extraction
            check_cancellation(request_code)
            
            # get duration
            try:
                video_duration = get_video_duration(video_path)
            except Exception:
                video_duration = 0

            update_status(request_code, message="Extracting subtitles...", status="extracting_subtitles", progress=25)
            transcript = extract_subtitles(video_path)
            
            # --- Save Subtitle File for Frontend ---
            try:
                # Based on extract_subtitles logic, the SRT is in tmp/basename.srt
                base_name = os.path.basename(video_path).split('.')[0]
                tmp_srt = os.path.join("tmp", f"{base_name}.srt")
                if os.path.exists(tmp_srt):
                    sub_dest = os.path.join("static", "subtitles", f"{request_code}.srt")
                    os.makedirs(os.path.dirname(sub_dest), exist_ok=True)
                    shutil.copy(tmp_srt, sub_dest)
                    # Update status with subtitle path immediately so user can see it
                    update_status(request_code, subtitle_file=f"/static/subtitles/{request_code}.srt")
            except Exception as e:
                print(f"Could not save subtitle file: {e}")
            # ---------------------------------------
            
            check_cancellation(request_code)

            update_status(request_code, message="Analyzing for viral segments (chunked)...", status="analyzing", progress=35)
            viral_result = analyze_transcript_in_chunks(transcript, video_duration, key, request_code=request_code)
            viral_segments = viral_result.get("segments", [])

            # store cache only if we got segments and it was a full auto run
            if viral_segments:
                update_cache(video_id or str(uuid.uuid4()), json.dumps(viral_result))

        # Create shorts
        # Create shorts
        check_cancellation(request_code)
        
        # Limit number of clips if not in custom mode
        segments_to_process = viral_segments
        if not custom_ranges and num_clips > 0:
            segments_to_process = viral_segments[:num_clips]
            
        generated_files = create_shorts(video_path, segments_to_process, request_code)
        update_status(request_code, message="All shorts created. Zipping...", status="zipping", progress=90)
        zip_rel = zip_the_files(generated_files, f"{request_code}.zip")
        update_status(request_code, status="completed", message="Completed successfully.", progress=100, zip=zip_rel, generated_files=[os.path.basename(x) for x in generated_files])

    except JobCancelledException as e:
        status_path = os.path.join("processing_status", f"{request_code}.json")
        try:
            with open(status_path, "r", encoding="utf-8") as fh: state = json.load(fh)
        except: state = {}
        state.update({"status": "cancelled", "message": "Job cancelled by user.", "progress": 0})
        with open(status_path, "w", encoding="utf-8") as fh: json.dump(state, fh)
        print(f"Job {request_code} cancelled.")

    except Exception as e:
        update_status(request_code, status="failed", message=str(e), error=str(e), progress=0)
        raise

# helper to get duration
def get_video_duration(video_path):
    with VideoFileClip(video_path) as v:
        return v.duration
