# app.py
import os
# set a safe runtime dir and dummy audio driver to reduce ALSA noise on headless systems
if os.name != 'nt':
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/tmp/runtime-{os.getuid()}")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flask import Flask, render_template, request, jsonify
from Viralize.main import ViralEm, ensure_dirs
import threading, uuid, json
import pytube
from yt_dlp import YoutubeDL

app = Flask(__name__)
app.secret_key = str(uuid.uuid4().hex)

ensure_dirs()  # make directories used by the worker & status files

def videoShort(videoLink, request_code, api_key, custom_ranges=None, num_clips=5):
    try:
        ViralEm(video_id=videoLink, key=api_key, request_code=request_code, custom_ranges=custom_ranges, num_clips=num_clips)
    except Exception as e:
        # write error to status
        try:
            status_path = os.path.join("processing_status", f"{request_code}.json")
            with open(status_path, "w") as fh:
                json.dump({"status": "failed", "error": str(e), "message": str(e)}, fh)
        except Exception as write_e:
            print(f"Failed to write error status for {request_code}: {write_e}")
        print("Background processing error:", e)

def probe_video_title(url):
    """Try pytube first (fast) and fallback to yt-dlp for robust metadata."""
    try:
        video = pytube.YouTube(url)
        return video.title
    except Exception as e_pytube:
        try:
            ydl_opts = {'skip_download': True, 'quiet': True, 'no_warnings': True}
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('title') or info.get('webpage_title') or info.get('id')
        except Exception as e_yd:
            raise RuntimeError(f"pytube error: {e_pytube}; yt-dlp error: {e_yd}")

@app.get('/')
def dashboard():
    job_id = request.args.get('job')
    downloads_dir = './static/downloads'
    os.makedirs(downloads_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(downloads_dir) if f.endswith('.zip')], reverse=True)
    zip_files = ["/static/downloads/" + f for f in files]
    
    # If job_id exists, we pass it to the template to resume polling
    return render_template('pages/dash.html', files=zip_files, request_code=job_id)

@app.post('/')
def dashboard_post():
    videoLink = request.form.get('vidLink')
    api_key = request.form.get('apiKey')
    mode = request.form.get('mode', 'auto')
    
    # Get lists for multiple clips
    start_times = request.form.getlist('startTime')
    end_times = request.form.getlist('endTime')

    custom_ranges = []
    if mode == 'custom':
        # zip and filter valid pairs
        for s, e in zip(start_times, end_times):
            if s and e:
                # Ensure start is less than end
                if s > e:
                    s, e = e, s
                # Avoid zero duration or duplicates if needed, but swap handles the order
                if s != e:
                     custom_ranges.append((s, e))
        
        if not custom_ranges:
            downloads_dir = './static/downloads'
            os.makedirs(downloads_dir, exist_ok=True)
            files = sorted([f for f in os.listdir(downloads_dir) if f.endswith('.zip')], reverse=True)
            zip_files = ["/static/downloads/" + f for f in files]
            return render_template('pages/dash.html', error="At least one start/end time pair is required for Custom Clip.", files=zip_files, request_code=None)
        
        if not api_key:
            api_key = "dummy_key_for_custom_clip"
    else:
        if not api_key:
            downloads_dir = './static/downloads'
            os.makedirs(downloads_dir, exist_ok=True)
            files = sorted([f for f in os.listdir(downloads_dir) if f.endswith('.zip')], reverse=True)
            zip_files = ["/static/downloads/" + f for f in files]
            return render_template('pages/dash.html', error="OpenAI API key is required.", files=zip_files, request_code=None)

    if not videoLink:
        downloads_dir = './static/downloads'
        os.makedirs(downloads_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(downloads_dir) if f.endswith('.zip')], reverse=True)
        zip_files = ["/static/downloads/" + f for f in files]
        return render_template('pages/dash.html', error="No video URL provided.", files=zip_files, request_code=None)

    # probe metadata (pytube -> yt-dlp fallback)
    try:
        videoTitle = probe_video_title(videoLink)
    except Exception as e:
        downloads_dir = './static/downloads'
        os.makedirs(downloads_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(downloads_dir) if f.endswith('.zip')], reverse=True)
        zip_files = ["/static/downloads/" + f for f in files]
        return render_template('pages/dash.html', error=f"Failed to fetch video metadata. Error: {str(e)}", files=zip_files, request_code=None)

    request_code = str(uuid.uuid4())

    # create initial status file
    status_dir = "processing_status"
    os.makedirs(status_dir, exist_ok=True)
    status_path = os.path.join(status_dir, f"{request_code}.json")
    initial_status = {
        "status": "queued",
        "message": "Task queued.",
        "progress": 0,
        "video_title": videoTitle,
        "download_percent": 0,
        "generated_files": [],
        "zip": None,
        "error": None
    }
    with open(status_path, "w", encoding="utf-8") as fh:
        json.dump(initial_status, fh)

    # start background processing
    # Pass 'custom_ranges' (list of tuples) instead of single 'custom_range'
    processing_thread = threading.Thread(target=videoShort, args=(videoLink, request_code, api_key, custom_ranges), daemon=True)
    processing_thread.start()

    # PRG: Redirect to the GET route with the job ID
    from flask import redirect, url_for
    return redirect(url_for('dashboard', job=request_code))

@app.post('/cancel/<request_code>')
def cancel_job(request_code):
    try:
        # Create a cancel marker file
        cancel_path = os.path.join("processing_status", f"{request_code}.cancel")
        with open(cancel_path, 'w') as f:
            f.write('cancelled')
        return jsonify({"status": "ok", "message": "Cancellation signal sent."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.get('/status/<request_code>')
def status(request_code):
    status_path = os.path.join("processing_status", f"{request_code}.json")
    if not os.path.exists(status_path):
        # return JSON 200 with not_found to avoid some proxies returning 503
        return jsonify({"status": "not_found", "message": "status file not found"}), 200
    try:
        with open(status_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


# ---------- cleanup helper ----------
import time

def cleanup_old_files():
    """
    Periodically delete files in 'videos', 'static/downloads', 'static/shorts'
    that are older than 1 hour.
    """
    dirs_to_clean = [
        'videos',
        os.path.join('static', 'downloads'),
        os.path.join('static', 'shorts'), # also clean generated shorts
        'processing_status' # maybe clean old status files too? yes
    ]
    
    print("Cleanup thread started...")
    
    # Run once immediately, then loop
    while True:
        try:
            now = time.time()
            cutoff = now - 3600  # 1 hour ago
            
            for d in dirs_to_clean:
                if not os.path.exists(d):
                    continue
                
                for filename in os.listdir(d):
                    filepath = os.path.join(d, filename)
                    if os.path.isfile(filepath):
                        # check modification time
                        try:
                            mtime = os.path.getmtime(filepath)
                            if mtime < cutoff:
                                os.remove(filepath)
                                print(f"[Cleanup] Deleted old file: {filepath}")
                        except Exception as e:
                            print(f"[Cleanup] Error deleting {filepath}: {e}")
        except Exception as e:
            print(f"[Cleanup] Error in loop: {e}")
        
        time.sleep(600) # check every 10 mins

if __name__ == "__main__":
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()

    # For local dev you can keep debug=True, but for heavy jobs prefer debug=False
    app.run(host='0.0.0.0', debug=False)
