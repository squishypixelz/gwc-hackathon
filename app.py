from flask import Flask, request, jsonify, render_template, session
import librosa
import numpy as np
import os
import json
from datetime import datetime
import subprocess
import tempfile
import threading

app = Flask(__name__)
app.secret_key = "yawns_be_gone_secret_420"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Keep track of transcriptions since Flask sessions don't work well with threads
transcription_cache = {}

def get_this_thicc_boi_silent(audio_path, threshold_db=-50, min_duration=0.3):
    """
    Find where the audio goes quiet - detects all the boring pauses
    
    Improved algorithm:
    - Uses RMS (Root Mean Square) energy for better silence detection
    - Works on smaller windows for better time resolution
    - Uses absolute threshold instead of relative to max
    - More aggressive silence detection for lectures
    """
    # Load the audio file
    y, sr_val = librosa.load(audio_path, sr=None)
    
    # Use smaller frame length for better time resolution (10ms windows)
    frame_length = int(0.01 * sr_val)  # 10ms
    hop_length = int(0.005 * sr_val)   # 5ms hop (50% overlap)
    
    # Calculate RMS energy for each frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB (using absolute reference, not relative to max)
    # This is more reliable for detecting actual silence
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    
    # Also calculate zero crossing rate to help detect speech vs silence
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize zero crossing rate
    zcr_normalized = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-10)
    
    # Mark frames as silent if BOTH conditions are met:
    # 1. Energy is below threshold
    # 2. Zero crossing rate is low (speech has higher ZCR)
    silent_frames = (rms_db < threshold_db) & (zcr_normalized < 0.3)
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr_val, hop_length=hop_length)
    
    # Find continuous silent regions
    pauses = []
    in_pause = False
    pause_start = 0
    
    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_pause:
            # Starting a quiet section
            in_pause = True
            pause_start = times[i]
        elif not is_silent and in_pause:
            # Quiet section ended, check if it was long enough
            pause_duration = times[i] - pause_start
            if pause_duration >= min_duration:
                pauses.append({
                    "start": float(pause_start),
                    "end": float(times[i]),
                    "duration": float(pause_duration)
                })
            in_pause = False
    
    # Don't forget to close the last pause if the audio ends in silence
    if in_pause and len(times) > 0:
        pause_duration = times[-1] - pause_start
        if pause_duration >= min_duration:
            pauses.append({
                "start": float(pause_start),
                "end": float(times[-1]),
                "duration": float(pause_duration)
            })
    
    return pauses


def big_bruh_moment_audio_removal(audio_path, pauses, output_path):
    """Takes out all the quiet parts and stitches the good parts back together"""
    import soundfile as sf
    
    # Load the audio
    y, sr_val = librosa.load(audio_path, sr=None)
    
    # Figure out which parts to keep
    segments_to_keep = []
    last_end = 0
    
    # Go through each pause and grab the audio before it
    for pause in pauses:
        if last_end < pause["start"]:
            # Keep everything from last_end to the start of this pause
            segments_to_keep.append((int(last_end * sr_val), int(pause["start"] * sr_val)))
        last_end = pause["end"]
    
    # Don't forget the audio after the last pause
    if last_end < len(y) / sr_val:
        segments_to_keep.append((int(last_end * sr_val), len(y)))
    
    # Combine all the pieces back together
    if segments_to_keep:
        y_trimmed = np.concatenate([y[start:end] for start, end in segments_to_keep])
    else:
        y_trimmed = y
    
    # Save the new audio file
    sf.write(output_path, y_trimmed, sr_val)
    return output_path


def no_cap_transcribe_this_banger(audio_path, max_duration=30):
    """Try to convert the audio to text using Google's speech recognition API"""
    try:
        from pydub import AudioSegment
        import speech_recognition as sr
        
        # Make sure it's a WAV file (Google API likes WAV)
        if not audio_path.lower().endswith('.wav'):
            try:
                sound = AudioSegment.from_file(audio_path)
                wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
                sound.export(wav_path, format="wav")
                audio_path = wav_path
            except Exception as convert_error:
                # If it doesn't work, just try anyway
                print(f"Warning: couldn't convert to WAV: {convert_error}")
        
        # Limit to first 30 seconds so it's not too slow
        y, sr_val = librosa.load(audio_path, sr=None)
        max_samples = int(max_duration * sr_val)
        if len(y) > max_samples:
            y = y[:max_samples]
            # Save the shortened version for transcription
            trimmed_path = audio_path.rsplit('.', 1)[0] + '_transcribe.wav'
            import soundfile as sf
            sf.write(trimmed_path, y, sr_val)
            audio_path = trimmed_path
        
        # Send to Google and get the text back
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        # If it fails, just return an error message
        return f"Transcription didn't work: {str(e)}"


def spit_out_the_stats_no_lie(original_duration, trimmed_duration, pauses):
    """Calculate how much time we saved and stuff"""
    time_removed = original_duration - trimmed_duration
    percentage_removed = (time_removed / original_duration * 100) if original_duration > 0 else 0
    num_pauses = len(pauses)
    
    return {
        "original_duration": float(original_duration),
        "trimmed_duration": float(trimmed_duration),
        "time_removed": float(time_removed),
        "percentage_removed": float(percentage_removed),
        "num_pauses": num_pauses,
        "pauses": pauses
    }


@app.route("/")
def yo_homie_gimme_home():
    """Show the landing page"""
    return render_template("index.html")


@app.route("/upload")
def bruh_where_the_upload_at():
    """Show the upload page"""
    return render_template("audio.html")


@app.route("/upload", methods=["POST"])
def yeet_this_audio_fam():
    """Handle file uploads - process the audio and skip the silence"""
    print("Upload endpoint called")
    print("Request files:", request.files)
    print("Request form:", request.form)
    
    if "file" not in request.files:
        print("No file in request.files")
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    print(f"File object: {file}")
    print(f"Filename: {file.filename}")
    
    if file.filename == "":
        print("Empty filename")
        return jsonify({"error": "You need to actually pick a file"}), 400
    
    try:
        # Save the uploaded file with a timestamp so we don't overwrite stuff
        filename = f"{datetime.now().timestamp()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")
        
        # Check if it's a video and extract audio if needed
        file_ext = filename.rsplit('.', 1)[-1].lower()
        audio_filepath = filepath
        
        # Handle video files
        if file_ext in ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv']:
            # It's a video file, so pull out the audio using ffmpeg directly
            try:
                # Try using ffmpeg directly first
                audio_filename = f"audio_{filename.rsplit('.', 1)[0]}.wav"
                audio_filepath = os.path.join(UPLOAD_FOLDER, audio_filename)
                
                # Use ffmpeg command line tool
                result = subprocess.run([
                    'ffmpeg', '-i', filepath,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # PCM 16-bit
                    '-ar', '44100',  # Sample rate
                    '-ac', '2',  # Stereo
                    '-y',  # Overwrite output file
                    audio_filepath
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    # Try using pydub as fallback
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(filepath)
                        audio.export(audio_filepath, format="wav")
                    except Exception as pydub_error:
                        return jsonify({
                            "error": f"Couldn't extract audio. Please install ffmpeg: 'sudo apt-get install ffmpeg' (Linux) or 'brew install ffmpeg' (Mac). Error: {str(pydub_error)}"
                        }), 400
                        
            except FileNotFoundError:
                # ffmpeg not found, try pydub
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(filepath)
                    audio.export(audio_filepath, format="wav")
                except Exception as pydub_error:
                    return jsonify({
                        "error": f"FFmpeg not found. Please install it: 'sudo apt-get install ffmpeg' (Linux) or 'brew install ffmpeg' (Mac). Or upload an audio file directly (.mp3, .wav, etc.)"
                    }), 400
        
        # Handle non-WAV audio files (FLAC, M4A, OGG, etc.)
        elif file_ext in ['flac', 'm4a', 'ogg', 'mp3', 'aac']:
            try:
                # Convert to WAV for processing
                audio_filename = f"audio_{filename.rsplit('.', 1)[0]}.wav"
                audio_filepath = os.path.join(UPLOAD_FOLDER, audio_filename)
                
                # Try ffmpeg first
                try:
                    result = subprocess.run([
                        'ffmpeg', '-i', filepath,
                        '-acodec', 'pcm_s16le',
                        '-ar', '44100',
                        '-ac', '2',
                        '-y',
                        audio_filepath
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise Exception(f"FFmpeg failed: {result.stderr}")
                        
                except (FileNotFoundError, Exception) as ffmpeg_error:
                    print(f"FFmpeg conversion failed, trying pydub: {ffmpeg_error}")
                    # Fallback to pydub
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(filepath)
                    audio.export(audio_filepath, format="wav")
                    
            except Exception as convert_error:
                return jsonify({
                    "error": f"Couldn't convert audio file. Try installing ffmpeg or upload a .wav file directly. Error: {str(convert_error)}"
                }), 400
        
        # Load the audio and figure out how long it is
        y, sr_val = librosa.load(audio_filepath, sr=None)
        original_duration = len(y) / sr_val
        
        # Find all the quiet parts - adjusted for lecture recordings
        # Lectures often have long pauses, so we use:
        # - Lower threshold (-55 to -60 dB) to catch more silence
        # - Shorter minimum (0.2s) to catch brief pauses too
        pauses = get_this_thicc_boi_silent(audio_filepath, threshold_db=-55, min_duration=0.2)
        
        print(f"Found {len(pauses)} pauses in the audio")
        if len(pauses) > 0:
            total_pause_time = sum(p['duration'] for p in pauses)
            print(f"Total silence time: {total_pause_time:.2f} seconds")
        
        
        # Remove the quiet parts
        output_filename = f"trimmed_{filename.rsplit('.', 1)[0]}.wav"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        big_bruh_moment_audio_removal(audio_filepath, pauses, output_path)
        
        # Load the new file and see how long it is now
        y_trim, _ = librosa.load(output_path, sr=None)
        trimmed_duration = len(y_trim) / sr_val
        
        # Calculate all the stats
        stats = spit_out_the_stats_no_lie(original_duration, trimmed_duration, pauses)
        
        # Save everything to the session so we can use it later
        session['processed_file'] = output_filename
        session['original_file'] = filename
        session['stats'] = stats
        session_id = f"trans_{output_filename}"
        session['transcription_id'] = session_id
        transcription_cache[session_id] = "Getting the text version..."
        
        # Do the transcription in the background so we don't make the user wait forever
        def do_transcription():
            try:
                transcription = no_cap_transcribe_this_banger(audio_filepath)
                transcription_cache[session_id] = transcription
            except Exception as transcribe_error:
                print(f"Transcription failed: {str(transcribe_error)}")
                transcription_cache[session_id] = "Couldn't transcribe, but your audio was processed!"
        
        # Start it in a separate thread
        transcribe_thread = threading.Thread(target=do_transcription, daemon=True)
        transcribe_thread.start()
        
        # Send back the stats right away (transcription will happen in the background)
        transcription_text = transcription_cache.get(session_id, 'Getting the text version...')
        if len(transcription_text) > 500:
            transcription_text = transcription_text[:500] + "..."
        
        return jsonify({
            "success": True,
            "stats": stats,
            "transcription": transcription_text
        })
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


@app.route("/result")
def gimme_that_yeet_result():
    """Show the results page with stats"""
    return render_template("result.html")


@app.route("/download")
def straight_up_lemme_download_this():
    """Let the user download their processed audio"""
    from flask import send_file
    
    if 'processed_file' not in session:
        return jsonify({"error": "No file to download"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, session['processed_file'])
    if not os.path.exists(filepath):
        return jsonify({"error": "File doesn't exist anymore"}), 404
    
    return send_file(filepath, as_attachment=True, download_name="lecture_no_pauses.wav")


@app.route("/get-stats")
def yo_give_me_them_stats():
    """API endpoint to retrieve stats about the processed audio"""
    if 'stats' not in session:
        return jsonify({"error": "No stats available"}), 400
    
    return jsonify(session.get('stats', {}))


@app.route("/get-transcription")
def lemme_see_what_was_said():
    """API endpoint to retrieve the transcribed text"""
    if 'transcription_id' not in session:
        return jsonify({"error": "No transcription available"}), 400
    
    session_id = session.get('transcription_id')
    transcription = transcription_cache.get(session_id, "Still transcribing...")
    return jsonify({"transcription": transcription})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)