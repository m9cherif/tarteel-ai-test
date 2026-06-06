#!/usr/bin/env python3
"""
Tarteel AI - Quran MP3 Transcription App
-----------------------------------------
Upload an MP3 file and get word-by-word Quran-style transcription.
"""

import os
import io
import time
import uuid
import json
import threading
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import whisper
import pydub
import pydub.playback
import logging

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "tarteel-ai/whisper-base-ar-quran"
SAMPLE_RATE = 16000
DEVICE = "cuda" if whisper.torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'}

print(f"🔊 Using device: {DEVICE}")
print(f"🤖 Loading model: {MODEL_NAME}")

# ============================================================
# Load Model
# ============================================================
print("⏳ Loading OpenAI Whisper (base - same as tarteel-ai)...")
model = whisper.load_model("base", device=DEVICE)
print(f"✅ Model loaded on {DEVICE}")

# ============================================================
# Flask Application
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tarteel-ai-secret-key-2024'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# In-memory task store
tasks = {}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# Audio Processing Functions
# ============================================================
def allowed_file(filename):
    """Check if file extension is allowed"""
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

def load_audio(filepath):
    """
    Load audio file and convert to numpy array at 16kHz mono.
    Uses pydub for format support (MP3, M4A, OGG, etc.)
    """
    print(f"🎵 Loading audio: {filepath}")
    
    # Load audio with pydub
    audio = pydub.AudioSegment.from_file(filepath)
    
    # Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Resample to 16kHz
    audio = audio.set_frame_rate(SAMPLE_RATE)
    
    # Convert to numpy array (float32, normalized to [-1, 1])
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Normalize based on sample width
    sample_width = audio.sample_width
    if sample_width == 2:  # 16-bit
        samples /= 32768.0
    elif sample_width == 4:  # 32-bit
        samples /= 2147483648.0
    elif sample_width == 1:  # 8-bit
        samples = (samples - 128) / 128.0
    
    duration_sec = len(samples) / SAMPLE_RATE
    print(f"✅ Audio loaded: {duration_sec:.1f}s, {len(samples)} samples at {SAMPLE_RATE}Hz")
    
    return samples, duration_sec

def process_audio_file(filepath, task_id):
    """
    Process audio file with Whisper and emit results via WebSocket.
    Runs in a separate thread.
    """
    global tasks
    
    try:
        # Update task status
        tasks[task_id]['status'] = 'processing'
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'processing',
            'message': '⏳ جارٍ تحميل ومعالجة الملف الصوتي...',
            'progress': 10
        })
        
        # Load audio
        audio_np, duration_sec = load_audio(filepath)
        
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'processing',
            'message': f'🎵 تم تحميل الملف ({duration_sec:.1f} ثانية). جارٍ الترجمة...',
            'progress': 30
        })
        
        # Calculate duration for progress reporting
        chunk_duration = 30.0  # Report progress every ~30s of audio
        
        # Transcribe with Whisper
        print(f"🔄 Transcribing audio ({duration_sec:.1f}s)...")
        
        result = model.transcribe(
            audio_np,
            language="ar",
            task="transcribe",
            temperature=0.0,
            fp16=False,
            verbose=False,
        )
        
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'processing',
            'message': '📝 جارٍ استخراج الكلمات...',
            'progress': 80
        })
        
        # Extract word-level data with timing
        words_with_timing = []
        all_words = []
        
        for seg in segments:
            seg_words = seg.get('text', '').strip().split()
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            seg_text = seg.get('text', '').strip()
            
            if not seg_words or not seg_text:
                continue
            
            # Distribute words evenly within the segment time
            word_duration = (seg_end - seg_start) / max(len(seg_words), 1)
            
            for i, word in enumerate(seg_words):
                word_start = seg_start + (i * word_duration)
                word_end = word_start + word_duration
                
                words_with_timing.append({
                    'word': word,
                    'start': round(word_start, 2),
                    'end': round(word_end, 2),
                })
                all_words.append(word)
        
        # Prepare result
        result_data = {
            'task_id': task_id,
            'status': 'completed',
            'text': text,
            'words': words_with_timing,
            'segments': [
                {
                    'text': s['text'],
                    'start': round(s['start'], 2),
                    'end': round(s['end'], 2),
                }
                for s in segments
            ],
            'duration': round(duration_sec, 1),
            'word_count': len(all_words),
        }
        
        # Store result
        tasks[task_id]['result'] = result_data
        tasks[task_id]['status'] = 'completed'
        
        print(f"✅ Transcription complete: {len(all_words)} words in {len(segments)} segments")
        
        # Emit completion via WebSocket
        socketio.emit('task_complete', result_data)
        
        # Clean up uploaded file after processing
        try:
            os.remove(filepath)
            print(f"🗑️ Deleted temporary file: {filepath}")
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        
        socketio.emit('task_update', {
            'task_id': task_id,
            'status': 'error',
            'message': f'❌ خطأ في المعالجة: {str(e)}',
            'progress': 0
        })
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass

# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'model': MODEL_NAME,
        'device': DEVICE,
        'status': 'ready',
        'ready': True,
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload an audio file for transcription"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'لم يتم رفع أي ملف'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'صيغة الملف غير مدعومة.-supported: mp3, wav, m4a, ogg'}), 400
    
    # Generate unique task ID and filename
    task_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename.lower())
    safe_filename = f"{task_id}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    
    # Save file
    file.save(filepath)
    print(f"📁 File saved: {filepath}")
    
    # Create task
    tasks[task_id] = {
        'id': task_id,
        'filename': file.filename,
        'status': 'uploaded',
        'created_at': time.time(),
        'result': None,
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_audio_file,
        args=(filepath, task_id),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'message': '✅ تم رفع الملف بنجاح. جارٍ المعالجة...',
        'status': 'uploaded'
    })

@app.route('/api/task/<task_id>')
def get_task(task_id):
    """Get task status and result"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'المهمة غير موجودة'}), 404
    
    response = {
        'task_id': task['id'],
        'filename': task['filename'],
        'status': task['status'],
        'created_at': task['created_at'],
    }
    
    if task['result']:
        response['result'] = task['result']
    
    if 'error' in task:
        response['error'] = task['error']
    
    return jsonify(response)

# ============================================================
# WebSocket Events
# ============================================================
@socketio.on('connect')
def handle_connect():
    print(f"🔗 Client connected")
    emit('status', {'message': '✅ متصل بالخادم', 'model': f'{MODEL_NAME}'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 Client disconnected")

# ============================================================
# Main Entry Point
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  🕌 Tarteel AI - تفريغ القرآن من ملفات MP3")
    print(f"  🤖 النموذج: {MODEL_NAME}")
    print(f"  💻 الجهاز: {DEVICE}")
    print(f"  🎵 معدل العينة: {SAMPLE_RATE} Hz")
    print(f"  📁 رفع الملفات: {UPLOAD_FOLDER}/")
    print("=" * 60)
    print("\n🌐 افتح http://localhost:5000 في المتصفح")
    print("🎯 ارفع ملف MP3 وسيتم تفريغه كلمات بكلمات\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)