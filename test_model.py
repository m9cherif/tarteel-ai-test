import whisper
import time

print("Loading OpenAI Whisper 'base' model...")
start = time.time()
model = whisper.load_model("base")
elapsed = time.time() - start
print(f"✅ Model loaded in {elapsed:.1f}s")
print(f"Model type: {type(model)}")