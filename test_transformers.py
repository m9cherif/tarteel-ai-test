from transformers import pipeline
import time

print("Loading tarteel-ai/whisper-base-ar-quran with transformers pipeline...")
start = time.time()
pipe = pipeline(
    "automatic-speech-recognition", 
    model="tarteel-ai/whisper-base-ar-quran",
    device=-1,
)
elapsed = time.time() - start
print(f"✅ Model loaded in {elapsed:.1f}s")
print(f"Pipeline ready: {type(pipe)}")