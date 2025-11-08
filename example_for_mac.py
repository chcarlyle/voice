import torch
import soundfile as sf
import time
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = "Listen, these Transformers — absolute game changers, okay? They don’t just read words straight down the line like robots; Nah! They look at how every word connects, what actually matters; Boom! That’s how they get freakin’ smart; Electric stuff, man!"

# Cache the target voice once so later generations skip re-embedding
AUDIO_PROMPT_PATH = "PatReference00.mp3"
model.prepare_conditionals(AUDIO_PROMPT_PATH, exaggeration=0.5)

start = time.perf_counter()
wav = model.generate(
    text,
    exaggeration=0.6,
    cfg_weight=0.01
    )
elapsed = time.perf_counter() - start
print(f"Generation time after voice cached: {elapsed:.2f}s")

sf.write("Patvoiceclonev3.wav", wav.squeeze(0).cpu().numpy(), model.sr)
