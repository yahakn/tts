import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import sounddevice as sd  # For audio playback


# Cihaz ayarı (GPU varsa kullanılır, aksi halde CPU kullanılır)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model ve tokenizer yükleme
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# prompt.txt dosyasından prompt'u okuma
with open("promp.txt", "r") as file:
    prompt = file.read().strip()  # Dosyayı okuyup, boşlukları temizler

# Seslendirme için açıklama
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

# Tokenizer ile input id'leri oluşturma
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Model ile ses üretimi
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

# Üretilen sesi kaydetme
audio_arr = generation.cpu().numpy().squeeze()
# Play and save the combined audio
sd.play(audio_arr, samplerate=model.config.sampling_rate)
sd.wait()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
print("Audio saved as 'parler_tts_out.wav'.")