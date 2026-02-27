from model import aasist3

# Load the model from Hugging Face Hub
model = aasist3.from_pretrained("MTUCI/AASIST3")
model.eval()

import torch
import torchaudio

# Load and preprocess audio
audio, sr = torchaudio.load("123.wav")
# Ensure audio is 16kHz and mono
if sr != 16000:
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)

# Prepare input (model expects ~4 seconds of audio at 16kHz)
# Pad or truncate to 64600 samples
if audio.shape[1] < 64600:
    audio = torch.nn.functional.pad(audio, (0, 64600 - audio.shape[1]))
else:
    audio = audio[:, :64600]

# Run inference
with torch.no_grad():
    output = model(audio)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1)

    # prediction: 0 = bonafide, 1 = spoof
    print(f"Prediction: {'Bonafide' if prediction.item() == 0 else 'Spoof'}")
    print(f"Confidence: {probabilities.max().item():.3f}")
