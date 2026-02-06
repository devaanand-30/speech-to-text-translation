import os
import soundfile as sf
from pydub import AudioSegment

def to_wav_16k_mono(in_path: str, out_path: str) -> str:
    """
    Convert any audio (mp3/mp4/m4a/wav) to 16kHz mono WAV for ASR.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(out_path, format="wav")
    return out_path
