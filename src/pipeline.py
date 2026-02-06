import os
from .utils_audio import to_wav_16k_mono
from .asr import ASR
from .translate import Translator

class VoiceTranslatePipeline:
    def __init__(self, use_gpu=True):
        device = "cuda" if use_gpu else "cpu"
        compute = "float16" if use_gpu else "int8"
        self.asr = ASR(model_size="small", device="cpu", compute_type="int8")
        self.te_en = Translator("te-en")
        self.en_te = Translator("en-te")

    def telugu_speech_to_english_text(self, in_audio_path: str) -> str:
        """
        Option A (recommended): single-step using Whisper translate (accurate, simple).
        """
        tmp = os.path.join("data","tmp","te_in.wav")
        wav = to_wav_16k_mono(in_audio_path, tmp)
        return self.asr.translate_te_to_en(wav)

    def telugu_speech_to_english_text_two_step(self, in_audio_path: str) -> str:
        """
        Option B: Telugu ASR -> Telugu text -> NMT Te->En.
        """
        tmp = os.path.join("data","tmp","te_in.wav")
        wav = to_wav_16k_mono(in_audio_path, tmp)
        te_text = self.asr.transcribe_te(wav)
        return self.te_en(te_text)

    def english_speech_to_telugu_text(self, in_audio_path: str) -> str:
        """
        English ASR -> English text -> NMT En->Te.
        """
        tmp = os.path.join("data","tmp","en_in.wav")
        wav = to_wav_16k_mono(in_audio_path, tmp)
        en_text = self.asr.transcribe_en(wav)
        return self.en_te(en_text)
