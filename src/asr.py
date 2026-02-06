from faster_whisper import WhisperModel
import os

class ASR:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        """
        Use smaller model to avoid download issues.
        device='cpu' is safer if GPU isn't configured.
        """
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            download_root=os.path.join("models", "whisper")
        )

    def transcribe_te(self, wav_path: str) -> str:
        """
        Telugu speech -> Telugu text
        """
        segments, info = self.model.transcribe(wav_path, language="te", task="transcribe")
        return "".join(seg.text for seg in segments).strip()

    def translate_te_to_en(self, wav_path: str) -> str:
        """
        Telugu speech -> English text (direct translation)
        """
        segments, info = self.model.transcribe(wav_path, language="te", task="translate")
        return "".join(seg.text for seg in segments).strip()

    def transcribe_en(self, wav_path: str) -> str:
        """
        English speech -> English text
        """
        segments, info = self.model.transcribe(wav_path, language="en", task="transcribe")
        return "".join(seg.text for seg in segments).strip()
