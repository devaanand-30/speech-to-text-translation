import argparse
from src.pipeline import VoiceTranslatePipeline


def main():
    parser = argparse.ArgumentParser(description="Multimodal Telugu↔English voice translator")
    parser.add_argument("--mode", choices=["te2en", "te2en_twostep", "en2te"], required=True)
    parser.add_argument("--audio", required=True, help="Path to input audio (wav/mp3/m4a...)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    pipe = VoiceTranslatePipeline(use_gpu=not args.cpu)

    if args.mode == "te2en":
        out = pipe.telugu_speech_to_english_text(args.audio)
    elif args.mode == "te2en_twostep":
        out = pipe.telugu_speech_to_english_text_two_step(args.audio)
    else:
        out = pipe.english_speech_to_telugu_text(args.audio)

    print("\n=== OUTPUT TEXT ===")
    print(out)

if __name__ == "__main__":
    main()
