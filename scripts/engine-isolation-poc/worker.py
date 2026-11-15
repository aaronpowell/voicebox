"""One-shot CPU-only Chatterbox Turbo worker for the isolation experiment."""

import json
import sys
from pathlib import Path


def main() -> None:
    request = json.load(sys.stdin)
    if set(request) != {"text", "model_dir", "audio_path", "result_path"}:
        raise ValueError("Unexpected worker request fields")
    if not isinstance(request["text"], str) or not 1 <= len(request["text"]) <= 100:
        raise ValueError("Worker text must contain 1 to 100 characters")

    import torch
    import soundfile as sf
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = ChatterboxTurboTTS.from_local(request["model_dir"], "cpu")
    with torch.inference_mode():
        audio = model.generate(request["text"]).detach().cpu().squeeze().numpy()
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Worker returned empty or non-mono audio")
    sf.write(request["audio_path"], audio, model.sr)
    Path(request["result_path"]).write_text(
        json.dumps({"sample_rate": model.sr, "samples": int(audio.size)}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
