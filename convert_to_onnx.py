import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
import os
from predict import write_mono_wav, process_text
import argparse

VOCODER_NAME = "hifigan_lj_ft_t2_v2"
TTS_MODEL = "lightspeech_new.pt"
INPUT_TEXT = "展览将对全体观众实行免费入场提供义务讲解"
SPEAKER_ID = 218


class ExportableModel(nn.Module):
    def __init__(self, base_model, vocoder):
        super().__init__()
        self.base_model = base_model
        self.vocoder = vocoder

    def forward(self, speakers, tokens, tones):
        mel = self.base_model(speakers, tokens, tones)[0].permute(0, 2, 1)
        return self.vocoder(mel)


def load_model(model_path, vocoder_name):
    try:
        if "lightspeech" in model_path:
            from lightspeech import Model
        else:
            from fastspeech2 import Model

        state_dict = torch.load(model_path, map_location="cpu")
        model = (
            Model(
                num_phones=state_dict["num_phones"],
                num_speakers=state_dict["num_speakers"],
                num_mel_bins=state_dict["num_mel_bins"],
                d_model=state_dict["d_model"],
            )
            .to("cpu")
            .eval()
        )

        model.load_state_dict(state_dict["state_dict"], strict=True)

        vocoder = torch.hub.load(
            "lars76/bigvgan-mirror",
            vocoder_name,
            trust_repo=True,
            pretrained=True,
            verbose=False,
        )

        return (
            ExportableModel(model, vocoder),
            state_dict["pinyin_dict"],
            state_dict["phone_dict"],
            model,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def prepare_inputs(sequence_length=8):
    return {
        "tokens": torch.randint(1, 50, (1, sequence_length), dtype=torch.long),
        "tones": torch.randint(1, 6, (1, sequence_length), dtype=torch.long),
        "speakers": torch.tensor([0], dtype=torch.long),
    }


def export_to_onnx(model, inputs, onnx_path, opset_version=17):
    try:
        torch.onnx.export(
            model,
            (inputs["speakers"], inputs["tokens"], inputs["tones"]),
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["speakers", "tokens", "tones"],
            output_names=["output"],
            dynamic_axes={
                "tokens": {0: "batch_size", 1: "sequence_length"},
                "tones": {0: "batch_size", 1: "sequence_length"},
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
        )
        print(f"Model exported to {onnx_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to export model to ONNX: {e}")


def main():
    parser = argparse.ArgumentParser(description="TTS model export and inference")
    parser.add_argument(
        "--tts_model", type=str, default=TTS_MODEL, help="Path to TTS model"
    )
    parser.add_argument(
        "--vocoder", type=str, default=VOCODER_NAME, help="Vocoder name"
    )
    parser.add_argument(
        "--text", type=str, default=INPUT_TEXT, help="Input text for inference"
    )
    parser.add_argument("--speaker_id", type=int, default=SPEAKER_ID, help="Speaker ID")
    parser.add_argument("--type", type=str, default="simplified", help="Type")
    parser.add_argument("--output", type=str, default="output.wav", help="Type")
    args = parser.parse_args()

    model, pinyin_to_ipa, ipa_to_token, base_model = load_model(
        args.tts_model, args.vocoder
    )

    # Export to ONNX
    inputs = prepare_inputs()
    export_name = f"{os.path.splitext(args.tts_model)[0]}_{args.vocoder}.onnx"
    export_to_onnx(model, inputs, export_name)

    # Inference
    token_ids, tone_ids, phonemes, ipa_text = process_text(
        args.text, args.type, pinyin_to_ipa, ipa_to_token
    )

    ort_session = ort.InferenceSession(export_name)
    results = ort_session.run(
        None,
        {
            "speakers": [args.speaker_id],
            "tokens": [[ipa_to_token["<sil>"]] + token_ids + [ipa_to_token["<sil>"]]],
            "tones": [[1] + tone_ids + [1]],
        },
    )

    write_mono_wav(
        args.output,
        model.vocoder.sampling_rate,
        results,
    )
    print("ONNX model ran successfully")


if __name__ == "__main__":
    main()
