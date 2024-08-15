import argparse
import torch
import re
import os
import wave
import numpy as np

from typing import Dict, List, Tuple


def load_model(
    model_class: str, model_path: str, device: torch.device
) -> Tuple[torch.nn.Module, Dict, Dict]:
    """
    Load the TTS model and associated dictionaries.

    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device to load the model on.

    Returns:
        Tuple[torch.nn.Module, Dict, Dict]: Loaded model, pinyin_to_ipa dict, ipa_to_token dict and speaker dict.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"The specified model file '{model_path}' does not exist."
        )

    if model_class == "fastspeech2":
        from fastspeech2 import Model
    elif model_class == "lightspeech":
        from lightspeech import Model
    else:
        raise ValueError(f"The specified model '{model_class}' does not exist.")

    state_dict = torch.load(model_path, map_location=device)
    model = (
        Model(
            num_phones=state_dict["num_phones"],
            num_speakers=state_dict["num_speakers"],
            num_mel_bins=state_dict["num_mel_bins"],
            d_model=state_dict["d_model"]
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(state_dict["state_dict"], strict=True)

    return (
        model,
        state_dict["pinyin_dict"],
        state_dict["phone_dict"],
        state_dict["speaker_dict"],
    )


def load_vocoder(vocoder_name: str, device: torch.device) -> torch.nn.Module:
    """
    Load the vocoder model.

    Args:
        vocoder_name (str): Name of the vocoder model.
        device (torch.device): Device to load the vocoder on.

    Returns:
        torch.nn.Module: Loaded vocoder model.
    """
    return torch.hub.load(
        "lars76/bigvgan-mirror",
        vocoder_name,
        trust_repo=True,
        pretrained=True,
        verbose=False,
    ).to(device)


def process_text(
    text: str, text_type: str, pinyin_to_ipa: Dict, ipa_to_token: Dict
) -> Tuple[List[int], List[int]]:
    """
    Process the input text and convert it to token and tone IDs.

    Args:
        text (str): Input text.
        text_type (str): Type of input text ('ipa', 'pinyin', 'simplified', or 'traditional').
        pinyin_to_ipa (Dict): Dictionary for converting Pinyin to IPA.
        ipa_to_token (Dict): Dictionary for converting IPA to token IDs.

    Returns:
        Tuple[List[int], List[int], List[str]]: Token IDs, tone IDs and phonemes.
    """
    text = text.lower()
    print(f"Input text: {text}")

    if not text:
        raise ValueError("Empty input string")

    if text_type in ("simplified", "traditional"):
        text = convert_characters_to_pinyin(text_type, text)
        if not text:
            raise ValueError("Conversion to Pinyin resulted in empty string")
        print(f"Pinyin text: {text}")
        text_type = "pinyin"

    if text_type == "pinyin":
        text = convert_pinyin_to_ipa(pinyin_to_ipa, text)
        if not text:
            raise ValueError("Conversion to IPA resulted in empty string")
        print(f"IPA: {text}")

    token_ids, tone_ids, phonemes = convert_ipa_to_tokens(ipa_to_token, text)

    return token_ids, tone_ids, phonemes, text


def convert_characters_to_pinyin(character_type: str, text: str) -> str:
    """
    Converts Chinese characters to Pinyin.

    Args:
        character_type (str): Type of Chinese characters used in the text, either "simplified" or "traditional".
        text (str): The input text containing Chinese characters to be converted to Pinyin.

    Returns:
        str: The Pinyin representation of the input text with spaces separating syllables.

    Raises:
        ImportError: If the `g2pw` package is not installed.
    """
    try:
        from g2pw import G2PWConverter
    except ImportError:
        raise ImportError(
            "The 'g2pw' package is required to use Chinese characters. Please install it using 'pip install g2pw'."
        )

    # Initialize the converter
    conv = G2PWConverter(
        style="pinyin", enable_non_tradional_chinese=(character_type == "simplified")
    )

    # Remove Chinese punctuation
    punctuation_pattern = r"[，。！？《》【】（）“”‘’、；：]"
    cleaned_text = re.sub(punctuation_pattern, "", text)

    # Convert characters to Pinyin
    predicted_pinyin = conv(cleaned_text)[0]

    parsed = []
    unknown_tokens = 0
    for pinyin in predicted_pinyin:
        if pinyin is None:
            unknown_tokens += 1
            continue
        parsed.append(pinyin)

    if unknown_tokens > 0:
        print(f"{unknown_tokens} unknown tokens found!")

    return " ".join(parsed)


def convert_pinyin_to_ipa(pinyin_to_ipa: Dict[str, str], text: str) -> str:
    """
    Converts Pinyin text into IPA notation with tones.

    Args:
        pinyin_to_ipa (dict): A dictionary mapping Pinyin syllables (without tone digits)
                              to their corresponding IPA representations.
        text (str): A string containing Pinyin syllables, separated by spaces.
                    Each syllable may optionally end with a digit representing a tone.
                    If no tone digit is present, tone 5 is assumed by default.

    Returns:
        str: A string containing the concatenated IPA notation for the input text,
             with each IPA segment followed by its corresponding tone digit.
    """
    ipa_string = ""

    for syllable in text.split():
        syllable = syllable.strip()

        # Ensure each syllable ends with a digit for tone
        if not syllable[-1].isdigit():
            if syllable == "<sil>":
                syllable += "1"
            else:
                syllable += "5"  # Default to neutral tone (tone 5)

        ipa_key = syllable[:-1]
        tone = syllable[-1]

        ipas = pinyin_to_ipa.get(ipa_key)
        if ipas is None:
            print(f"Unknown token: {syllable}")
            continue

        # Append IPA and tone to ipa_string
        ipa_string += ipas.replace(" ", "") + tone + " "

    return ipa_string[:-1]


def convert_ipa_to_tokens(
    ipa_to_id: Dict[str, int], text: str
) -> Tuple[List[int], List[int]]:
    """
    Converts IPA text into tokens and tone IDs based on a mapping.

    Args:
        ipa_to_id (dict): A dictionary mapping IPA phonemes to token IDs.
        text (str): A string containing IPA phonemes, separated by spaces.
                    Each phoneme can optionally be followed by a digit representing a tone.

    Returns:
        tuple: A tuple containing two lists:
            - token_ids (list): List of token IDs corresponding to the IPA phonemes.
            - tone_ids (list): List of tone IDs corresponding to the phonemes, derived from trailing digits.
            - phonemes (list): List of phonemes.
    """
    token_ids = []
    tone_ids = []
    phonemes = []

    # Sort phonemes by length in descending order
    sorted_phonemes = sorted(ipa_to_id.keys(), key=len, reverse=True)

    # Process each token in the text
    for k in text.split():
        # Determine tone and base IPA key
        if k[-1].isdigit():
            tone_id = int(k[-1]) + 1
            ipa_key = k[:-1]
        else:
            print(f"No tone id: {k}")
            continue

        i = 0
        while i < len(ipa_key):
            matched = False
            for phoneme in sorted_phonemes:
                if ipa_key[i:].startswith(phoneme):
                    phonemes.append(phoneme)
                    token_ids.append(ipa_to_id[phoneme])
                    tone_ids.append(tone_id)
                    i += len(phoneme)
                    matched = True
                    break
            if not matched:
                print(f"Unmatched sequence in token '{k}': '{ipa_key[i:]}'")
                break

    return token_ids, tone_ids, phonemes


def display_speaker_info(speakers: List[Dict[str, str]]):
    if not speakers:
        print("No speaker information available.")
        return

    # Determine available fields
    available_fields = set().union(*speakers)
    field_order = ["speaker_id", "name", "age group", "gender", "accent"]
    display_fields = [field for field in field_order if field in available_fields]

    # Create format string and header
    format_string = " ".join("{:<15}" for _ in display_fields)
    header = format_string.format(*[field.capitalize() for field in display_fields])

    print("Available speakers:")
    print(header)
    print("-" * (15 * len(display_fields)))

    for speaker in speakers:
        row_data = [speaker.get(field, "N/A") for field in display_fields]
        row_data = [
            str(item).capitalize() if item != "N/A" else item for item in row_data
        ]
        print(format_string.format(*row_data))


def plot_mel_spectrogram(
    mel_spectrogram,
    pitch,
    dur,
    phonemes,
    ipa_text,
    vocoder,
    save_path=None,
):
    """
    Plots and optionally saves a mel spectrogram and pitch with segment labels.

    Parameters:
    mel_spectrogram (numpy.ndarray or torch.Tensor): The mel spectrogram to plot.
    pitch (numpy.ndarray or torch.Tensor): The pitch values.
    dur (numpy.ndarray or torch.Tensor): The durations of the segments.
    phonemes (list of str): The phoneme labels for each segment.
    ipa_text (str): Text for title.
    vocoder (torch.nn.Module): Vocoder model for audio synthesis.
    save_path (str, optional): Path to save the spectrogram image. If None, the image is not saved.
    """
    try:
        import librosa
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "The packages 'librosa' and 'matplotlib' are required to plot the spectogram. Please install it using 'pip install librosa matplotlib'."
        )

    # Convert inputs to numpy arrays if they are torch tensors
    if not isinstance(mel_spectrogram, np.ndarray):
        mel_spectrogram = mel_spectrogram.numpy().squeeze()
    if not isinstance(pitch, np.ndarray):
        pitch = pitch.numpy().squeeze()
    if not isinstance(dur, np.ndarray):
        dur = dur.numpy().squeeze()

    # Calculate the time axis
    num_frames = mel_spectrogram.shape[0]
    time_axis = np.arange(num_frames) * vocoder.hop_size / vocoder.sampling_rate

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    plt.suptitle(f"Text: {ipa_text}")

    # Plot the mel spectrogram
    librosa.display.specshow(
        mel_spectrogram.T,
        sr=vocoder.sampling_rate,
        hop_length=vocoder.hop_size,
        x_axis="time",
        y_axis="mel",
        n_fft=vocoder.n_fft,
        win_length=vocoder.win_size,
        fmin=vocoder.fmin,
        fmax=vocoder.fmax,
        ax=axs[0],
    )
    axs[0].set_title("Mel-frequency spectrogram")

    # Plot the pitch values
    axs[1].plot(time_axis, pitch, marker="o", label="Pitch")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Pitch")
    axs[1].set_title("Pitch and Phoneme Segments")

    # Highlight the segments and add phoneme labels
    current_time = 0
    for i, duration in enumerate(dur):
        segment_end = current_time + duration * vocoder.hop_size / vocoder.sampling_rate
        axs[1].axvspan(current_time, segment_end, color="gray", alpha=0.3)

        # Calculate the midpoint of the segment for the label
        midpoint = (current_time + segment_end) / 2
        if phonemes[i]:
            axs[1].text(
                midpoint,
                np.min(pitch) + 0.1,
                phonemes[i],
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.5),
            )

        current_time = segment_end

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot as a jpg file if a save path is provided
    if save_path:
        plt.savefig(save_path, format="jpg")

    # Show the plot
    plt.show()
    plt.close()


def write_mono_wav(filename, sample_rate, samples):
    """
    Writes a mono 16-bit integer WAV file.

    Parameters:
        filename (str): The output WAV file name.
        sample_rate (int): The sample rate (samples per second).
        samples (numpy array or list): The sound data as an array of float samples.
    """
    # Ensure the samples array is in the correct format
    if not isinstance(samples, (list, np.ndarray)):
        raise ValueError("samples must be a list or a numpy array")

    # Convert to numpy array if it's a list
    samples = np.asarray(samples)

    # Check sample rate
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    # Normalize and convert samples to 16-bit integers
    samples_int16 = np.int16(samples * 32767)

    # Write to WAV file
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes for 16-bit integer
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert input text to speech using a specified Chinese TTS model. "
            "The input text can be in IPA (International Phonetic Alphabet), Pinyin, or Chinese characters.\n\n"
            "Example usage:\n"
            '  python predict.py "开车慢慢前行" --type simplified --model fastspeech2.pt --model_class "FastSpeech2" --speaker 0\n'
            '  python predict.py "kai1 che1 man4 man4 qian2 xing2" --type pinyin --model lightspeech.pt --model_class "LightSpeech" --speaker 1\n'
            '  python predict.py "hau̯2 aɻ2 ju2" --type ipa --model fastspeech2.pt --speaker 5 --model_class "FastSpeech2" --device cuda:0\n'
            "  python predict.py --list-speakers --model fastspeech2.pt\n\n"
            "Required arguments:\n"
            "  text           The input text in IPA, Pinyin, or Chinese characters format. Leave blank if using --list-speakers.\n"
            "  --type         Specify the input format: 'ipa' for IPA, 'pinyin' for Pinyin, or 'simplified'/'traditional' for Chinese characters. Default is 'simplified'.\n"
            "  --model_class  Name of the model class (FastSpeech2 or LightSpeech).\n"
            "  --model        Path to the TTS model file (.pt or .pth).\n"
            "  --speaker      Speaker ID, an integer between 0 and 217. Default is 0.\n\n"
            "Optional arguments:\n"
            "  --silence / --no-silence\n"
            "                 Include or exclude a brief silence at the start and end of the synthesized audio. Default is to include silence.\n"
            "  --spectogram / --no-spectogram\n"
            "                 Plot the pitch and spectogram of the synthesized audio. Default is to not plot.\n"
            "  --device       Device to use for processing: 'cpu' or 'cuda:k' where k is the GPU number. Default is 'cpu'.\n"
            "  --output       Filename for the output audio file (WAV format). Default is 'output.wav'.\n"
            "  --vocoder      Vocoder model for audio synthesis. Default is 'hifigan_lj_ft_t2_v1'.\n"
            "  --list-speakers Display information about available speakers and exit."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)

    # Add --list-speakers to the mutually exclusive group
    group.add_argument(
        "--list-speakers",
        action="store_true",
        help="Display information about available speakers and exit.",
    )

    # Add text argument to the mutually exclusive group
    group.add_argument(
        "text",
        nargs="?",  # Makes it optional
        type=str,
        help=(
            "The text to convert to speech. It should match the specified --type format:\n"
            "  - 'ipa' for International Phonetic Alphabet,\n"
            "  - 'pinyin' for Romanized Chinese,\n"
            "  - 'simplified' or 'traditional' for Chinese characters.\n"
            "Leave this blank if using --list-speakers."
        ),
    )

    parser.add_argument(
        "--type",
        type=str,
        default="simplified",
        choices=["ipa", "pinyin", "simplified", "traditional"],
        help=(
            "Specify the format of the input text:\n"
            "  - 'ipa' for International Phonetic Alphabet,\n"
            "  - 'pinyin' for Romanized Chinese,\n"
            "  - 'simplified' for Simplified Chinese characters,\n"
            "  - 'traditional' for Traditional Chinese characters.\n"
            "This determines how the model interprets the input.\n"
            "Default is 'simplified'."
        ),
    )

    parser.add_argument(
        "--model_class",
        type=str,
        default="LightSpeech",
        help=("Name of the model class (FastSpeech2 or LightSpeech)."),
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to the TTS model file (.pt or .pth). This file contains the trained model weights necessary for speech synthesis."
        ),
    )

    parser.add_argument(
        "--speaker",
        type=int,
        default=218,
        help=(
            "Speaker ID to use for synthesis. An integer between 0 and 218, each representing a unique speaker.\n"
            "Default is 0. Use --list-speakers to see available options."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Device for processing the text-to-speech conversion:\n"
            "  - 'cpu' for Central Processing Unit,\n"
            "  - 'cuda:k' for GPU (where k is the GPU number).\n"
            "Default is 'cpu'."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help=(
            "Filename for the output audio file. The audio will be saved in WAV format.\n"
            "Default is 'output.wav'."
        ),
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        default="hifigan_lj_ft_t2_v1",
        help=(
            "Vocoder model for final audio synthesis. Vocoders convert the model output into a waveform.\n"
            "Default is 'hifigan_lj_ft_t2_v1'."
        ),
    )

    parser.add_argument(
        "--silence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include a brief silence at the end and start of the synthesized audio.\n"
            "Use '--silence' to enable or '--no-silence' to disable.\n"
            "Default is to include silence."
        ),
    )

    parser.add_argument(
        "--spectogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Plot also a mel spectogram.\n"
            "Use '--spectogram' to enable or '--no-spectogram' to disable.\n"
            "Default is to not plot a spectogram."
        ),
    )

    args = parser.parse_args()

    # Validate model file existence
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"The specified model file '{args.model}' does not exist."
        )

    device = torch.device(args.device)

    model, pinyin_to_ipa, ipa_to_token, speaker_info = load_model(
        args.model_class.lower(), args.model, device
    )

    if args.list_speakers:
        display_speaker_info(speaker_info)
        return

    # Validate speaker ID
    if args.speaker not in [speaker["speaker_id"] for speaker in speaker_info]:
        print(
            "Error: Invalid speaker ID. Use --list-speakers to see available options."
        )
        return

    vocoder_predictor = load_vocoder(args.vocoder, device)

    token_ids, tone_ids, phonemes, ipa_text = process_text(
        args.text, args.type, pinyin_to_ipa, ipa_to_token
    )

    if args.silence:
        sil = ipa_to_token["<sil>"]
        token_ids = [sil] + token_ids + [sil]
        tone_ids = [1] + tone_ids + [1]
        phonemes = [""] + phonemes + [""]

    token_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    tone_ids = torch.tensor([tone_ids], dtype=torch.long).to(device)
    speaker_id = torch.tensor([args.speaker], dtype=torch.long).to(device)

    with torch.inference_mode():
        mel, dur, pitch, _ = model(speaker_id, token_ids, tone_ids)
        predicted_wav = vocoder_predictor(mel.transpose(1, 2))

    write_mono_wav(
        args.output,
        vocoder_predictor.sampling_rate,
        predicted_wav.flatten().cpu().numpy(),
    )
    print(f"Audio saved to {args.output}")

    if args.spectogram:
        plot_mel_spectrogram(mel, pitch, dur, phonemes, ipa_text, vocoder_predictor)


if __name__ == "__main__":
    main()
