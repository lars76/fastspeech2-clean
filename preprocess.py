from pathlib import Path
from typing import List, Dict, Tuple

import librosa
import numpy as np
import pandas as pd
import penn
import tgt
import torch
from tqdm import tqdm

DATASET_PATH = "dataset/"
OUTPUT_PATH = "processed/"

VOCODER_NAME = "bigvgan_base_22khz_80band"
VOCODER = torch.hub.load(
    "lars76/bigvgan-mirror",
    VOCODER_NAME,
    source="github",
    trust_repo=True,
    pretrained=True,
)

# Pitch
PITCH_FMIN = 75
PITCH_FMAX = 500
DEVICE = "cuda:0"
BATCH_SIZE = 2048

SILENCE_TOKEN = "<sil>"
# 1000 * HOP_LENGTH / SAMPLE_RATE = 1000 * 256 / 22050 = 12ms
# specifies how long one <sil> token can be
MAX_SILENCE_LENGTH = 40 * VOCODER.hop_size
# Remove start and stop silence
TRIM_SILENCE = False
# Set silence in the waveforms to zero
SILENT_TO_ZERO = True


def approximate_integer_sum(
    rational_numbers: np.ndarray, target_sum: int, valid_indices: np.ndarray
) -> np.ndarray:
    """
    Adjusts a list of rational numbers to sum to a specified target integer sum.

    This function resolves the discrepancy between the sum of rounded phoneme durations
    and the total length of a STFT. WAV files store audio as 16-bit integers.
    Phoneme timings are initially calculated by multiplying by SAMPLE_RATE. Applying
    Short-Time Fourier Transform (STFT) with HOP_LENGTH and WIN_LENGTH introduces rounding
    errors, causing the sum of phoneme durations to deviate from the total sequence length.

    This function rounds each phoneme duration and adjusts the rounding to ensure the total
    sum matches the target sum (the total length of the STFT).

    Args:
        rational_numbers: The rational numbers representing phoneme durations.
        target_sum: The target sum that the adjusted integers should sum to.
        valid_indices: Indices to consider for adjusting the rounding.

    Returns:
        The adjusted list of integers summing to the target_sum.
    """
    rounded_integers = np.maximum(np.rint(rational_numbers), 1)
    float_differences = rational_numbers - rounded_integers
    current_sum = np.sum(rounded_integers)

    error = int(current_sum - target_sum)

    if error > 0:
        to_round_down = np.argsort(float_differences)
        to_round_down = to_round_down[valid_indices]
        to_round_down = to_round_down[:error]
        rounded_integers[to_round_down] -= 1
    elif error < 0:
        to_round_up = np.argsort(-float_differences)
        to_round_up = to_round_up[valid_indices]
        to_round_up = to_round_up[: abs(error)]
        rounded_integers[to_round_up] += 1

    current_sum = np.sum(rounded_integers)
    final_error = int(current_sum - target_sum)

    if final_error < 0:
        smallest_idx = np.argmin(rounded_integers)
        rounded_integers[smallest_idx] += abs(final_error)
    elif final_error > 0:
        largest_idx = np.argmax(rounded_integers)
        rounded_integers[largest_idx] -= final_error

    return rounded_integers


def pinyin_to_phones_tones(
    pinyin_tier: List[Dict], phones_tier: List[Dict]
) -> Tuple[List[Dict], List[int]]:
    mapper = []
    tones = []
    for pinyin in pinyin_tier:
        phone_text = ""
        for phone in phones_tier:
            if (
                phone["start_frames"] >= pinyin["start_frames"]
                and phone["end_frames"] <= pinyin["end_frames"]
            ):
                if pinyin["text"] in [SILENCE_TOKEN]:
                    tones.append(1)
                else:
                    # tones 1-5 to 2-6
                    # 0 is padding
                    tones.append(int(pinyin["text"][-1]) + 1)
                phone_text += f"{phone['text']} "
        pinyin_text = pinyin["text"]
        if pinyin_text[-1].isdigit():
            pinyin_text = pinyin_text[:-1]
        mapper.append({"text": pinyin_text, "phones": phone_text.strip()})

    return mapper, tones


def tier_to_dict(
    textgrid: tgt.TextGrid, tier_name: str
) -> Tuple[List[Dict], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Converts a tier from a TextGrid to a list of dictionaries containing text and frame information.
    Additionally, limits the maximum silence and identifies silent regions.
    Args:
        textgrid: The TextGrid object containing the tier.
        tier_name: The name of the tier to extract.
    Returns:
        A list of dictionaries with text and frame information for each interval in the tier.
        A list of frames containing the locations where there is a phoneme or silence.
        A list of silent regions.
    """
    tier = textgrid.get_tier_by_name(tier_name)
    result = []
    textgrid_offset = 0
    phone_regions = []
    silent_regions = []
    for i, interval in enumerate(tier):
        start_frames = textgrid_offset + int(
            np.ceil(interval.start_time * VOCODER.sampling_rate)
        )
        end_frames = textgrid_offset + int(
            np.ceil(interval.end_time * VOCODER.sampling_rate)
        )
        wav_start_frames = int(np.ceil(interval.start_time * VOCODER.sampling_rate))
        wav_end_frames = int(np.ceil(interval.end_time * VOCODER.sampling_rate))
        duration_frames = end_frames - start_frames
        if not interval.text:
            text = SILENCE_TOKEN
            silent_regions.append((wav_start_frames, wav_end_frames))
            if (i == 0 or i == len(tier) - 1) and TRIM_SILENCE:
                textgrid_offset -= duration_frames
                continue
            if MAX_SILENCE_LENGTH < duration_frames:
                textgrid_offset -= duration_frames - MAX_SILENCE_LENGTH
                duration_frames = MAX_SILENCE_LENGTH
                end_frames = start_frames + duration_frames
                if i == 0:
                    wav_start_frames = wav_end_frames - MAX_SILENCE_LENGTH
                else:
                    wav_end_frames = wav_start_frames + MAX_SILENCE_LENGTH
        else:
            text = interval.text.strip()
        duration_stft_frames = duration_frames / VOCODER.hop_size
        result.append(
            {
                "text": text,
                "start_frames": start_frames,
                "end_frames": end_frames,
                "duration_stft_frames": duration_stft_frames,
            }
        )
        phone_regions.append((wav_start_frames, wav_end_frames))
    return result, phone_regions, silent_regions


def compute_mel(y: np.ndarray) -> np.ndarray:
    mel_basis = librosa.filters.mel(
        sr=VOCODER.sampling_rate,
        n_fft=VOCODER.n_fft,
        n_mels=VOCODER.num_mels,
        fmin=VOCODER.fmin,
        fmax=VOCODER.fmax,
    )

    pad_length = int((VOCODER.n_fft - VOCODER.hop_size) / 2)
    y = np.pad(y, (pad_length, pad_length), mode="reflect")

    D = librosa.stft(
        y,
        n_fft=VOCODER.n_fft,
        hop_length=VOCODER.hop_size,
        win_length=VOCODER.win_size,
        window="hann",
        center=False,
        pad_mode="reflect",
    )

    S = np.sqrt(np.abs(D) ** 2 + 1e-9)
    S = np.dot(mel_basis, S)
    S = np.log(np.maximum(S, 1e-5))

    return S


def compute_pitch(
    audio: np.ndarray, resample: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    audio = torch.from_numpy(audio)[None].to(DEVICE)

    # Penn downsamples our frequency to 16000Hz and uses a hop size of 160
    pitch, periodicity = penn.from_audio(
        audio=audio,
        sample_rate=VOCODER.sampling_rate,
        fmin=PITCH_FMIN,
        fmax=PITCH_FMAX,
        gpu=None if DEVICE == "cpu" else DEVICE.replace("cuda:", ""),
        batch_size=BATCH_SIZE,
    )
    pitch = pitch.cpu().squeeze().numpy()
    periodicity = periodicity.cpu().squeeze().numpy()

    if resample > 0:
        pitch = np.interp(
            np.linspace(0, pitch.shape[0], resample), np.arange(pitch.shape[0]), pitch
        )
        periodicity = np.interp(
            np.linspace(0, periodicity.shape[0], resample),
            np.arange(periodicity.shape[0]),
            periodicity,
        )

    return pitch, periodicity


def read_audio(file_path: str, subtract_dc: bool = False) -> np.ndarray:
    # Load the audio file
    y, _ = librosa.load(file_path, sr=VOCODER.sampling_rate)

    # Subtract the mean to remove DC offset
    if subtract_dc:
        y = y - np.mean(y)

    # Compute the RMS of the current signal
    rms = np.sqrt(np.mean(y**2))

    # Desired RMS in linear scale for -20 dBFS
    desired_rms = 10 ** (-20 / 20)

    # Compute the required gain to reach the desired RMS
    gain = desired_rms / rms

    # Constrain the gain within -3 to 3 dB
    gain = np.clip(gain, 10 ** (-3 / 20), 10 ** (3 / 20))

    # Apply the gain
    y = y * gain

    # Normalize the waveform to range between -1 and 1
    y = y / np.max(np.abs(y))

    # Set the sample width to 16-bit
    # Convert the float32 waveform to int16 format and back to float32 to simulate 16-bit depth
    y = (y * 32767).astype(np.int16).astype(np.float32) / 32767

    return y


def process_file(
    filename: Path, speaker_info: Dict, phone_to_id: Dict, cur_phone_id: int
) -> Tuple[Dict, List, List, int]:
    wav_filename = filename.with_suffix(".wav")
    if not wav_filename.exists():
        raise FileNotFoundError(f"{wav_filename} does not exist")

    speaker_id = filename.parent.name
    if speaker_id not in speaker_info:
        speaker_info[speaker_id] = {"num_files": 0, "speaker_index": len(speaker_info)}

    speaker_output_path = Path(OUTPUT_PATH) / speaker_id
    speaker_output_path.mkdir(parents=True, exist_ok=True)

    num_files = speaker_info[speaker_id]["num_files"]
    output_filepath = speaker_output_path / f"{num_files:03}.pt"

    speaker_info[speaker_id]["num_files"] += 1

    textgrid = tgt.io.read_textgrid(filename, include_empty_intervals=True)
    pinyins, phone_regions, silent_regions = tier_to_dict(textgrid, "pinyins")
    hanzis, _, _ = tier_to_dict(textgrid, "hanzis")
    phones, _, _ = tier_to_dict(textgrid, "phones")

    wav = read_audio(str(wav_filename))

    # Set silent regions to zero
    if SILENT_TO_ZERO:
        for silent_region_start, silent_region_stop in silent_regions:
            wav[silent_region_start:silent_region_stop] = 0

    mask = np.zeros(len(wav), dtype=bool)
    for start, stop in phone_regions:
        mask[start:stop] = True
    wav = wav[mask]

    phone_stats = phones
    pinyin_to_phones, tones = pinyin_to_phones_tones(pinyins, phones)
    pinyin_stats = pinyin_to_phones

    mel = compute_mel(wav)
    pitch, periodicity = compute_pitch(wav, resample=mel.shape[1])

    assert pitch.shape[0] == mel.shape[1]

    durations = np.array([p["duration_stft_frames"] for p in phones])
    valid_indices = np.array([p["text"] != SILENCE_TOKEN for p in phones])
    rounded_durations = approximate_integer_sum(durations, mel.shape[1], valid_indices)

    assert (rounded_durations > 0).all()
    assert np.sum(rounded_durations) == mel.shape[1]

    phone_text = []
    for p in phones:
        text = p["text"]
        if text not in phone_to_id:
            phone_to_id[text] = cur_phone_id
            cur_phone_id += 1
        phone_text.append(phone_to_id[text])

    torch.save(
        {
            "speaker": torch.tensor(
                speaker_info[speaker_id]["speaker_index"], dtype=torch.long
            ),
            "encoded_text": torch.tensor(phone_text, dtype=torch.long),
            "encoded_tone": torch.tensor(tones, dtype=torch.long),
            "pitch": torch.from_numpy(pitch).float(),
            "pitch_periodicity": torch.from_numpy(periodicity).float(),
            "duration": torch.from_numpy(rounded_durations).long(),
            "mel": torch.from_numpy(mel.T),
            "original_text": "".join([k["text"] for k in hanzis]),
        },
        output_filepath,
    )

    return speaker_info, phone_stats, pinyin_stats, cur_phone_id


def create_statistics(
    speaker_info: Dict, phone_stats: List, pinyin_stats: List, phone_to_id: Dict
) -> None:
    output_path = Path(OUTPUT_PATH)

    speaker_df = pd.DataFrame(
        [
            {
                "speaker_id": info["speaker_index"],
                "name": speaker_id,
                "num_files": info["num_files"],
            }
            for speaker_id, info in speaker_info.items()
        ]
    )
    speaker_df = speaker_df[["speaker_id", "name", "num_files"]]
    speaker_df.to_csv(output_path / "speakers.tsv", index=False, sep="\t")

    phone_stats_df = pd.DataFrame(phone_stats)
    phone_stats_df.insert(0, "phone_id", phone_stats_df["text"].map(phone_to_id))
    phone_stats_df = (
        phone_stats_df.groupby(["phone_id", "text"])
        .agg(
            occurrences=("text", "count"),
        )
        .sort_values("occurrences", ascending=False)
        .reset_index()
    )
    phone_stats_df.to_csv(output_path / "phones.tsv", index=False, sep="\t")

    pinyin_stats_df = pd.DataFrame(pinyin_stats)
    pinyin_stats_df = (
        pinyin_stats_df.groupby(["text", "phones"])
        .agg(
            occurrences=("text", "count"),
        )
        .sort_values("occurrences", ascending=False)
        .reset_index()
    )
    pinyin_stats_df.to_csv(output_path / "pinyins.tsv", index=False, sep="\t")


def main():
    dataset_path = Path(DATASET_PATH)
    phone_to_id = {}
    cur_phone_id = 1

    phone_stats = []
    pinyin_stats = []

    speaker_info = {}

    textgrid_files = sorted(dataset_path.glob("*/*.TextGrid"))
    for filename in tqdm(textgrid_files):
        try:
            speaker_info, file_phone_stats, file_pinyin_stats, cur_phone_id = (
                process_file(filename, speaker_info, phone_to_id, cur_phone_id)
            )
            phone_stats.extend(file_phone_stats)
            pinyin_stats.extend(file_pinyin_stats)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    create_statistics(speaker_info, phone_stats, pinyin_stats, phone_to_id)


if __name__ == "__main__":
    main()
