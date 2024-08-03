import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import os
import glob
import random
from torch import optim
import pandas as pd
import whisper
import jiwer
import re
from typing import List, Tuple
from scipy.io import wavfile
from preprocess import VOCODER, VOCODER_NAME, OUTPUT_PATH, TRIM_SILENCE
from collections import defaultdict
from sklearn.model_selection import train_test_split

from torch.nn import MSELoss, L1Loss

import logging
import sys

# from lightspeech import Model
from fastspeech2 import Model

DEVICE = "cuda:0"
SEED = 3
EPOCHS = 200
WARMUP = 5
LR_RATE = 1e-3
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAINING_SPLIT = 0.2

WHISPER_SIZE = "tiny"


def setup_logger(log_file="training.log"):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Set up the logger
logger = setup_logger()


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WarmupLinearSchedule(optim.lr_scheduler.LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


class CustomDataset(Dataset):
    def __init__(self, files: List[str], periodicity_range=[], pitch_mean_std=[]):
        self.files = files

        if not periodicity_range or not pitch_mean_std:
            self.periodicity_range = [float("inf"), float("-inf")]
            self.pitch_mean_std = [0.0, 0.0]
            self._compute_statistics()
            logger.info(
                f"Pitch mean/std: {self.pitch_mean_std[0]:.4f}, {self.pitch_mean_std[1]:.4f}"
            )
            logger.info(
                f"Periodicity range min/max: {self.periodicity_range[0]:.4f}, {self.periodicity_range[1]:.4f}"
            )
        else:
            self.periodicity_range = periodicity_range
            self.pitch_mean_std = pitch_mean_std

    def _compute_statistics(self):
        count = 0
        mean = 0.0
        M2 = 0.0

        for filename in tqdm(self.files, desc="Computing statistics"):
            try:
                pt_file = torch.load(filename)
                pitch_periodicity = pt_file["pitch_periodicity"]
                pitch = pt_file["pitch"]

                # Update periodicity range
                self.periodicity_range[0] = min(
                    pitch_periodicity.min().item(), self.periodicity_range[0]
                )
                self.periodicity_range[1] = max(
                    pitch_periodicity.max().item(), self.periodicity_range[1]
                )

                # Vectorized Welford's online algorithm
                new_count = count + pitch.size(0)
                delta = pitch - mean
                mean_update = (delta / new_count).sum()
                mean += mean_update
                delta2 = pitch - mean
                M2 += (delta * delta2).sum()
                count = new_count

            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

        if count > 1:
            variance = M2 / (count - 1)
            std_dev = np.sqrt(variance.item())
            self.pitch_mean_std = [mean.item(), std_dev]
        else:
            logger.warning("Warning: Insufficient data to compute statistics.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx])

    @staticmethod
    def pad_tensors(data: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        if not data:
            raise ValueError("Data must contain at least one tensor.")

        max_len = max(d.shape[0] for d in data)
        if data[0].dim() == 1:  # 1D tensors
            padded_data = torch.stack(
                [
                    torch.nn.functional.pad(
                        d, (0, max_len - d.shape[0]), value=pad_value
                    )
                    for d in data
                ]
            )
        elif data[0].dim() == 2:  # 2D tensors
            padded_data = torch.stack(
                [
                    torch.nn.functional.pad(
                        d, (0, 0, 0, max_len - d.shape[0]), value=pad_value
                    )
                    for d in data
                ]
            )
        else:
            raise ValueError("Tensors must be 1D or 2D.")

        return padded_data

    def collate_fn(self, batch: List[dict]) -> Tuple[torch.Tensor, ...]:
        speakers = torch.tensor([b["speaker"] for b in batch])
        texts = self.pad_tensors([b["encoded_text"] for b in batch])
        tones = self.pad_tensors([b["encoded_tone"] for b in batch])

        pitches = self.pad_tensors(
            [
                (b["pitch"] - self.pitch_mean_std[0]) / self.pitch_mean_std[1]
                for b in batch
            ]
        )
        periodicity = self.pad_tensors(
            [
                (b["pitch_periodicity"] - self.periodicity_range[0])
                / (self.periodicity_range[1] - self.periodicity_range[0])
                for b in batch
            ]
        ).float()
        durations_rounded = self.pad_tensors([b["duration"] for b in batch])
        mels = self.pad_tensors([b["mel"] for b in batch])

        padding_mask_pitch = self.pad_tensors(
            [torch.ones_like(b["pitch"]) for b in batch]
        ).bool()
        padding_mask_mel = self.pad_tensors(
            [torch.ones_like(b["mel"]) for b in batch]
        ).bool()
        padding_mask_dur = self.pad_tensors(
            [torch.ones_like(b["duration"]) for b in batch]
        ).bool()

        return (
            speakers,
            texts,
            tones,
            pitches,
            periodicity,
            durations_rounded,
            mels,
            padding_mask_pitch,
            padding_mask_mel,
            padding_mask_dur,
        )


def train_one_epoch(model, train_loader, optimizer, scaler, scheduler):
    model.train()
    mse_loss = MSELoss(reduction="none")
    l1_loss = L1Loss(reduction="none")
    total_losses = defaultdict(float)

    for audio in tqdm(train_loader, desc="Training"):
        audio = [k.to(DEVICE) for k in audio]
        (
            speakers,
            texts,
            tones,
            pitches,
            periodicity,
            durations,
            mels,
            padding_mask_pitch,
            padding_mask_mel,
            padding_mask_dur,
        ) = audio

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            mel_pred, dur_pred, pitch_pred, periodicity_pred = model(
                speakers, texts, tones, pitches, periodicity, durations, mels
            )

            mel_loss = l1_loss(mel_pred, mels)[padding_mask_mel].mean()
            dur_loss = mse_loss(dur_pred, torch.log1p(durations.float()))[
                padding_mask_dur
            ].mean()
            pitch_loss = (periodicity * mse_loss(pitch_pred, pitches))[
                padding_mask_pitch
            ].mean()
            periodicity_loss = mse_loss(periodicity_pred, periodicity)[
                padding_mask_pitch
            ].mean()

            loss_all = mel_loss + dur_loss + pitch_loss + periodicity_loss

        scaler.scale(loss_all).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = speakers.size(0)
        for loss_name, loss_value in [
            ("train_mel_loss", mel_loss),
            ("train_dur_loss", dur_loss),
            ("train_pitch_loss", pitch_loss),
            ("train_periodicity_loss", periodicity_loss),
        ]:
            total_losses[loss_name] += loss_value.item() * batch_size

    total_samples = len(train_loader.dataset)
    return {k: v / total_samples for k, v in total_losses.items()}


def val_one_epoch(model, val_loader):
    model.eval()
    mse_loss = MSELoss(reduction="none")
    l1_loss = L1Loss(reduction="none")
    total_losses = defaultdict(float)

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=True, dtype=torch.float16
    ):
        for audio in tqdm(val_loader, desc="Validation"):
            audio = [k.to(DEVICE) for k in audio]
            (
                speakers,
                texts,
                tones,
                pitches,
                periodicity,
                durations,
                mels,
                padding_mask_pitch,
                padding_mask_mel,
                padding_mask_dur,
            ) = audio

            mel_pred, dur_pred, pitch_pred, periodicity_pred = model(
                speakers, texts, tones, pitches, periodicity, durations, mels
            )

            mel_loss = l1_loss(mel_pred, mels)[padding_mask_mel].mean()
            dur_loss = mse_loss(dur_pred, torch.log1p(durations.float()))[
                padding_mask_dur
            ].mean()
            pitch_loss = (periodicity * mse_loss(pitch_pred, pitches))[
                padding_mask_pitch
            ].mean()
            periodicity_loss = mse_loss(periodicity_pred, periodicity)[
                padding_mask_pitch
            ].mean()

            batch_size = speakers.size(0)
            for loss_name, loss_value in [
                ("val_mel_loss", mel_loss),
                ("val_dur_loss", dur_loss),
                ("val_pitch_loss", pitch_loss),
                ("val_periodicity_loss", periodicity_loss),
            ]:
                total_losses[loss_name] += loss_value.item() * batch_size

    total_samples = len(val_loader.dataset)
    losses = {k: v / total_samples for k, v in total_losses.items()}
    losses["val_total_loss"] = sum(losses.values())

    return losses


def evaluate_cer_mos(model, val_files, use_gt=False):
    mos_predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    ).to(DEVICE)
    asr_predictor = whisper.load_model(WHISPER_SIZE, device=DEVICE)

    if use_gt:
        vocoder_predictor = torch.hub.load(
            "lars76/bigvgan-mirror",
            "bigvgan_v2_22khz_80band_fmax8k_256x",
            source="github",
            trust_repo=True,
            pretrained=True,
        ).to(DEVICE)
    else:
        vocoder_predictor = VOCODER.to(DEVICE)

    model.eval()
    metrics = defaultdict(float)

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=True, dtype=torch.float16
    ):
        for pt_file in tqdm(val_files, desc="Evaluating metrics"):
            pt_file = torch.load(pt_file)
            inputs = {
                k: v[None].to(DEVICE)
                for k, v in pt_file.items()
                if k != "original_text"
            }

            if use_gt:
                mel_pred, dur_pred = inputs["mel"], inputs["duration"]
            else:
                mel_pred, dur_pred, _, _ = model(
                    inputs["speaker"], inputs["encoded_text"], inputs["encoded_tone"]
                )

            # to be consistent, remove start and stop silence if found
            # it can affect the evaluation
            if not TRIM_SILENCE:
                dur_pred = dur_pred.cpu().numpy().flatten()
                mel_pred = mel_pred[:, int(dur_pred[0]) : -int(dur_pred[-1])]

            pred_wav = (
                vocoder_predictor(mel_pred.transpose(1, 2))
                .float()
                .flatten()
                .cpu()
                .numpy()
            )

            tmp_wav_file = "tmp.wav"
            wavfile.write(tmp_wav_file, vocoder_predictor.sampling_rate, pred_wav)

            metrics["val_character_error_rate"] += calculate_cer(
                asr_predictor, tmp_wav_file, pt_file["original_text"]
            ).item() / len(val_files)
            metrics["val_mean_opinion_score"] += calculate_mos(
                mos_predictor, pred_wav, vocoder_predictor.sampling_rate
            ).item() / len(val_files)

    return dict(metrics)


def calculate_cer(asr_predictor, audio_file, original_text):
    pred_hanzi = asr_predictor.transcribe(
        audio=audio_file,
        language="zh",
        initial_prompt="以下是普通话的句子，请以简体输出。",
    )["text"]
    pred_hanzi = re.sub(r"[^\u4e00-\u9fff]+", "", pred_hanzi)
    return np.clip(jiwer.cer(original_text.replace("<sil>", ""), pred_hanzi), 0.0, 1.0)


def calculate_mos(mos_predictor, wav_data, sampling_rate):
    try:
        wav_tensor = torch.from_numpy(wav_data).unsqueeze(0).to(DEVICE)
        mos_score = mos_predictor(wav_tensor, sampling_rate).item()
        if not np.isfinite(mos_score):
            raise ValueError("Invalid MOS score: non-finite number encountered")
        return np.clip(mos_score, 1, 5)
    except Exception as e:
        logger.error(f"Error calculating MOS score: {e}")
        return 1


def parse_speakers(filename):
    speaker_df = pd.read_csv(filename, sep="\t")
    dicts_list = speaker_df.to_dict(orient="records")

    return dicts_list, speaker_df["name"]


def get_train_val_files(
    file_list, speaker_ids, unique_speakers, test_size, random_state=42
):
    """
    Generate train and validation files using a stratified split.

    Args:
        file_list: List of all file paths
        speaker_ids: Array of speaker IDs corresponding to file_list
        unique_speakers: List of unique speaker IDs
        test_size: Proportion of the dataset to include in the validation split
        random_state: Random state for reproducibility

    Returns:
        train_files: List of file paths for training
        val_files: List of file paths for validation
    """
    train_files = []
    val_files = []

    def text_length_score(text):
        """Calculate the text length score, excluding specific characters."""
        return len(text.replace("<sil>", ""))

    for speaker_id in tqdm(unique_speakers, desc="Splitting files"):
        speaker_files = np.array(file_list)[speaker_ids == speaker_id]
        speaker_data = []

        for file_path in speaker_files:
            file_content = torch.load(file_path)
            cleaned_text = file_content["original_text"].replace("<sil>", "")

            # Check if file contains specific characters
            if any(char in cleaned_text for char in "零二三四五六七八九十百123456789"):
                train_files.append(file_path)
            else:
                speaker_data.append((file_path, text_length_score(cleaned_text)))

        if speaker_data:
            # Extract file paths and text length scores
            file_paths, text_lengths = zip(*speaker_data)

            # Create bins and ensure each bin has at least two members
            n_bins = min(len(text_lengths) // 2, 10)
            text_length_bins = np.percentile(
                text_lengths, np.linspace(0, 100, n_bins + 1)
            )
            bin_indices = np.digitize(text_lengths, text_length_bins, right=True)

            # Merge small bins
            unique_bins, counts = np.unique(bin_indices, return_counts=True)
            for bin_val, count in zip(unique_bins, counts):
                if count < 2:
                    bin_indices[bin_indices == bin_val] = unique_bins[counts > 1][
                        0
                    ]  # Merge to the first larger bin

            train_idx, val_idx = train_test_split(
                range(len(speaker_data)),
                test_size=test_size,
                stratify=bin_indices,
                random_state=random_state,
            )

            train_files.extend([file_paths[i] for i in train_idx])
            val_files.extend([file_paths[i] for i in val_idx])

    return train_files, val_files


def main():
    start_time = time.time()

    file_list = np.asarray(
        sorted(glob.glob(os.path.join(OUTPUT_PATH, "**", "*pt"), recursive=True))
    )
    speaker_ids = np.asarray([os.path.basename(os.path.dirname(f)) for f in file_list])
    speaker_dict, unique_speakers = parse_speakers(
        os.path.join(OUTPUT_PATH, "speakers.tsv")
    )
    num_speakers = len(unique_speakers)
    logger.info(f"Number of speakers: {num_speakers}")

    train_files, val_files = get_train_val_files(
        file_list, speaker_ids, unique_speakers, TRAINING_SPLIT
    )
    logger.info(
        f"Training files: {len(train_files)}, validation files: {len(val_files)}"
    )

    seed_all(SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = CustomDataset(train_files)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=train_dataset.collate_fn,
        generator=g,
    )

    val_dataset = CustomDataset(
        val_files, train_dataset.periodicity_range, train_dataset.pitch_mean_std
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        collate_fn=val_dataset.collate_fn,
        generator=g,
    )

    train_epoch_steps = len(train_loader)

    # keep_default_na=False to force "nan" not to be interpreted as NaN
    pinyin_df = pd.read_csv(
        f"{OUTPUT_PATH}/pinyins.tsv", sep="\t", keep_default_na=False
    )
    pinyin_dict = pinyin_df.set_index("text")["phones"].to_dict()

    phone_df = pd.read_csv(f"{OUTPUT_PATH}/phones.tsv", sep="\t", keep_default_na=False)
    phone_dict = phone_df.set_index("text")["phone_id"].to_dict()
    num_phones = phone_df["phone_id"].max() + 1
    logger.info(f"Number of phones: {num_phones}")

    model = Model(
        num_phones=num_phones,
        num_speakers=num_speakers,
        num_mel_bins=VOCODER.num_mels,
    ).to(DEVICE)
    logger.info(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")

    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=LR_RATE)

    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=WARMUP * train_epoch_steps,
        t_total=EPOCHS * train_epoch_steps,
    )

    best_loss = float("inf")
    log_file = []
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch: {epoch}/{EPOCHS}")
        epoch_start_time = time.time()

        epoch_info = {"epoch": epoch}

        epoch_info |= train_one_epoch(model, train_loader, optimizer, scaler, scheduler)
        epoch_info |= val_one_epoch(model, val_loader)

        if epoch_info["val_total_loss"] < best_loss:
            best_loss = epoch_info["val_total_loss"]
            logger.info("New best val_total_loss")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "phone_dict": phone_dict,
                    "pinyin_dict": pinyin_dict,
                    "speaker_dict": speaker_dict,
                    "vocoder_name": VOCODER_NAME,
                    "num_phones": num_phones,
                    "num_speakers": num_speakers,
                    "num_mel_bins": VOCODER.num_mels,
                }
                | epoch_info,
                "model.pt",
            )

        log_file.append(
            epoch_info
            | {
                "elapsed": (time.time() - epoch_start_time) / 60,
                "elapsed_total": (time.time() - start_time) / 60,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        logger.info(log_file[-1])

    pd.DataFrame(log_file).to_csv("model.csv", index=False)

    logger.info(f"Best loss: {best_loss}")
    run_time = (time.time() - start_time) / 60

    model.load_state_dict(torch.load("model.pt")["state_dict"])

    logger.info("Predicting ground truth mel spectrograms...")
    result_gt = evaluate_cer_mos(model, val_files, use_gt=True)
    logger.info(f"Result: {result_gt}")

    logger.info("Predicting model spectrograms...")
    result_model = evaluate_cer_mos(model, val_files, use_gt=False)
    logger.info(f"Result: {result_model}")

    logger.info(f"Run time: {run_time} minutes")


if __name__ == "__main__":
    main()
