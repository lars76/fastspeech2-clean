# FastSpeech2-Clean

This repository provides a clean and modernized implementation of [FastSpeech2](https://arxiv.org/abs/2006.04558) and [LightSpeech](https://arxiv.org/abs/2102.04040). Existing repositories often have problems with reproducibility or contain bugs. This version aims to solve these problems by providing a cleaner and more up-to-date code base.

Currently, pre-processing and training are only implemented for [AISHELL-3](https://www.openslr.org/93/) (Mandarin Chinese, Apache v2.0 license). However, the scripts are designed to be easily adapted to other datasets.

![Alt text](example.jpg?raw=true)

## Available Models

If you discover something that can further improve the speech quality, please contribute by opening an issue or pull request.

| Model        | UTMOS  | CER    | Val loss | Params |
|--------------|--------|--------|----------|--------|
| [LightSpeech](https://github.com/lars76/fastspeech2-clean/releases/download/models/lightspeech.pt)  | 2.3098 | 0.2594 | 0.6640   | 3.37M  |
| [FastSpeech2](https://github.com/lars76/fastspeech2-clean/releases/download/models/fastspeech2.pt)  | **2.5620** | **0.2550** | 0.6374   | 25.36M |
| Ground truth | 2.4276 | 0.2917 | **0.0**      |   -    |

MOS is calculated using UTMOS (higher is better), and CER is calculated using Whisper (lower is better). The "ground truth" refers to the reconstruction of the real mel spectrograms by the vocoder `bigvgan_v2_22khz_80band_fmax8k_256x`. For predicting the generated spectrograms, we use `bigvgan_base_22khz_80band` due to its superior performance on distorted spectograms. See also my other [repository](https://github.com/lars76/bigvgan-mirror/). For validation, 14415 files are used (20% of the whole dataset).

### Audio Samples

| **Hanzi**                          | **Pinyin**                                                | **IPA**                                             |
|------------------------------------|-----------------------------------------------------------|-----------------------------------------------------|
| 按被征农地的原有用途来确定补偿         | an4 bei4 zheng1 nong2 di4 de5 yuan2 you3 yong4 tu2 lai2 que4 ding4 bu3 chang2 | an4 pei̯4 ʈʂəŋ1 nʊŋ2 ti4 tɤ5 ɥɛn2 jou̯3 jʊŋ4 tʰu2 lai̯2 tɕʰɥe4 tiŋ4 pu3 ʈʂʰaŋ2 |


#### LightSpeech

https://github.com/user-attachments/assets/b4e8bbd1-070b-405c-9c01-a941dffb1a74

#### FastSpeech2

https://github.com/user-attachments/assets/01cb62b7-f801-4584-8a65-3de647a1cc1e

#### Ground truth

https://github.com/user-attachments/assets/09a4659c-c455-47cc-9032-611c3f0cc23d

## Prediction

After [downloading a model](https://github.com/lars76/fastspeech2-clean/releases), you can generate speech using Chinese characters, pinyin, or International Phonetic Alphabet (IPA). Only PyTorch is required, but optionally matplotlib, librosa and g2pw are needed.

### Example Commands

See `python predict.py --help` for all commands.

- **Simplified Chinese:** `python predict.py "开车慢慢前行" --type simplified --model fastspeech2.pt --model_class "FastSpeech2" --speaker 0`
- **Pinyin:** `python predict.py "kai1 che1 man4 man4 qian2 xing2" --type pinyin --model fastspeech2.pt --model_class "FastSpeech2" --speaker 1`
- **IPA:** `python predict.py "kʰai̯1 ʈʂʰɤ1 man4 man4 tɕʰjɛn2 ɕiŋ2" --type ipa --model fastspeech2.pt --model_class "FastSpeech2" --speaker 1`
- **Listing available speakers:** `python predict.py --list-speakers --model fastspeech2.pt`
- **Simulating Other Languages:** Since the model is trained on phonemes, it can simulate other languages. For example, "how are you?" could be transcribed in IPA as "hau̯2 aɻ2 ju2". However, the quality for out-of-distribution words is not as good.

The supported phones are `['<sil>', 'n', 'a', 'ŋ', 'j', 'i', 'w', 't', 'ɤ', 'ʂ', 'ə', 'tɕ', 'u', 'ɛ', 'ou̯', 'l', 'ʈʂ', 'ɕ', 'p', 'au̯', 'k', 'ei̯', 'ai̯', 'o', 'tɕʰ', 'm', 'ʊ', 'tʰ', 'ts', 'ʐ̩', 'ʈʂʰ', 's', 'y', 'f', 'e', 'ɻ̩', 'x', 'ɥ', 'ɹ̩', 'h', 'kʰ', 'pʰ', 'tsʰ', 'ɻ', 'ʐ', 'aɚ̯', 'ɚ', 'z̩', 'ɐ', 'ou̯˞', 'ɔ', 'ɤ̃', 'u˞', 'œ', 'ɑ̃', 'ʊ̃']`

Here, `<sil>` denotes a silence marker.

## Instructions for training

### Download dataset

1. Download [data_aishell3.tgz](https://www.openslr.org/93/). 
2. Extract the archive to the chosen path: `tar xzf data_aishell3.tgz -C DATASET_PATH` where `DATASET_PATH` is your output path.
3. Download the [TextGrid files](https://github.com/lars76/forced-alignment-aishell/releases/download/textgrid_files/aishell3_textgrid_files.zip) or create the files on your own by following the instruction in this [repository](https://github.com/lars76/forced-alignment-chinese).
4. Extract the TextGrid files by running `unzip -q aishell3_textgrid_files.zip` and copy the files using `cp -r test train DATASET_PATH`.

### Installing packages

1. `conda create --name fastspeech2-clean python=3.11 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
2. `conda activate fastspeech2-clean`
3. `pip install librosa pandas penn tgt openai-whisper jiwer`

### Preprocessing

Preprocess your dataset with `preprocess.py`. This script is tailored for the AISHELL-3 dataset but can be adapted for other datasets. Make sure to change `DATASET_PATH` and `OUTPUT_PATH` in `preprocess.py`.

### Training

Run `CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py` to train the network. The `CUBLAS_WORKSPACE_CONFIG=:4096:8` flag is only necessary because of the use of `torch.use_deterministic_algorithms(True)`.

## Literature

- **LightSpeech**: [LightSpeech](https://arxiv.org/abs/2102.04040) has demonstrated that CNN architectures can achieve similar performance to transformers with reduced computational overhead.
- **BigVGAN Vocoder**: I use [BigVGAN](https://arxiv.org/abs/2206.04658) for better vocoding quality over Hifi-GAN.
- **Pitch estimation**: Many FastSpeech implementations use DIO + StoneMask, but these perform significantly worse than neural network based approaches. Here I use [PENN](https://arxiv.org/pdf/2301.12258), the current state of the art.
- **Objective Metrics**: Instead of looking only at the mel spectrogram loss, we employ [UTMOS](https://arxiv.org/abs/2204.02152) for MOS estimation and [Whisper](https://arxiv.org/abs/2212.04356) for Character Error Rate (CER). The best parameters are selected based on speech quality (MOS), intelligibility (CER) and validation loss. I have found that MOS alone is only weakly correlated with speech quality. [This paper](https://www.arxiv.org/abs/2407.12707) also came to the same conclusion.
