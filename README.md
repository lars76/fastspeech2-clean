# FastSpeech2-Clean

This repository provides a clean and modernized implementation of [FastSpeech2](https://arxiv.org/abs/2006.04558) and [LightSpeech](https://arxiv.org/abs/2102.04040). Existing repositories often have problems with reproducibility or contain bugs. This version aims to solve these problems by providing a cleaner and more up-to-date code base.

Currently, pre-processing and training are only implemented for Chinese speech datasets. However, the scripts are designed to be easily adapted to other languages.

![Alt text](example.jpg?raw=true)

## Models

If you have suggestions to further enhance speech quality, please contribute by opening an issue or pull request.

### Models trained on AISHELL-3

| Model | UTMOS | CER | Val Loss | Params |
|-------|-------|-----|----------|--------|
| [LightSpeech](https://github.com/lars76/fastspeech2-clean/releases/download/models/lightspeech.pt) | 2.3098 | 0.2594 | 0.6640 | 3.37M |
| [FastSpeech2](https://github.com/lars76/fastspeech2-clean/releases/download/models/fastspeech2.pt) | **2.5620** | **0.2550** | 0.6374 | 25.36M |
| Ground Truth | 2.4276 | 0.2917 | **0.0** | - |

### Models trained on AISHELL-3 and biaobei combined

| Model | UTMOS | CER | Val Loss | Params |
|-------|-------|-----|----------|--------|
| [FastSpeech2](https://github.com/lars76/fastspeech2-clean/releases/download/models/fastspeech2_ab_combined.pt) | **2.6578** | **0.2529** | 0.6335 | 25.36M |
| Ground Truth | 2.5482 | 0.2909 | **0.0** | - |

### Notes

- MOS is calculated using UTMOS (higher is better), and CER is calculated using Whisper (lower is better).
- The "ground truth" refers to the reconstruction of the true mel spectrograms by the vocoder `bigvgan_v2_22khz_80band_fmax8k_256x`.
- For predicting generated spectrograms, `bigvgan_base_22khz_80band` is used due to its superior performance on distorted spectrograms. See also my other [repository](https://github.com/lars76/bigvgan-mirror/).
- Approximately 20% of the dataset is used for validation.

## Audio Samples

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

### Preparing training data

Organize all audio files in a directory named `dataset` with the following structure: `dataset/SPEAKER_NAME/FILENAME.wav` and `dataset/SPEAKER_NAME/FILENAME.TextGrid`. For instance, the file `SSB00050001.wav` from AISHELL-3 would be located at `dataset/SSB0005/SSB00050001.wav`.

For [AISHELL-3](https://www.openslr.org/93/) (Apache v2.0 license) and [biaobei](https://en.data-baker.com/datasets/freeDatasets/) (non-commercial use only), pretrained TextGrid files are available in this [repository](https://github.com/lars76/forced-alignment-chinese). However, you can also generate your own annotations if needed.

Make sure that the .TextGrid files have the following sections: "hanzis", "pinyins", "phones".

### Installing packages

1. `conda create --name fastspeech2-clean python=3.11 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
2. `conda activate fastspeech2-clean`
3. `pip install librosa pandas penn tgt openai-whisper jiwer`

### Preprocessing

Preprocess your dataset with `preprocess.py`. This script is tailored to the Chinese language, but can also be adapted for other languages. Make sure to change `DATASET_PATH` and `OUTPUT_PATH` in `preprocess.py` if your input/output files should be in a different folder.

### Training

Run `CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py` to train the network. The `CUBLAS_WORKSPACE_CONFIG=:4096:8` flag is only necessary because of the use of `torch.use_deterministic_algorithms(True)`.

## Literature

- **LightSpeech**: [LightSpeech](https://arxiv.org/abs/2102.04040) has demonstrated that CNN architectures can achieve similar performance to transformers with reduced computational overhead.
- **BigVGAN Vocoder**: I use [BigVGAN](https://arxiv.org/abs/2206.04658) for better vocoding quality over Hifi-GAN.
- **Pitch estimation**: Many FastSpeech implementations use DIO + StoneMask, but these perform significantly worse than neural network based approaches. Here I use [PENN](https://arxiv.org/pdf/2301.12258), the current state of the art.
- **Objective Metrics**: Instead of looking only at the mel spectrogram loss, we employ [UTMOS](https://arxiv.org/abs/2204.02152) for MOS estimation and [Whisper](https://arxiv.org/abs/2212.04356) for Character Error Rate (CER). The best parameters are selected based on speech quality (MOS), intelligibility (CER) and validation loss. I have found that MOS alone is only weakly correlated with speech quality. [This paper](https://www.arxiv.org/abs/2407.12707) also came to the same conclusion.

## Disclaimer

This Text-to-Speech (TTS) system is provided as-is, without any guarantees or warranty. By using this system, you agree that the developers hold no responsibility or liability for any harm or damages that may result from the use of the generated speech.

### Responsibility for Generated Content

The developers of this TTS system are not responsible for the content generated by the system. Users are solely responsible for any speech generated using this tool and must ensure that their use complies with all applicable laws and regulations.

### Use of Voice Data

All voice data used in this TTS system is the property of the original voice actors or respective owners. The use of this data is subject to the terms and conditions set by the original owners. Users must obtain appropriate permissions or licenses for any commercial use of the generated speech or underlying voice data.

### Ethical Use

We encourage ethical use of this TTS system. The generated speech should not be used for any malicious activities, including but not limited to, spreading misinformation, creating deepfakes, impersonation, or any other activities that could harm individuals or society.