# MTAD: Large-scale Multi-Topic Audio Deepfake Benchmark
[![Hugging Face%20-%20MTAD](https://img.shields.io/badge/🤗%20Hugging%20Face%20-%20MTAD-blue)](https://huggingface.co/datasets/Lioy/MTAD)

MTAD is a large-scale benchmark for both evaluating audio deepfake detection and synthesizing realistic, dataset-aligned deepfake audio across diverse topics. This repository includes an evaluation suite with state-of-the-art baselines and a modular generation toolkit (TTS + Voice Conversion) to synthesize deepfake audio.

## 🔥 Features

- **Multi-topic Coverage**: 13 diverse datasets spanning different audio topics (news, interviews, movies, audiobooks, etc.)
- **Comprehensive Evaluation**: Support for both cross-domain and in-domain evaluation scenarios
- **State-of-the-art Baselines**: 6 advanced audio deepfake detection models
- **Rich Audio Content**: Over 422 hours of audio data with balanced real and fake samples
- **Integrated Generation**: TTS + VC toolkit to synthesize domain-specific deepfake audio with automatic metadata updates
- **Flexible Framework**: Easy-to-use, modular pipelines for both evaluation and generation

## 📚 Table of Contents

- **Installation**: [Installation](#installation)
- **Quick Start**: [Quick Start](#quick-start)
- **Evaluation Guide**: [Evaluation Guide](#evaluation-guide)
- **Generation Guide**: [Generation Guide](#generation-guide)
- **Datasets**: [Dataset Overview](#dataset-overview)
- **Baselines**: [Available Baselines](#available-baselines)
- **Metrics**: [Evaluation Metrics](#evaluation-metrics)
- **Advanced Usage**: [Advanced Usage](#advanced-usage)
- **Repository Structure**: [Repository Structure](#repository-structure)
- **Logging and Results**: [Logging and Results](#logging-and-results)
- **Citation**: [Citation](#citation)
- **License**: [License](#license)

## 🏗️ Installation

### Prerequisites
- Python 3.12+
- CUDA 12.4+
- 75GB+ free disk space for full dataset

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/wtalioy/MTAD.git
cd MTAD
```

2. **Install dependencies**:
```bash
conda create -n mtad python=3.12 -y
conda activate mtad
pip install -r requirements.txt
python -m unidic download
cd src/generation/models/tts/vits/monotonic_align
python setup.py build_ext --inplace
cd ../../../../../..
```

3. **Download datasets**: 
Place MTAD dataset files in the `data/` directory following this structure:
```
data/
├── Audiobook/
├── Emotional/
├── Interview/
├── Movie/
├── News/
├── NoisySpeech/
├── PartialFake/
├── PhoneCall/
├── Podcast/
├── PublicFigure/
└── PublicSpeech/
```

## 🚀 Quick Start

Run a quick evaluation or generation with the most common options.

### Evaluation
```bash
# Cross-domain evaluation on two datasets
python src/eval/main.py --baseline aasist --dataset interview publicspeech --mode cross --metric eer

# In-domain train+eval
python src/eval/main.py --baseline aasist --dataset interview --mode in --metric eer
```

### Generation
```bash
# Generate English podcast samples using reference-speaker TTS
python src/generation/main.py -d podcast -t xttsv2 yourtts -s en
```

## 🧪 Evaluation Guide

Evaluate baseline models across domains with unified CLI.

### CLI
```bash
python src/eval/main.py \
  -b aasist rawnet2 \
  -d phonecall publicspeech interview \
  -m cross \
  --metric eer
```

### Arguments
- **-b / --baseline**: one or more of: `aasist`, `aasist-l`, `ardetect`, `res-tssdnet`, `inc-tssdnet`, `rawnet2`, `rawgat-st`
- **-d / --dataset**: one or more of: `publicfigure`, `news`, `podcast`, `partialfake`, `audiobook`, `noisyspeech`, `phonecall`, `interview`, `publicspeech`, `movie`, `emotional`, `asvspoof2021`, `in-the-wild`
- **-m / --mode**: `cross` or `in`
- **-s / --subset**: dataset-specific subset (if applicable)
- **--metric**: one or more metrics, e.g. `eer`, `auroc`
- **--train_only / --eval_only**: restrict to one stage in `in` mode
- **--data_dir**: path to data root (default: `data`)

### Modes
- **Cross-domain**: evaluate without training
  ```bash
  python src/eval/main.py -b aasist rawnet2 -d phonecall publicspeech interview -m cross --metric eer
  ```
- **In-domain**: optional training followed by evaluation
  ```bash
  # Train only
  python src/eval/main.py -b aasist -d interview -m in --train_only
  # Eval only (using existing checkpoints)
  python src/eval/main.py -b aasist -d interview -m in --eval_only --metric eer
  # Train + Eval
  python src/eval/main.py -b aasist -d interview -m in --metric eer
  ```

### Outputs
- Metrics printed to console
- Logs written to `logs/eval.log`

## 🔊 Generation Guide

Generate synthetic audio for raw domains using TTS and optional Voice Conversion (VC) models.

### CLI
```bash
python src/generation/main.py \
  -d podcast \
  -t xttsv2 yourtts \
  -v openvoice \
  -s en
```

### Arguments
- **-d / --dataset**: one or more of: `news`, `podcast`, `movie`, `phonecall`, `interview`, `publicspeech`, `partialfake`, `noisyspeech`
- **-t / --tts_model**: one or more of: `vits`, `xttsv2`, `yourtts`, `tacotron2`, `bark`, `melotts`, `elevenlabs`, `geminitts`, `gpt4omini`
- **-v / --vc_model**: optional VC models: `knnvc`, `freevc`, `openvoice`
- **-s / --subset**: language/subset for certain datasets (e.g., `phonecall`). Common values: `en`, `zh-cn` (default: `zh-cn`)
- **--data_dir**: custom data root for the selected dataset(s). If omitted, defaults to `data/{Dataset}`

Notes:
- Some TTS models require VC (their voices are not speaker-conditioned). These are marked internally and will be paired with provided VC models if any.
- TTS models that support reference audio (e.g., `xttsv2`, `yourtts`) can run without VC.

### Dataset expectations
- Each dataset directory should contain a `meta.json` describing items and real audio paths, e.g. `data/Podcast/meta.json` with `audio/real/...` entries.
- Generated audio is saved under `audio/fake/...` and `meta.json` is updated with a mapping per model.
- `phonecall` expects a subfolder by subset: `data/PhoneCall/en` or `data/PhoneCall/zh-cn`.
- `partialfake` will build its own `meta.json` by sampling from `Interview`, `Podcast`, and `PublicSpeech` test metadata. Ensure these exist at `data/{Domain}/meta_test.json`.

### Environment variables for cloud TTS
Set the following if you use those providers:
- **OPENAI_API_KEY**: required for `gpt4omini`
- **GOOGLE_API_KEY**: required for `geminitts`
- **ELEVENLABS_API_KEY**: required for `elevenlabs`

Examples:
```bash
# Bash
export OPENAI_API_KEY=... \
       GOOGLE_API_KEY=... \
       ELEVENLABS_API_KEY=...

# PowerShell
$env:OPENAI_API_KEY="..."; $env:GOOGLE_API_KEY="..."; $env:ELEVENLABS_API_KEY="..."
```

### Examples
- **TTS-only English podcast generation** (reference-speaker TTS):
```bash
python src/generation/main.py -d podcast -t xttsv2 yourtts -s en
```

- **Chinese news with TTS+VC** (pairs TTS that require VC with a VC model):
```bash
python src/generation/main.py -d news -t gpt4omini melotts bark -v openvoice -s zh-cn
```

- **PartialFake composition across domains**:
```bash
python src/generation/main.py -d partialfake -t xttsv2 -v openvoice
```

### Outputs and logs
- Generated files: `data/{Dataset}/audio/fake/...`
- Updated metadata: `data/{Dataset}/meta.json`
- Logs: `logs/generation*.log`

## 📊 Dataset Overview

MTAD includes 13 diverse datasets across multiple domains:

| Dataset | Domain | Real Duration | Fake Duration | Total Duration | Real Files | Fake Files |
|---------|--------|---------------|---------------|----------------|------------|------------|
| **Audiobook** | Audiobooks | 19h 24m | 27h 48m | 47h 13m | 9,425 | 13,612 |
| **Emotional** | Emotional Speech | 29h 4m | 29h 51m | 58h 55m | 35,000 | 36,000 |
| **Interview** | Interviews | 27h 20m | 31h 52m | 59h 13m | 12,095 | 12,096 |
| **Movie** | Movie Dialogues | 8h 6m | 11h 51m | 19h 57m | 9,167 | 9,115 |
| **News** | News Reports | 20h 42m | 19h 9m | 39h 52m | 4,910 | 4,910 |
| **NoisySpeech** | Noisy Environments | 0h | 18h 0m | 18h 0m | 0 | 7,505 |
| **PartialFake** | Partial Synthesis | 0h | 22h 57m | 22h 57m | 0 | 7,281 |
| **PhoneCall** | Phone Conversations | 12h 3m | 14h 47m | 26h 51m | 8,268 | 8,237 |
| **Podcast** | Podcasts | 22h 42m | 24h 16m | 46h 58m | 4,948 | 4,942 |
| **PublicFigure** | Public Figures | 15h 17m | 14h 45m | 30h 2m | 9,931 | 7,631 |
| **PublicSpeech** | Public Speeches | 26h 10m | 26h 35m | 52h 45m | 9,651 | 9,651 |

**Total**: 180h 49m real + 241h 51m fake = **422h 41m** (103,395 real + 120,980 fake files)

## 🎯 Available Baselines

MTAD includes 6 state-of-the-art audio deepfake detection models:

| Baseline | Description | Paper |
|----------|-------------|-------|
| **AASIST** | Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks | [ICASSP 2022](https://arxiv.org/abs/2110.01200) |
| **AASIST-L** | Lightweight variant of AASIST | [ICASSP 2022](https://arxiv.org/abs/2110.01200) |
| **RawNet2** | End-to-end anti-spoofing using raw waveforms | [ICASSP 2021](https://arxiv.org/abs/2011.01108) |
| **Res-TSSDNet** | Time-domain synthetic speech detection net (Resnet Net Style) | [IEEE 2021](https://arxiv.org/abs/2106.06341) |
| **Inc-TSSDNet** | Time-domain synthetic speech detection net (Inception Net Style) | [IEEE 2021](https://arxiv.org/abs/2106.06341) |
| **RawGAT-ST** | End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection | [ASVspoof 2021 Workshop](https://arxiv.org/abs/2107.12710) |
| **ARDetect** | Maximum Mean Discrepancy based detection | [Published soon]() |

## 📈 Evaluation Metrics

MTAD supports the following evaluation metrics:

- **EER** (Equal Error Rate): Primary metric for audio deepfake detection
- **AUROC** (Area Under the Receiver Operating Characteristic Curve): Secondary metric for audio deepfake detection

## 🔧 Advanced Usage

### Custom Dataset

To add a new dataset, create a class inheriting from `BaseDataset`:

```python
from cmad_datasets.base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "MyDataset"), *args, **kwargs)
        self.name = "MyDataset"
```

### Custom Baseline

To add a new baseline model, inherit from the `Baseline` class:

```python
from baselines.base import Baseline

class MyBaseline(Baseline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "MyBaseline"
        self.supported_metrics = ["eer", "acc"]
    
    def evaluate(self, data, labels, metrics, **kwargs):
        # Implementation
        pass
```

### Configuration

Model configurations are stored in `src/eval/baselines/{model}/config/`:
- `model.yaml`: Model architecture configuration
- `train_default.yaml`: Default training configuration
- `train_{dataset}.yaml`: Dataset-specific training configuration

## 📁 Repository Structure

```
MTAD/
├── data/                          # Dataset directory
├── src/
│   ├── eval/
│   │   ├── main.py               # Main evaluation script
│   │   ├── config.py             # Configuration definitions
│   │   ├── baselines/            # Baseline model implementations
│   │   │   ├── aasist/
│   │   │   ├── rawnet2/
│   │   │   ├── TSSDNet/
│   │   │   ├── RawGAT_ST/
│   │   │   └── ardetect/
│   │   └── cmad_datasets/        # Dataset loading implementations
│   ├── generation/               # Audio generation tools
│   │   ├── main.py               # Main generation script
│   │   ├── models/               # TTS and Voice Conversion models
│   │   │   ├── tts/              # Text-to-Speech models
│   │   │   │   ├── bark.py       # Bark TTS model
│   │   │   │   ├── elevenlabs_tts.py # ElevenLabs TTS API
│   │   │   │   ├── gemini_tts.py # Gemini TTS model
│   │   │   │   ├── gpt4omini_tts.py # GPT-4o Mini TTS
│   │   │   │   ├── melotts.py    # MeloTTS model
│   │   │   │   ├── tacotron2.py  # Tacotron2 model
│   │   │   │   ├── vits/         # VITS model implementation
│   │   │   │   ├── xttsv2.py     # XTTS v2 model
│   │   │   │   └── yourtts.py    # YourTTS model
│   │   │   └── vc/               # Voice Conversion models
│   │   │       ├── freevc.py     # FreeVC model
│   │   │       ├── knnvc.py      # kNN-VC model
│   │   │       └── openvoice.py  # OpenVoice model
│   │   ├── noise/                # Background noise samples
│   │   ├── samples/              # Sample audio files
│   │   │   ├── en.wav            # English sample
│   │   │   └── zh-cn.wav         # Chinese sample
│   │   └── transcription/        # Speech-to-text tools
│   │       ├── parakeet.py       # Parakeet transcription
│   │       └── voxtral.py        # Voxtral transcription
│   └── utils/                    # Utility functions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📊 Logging and Results

Evaluation results are automatically logged to:
- Console output with detailed metrics
- `logs/eval.log`: Comprehensive evaluation logs with rotation

Example output:
```
(AASIST on Interview) eer: 0.1234
(RawNet2 on PublicSpeech) eer: 0.2345
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use MTAD in your research, please cite:

```bibtex
@inproceedings{mtad2024,
  title={MTAD: Large-scale Multi-topic Audio Deepfake Benchmark},
  author={Your Name and Collaborators},
  booktitle={Conference Name},
  year={2024}
}
```

---

<div align="center">
Made with ❤️ for advancing audio deepfake detection research
</div>
