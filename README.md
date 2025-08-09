# CMAD: Comprehensive Multi-domain Audio Deepfake Benchmark

CMAD (Comprehensive Multi-domain Audio Deepfake benchmark) is a large-scale benchmark for evaluating audio deepfake detection methods across diverse domains and scenarios. This repository provides evaluation tools, baseline models, and comprehensive datasets for advancing research in audio deepfake detection.

## 🔥 Features

- **Multi-domain Coverage**: 13 diverse datasets spanning different audio domains (news, interviews, movies, audiobooks, etc.)
- **Comprehensive Evaluation**: Support for both cross-domain and in-domain evaluation scenarios
- **State-of-the-art Baselines**: 6 advanced audio deepfake detection models
- **Rich Audio Content**: Over 422 hours of audio data with balanced real and fake samples
- **Flexible Framework**: Easy-to-use evaluation pipeline with modular design

## 📊 Dataset Overview

CMAD includes 13 diverse datasets across multiple domains:

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

## 🏗️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 64GB+ free disk space for full dataset

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/wtalioy/CMAD.git
cd CMAD
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download datasets**: 
Place your dataset files in the `data/` directory following this structure:
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

### Basic Evaluation

Evaluate a baseline model on multiple datasets:

```bash
cd src/eval
python main.py --baseline aasist --dataset interview publicspeech --mode cross --metric eer
```

### Cross-domain Evaluation

Evaluate multiple baselines across domains (no training required):

```bash
python main.py \
    --baseline aasist rawnet2 res-tssdnet \
    --dataset phonecall publicspeech interview \
    --mode cross \
    --metric eer
```

### In-domain Evaluation

Train and evaluate on the same domain:

```bash
python main.py \
    --baseline aasist \
    --dataset interview \
    --mode in \
    --metric eer
```

### Training Only

Train a model without evaluation:

```bash
python main.py \
    --baseline aasist \
    --dataset interview \
    --mode in \
    --train_only
```

### Evaluation Only

Evaluate a pre-trained model:

```bash
python main.py \
    --baseline aasist \
    --dataset interview \
    --mode in \
    --eval_only
```

## 🎯 Available Baselines

CMAD includes 6 state-of-the-art audio deepfake detection models:

| Baseline | Description | Paper |
|----------|-------------|-------|
| **AASIST** | Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks | [ICASSP 2021](https://arxiv.org/abs/2110.01200) |
| **AASIST-L** | Large variant of AASIST | [ICASSP 2021](https://arxiv.org/abs/2110.01200) |
| **RawNet2** | End-to-end anti-spoofing using raw waveforms | [Interspeech 2020](https://arxiv.org/abs/2011.01108) |
| **Res-TSSDNet** | Residual Time-frequency Squeeze-and-Scale Network | [ICASSP 2023](https://arxiv.org/abs/2210.06694) |
| **Inc-TSSDNet** | Inception Time-frequency Squeeze-and-Scale Network | [ICASSP 2023](https://arxiv.org/abs/2210.06694) |
| **RawGAT-ST** | Graph Attention Networks with Spectro-Temporal features | [ICASSP 2022](https://arxiv.org/abs/2203.06028) |
| **ARDetect** | Maximum Mean Discrepancy based detection | [ICLR 2024](https://arxiv.org/abs/2309.15603) |

## 📈 Evaluation Metrics

CMAD supports the following evaluation metrics:

- **EER** (Equal Error Rate): Primary metric for audio deepfake detection

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
CMAD/
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

If you use CMAD in your research, please cite:

```bibtex
@inproceedings{cmad2024,
  title={CMAD: Comprehensive Multi-domain Audio Deepfake Benchmark},
  author={Your Name and Collaborators},
  booktitle={Conference Name},
  year={2024}
}
```

## 🔗 Links

- **Dataset on Hugging Face**: [Lioy/CMAD](https://huggingface.co/datasets/Lioy/CMAD)
- **Paper**: [Coming Soon]
- **Demo**: [Coming Soon]

---

<div align="center">
Made with ❤️ for advancing audio deepfake detection research
</div>
