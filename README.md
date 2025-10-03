# AI Text-to-Speech


[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> A simple desktop application for generating high-quality AI speech from text using Suno's Bark model, featuring a modern dark-themed interface with real-time waveform visualization. Despite the models ability to produce decent sounding audio, the system is not good with continuity or consistent preset selection/seeding. 

![Untitled video - Made with Clipchamp (12)](https://github.com/user-attachments/assets/6425fdbf-2c03-47b1-b1c2-409a3a024beb)

<img width="800" height="623" alt="Screenshot 2025-10-02 101413" src="https://github.com/user-attachments/assets/347f5119-74f0-48a2-80f1-a47cdc59953e" />


## Features

- **High-Quality Speech Generation** - Powered by Suno's Bark transformer model
- **Real-Time Waveform Visualization** - See your audio as it's generated
- **Modern Dark UI** - Sleek, professional interface with custom title bar
- **Multiple Voice Presets** - 10+ English voice options to choose from
- **GPU Acceleration** - CUDA support for faster generation
- **Audio Playback Controls** - Built-in player with seek functionality
- **Export to WAV** - Save your generated audio in high quality
- **Sentence Counter** - Track text length with automatic tokenization
- **Adjustable Sample Rate** - Customize audio quality (default 24kHz)
- **Non-Speech Sounds** - Support for laughter, music, sighs, and more

## Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster generation)
- 4GB+ RAM (8GB+ recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dovvnloading/Suno-Bark.git
   cd Suno-Bark
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python echoweave.py
   ```

### Dependencies

```
PySide6>=6.4.0
transformers>=4.30.0
torch>=2.0.0
soundfile>=0.12.0
sounddevice>=0.4.6
pydub>=0.25.1
nltk>=3.8
numpy>=1.24.0
```

## Usage

1. **Enter Your Text** - Type or paste the text you want to convert to speech
2. **Select Voice** - Choose from available English voice presets
3. **Configure Settings** - Adjust sample rate if desired (default: 24000 Hz)
4. **Generate** - Click "Generate Audio" and wait for processing
5. **Preview** - Use the waveform viewer and playback controls
6. **Export** - Save your audio as a WAV file

### Special Text Markers

EchoWeave supports special markers for non-speech sounds:

- `[laughter]` or `[laughs]` - Adds laughter
- `[sighs]` - Adds a sigh
- `[music]` - Adds musical notes
- `[gasps]` - Adds a gasp
- `[clears throat]` - Adds throat clearing


The complete Visual Studio project files and all source code are included in the main repository for easy access and development.

## Interface Components

- **Custom Title Bar** - Frameless window with minimize, maximize, and close buttons
- **Text Editor** - Multi-line input with syntax support for special markers
- **Settings Panel** - Voice selection and sample rate configuration
- **Waveform Viewer** - Real-time audio visualization with playback indicator
- **Seek Slider** - Interactive timeline scrubbing
- **Control Buttons** - Generate, Play/Stop, and Save functionality
- **Status Bar** - Real-time feedback and progress indication

## Technical Details

- **Framework**: PySide6 (Qt for Python)
- **ML Model**: Suno Bark (HuggingFace Transformers)
- **Audio Processing**: soundfile, sounddevice, pydub
- **Text Processing**: NLTK for sentence tokenization
- **Visualization**: Custom QPainter-based waveform rendering
- **Threading**: QThread for non-blocking audio generation and playback

## Advanced Configuration

### GPU Memory Optimization

The application automatically enables CPU offloading for CUDA devices to manage memory efficiently. For systems with limited GPU memory, the model uses FP16 precision when available.

### Custom Sample Rates

While 24kHz is recommended, you can adjust the sample rate for different use cases:
- **16kHz** - Lower quality, smaller file sizes
- **24kHz** - Balanced quality and size (default)
- **48kHz** - Higher quality, larger file sizes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Suno AI](https://github.com/suno-ai/bark) for the Bark text-to-speech model
- [HuggingFace](https://huggingface.co/) for the transformers library
- [Qt/PySide6](https://www.qt.io/qt-for-python) for the UI framework

## Contact

Project Link: [https://github.com/dovvnloading/Suno-Bark](https://github.com/dovvnloading/Suno-Bark)

## Known Issues

- First-time model loading may take several minutes
- critical: No continuity of presets (the presets for this model do not seem to work)
- GPU memory usage can be high (4GB+ VRAM recommended)
- Long text inputs (50+ sentences) may take significant time to process


---

<p align="center">Made with care</p>
<p align="center">Star me on GitHub — it helps!</p>
