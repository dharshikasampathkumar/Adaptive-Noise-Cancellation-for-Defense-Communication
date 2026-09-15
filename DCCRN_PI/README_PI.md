# Raspberry Pi 3B+ Deployment — Experiment 3 Tiny DCCRN

This package provides the complete, isolated inference module for the **Experiment 3 Tiny DCCRN (Identity Residual)** speech enhancement model.

---

## 1. Quick Start Guide

### Prerequisites (Raspberry Pi 3B+ or Laptop)

```bash
# Core dependencies
pip install torch numpy soundfile
```

---

## 2. Integration Boundary (Team Hand-off Interface)

### Standard Whole-File Interface (`DCCRNInference`)

For processing complete audio files or buffers:

```python
import soundfile as sf
from dccrn_model import load_checkpoint, load_quantized_model
from dccrn_interface import DCCRNInference

# Load model (FP32 or INT8 Quantized)
model, stft_config = load_checkpoint("dccrn_fp32_best.pth", device="cpu")
# OR for quantized model:
# model, stft_config = load_quantized_model("dccrn_quantized.pth")

# Initialize inference engine
engine = DCCRNInference(model=model, stft_config=stft_config, device="cpu")

# Load noisy audio (16 kHz mono float32)
noisy_audio, sr = sf.read("input_noisy.wav", dtype="float32")

# Enhance audio
enhanced_audio = engine.process_waveform(noisy_audio)

# Save result
sf.write("output_enhanced.wav", enhanced_audio, sr)
```

---

### Real-Time Block-by-Block Streaming Interface (`DCCRNStreamer`)

For streaming audio frame-by-frame (e.g. from primary microphone stream before downstream NLMS/NLSM processing):

- **Frame Size**: 512 samples (32 ms at 16 kHz)
- **Hop Size**: 128 samples (8 ms at 16 kHz)

```python
import numpy as np
from dccrn_model import load_checkpoint
from dccrn_interface import DCCRNStreamer

# Load model & create streamer
model, stft_config = load_checkpoint("dccrn_fp32_best.pth", device="cpu")
streamer = DCCRNStreamer(model=model, stft_config=stft_config, device="cpu")

# Stream processing loop (example chunk of 512 samples)
# In real application, pass incoming 512-sample buffer from primary mic
mic_chunk = np.zeros(512, dtype=np.float32) 
enhanced_chunk = streamer.process_chunk(mic_chunk)

# Reset internal GRU hidden state and STFT buffer between separate recordings/calls
streamer.reset()
```

---

## 3. Verification & Testing Commands

To test deployment on laptop or Pi:

```bash
# Run baseline FP32 deployment test
python test_deployment.py --checkpoint dccrn_fp32_best.pth --audio-dir test_samples/

# Run quantized INT8 deployment test
python test_deployment.py --checkpoint dccrn_quantized.pth --quantized --audio-dir test_samples/
```

---

## 4. Performance Specifications

- **Sampling Rate**: 16,000 Hz (16 kHz)
- **FFT Size**: 512
- **Hop Size**: 128 (75% overlap)
- **Model Parameters**: ~4.41M
- **FP32 Size**: ~16.83 MB
- **INT8 Quantized Size**: ~4.35 MB
