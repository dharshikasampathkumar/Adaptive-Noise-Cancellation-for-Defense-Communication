      # Raspberry Pi 3B+ Deployment Package — Experiment 3 Dynamic INT8 DCCRN

      This package contains the complete, isolated inference module and testing suite for the **Experiment 3 Tiny DCCRN (Identity Residual)** speech enhancement model.

      ---

      ## Package Directory Layout

      ```text
      RaspberryPi_Development/
      ├── model/
      │   └── dccrn_quantized.pth       # Dynamic INT8 Quantized Checkpoint (4.52 MB)
      ├── src/
      │   ├── config.py                 # Audio & STFT parameters
      │   ├── dccrn_model.py            # Experiment 3 TinyDCCRN architecture
      │   └── dccrn_interface.py        # Offline (DCCRNInference) & Streaming (DCCRNStreamer)
      ├── tests/
      │   ├── test_pi_inference.py      # Offline WAV evaluation script
      │   └── test_streaming.py        # 512-sample real-time streaming test
      ├── requirements.txt              # Minimal Pi dependencies
      ├── README_PI.md                  # Integration & execution documentation
      └── run_pi_test.py                # Single-command Pi validation runner
      ```

      ---

      ## 1. System Preparation on Raspberry Pi 3B+

      ### Step 1.1: Verify 64-Bit OS Architecture

      PyTorch official ARM wheels require a **64-bit Linux OS** (aarch64 / arm64).

      Run on Raspberry Pi terminal:

      ```bash
      python3 --version
      uname -m
      ```

      *Required output for `uname -m`*: `aarch64` or `arm64`.

      ---

      ### Step 1.2: Install PyTorch & Dependencies

      Refer to official PyTorch installation guidance for ARM 64-bit devices:

      ```bash
      # Update system packages
      sudo apt-get update
      sudo apt-get install -y python3-pip libsndfile1

      # Install requirements
      pip3 install -r requirements.txt
      ```

      ---

      ## 2. Running Deployment Tests on Raspberry Pi

      Run the single-command validation launcher with any 16 kHz mono test WAV:

      ```bash
      python3 run_pi_test.py --wav sample.wav
      ```

      The runner executes:
      1. System hardware audit (OS, architecture, Python, PyTorch).
      2. Offline WAV enhancement & diagnostics (load time, inference time, RTF, RAM usage).
      3. 512-sample streaming chunk evaluation (chunk latency, Max compute time, Streaming RTF).
      4. Prints the final classification: **REAL-TIME CAPABLE ON RASPBERRY PI 3B+** or **NOT REAL-TIME CAPABLE**.

      ---

      ## 3. Final Team Integration Interface

      For teammates integrating with primary microphone input and downstream NLMS / NLSM processing:

      ### Input / Output Specification

      - **Input format**: 16,000 Hz, 16-bit PCM converted to `float32`, mono, **512 samples** (32 ms).
      - **Output format**: Enhanced 16,000 Hz `float32`, mono, **512 samples**.

      ### Team Code Integration Example

      ```python
      import numpy as np
      from src.dccrn_model import load_quantized_model
      from src.dccrn_interface import DCCRNStreamer

      # 1. Initialize model & streamer once during startup
      model, _ = load_quantized_model("model/dccrn_quantized.pth")
      streamer = DCCRNStreamer(model=model, device="cpu")

      # 2. In audio callback loop (512-sample incoming buffer from primary mic):
      def on_primary_mic_audio(primary_mic_chunk_512: np.ndarray):
      # Process through DCCRN INT8
      enhanced_chunk = streamer.process_chunk(primary_mic_chunk_512)
      
      # Pass enhanced_chunk to downstream NLMS/NLSM filter with reference mic
      nlms_filter.process(primary_enhanced=enhanced_chunk, reference_mic=ref_mic_chunk)
      ```

      ---

      ## 4. End-to-End System Audio Pipeline

      ```text
      Primary INMP441 Mic
            ↓
      16 kHz / Mono / 512 Samples
            ↓
      DCCRN INT8 (dccrn_quantized.pth)
            ↓
      Enhanced Speech (512 Samples)
            ↓
      NLMS / NLSM Adaptive Filter ← Reference INMP441 Mic
            ↓
      DAC / Audio Output
            ↓
      Headset
      ```
