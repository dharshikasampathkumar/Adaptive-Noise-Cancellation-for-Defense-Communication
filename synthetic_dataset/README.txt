DEFENCE SPEECH ENHANCEMENT SYNTHETIC DATASET
=============================================
Generated : 2026-09-11 23:51:31
Seed      : 42

PURPOSE
-------
Synthetic noisy speech dataset for training a Tiny DCCRN speech enhancement
model targeting Indian-accented English in defence environments.

CLEAN SPEECH SOURCE
-------------------
Indian-accented English speech (male speakers, multiple languages & states).
Files used: 1000

NOISE CATEGORIES
----------------
  - armored_vehicle
  - artillery
  - drone
  - siren
  - gunshot
  - helicopter
  - wind

AUDIO FORMAT
------------
  WAV, PCM 16-bit, Mono, 16000 Hz

SNR LEVELS
----------
  -5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB
  (~166 samples per level, total 1000)

DIRECTORY STRUCTURE
-------------------
  synthetic_dataset/
      noisy_speech/
          snr_-5dB/     snr_0dB/     snr_5dB/
          snr_10dB/     snr_15dB/    snr_20dB/
      clean_reference/
      metadata/
          dataset_metadata.csv
          generation_summary.txt
      splits/
          train.csv   validation.csv   test.csv
      README.txt

FILE NAMING
-----------
  Noisy : S<ID>_snr_<SNR>dB_<noise_category>.wav
  Clean : S<ID>_clean_ref.wav

MIXING METHODOLOGY
------------------
  1. All audio standardised to 16kHz/mono/16-bit WAV.
  2. Noise looped or trimmed to match speech duration.
  3. Scale factor: k = sqrt(P_speech / (P_noise * 10^(SNR/10)))
  4. NOISY = CLEAN + k * NOISE
  5. Anti-clip: uniform scale-down if peak > 0.99 (SNR preserved).
  6. Actual SNR independently measured after mixing.

TRAIN/VAL/TEST SPLIT
--------------------
  Train: 800 | Validation: 100 | Test: 100
  (Speaker-aware: same clean file not in multiple splits where possible)

LIMITATIONS
-----------
  - Synthetic mixing only; no real field conditions.
  - Some gunshot files very short; looped to match speech.
  - Speaker identity metadata not available beyond language/state.
  - NOT real military communication recordings.

DISCLAIMER
----------
  Synthetically generated for research/development purposes only.
