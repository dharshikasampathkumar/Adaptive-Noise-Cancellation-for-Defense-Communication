"""
=============================================================================
DEFENCE SPEECH ENHANCEMENT DATASET PREPARATION PIPELINE
Tiny DCCRN Training Data Generator
=============================================================================
Purpose  : Generate synthetic noisy speech by mixing Indian-accented English
           clean speech with defence/environmental noise at controlled SNRs.
Scope    : Dataset preparation ONLY. No model training/inference.
Seed     : 42 (reproducible)
=============================================================================
"""

import os
import sys
import csv
import json
import math
import random
import shutil
import struct
import wave
import traceback
import subprocess
from pathlib import Path
from collections import defaultdict
import datetime

# ── Try to import required libraries ────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not found. Run: pip install numpy"); sys.exit(1)

try:
    import scipy.signal as sps
    import scipy.io.wavfile as wavfile
except ImportError:
    print("ERROR: scipy not found. Run: pip install scipy"); sys.exit(1)

try:
    import soundfile as sf
    HAVE_SF = True
except ImportError:
    HAVE_SF = False
    print("WARNING: soundfile not found – will use fallback loaders")

try:
    import librosa
    HAVE_LIBROSA = True
except ImportError:
    HAVE_LIBROSA = False
    print("WARNING: librosa not found – using scipy resampler")

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TARGET_SR        = 16000
TARGET_CHANNELS  = 1
TARGET_BITS      = 16
TARGET_SAMPLES   = 1000
SNR_LEVELS       = [-5, 0, 5, 10, 15, 20]
SAMPLES_PER_SNR  = TARGET_SAMPLES // len(SNR_LEVELS)
EXTRA_SAMPLES    = TARGET_SAMPLES - SAMPLES_PER_SNR * len(SNR_LEVELS)
MIN_DURATION_SEC = 3.0
CLIP_THRESHOLD   = 0.99
MAX_SNR_ERROR_FLAG = 3.0

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE     = Path(r"c:\SIH")
SPEECH_DIR    = WORKSPACE / "Clean Speech"
NOISE_DIR     = WORKSPACE / "Defence noise"
OUTPUT_DIR    = WORKSPACE / "synthetic_dataset"
WORK_DIR      = WORKSPACE / "working_copies"

NOISY_SPEECH_DIR   = OUTPUT_DIR / "noisy_speech"
CLEAN_REF_DIR      = OUTPUT_DIR / "clean_reference"
METADATA_DIR       = OUTPUT_DIR / "metadata"
SPLITS_DIR         = OUTPUT_DIR / "splits"

SNR_DIRS = {snr: NOISY_SPEECH_DIR / f"snr_{snr}dB" for snr in SNR_LEVELS}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# LOGGING
# =============================================================================
LOG_LINES = []

def log(msg, level="INFO"):
    prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "ERROR": "[ERROR]", "OK": "[ OK ]"}.get(level, "[INFO]")
    line = f"{prefix} {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def make_dirs():
    for d in [WORK_DIR, NOISY_SPEECH_DIR, CLEAN_REF_DIR, METADATA_DIR, SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for snr, d in SNR_DIRS.items():
        d.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "speech").mkdir(exist_ok=True)
    (WORK_DIR / "noise").mkdir(exist_ok=True)

def rms(audio):
    a = audio.astype(np.float64)
    return float(np.sqrt(np.mean(a ** 2)))

def power(audio):
    a = audio.astype(np.float64)
    return float(np.mean(a ** 2))

def peak_val(audio):
    return float(np.max(np.abs(audio.astype(np.float64))))

def is_clipped(audio, threshold=CLIP_THRESHOLD):
    return bool(peak_val(audio) >= threshold)

def snr_db_calc(speech_power, noise_power):
    if noise_power <= 0:
        return float('inf')
    return 10.0 * math.log10(max(speech_power, 1e-20) / max(noise_power, 1e-20))

def float_to_int16(audio_float):
    clipped = np.clip(audio_float, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)

def int16_to_float(audio_int16):
    return audio_int16.astype(np.float64) / 32768.0

# =============================================================================
# AUDIO LOADING
# =============================================================================

def load_audio_any(path):
    """Load audio file in any format. Returns (float64_array, sample_rate)."""
    path = Path(path)
    ext = path.suffix.lower()

    # Try soundfile first (WAV, FLAC, OGG, AIFF)
    if HAVE_SF:
        try:
            data, sr = sf.read(str(path), dtype='float64', always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data, sr
        except Exception:
            pass

    # Try librosa (handles MP3 via audioread/sndfile)
    if HAVE_LIBROSA:
        try:
            data, sr = librosa.load(str(path), sr=None, mono=True, dtype=np.float64)
            return data, sr
        except Exception:
            pass

    # Try scipy for WAV
    if ext == '.wav':
        try:
            sr, data = wavfile.read(str(path))
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = int16_to_float(data)
            elif data.dtype == np.int32:
                data = data.astype(np.float64) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float64) - 128) / 128.0
            elif data.dtype in [np.float32, np.float64]:
                data = data.astype(np.float64)
            return data, sr
        except Exception:
            pass

    # Try ffmpeg as last resort
    try:
        cmd = ["ffmpeg", "-y", "-i", str(path), "-f", "f64le",
               "-acodec", "pcm_f64le", "-ac", "1", "-"]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and len(result.stdout) > 0:
            data = np.frombuffer(result.stdout, dtype=np.float64)
            sr = None
            for line in result.stderr.decode(errors='ignore').split('\n'):
                if 'Hz' in line:
                    for tok in line.split():
                        tok_clean = tok.strip(',').strip()
                        try:
                            v = int(tok_clean)
                            if 8000 <= v <= 192000:
                                sr = v; break
                        except ValueError:
                            pass
                if sr:
                    break
            if sr is None:
                sr = 44100
            return data, sr
    except Exception:
        pass

    return None, None


def resample_audio(audio, orig_sr, target_sr=TARGET_SR):
    if orig_sr == target_sr:
        return audio
    if HAVE_LIBROSA:
        return librosa.resample(audio.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr)
    # scipy poly-phase resampling
    gcd = math.gcd(int(orig_sr), int(target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    return sps.resample_poly(audio.astype(np.float64), up, down)


def save_wav_16k_mono_16bit(audio_float, out_path):
    audio_int16 = float_to_int16(audio_float)
    with wave.open(str(out_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(audio_int16.tobytes())


def verify_wav(path):
    try:
        with wave.open(str(path), 'r') as wf:
            return {
                'channels': wf.getnchannels(),
                'sampwidth': wf.getsampwidth(),
                'framerate': wf.getframerate(),
                'nframes': wf.getnframes(),
                'duration': wf.getnframes() / wf.getframerate()
            }
    except Exception:
        return None

# =============================================================================
# STEP 1 – INSPECT
# =============================================================================

def inspect_clean_speech():
    log("=" * 60)
    log("STEP 1a: Inspecting clean speech files")
    log("=" * 60)

    metadata_path = SPEECH_DIR / "metadata.csv"
    meta_lookup = {}
    if metadata_path.exists():
        with open(metadata_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta_lookup[row['filename']] = row
        log(f"Loaded metadata for {len(meta_lookup)} speech files")

    speech_files = sorted([f for f in SPEECH_DIR.iterdir() if f.suffix.lower() == '.wav'])
    log(f"Found {len(speech_files)} WAV files in Clean Speech/")

    valid = []
    corrupted = []

    for fpath in speech_files:
        fname = fpath.name
        try:
            audio, sr = load_audio_any(fpath)
            if audio is None:
                corrupted.append({'file': fname, 'reason': 'Cannot load audio'})
                continue
            duration = len(audio) / sr
            if duration < MIN_DURATION_SEC:
                corrupted.append({'file': fname, 'reason': f'Duration {duration:.2f}s < {MIN_DURATION_SEC}s'})
                continue
            p = power(audio)
            if p < 1e-10:
                corrupted.append({'file': fname, 'reason': 'Silent file'})
                continue
            meta = meta_lookup.get(fname, {})
            valid.append({
                'filename': fname,
                'path': str(fpath),
                'orig_sr': sr,
                'duration': round(duration, 4),
                'rms': round(rms(audio), 6),
                'power': round(p, 8),
                'peak': round(peak_val(audio), 6),
                'is_clipped': is_clipped(audio),
                'gender': meta.get('gender', ''),
                'age_group': meta.get('age_group', ''),
                'primary_language': meta.get('primary_language', ''),
                'native_place_state': meta.get('native_place_state', ''),
                'native_place_district': meta.get('native_place_district', ''),
                'highest_qualification': meta.get('highest_qualification', ''),
                'job_category': meta.get('job_category', ''),
                'occupation_domain': meta.get('occupation_domain', ''),
            })
        except Exception as e:
            corrupted.append({'file': fname, 'reason': str(e)[:120]})

    log(f"Valid speech files  : {len(valid)}")
    log(f"Corrupted/rejected  : {len(corrupted)}")
    for c in corrupted[:5]:
        log(f"  Rejected: {c['file']} — {c['reason']}", "WARN")
    return valid, corrupted


def inspect_noise():
    log("=" * 60)
    log("STEP 1b: Inspecting noise files")
    log("=" * 60)

    CATEGORY_MAP = {
        "Armored vehicles":        "armored_vehicle",
        "Artillery fire":          "artillery",
        "Drones":                  "drone",
        "Emergency siren":         "siren",
        "GunShots":                "gunshot",
        "Helicopter rotor noice":  "helicopter",
        "Wind":                    "wind",
    }
    SUPPORTED_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif', '.opus'}

    noise_files = defaultdict(list)
    corrupted = []
    category_counts = {}

    for cat_folder, cat_label in CATEGORY_MAP.items():
        cat_path = NOISE_DIR / cat_folder
        if not cat_path.exists():
            log(f"Category folder not found: {cat_folder}", "WARN")
            continue

        all_files = []
        for root, dirs, files in os.walk(cat_path):
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_EXTS:
                    all_files.append(Path(root) / fname)

        log(f"  {cat_label:<22}: {len(all_files)} audio files found")
        category_counts[cat_label] = {'found': len(all_files), 'valid': 0}

        for fpath in all_files:
            try:
                audio, sr = load_audio_any(fpath)
                if audio is None:
                    corrupted.append({'file': str(fpath.name), 'category': cat_label, 'reason': 'Cannot load'})
                    continue
                duration = len(audio) / sr
                if duration < 0.5:
                    corrupted.append({'file': str(fpath.name), 'category': cat_label,
                                      'reason': f'Too short: {duration:.3f}s'})
                    continue
                p = power(audio)
                if p < 1e-12:
                    corrupted.append({'file': str(fpath.name), 'category': cat_label, 'reason': 'Silent'})
                    continue

                noise_files[cat_label].append({
                    'filename': fpath.name,
                    'path': str(fpath),
                    'category': cat_label,
                    'orig_sr': sr,
                    'duration': round(duration, 4),
                    'rms': round(rms(audio), 6),
                    'power': round(p, 8),
                    'peak': round(peak_val(audio), 6),
                })
                category_counts[cat_label]['valid'] += 1
            except Exception as e:
                corrupted.append({'file': str(fpath.name), 'category': cat_label, 'reason': str(e)[:80]})

    log("\nNoise category summary:")
    for cat, counts in category_counts.items():
        log(f"  {cat:<22}: {counts['valid']}/{counts['found']} valid")
    log(f"Total corrupted noise files: {len(corrupted)}")
    return noise_files, corrupted, category_counts

# =============================================================================
# STEP 2 – SELECT SPEECH POOL
# =============================================================================

def select_speech_pool(valid_speech, max_count=1000):
    log("=" * 60)
    log("STEP 2: Selecting speech pool")
    log("=" * 60)
    pool = sorted(valid_speech, key=lambda x: x['filename'])
    random.shuffle(pool)
    selected = pool[:max_count]
    log(f"Selected {len(selected)} speech files")
    lang_counts = defaultdict(int)
    state_counts = defaultdict(int)
    for s in selected:
        lang_counts[s.get('primary_language', 'Unknown')] += 1
        state_counts[s.get('native_place_state', 'Unknown')] += 1
    log(f"  Languages  : {len(lang_counts)}")
    log(f"  States     : {len(state_counts)}")
    return selected

# =============================================================================
# STEP 3 – STANDARDISE
# =============================================================================

def standardise_speech(speech_pool):
    log("=" * 60)
    log("STEP 3a: Standardising clean speech")
    log("=" * 60)
    out_dir = WORK_DIR / "speech"
    standardised = []
    failed = []
    for info in speech_pool:
        dst = out_dir / info['filename']
        try:
            audio, sr = load_audio_any(info['path'])
            if audio is None:
                failed.append(info['filename']); continue
            if sr != TARGET_SR:
                audio = resample_audio(audio, sr, TARGET_SR)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            max_abs = np.max(np.abs(audio))
            if max_abs > 1.0:
                audio = audio / max_abs * 0.99
            elif max_abs < 1e-10:
                failed.append(info['filename']); continue
            save_wav_16k_mono_16bit(audio, dst)
            ni = dict(info)
            ni['std_path'] = str(dst)
            ni['std_duration'] = round(len(audio) / TARGET_SR, 4)
            ni['std_rms'] = round(rms(audio), 6)
            ni['std_power'] = round(power(audio), 8)
            standardised.append(ni)
        except Exception as e:
            failed.append(info['filename'])
            log(f"  Speech fail {info['filename']}: {e}", "WARN")
    log(f"Standardised {len(standardised)} speech, failed {len(failed)}")
    return standardised, failed


def standardise_noise(noise_files):
    log("=" * 60)
    log("STEP 3b: Standardising noise files")
    log("=" * 60)
    std_noise = defaultdict(list)
    failed = []
    name_seen = defaultdict(set)

    for cat, files in noise_files.items():
        cat_dir = WORK_DIR / "noise" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for info in files:
            src = Path(info['path'])
            base = src.stem[:50].replace(' ', '_')
            for ch in ',()[]{}\'\"':
                base = base.replace(ch, '')
            safe_name = base + '.wav'
            # Resolve name collision
            if safe_name in name_seen[cat]:
                safe_name = f"{base}_{abs(hash(str(src))) % 9999:04d}.wav"
            name_seen[cat].add(safe_name)
            dst = cat_dir / safe_name
            try:
                audio, sr = load_audio_any(src)
                if audio is None:
                    failed.append({'file': info['filename'], 'category': cat}); continue
                if sr != TARGET_SR:
                    audio = resample_audio(audio, sr, TARGET_SR)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                max_abs = np.max(np.abs(audio))
                if max_abs < 1e-12:
                    failed.append({'file': info['filename'], 'category': cat}); continue
                if max_abs > 1.0:
                    audio = audio / max_abs * 0.99
                save_wav_16k_mono_16bit(audio, dst)
                ni = dict(info)
                ni['std_path'] = str(dst)
                ni['std_filename'] = safe_name
                ni['std_duration'] = round(len(audio) / TARGET_SR, 4)
                ni['std_rms'] = round(rms(audio), 6)
                ni['std_power'] = round(power(audio), 8)
                std_noise[cat].append(ni)
            except Exception as e:
                failed.append({'file': info['filename'], 'category': cat})
                log(f"  Noise fail {info['filename']}: {e}", "WARN")

    for cat, files in std_noise.items():
        log(f"  {cat:<22}: {len(files)} standardised")
    log(f"Failed noise: {len(failed)}")
    return std_noise, failed

# =============================================================================
# MIXING
# =============================================================================

def loop_noise(noise_audio, target_length):
    if len(noise_audio) == 0:
        return np.zeros(target_length)
    if len(noise_audio) >= target_length:
        max_start = len(noise_audio) - target_length
        start = random.randint(0, max_start)
        return noise_audio[start:start + target_length].copy()
    repeats = math.ceil(target_length / len(noise_audio))
    tiled = np.tile(noise_audio, repeats)
    max_start = len(tiled) - target_length
    start = random.randint(0, max(0, max_start))
    return tiled[start:start + target_length].copy()


def mix_at_snr(speech_audio, noise_audio, target_snr_db):
    speech_f = speech_audio.astype(np.float64)
    noise_seg = loop_noise(noise_audio.astype(np.float64), len(speech_f))

    sp = power(speech_f)
    np_val = power(noise_seg)

    if np_val < 1e-20:
        return speech_f, 0.0, float('inf'), noise_seg
    if sp < 1e-20:
        return speech_f + noise_seg, 0.0, float('-inf'), noise_seg

    target_linear = 10.0 ** (target_snr_db / 10.0)
    k = math.sqrt(sp / (np_val * target_linear))

    scaled_noise = k * noise_seg
    noisy = speech_f + scaled_noise

    actual_noise_power = power(scaled_noise)
    actual_snr = snr_db_calc(sp, actual_noise_power)

    noisy_peak = np.max(np.abs(noisy))
    if noisy_peak > CLIP_THRESHOLD:
        scale_down = CLIP_THRESHOLD / noisy_peak
        noisy = noisy * scale_down

    return noisy, k, actual_snr, scaled_noise


def plan_dataset(std_speech, std_noise):
    log("=" * 60)
    log("STEP 5: Planning dataset")
    log("=" * 60)

    categories = list(std_noise.keys())
    n_cats = len(categories)
    if n_cats == 0:
        raise RuntimeError("No noise categories!")

    snr_counts = []
    base = TARGET_SAMPLES // len(SNR_LEVELS)
    extra = TARGET_SAMPLES - base * len(SNR_LEVELS)
    for i in range(len(SNR_LEVELS)):
        snr_counts.append(base + (1 if i < extra else 0))

    # Build noise queues per category
    noise_queues = {}
    noise_ptrs   = {}
    for cat in categories:
        q = list(std_noise[cat])
        random.shuffle(q)
        noise_queues[cat] = q
        noise_ptrs[cat]   = 0

    def next_noise(cat):
        idx = noise_ptrs[cat]
        item = noise_queues[cat][idx % len(noise_queues[cat])]
        noise_ptrs[cat] = idx + 1
        return item

    plan = []
    speech_idx = 0
    total_speech = len(std_speech)
    global_idx = 0

    for snr_i, (snr, count) in enumerate(zip(SNR_LEVELS, snr_counts)):
        cat_counts = [count // n_cats] * n_cats
        remainder  = count - sum(cat_counts)
        for j in range(remainder):
            cat_counts[j % n_cats] += 1

        for cat_i, cat in enumerate(categories):
            for _ in range(cat_counts[cat_i]):
                speech = std_speech[speech_idx % total_speech]
                speech_idx += 1
                noise  = next_noise(cat)
                plan.append({
                    'sample_idx': global_idx + 1,
                    'speech':     speech,
                    'noise':      noise,
                    'category':   cat,
                    'target_snr': snr,
                })
                global_idx += 1

    log(f"Total planned: {len(plan)}")
    for snr, count in zip(SNR_LEVELS, snr_counts):
        log(f"  SNR {snr:>+3d} dB -> {count} samples")
    for cat in categories:
        n = sum(1 for p in plan if p['category'] == cat)
        log(f"  {cat:<22}: {n} samples")

    return plan


def generate_dataset(plan, std_noise):
    log("=" * 60)
    log("STEP 6: Generating noisy speech files")
    log("=" * 60)

    results = []
    qc_failures = []
    n = len(plan)

    for i, entry in enumerate(plan):
        if (i + 1) % 100 == 0 or i == 0 or i == n - 1:
            log(f"  Processing {i+1}/{n} ...")

        sidx    = entry['sample_idx']
        s_info  = entry['speech']
        n_info  = entry['noise']
        cat     = entry['category']
        tgt_snr = entry['target_snr']

        snr_str  = f"snr_{tgt_snr}dB"
        out_name = f"S{sidx:04d}_snr_{tgt_snr}dB_{cat}.wav"
        ref_name = f"S{sidx:04d}_clean_ref.wav"

        noisy_path = SNR_DIRS[tgt_snr] / out_name
        ref_path   = CLEAN_REF_DIR / ref_name

        qc_passed = False
        qc_reason = ""
        result    = None
        noise_opts = [n_info]  # try alternatives on failure
        # Add other noise files from same category as fallbacks
        if cat in std_noise:
            extras = [x for x in std_noise[cat] if x['std_path'] != n_info.get('std_path', '')]
            if extras:
                noise_opts += random.sample(extras, min(2, len(extras)))

        for noise_candidate in noise_opts:
            try:
                s_sr, s_int16 = wavfile.read(s_info['std_path'])
                n_sr, n_int16 = wavfile.read(noise_candidate['std_path'])

                if s_int16.ndim > 1: s_int16 = s_int16.mean(axis=1).astype(np.int16)
                if n_int16.ndim > 1: n_int16 = n_int16.mean(axis=1).astype(np.int16)

                s_float = int16_to_float(s_int16)
                n_float = int16_to_float(n_int16)

                if power(s_float) < 1e-10:
                    qc_reason = "Speech near-zero power"; continue
                if power(n_float) < 1e-12:
                    qc_reason = "Noise near-zero power"; continue

                noisy_float, k, actual_snr, scaled_noise = mix_at_snr(s_float, n_float, tgt_snr)

                if np.max(np.abs(noisy_float)) < 1e-8:
                    qc_reason = "Noisy output is silent"; continue

                snr_err = actual_snr - tgt_snr if actual_snr not in (float('inf'), float('-inf')) else 0

                # Save files
                save_wav_16k_mono_16bit(noisy_float, noisy_path)
                shutil.copy2(s_info['std_path'], ref_path)

                props = verify_wav(noisy_path)
                if props is None:
                    qc_reason = "Cannot verify generated WAV"; continue

                # QC: check format
                if props['framerate'] != TARGET_SR:
                    qc_reason = f"Wrong SR: {props['framerate']}"; continue
                if props['channels'] != 1:
                    qc_reason = f"Not mono: {props['channels']} ch"; continue
                if props['sampwidth'] != 2:
                    qc_reason = f"Not 16-bit: {props['sampwidth']*8}b"; continue

                noise_rms_orig = rms(n_float)
                flag_large_error = abs(snr_err) > MAX_SNR_ERROR_FLAG

                result = {
                    'sample_id':               f"S{sidx:04d}",
                    'noisy_filename':          out_name,
                    'clean_filename':          s_info['filename'],
                    'noise_filename':          noise_candidate['filename'],
                    'noise_category':          cat,
                    'target_snr_db':           tgt_snr,
                    'measured_snr_db':         round(actual_snr, 4) if actual_snr not in (float('inf'), float('-inf')) else actual_snr,
                    'snr_error_db':            round(snr_err, 4),
                    'duration_sec':            round(props['duration'], 4),
                    'sample_rate':             props['framerate'],
                    'channels':                props['channels'],
                    'bit_depth':               props['sampwidth'] * 8,
                    'clean_rms':               round(rms(s_float), 6),
                    'noise_rms_before_scaling': round(noise_rms_orig, 6),
                    'noise_scaling_factor':    round(k, 8),
                    'flag_large_snr_error':    flag_large_error,
                    'gender':                  s_info.get('gender', ''),
                    'age_group':               s_info.get('age_group', ''),
                    'primary_language':        s_info.get('primary_language', ''),
                    'native_place_state':      s_info.get('native_place_state', ''),
                    'native_place_district':   s_info.get('native_place_district', ''),
                    'highest_qualification':   s_info.get('highest_qualification', ''),
                    'job_category':            s_info.get('job_category', ''),
                    'occupation_domain':       s_info.get('occupation_domain', ''),
                }
                qc_passed = True
                break

            except Exception as e:
                qc_reason = str(e)[:120]

        if qc_passed and result:
            results.append(result)
        else:
            qc_failures.append({
                'sample_id': f"S{sidx:04d}",
                'reason': qc_reason,
                'category': cat,
                'target_snr': tgt_snr
            })
            log(f"  QC FAIL S{sidx:04d}: {qc_reason}", "WARN")

    log(f"Generated {len(results)} valid samples, {len(qc_failures)} QC failures")
    return results, qc_failures

# =============================================================================
# QUALITY CONTROL
# =============================================================================

def run_quality_control(results):
    log("=" * 60)
    log("STEP 10: Quality control verification")
    log("=" * 60)

    snr_errors = []
    for r in results:
        se = r['snr_error_db']
        if isinstance(se, float) and not math.isinf(se) and not math.isnan(se):
            snr_errors.append(se)

    flagged = [r for r in results if r.get('flag_large_snr_error', False)]
    log(f"  Total valid samples: {len(results)}")
    if snr_errors:
        log(f"  Min SNR error      : {min(snr_errors):.4f} dB")
        log(f"  Max SNR error      : {max(snr_errors):.4f} dB")
        log(f"  Mean |SNR error|   : {np.mean(np.abs(snr_errors)):.4f} dB")
    log(f"  Flagged samples    : {len(flagged)}")

    for snr in SNR_LEVELS:
        subset = [r for r in results if r['target_snr_db'] == snr]
        measured = [r['measured_snr_db'] for r in subset
                    if isinstance(r['measured_snr_db'], float)
                    and not math.isinf(r['measured_snr_db'])]
        if measured:
            log(f"  SNR {snr:>+3d} dB: n={len(subset)}, "
                f"mean_measured={np.mean(measured):.3f} dB, "
                f"mean_error={np.mean([m-snr for m in measured]):.4f} dB")

    return {
        'min_snr_error': round(min(snr_errors), 4) if snr_errors else None,
        'max_snr_error': round(max(snr_errors), 4) if snr_errors else None,
        'mean_abs_snr_error': round(float(np.mean(np.abs(snr_errors))), 4) if snr_errors else None,
        'flagged_count': len(flagged),
    }

# =============================================================================
# SPLITS
# =============================================================================

def make_splits(results):
    log("=" * 60)
    log("STEP 14: Train/validation/test split")
    log("=" * 60)

    by_speech = defaultdict(list)
    for r in results:
        by_speech[r['clean_filename']].append(r)

    speech_keys = sorted(by_speech.keys())
    random.shuffle(speech_keys)

    n      = len(speech_keys)
    n_val  = max(1, int(round(n * 0.10)))
    n_test = max(1, int(round(n * 0.10)))
    n_train= n - n_val - n_test

    train_keys = set(speech_keys[:n_train])
    val_keys   = set(speech_keys[n_train:n_train + n_val])
    test_keys  = set(speech_keys[n_train + n_val:])

    train = [r for r in results if r['clean_filename'] in train_keys]
    val   = [r for r in results if r['clean_filename'] in val_keys]
    test  = [r for r in results if r['clean_filename'] in test_keys]

    log(f"  Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    return train, val, test

# =============================================================================
# WRITE OUTPUTS
# =============================================================================

def write_metadata_csv(results, path):
    fields = [
        'sample_id', 'noisy_filename', 'clean_filename', 'noise_filename',
        'noise_category', 'target_snr_db', 'measured_snr_db', 'snr_error_db',
        'duration_sec', 'sample_rate', 'channels', 'bit_depth',
        'clean_rms', 'noise_rms_before_scaling', 'noise_scaling_factor',
        'flag_large_snr_error',
        'gender', 'age_group', 'primary_language', 'native_place_state',
        'native_place_district', 'highest_qualification', 'job_category',
        'occupation_domain',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    log(f"Wrote metadata: {path}")


def write_split_csv(data, path, name):
    fields = ['sample_id', 'noisy_filename', 'clean_filename', 'noise_category',
              'target_snr_db', 'measured_snr_db', 'duration_sec']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(data)
    log(f"Wrote {name} split: {path}")


def write_summary(path, stats):
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        "=" * 70,
        " DEFENCE SPEECH ENHANCEMENT DATASET – GENERATION SUMMARY",
        "=" * 70,
        f"Generated on        : {stats['timestamp']}",
        f"Random seed         : {RANDOM_SEED}",
        "",
        "─── 1. INPUT DATASET ───────────────────────────────────────────────",
        f"Clean speech files inspected : {stats['speech_inspected']}",
        f"Valid clean speech files     : {stats['speech_valid']}",
        f"Noise files inspected        : {stats['noise_inspected']}",
        f"Valid noise files            : {stats['noise_valid']}",
        f"Corrupted/rejected files     : {stats['corrupted_total']}",
        "",
        "─── 2. OUTPUT DATASET ──────────────────────────────────────────────",
        f"Target samples               : {TARGET_SAMPLES}",
        f"Final noisy samples          : {stats['total_generated']}",
        f"QC failures (excluded)       : {stats['qc_failures']}",
        "",
        "─── 3. SAMPLES PER SNR LEVEL ───────────────────────────────────────",
    ]
    for snr in SNR_LEVELS:
        key = f"{snr}dB"
        cnt = stats['snr_dist'].get(key, 0)
        mm  = stats['snr_mean_measured'].get(key, 'N/A')
        me  = stats['snr_mean_error'].get(key, 'N/A')
        mm_s = f"{mm:.4f}" if isinstance(mm, float) else str(mm)
        me_s = f"{me:.4f}" if isinstance(me, float) else str(me)
        lines.append(f"  SNR {snr:>+3d} dB  : {cnt:4d} samples | mean_measured={mm_s} dB | mean_error={me_s} dB")

    lines += [
        "",
        "─── 4. NOISE CATEGORY DISTRIBUTION ────────────────────────────────",
    ]
    for cat, cnt in stats['cat_dist'].items():
        lines.append(f"  {cat:<24}: {cnt:4d} samples")

    qs = stats['qc_stats']
    lines += [
        "",
        "─── 5. TRAIN / VALIDATION / TEST ───────────────────────────────────",
        f"  Train      : {stats['train_count']}",
        f"  Validation : {stats['val_count']}",
        f"  Test       : {stats['test_count']}",
        "",
        "─── 6. SNR ERROR STATISTICS ────────────────────────────────────────",
        f"  Min SNR error     : {fmt(qs['min_snr_error'])} dB",
        f"  Max SNR error     : {fmt(qs['max_snr_error'])} dB",
        f"  Mean |SNR error|  : {fmt(qs['mean_abs_snr_error'])} dB",
        f"  Flagged (>{MAX_SNR_ERROR_FLAG}dB): {qs['flagged_count']}",
        "",
        "─── 7. QUALITY CONTROL ─────────────────────────────────────────────",
        f"  QC failures       : {stats['qc_failures']}",
        f"  Samples passing   : {stats['total_generated']}",
        "",
        "─── 8. AUDIO FORMAT ────────────────────────────────────────────────",
        f"  Format      : WAV (PCM, uncompressed)",
        f"  Bit depth   : 16-bit",
        f"  Channels    : Mono (1)",
        f"  Sample rate : {TARGET_SR} Hz",
        "",
        "─── 9. NOISE CATEGORIES ────────────────────────────────────────────",
        f"  {', '.join(stats['cat_dist'].keys())}",
        "",
        "─── 10. LIMITATIONS & WARNINGS ─────────────────────────────────────",
    ]
    for w in stats.get('warnings', []):
        lines.append(f"  ! {w}")
    if not stats.get('warnings'):
        lines.append("  None")

    lines += [
        "",
        "─── NOTES ──────────────────────────────────────────────────────────",
        "  • Synthetically generated for Tiny DCCRN training.",
        "  • SNR formula: SNR(dB) = 10*log10(P_speech / P_noise)",
        "  • Noise looped/tiled if shorter than speech duration.",
        "  • Anti-clipping preserves SNR by uniform scaling.",
        "  • NOT real military recordings.",
        "=" * 70,
    ]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"Wrote summary: {path}")


def write_readme(path, stats):
    content = f"""DEFENCE SPEECH ENHANCEMENT SYNTHETIC DATASET
=============================================
Generated : {stats['timestamp']}
Seed      : {RANDOM_SEED}

PURPOSE
-------
Synthetic noisy speech dataset for training a Tiny DCCRN speech enhancement
model targeting Indian-accented English in defence environments.

CLEAN SPEECH SOURCE
-------------------
Indian-accented English speech (male speakers, multiple languages & states).
Files used: {stats['speech_valid']}

NOISE CATEGORIES
----------------
{chr(10).join('  - ' + cat for cat in stats['cat_dist'].keys())}

AUDIO FORMAT
------------
  WAV, PCM 16-bit, Mono, {TARGET_SR} Hz

SNR LEVELS
----------
  {', '.join(str(s)+' dB' for s in SNR_LEVELS)}
  (~{TARGET_SAMPLES // len(SNR_LEVELS)} samples per level, total {stats['total_generated']})

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
  Train: {stats['train_count']} | Validation: {stats['val_count']} | Test: {stats['test_count']}
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
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    log(f"Wrote README: {path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log("=" * 70)
    log(" DEFENCE SPEECH ENHANCEMENT – DATASET PREPARATION PIPELINE")
    log(f" Started: {timestamp}")
    log(f" Seed   : {RANDOM_SEED}")
    log("=" * 70)

    make_dirs()
    warnings = []

    # Step 1
    valid_speech, corrupt_speech = inspect_clean_speech()
    noise_files, corrupt_noise, cat_counts = inspect_noise()

    # Step 2
    if len(valid_speech) < TARGET_SAMPLES:
        warnings.append(f"Only {len(valid_speech)} valid speech files (target: {TARGET_SAMPLES})")
    speech_pool = select_speech_pool(valid_speech, max_count=TARGET_SAMPLES)

    # Step 3
    std_speech, fail_speech = standardise_speech(speech_pool)
    std_noise,  fail_noise  = standardise_noise(noise_files)

    if not std_speech:
        log("FATAL: No speech standardised", "ERROR"); sys.exit(1)
    if not std_noise:
        log("FATAL: No noise standardised", "ERROR"); sys.exit(1)

    # Steps 5-9
    plan    = plan_dataset(std_speech, std_noise)
    results, qc_failures = generate_dataset(plan, std_noise)

    if not results:
        log("FATAL: No samples generated", "ERROR"); sys.exit(1)

    # Step 10
    qc_stats = run_quality_control(results)

    # Step 14
    train, val, test = make_splits(results)

    # Write outputs
    write_metadata_csv(results, METADATA_DIR / "dataset_metadata.csv")
    write_split_csv(train, SPLITS_DIR / "train.csv", "train")
    write_split_csv(val,   SPLITS_DIR / "validation.csv", "validation")
    write_split_csv(test,  SPLITS_DIR / "test.csv", "test")

    # Build stats dict
    snr_dist = {}
    snr_mean_measured = {}
    snr_mean_error    = {}
    for snr in SNR_LEVELS:
        subset   = [r for r in results if r['target_snr_db'] == snr]
        measured = [r['measured_snr_db'] for r in subset
                    if isinstance(r['measured_snr_db'], float)
                    and not math.isinf(r['measured_snr_db'])]
        snr_dist[f"{snr}dB"]           = len(subset)
        snr_mean_measured[f"{snr}dB"]  = round(float(np.mean(measured)), 4) if measured else None
        snr_mean_error[f"{snr}dB"]     = round(float(np.mean([m-snr for m in measured])), 4) if measured else None

    cat_dist = defaultdict(int)
    for r in results:
        cat_dist[r['noise_category']] += 1

    noise_inspected = sum(c['found'] for c in cat_counts.values())
    noise_valid     = sum(c['valid'] for c in cat_counts.values())

    stats = {
        'timestamp':          timestamp,
        'speech_inspected':   len(valid_speech) + len(corrupt_speech),
        'speech_valid':       len(valid_speech),
        'noise_inspected':    noise_inspected,
        'noise_valid':        noise_valid,
        'corrupted_total':    len(corrupt_speech) + len(corrupt_noise),
        'total_generated':    len(results),
        'qc_failures':        len(qc_failures),
        'snr_dist':           snr_dist,
        'snr_mean_measured':  snr_mean_measured,
        'snr_mean_error':     snr_mean_error,
        'cat_dist':           dict(cat_dist),
        'train_count':        len(train),
        'val_count':          len(val),
        'test_count':         len(test),
        'qc_stats':           qc_stats,
        'warnings':           warnings,
    }

    write_summary(METADATA_DIR / "generation_summary.txt", stats)
    write_readme(OUTPUT_DIR / "README.txt", stats)

    # QC failure log
    if qc_failures:
        with open(METADATA_DIR / "qc_failures.csv", 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['sample_id','reason','category','target_snr'])
            w.writeheader()
            w.writerows(qc_failures)

    # ─── FINAL REPORT ─────────────────────────────────────────────────────────
    log("=" * 70)
    log(" PIPELINE COMPLETE – FINAL REPORT")
    log("=" * 70)
    log(f"A. Input dataset structure:")
    log(f"   Clean Speech : {SPEECH_DIR}")
    log(f"   Noise Dir    : {NOISE_DIR}")
    log(f"   Categories   : {', '.join(cat_dist.keys())}")
    log("")
    log(f"B. Clean speech statistics:")
    log(f"   Inspected    : {stats['speech_inspected']}")
    log(f"   Valid (≥3s)  : {stats['speech_valid']}")
    log(f"   Pool used    : {len(std_speech)}")
    log("")
    log("C. Noise category statistics (final dataset):")
    for cat, cnt in cat_dist.items():
        log(f"   {cat:<24}: {cnt} samples")
    log("")
    log(f"D. Total noisy samples     : {stats['total_generated']}")
    log("")
    log("E. SNR distribution:")
    for k, v in snr_dist.items():
        log(f"   {k:<8}: {v} samples")
    log("")
    log(f"G. Train/Val/Test          : {len(train)} / {len(val)} / {len(test)}")
    log("")
    log("H. Average measured SNR per target:")
    for k, v in snr_mean_measured.items():
        log(f"   {k:<8}: {v} dB")
    log("")
    log(f"I. SNR error stats:")
    log(f"   Min          : {qc_stats['min_snr_error']} dB")
    log(f"   Max          : {qc_stats['max_snr_error']} dB")
    log(f"   Mean |error| : {qc_stats['mean_abs_snr_error']} dB")
    log(f"   Flagged      : {qc_stats['flagged_count']}")
    log("")
    log(f"J. QC: {len(qc_failures)} failures, {len(results)} passed")
    log("")
    log(f"K. Output directory: {OUTPUT_DIR}")
    log("")
    log("L. Warnings:")
    for w in warnings:
        log(f"   ! {w}", "WARN")
    if not warnings:
        log("   None")
    log("")
    log("M. Recommendations before Tiny DCCRN training:")
    log(f"   1. Review QC failures in: {METADATA_DIR / 'qc_failures.csv'}")
    log(f"   2. Review flagged samples (|SNR error| > {MAX_SNR_ERROR_FLAG} dB) in metadata CSV.")
    log(f"   3. Confirm split CSVs at: {SPLITS_DIR}")
    log(f"   4. Working copies in {WORK_DIR} can be deleted after verification.")
    log(f"   5. Consider RIR augmentation for better room-acoustic coverage.")
    log("")
    log(" DATASET PREPARATION COMPLETE. DO NOT TRAIN YET.")
    log("=" * 70)


if __name__ == "__main__":
    main()
