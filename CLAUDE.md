# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RunPod-based Docker container for LoRA training on diffusion/video models using [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe). Designed as a RunPod GPU pod template with interactive setup, automated captioning, and training orchestration.

## Architecture

**Execution flow on RunPod:**
1. `Dockerfile` builds image with CUDA 12.8.1 + diffusion-pipe dependencies (flash-attn excluded)
2. `src/start_script.sh` (entrypoint) clones this repo at runtime, runs `src/start.sh`
3. `src/start.sh` sets up `$NETWORK_VOLUME`, installs flash-attn (prebuilt wheel or source compile per GPU arch), starts Jupyter, moves files into place, updates TOML paths with `sed`
4. User runs `bash interactive_start_training.sh` which orchestrates the full workflow

**Two training pipelines:**
- **diffusion-pipe** (primary): `interactive_start_training.sh` → configures TOML → `deepspeed --num_gpus=1 train.py --deepspeed --config examples/$TOML_FILE`
- **Musubi** (alternative): `{wan2.2,z_image_musubi,qwen_image_musubi}_training/` directories each have `setup_and_train_*.sh` + `*_config.sh` pairs that generate TOML dynamically from bash configs

**Supported models** (each has a TOML in `toml_files/`):
- Flux (`flux.toml`), SDXL (`sdxl.toml`)
- Wan 1.3B (`wan13_video.toml`), Wan 14B T2V (`wan14b_t2v.toml`), Wan 14B I2V (`wan14b_i2v.toml`)
- Qwen (`qwen_toml.toml`), Z Image Turbo (`z_image_toml.toml`)

**Captioning layer:** `Captioning/JoyCaption/` for images, `Captioning/video_captioner.sh` for videos (uses Gemini API via TripleX)

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `NETWORK_VOLUME` | Persistent storage root. Set by `start.sh`: `/workspace/diffusion_pipe_working_folder` if `/workspace` exists, else `/diffusion_pipe_working_folder` |
| `HUGGING_FACE_TOKEN` | Required for Flux model downloads |
| `GEMINI_API_KEY` | Required for video captioning |
| `IS_DEV` / `is_dev` | Uses `dev` branch; also enables Qwen/Z-Image musubi training folders |
| `git_branch` | Custom branch override for repo clone |

## Configuration

- **TOML files** (`toml_files/`): Training configs with `[model]`, `[adapter]`, `[optimizer]` sections. `start.sh` rewrites paths at runtime via `sed` to use `$NETWORK_VOLUME`.
- **`dataset.toml`**: Controls resolutions, aspect ratio buckets, frame buckets, and dataset directory paths. Video datasets are commented out by default.
- **Musubi bash configs** (e.g., `z_image_musubi_config.sh`): Shell variables sourced by training scripts, converted to TOML at runtime.

## Build

```bash
docker build -t runpod-lora-trainer .
```

flash-attn is intentionally excluded from the Docker build and installed at container start by `src/start.sh` based on detected GPU architecture.

## Runtime Directory Structure (on RunPod)

```
$NETWORK_VOLUME/
├── image_dataset_here/     # User's training images + .txt captions
├── video_dataset_here/     # User's training videos + .txt captions
├── models/                 # Downloaded model weights
├── diffusion_pipe/         # Cloned diffusion-pipe with train.py
│   └── examples/           # Active TOML configs copied here
├── training_outputs/       # LoRA output directories
└── logs/                   # Download, captioning, flash-attn logs
```

## Notes

- Training launch command pattern: `NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/<model>.toml`
- `start_training_scripts/` contains direct-launch scripts that bypass interactive mode
- `interactive_start_training.sh` is ~1200 lines and handles model download, captioning, config editing, and training launch with background process management
- The repo is cloned fresh on each pod start via `start_script.sh`, so local changes on a pod are ephemeral unless on network volume
