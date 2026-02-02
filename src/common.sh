#!/usr/bin/env bash
# common.sh — shared shell library for runpod-diffusion_pipe scripts
# Sourced at runtime from /usr/local/lib/runpod_common.sh
# (copied there by start_script.sh after cloning the repo)

########################################
# Colors
########################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
WHITE='\033[0;37m'
NC='\033[0m'

########################################
# Print helpers
########################################
print_header() {
    echo -e "${CYAN}================================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${CYAN}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

########################################
# GPU helpers
########################################
gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}'
    elif [ -n "${CUDA_VISIBLE_DEVICES-}" ] && [ "${CUDA_VISIBLE_DEVICES-}" != "" ]; then
        echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}'
    else
        echo 0
    fi
}

warn_blackwell_gpu() {
    if [ -f /tmp/gpu_arch_type ]; then
        local gpu_arch_type detected_gpu
        gpu_arch_type=$(cat /tmp/gpu_arch_type)
        detected_gpu=$(cat /tmp/detected_gpu 2>/dev/null || echo "Unknown")
        if [ "$gpu_arch_type" = "blackwell" ]; then
            echo ""
            echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════════${NC}"
            echo -e "${BOLD}${RED}⚠️  WARNING: BLACKWELL GPU DETECTED ⚠️${NC}"
            echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════════${NC}"
            echo -e "${BOLD}${RED}Detected GPU: $detected_gpu${NC}"
            echo -e "${BOLD}${RED}${NC}"
            echo -e "${BOLD}${RED}Blackwell GPUs (B100, B200, RTX 5090, etc.) are very new and${NC}"
            echo -e "${BOLD}${RED}may not be fully supported by all ML libraries yet.${NC}"
            echo -e "${BOLD}${RED}${NC}"
            echo -e "${BOLD}${RED}For best compatibility, use H100 or H200 GPUs.${NC}"
            echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════════${NC}"
            echo ""
            echo -n "Continuing in "
            for i in 10 9 8 7 6 5 4 3 2 1; do
                echo -n "$i.."
                sleep 1
            done
            echo ""
            echo ""
        fi
    fi
}

########################################
# CUDA compatibility check
########################################
check_cuda_compatibility() {
    local python_bin="${1:-python3}"
    "$python_bin" << 'PYTHON_EOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
        x = torch.randn(1, device='cuda')
        y = x * 2
        print("CUDA compatibility check passed")
    else:
        print("\n" + "="*70)
        print("CUDA NOT AVAILABLE")
        print("="*70)
        print("\nCUDA is not available on this system.")
        print("This script requires CUDA to run.")
        print("\nSOLUTION:")
        print("  Please deploy with CUDA 12.8 when selecting your GPU on RunPod")
        print("  This template requires CUDA 12.8")
        print("\n" + "="*70)
        sys.exit(1)
except RuntimeError as e:
    error_msg = str(e).lower()
    if "no kernel image" in error_msg or "cuda error" in error_msg:
        print("\n" + "="*70)
        print("CUDA KERNEL COMPATIBILITY ERROR")
        print("="*70)
        print("\nThis error occurs when your GPU architecture is not supported")
        print("by the installed CUDA kernels. This typically happens when:")
        print("  • Your GPU model is older or different from what was expected")
        print("  • The PyTorch/CUDA build doesn't include kernels for your GPU")
        print("\nSOLUTIONS:")
        print("  1. Use a newer GPU model (recommended):")
        print("     • H100 or H200 GPUs are recommended for best compatibility")
        print("  2. Ensure correct CUDA version:")
        print("     • Filter for CUDA 12.8 when selecting your GPU on RunPod")
        print("     • This template requires CUDA 12.8")
        print("\n" + "="*70)
        sys.exit(1)
    else:
        raise
PYTHON_EOF
    if [ $? -ne 0 ]; then
        exit 1
    fi
}

########################################
# Numeric CSV normalization
########################################
normalize_numeric_csv() {
    local s="$1"
    s="$(echo "$s" | tr -d '[]"' )"
    s="$(echo "$s" | sed -E 's/[[:space:]]*,[[:space:]]*/, /g; s/^[[:space:]]+|[[:space:]]+$//g')"
    echo "$s"
}

########################################
# Disk space pre-flight check
########################################
check_disk_space() {
    local path="$1" required_gb="$2" label="${3:-download}"
    local available_gb
    available_gb=$(df -BG "$path" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
    if [ -n "$available_gb" ] && [ "$available_gb" -lt "$required_gb" ] 2>/dev/null; then
        print_error "Insufficient disk space for $label. Need ${required_gb}GB, have ${available_gb}GB at $path"
        return 1
    fi
    print_info "Disk check: ${available_gb}GB available (need ~${required_gb}GB for $label)"
}
