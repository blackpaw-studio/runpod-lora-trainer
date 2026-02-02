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
# Skippable countdown
########################################
# Usage: countdown_or_enter SECONDS [MESSAGE]
# Shows a countdown that can be skipped by pressing Enter.
countdown_or_enter() {
    local seconds="${1:-10}" msg="${2:-Press Enter to skip...}"
    echo -e "${YELLOW}${msg}${NC}"
    for (( i=seconds; i>=1; i-- )); do
        printf "\r  %d... " "$i"
        # read with 1-second timeout; if user presses Enter, break
        if read -t 1 -r -s -n 1 2>/dev/null; then
            break
        fi
    done
    printf "\r         \r"
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
            countdown_or_enter 10 "Press Enter to continue, or waiting 10 seconds..."
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
# Wait for a background process while streaming its log
########################################
# Usage: wait_with_log PID LOG_FILE LABEL TIMEOUT_SECONDS [ERROR_PATTERN] [DONE_PATTERN]
wait_with_log() {
    local pid="$1" log_file="$2" label="$3" max_timeout="${4:-10800}"
    local error_pattern="${5:-}"
    local done_pattern="${6:-}"
    local timeout_counter=0 tail_pid

    # Stream the log to the user in real time
    touch "$log_file"
    tail -f "$log_file" 2>/dev/null &
    tail_pid=$!

    while kill -0 "$pid" 2>/dev/null; do
        # Check for early completion marker
        if [ -n "$done_pattern" ]; then
            if tail -n 1 "$log_file" 2>/dev/null | grep -q "$done_pattern"; then
                break
            fi
        fi
        # Check for errors
        if [ -n "$error_pattern" ]; then
            if tail -n 20 "$log_file" 2>/dev/null | grep -qiE "$error_pattern"; then
                kill "$tail_pid" 2>/dev/null || true
                wait "$tail_pid" 2>/dev/null || true
                print_error "$label encountered errors. Check log: $log_file"
                kill "$pid" 2>/dev/null || true
                return 1
            fi
        fi
        sleep 3
        timeout_counter=$((timeout_counter + 3))
        if [ "$timeout_counter" -ge "$max_timeout" ]; then
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            print_error "$label timed out. Check log: $log_file"
            kill "$pid" 2>/dev/null || true
            return 1
        fi
    done

    # Give tail a moment to catch up with final output, then stop it
    sleep 1
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true

    wait "$pid"
    return $?
}

########################################
# Disk space pre-flight check
########################################
check_disk_space() {
    local path="$1" required_gb="$2" label="${3:-download}"
    local available_gb
    available_gb=$(df -BG "$path" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
    if [ -z "$available_gb" ]; then
        print_warning "Could not determine available disk space at $path — skipping check for $label"
        return 0
    fi
    if [ "$available_gb" -lt "$required_gb" ] 2>/dev/null; then
        print_error "Insufficient disk space for $label. Need ${required_gb}GB, have ${available_gb}GB at $path"
        return 1
    fi
    print_info "Disk check: ${available_gb}GB available (need ~${required_gb}GB for $label)"
}
