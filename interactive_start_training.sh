#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Source shared library
if [ -f /usr/local/lib/runpod_common.sh ]; then
    source /usr/local/lib/runpod_common.sh
else
    echo "ERROR: /usr/local/lib/runpod_common.sh not found. Was start_script.sh run first?"
    exit 1
fi

# Cleanup background processes on interrupt
BACKGROUND_PIDS=()
cleanup() {
    for pid in "${BACKGROUND_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

# Welcome message
clear
print_header "Welcome to Blackpaw Studio LoRA Trainer"
echo ""
echo -e "${PURPLE}This interactive script will guide you through setting up and starting a LoRA training session.${NC}"
echo -e "${RED}Before you start, make sure to add your datasets to their respective folders.${NC}"
echo ""

# Check for Blackwell GPU and warn user
warn_blackwell_gpu

# Create logs directory
mkdir -p "$NETWORK_VOLUME/logs"

# Check if flash-attn installation is still running
# Skip check if wheel was successfully installed in foreground
if [ -f /tmp/flash_attn_wheel_success ]; then
    print_success "flash-attn is installed and ready (installed from prebuilt wheel)."
    echo ""
elif [ -f /tmp/flash_attn_pid ]; then
    FLASH_ATTN_PID=$(cat /tmp/flash_attn_pid)
    if kill -0 "$FLASH_ATTN_PID" 2>/dev/null; then
        print_warning "flash-attn is still being compiled from source (PID: $FLASH_ATTN_PID)"
        print_info "Waiting for flash-attn compilation to complete..."
        echo ""
        if wait_with_log "$FLASH_ATTN_PID" "$NETWORK_VOLUME/logs/flash_attn_install.log" \
                "flash-attn compilation" 1800; then
            print_success "flash-attn compilation completed successfully!"
        else
            print_warning "flash-attn compilation may have failed. Check log: $NETWORK_VOLUME/logs/flash_attn_install.log"
        fi
        rm -f /tmp/flash_attn_pid
        echo ""
    else
        # Process finished, clean up PID file
        rm -f /tmp/flash_attn_pid
        print_success "flash-attn is installed and ready."
        echo ""
    fi
fi

# Model selection
echo -e "${BOLD}Please select the model you want to train:${NC}"
echo ""
echo "1) Flux"
echo "2) SDXL"
echo "3) Wan 1.3B"
echo "4) Wan 14B Text-To-Video (Supports both T2V and I2V)"
echo "5) Wan 14B Image-To-Video (Not recommended, for advanced users only)"
echo "6) Qwen Image"
echo "7) Z Image Turbo"
echo ""

while true; do
    read -p "Enter your choice (1-7): " model_choice
    case $model_choice in
        1)
            MODEL_TYPE="flux"
            MODEL_NAME="Flux"
            TOML_FILE="flux.toml"
            break
            ;;
        2)
            MODEL_TYPE="sdxl"
            MODEL_NAME="SDXL"
            TOML_FILE="sdxl.toml"
            break
            ;;
        3)
            MODEL_TYPE="wan13"
            MODEL_NAME="Wan 1.3B"
            TOML_FILE="wan13_video.toml"
            break
            ;;
        4)
            MODEL_TYPE="wan14b_t2v"
            MODEL_NAME="Wan 14B Text-To-Video"
            TOML_FILE="wan14b_t2v.toml"
            break
            ;;
        5)
            MODEL_TYPE="wan14b_i2v"
            MODEL_NAME="Wan 14B Image-To-Video"
            TOML_FILE="wan14b_i2v.toml"
            break
            ;;
        6)
            MODEL_TYPE="qwen"
            MODEL_NAME="Qwen Image"
            TOML_FILE="qwen_toml.toml"
            break
            ;;
        7)
            MODEL_TYPE="z_image_turbo"
            MODEL_NAME="Z Image Turbo"
            TOML_FILE="z_image_toml.toml"
            break
            ;;
        *)
            print_error "Invalid choice. Please enter a number between 1-7."
            ;;
    esac
done

echo ""
print_success "Selected model: $MODEL_NAME"
echo ""

# LoRA output name
DEFAULT_LORA_NAME="${MODEL_TYPE}_lora"
echo -e "${BOLD}Enter a name for your LoRA (or press Enter for default: $DEFAULT_LORA_NAME):${NC}"
while true; do
    read -p "> " LORA_NAME_INPUT
    if [ -z "$LORA_NAME_INPUT" ]; then
        LORA_NAME="$DEFAULT_LORA_NAME"
        break
    elif [[ "$LORA_NAME_INPUT" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        LORA_NAME="$LORA_NAME_INPUT"
        break
    else
        print_error "Invalid name. Use only letters, numbers, underscores, and hyphens (no spaces)."
    fi
done
print_success "LoRA name: $LORA_NAME"
echo ""

# Check and set required API keys
if [ "$MODEL_TYPE" = "flux" ]; then
    if [ -z "$HUGGING_FACE_TOKEN" ] || [ "$HUGGING_FACE_TOKEN" = "token_here" ]; then
        print_warning "Hugging Face token is required for Flux model."
        echo ""
        echo "You can get your token from: https://huggingface.co/settings/tokens"
        echo ""
        read -p "Please enter your Hugging Face token: " hf_token
        if [ -z "$hf_token" ]; then
            print_error "Token cannot be empty. Exiting."
            exit 1
        fi
        export HUGGING_FACE_TOKEN="$hf_token"
        print_success "Hugging Face token set successfully."
    else
        print_success "Hugging Face token already set."
    fi
fi

echo ""

# Dataset selection
print_header "Dataset Configuration"
echo ""
echo -e "${BOLD}Do you want to caption images and/or videos?${NC}"
echo ""
echo "1) Images only"
echo "2) Videos only"
echo "3) Both images and videos"
echo "4) Skip captioning (use existing captions)"
echo ""

while true; do
    read -p "Enter your choice (1-4): " caption_choice
    case $caption_choice in
        1)
            CAPTION_MODE="images"
            break
            ;;
        2)
            CAPTION_MODE="videos"
            break
            ;;
        3)
            CAPTION_MODE="both"
            break
            ;;
        4)
            CAPTION_MODE="skip"
            break
            ;;
        *)
            print_error "Invalid choice. Please enter a number between 1-4."
            ;;
    esac
done

echo ""

# Check dataset directories
if [ "$CAPTION_MODE" != "skip" ]; then
    IMAGE_DIR="$NETWORK_VOLUME/image_dataset_here"
    VIDEO_DIR="$NETWORK_VOLUME/video_dataset_here"

    # Check Gemini API key if video captioning is needed
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "token_here" ]; then
            print_warning "Gemini API key is required for video captioning."
            echo ""
            echo "You can get your API key from: https://aistudio.google.com/app/apikey"
            echo ""
            read -p "Please enter your Gemini API key: " gemini_key
            if [ -z "$gemini_key" ]; then
                print_error "API key cannot be empty. Exiting."
                exit 1
            fi
            export GEMINI_API_KEY="$gemini_key"
            print_success "Gemini API key set successfully."
        else
            print_success "Gemini API key already set."
        fi
        echo ""
    fi

    # Ask for trigger word if image captioning is needed
    TRIGGER_WORD=""
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        echo -e "${BOLD}Image Captioning Configuration:${NC}"
        echo ""
        read -p "Enter a trigger word for image captions (or press Enter for none): " TRIGGER_WORD
        if [ -n "$TRIGGER_WORD" ]; then
            print_success "Trigger word set: '$TRIGGER_WORD'"
        else
            print_info "No trigger word set"
        fi
        echo ""
    fi

    # Function to check if directory has files
    check_directory() {
        local dir=$1
        local type=$2

        if [ ! -d "$dir" ]; then
            print_error "$type directory does not exist: $dir"
            return 1
        fi

        # Check for files (not just directories)
        if [ "$type" = "Image" ]; then
            file_count=$(find "$dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        else
            file_count=$(find "$dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | wc -l)
        fi

        if [ "$file_count" -eq 0 ]; then
            print_error "No $type files found in: $dir"
            return 1
        fi

        print_success "Found $file_count $type file(s) in: $dir"
        return 0
    }

    # Check based on caption mode
    case $CAPTION_MODE in
        "images")
            if ! check_directory "$IMAGE_DIR" "Image"; then
                echo ""
                print_error "Please add images to $IMAGE_DIR and re-run this script."
                exit 1
            fi
            ;;
        "videos")
            if ! check_directory "$VIDEO_DIR" "Video"; then
                echo ""
                print_error "Please add videos to $VIDEO_DIR and re-run this script."
                exit 1
            fi
            ;;
        "both")
            images_ok=true
            videos_ok=true

            if ! check_directory "$IMAGE_DIR" "Image"; then
                images_ok=false
            fi

            if ! check_directory "$VIDEO_DIR" "Video"; then
                videos_ok=false
            fi

            if [ "$images_ok" = false ] || [ "$videos_ok" = false ]; then
                echo ""
                print_error "Please add the missing files and re-run this script."
                if [ "$images_ok" = false ]; then
                    echo "  - Add images to: $IMAGE_DIR"
                fi
                if [ "$videos_ok" = false ]; then
                    echo "  - Add videos to: $VIDEO_DIR"
                fi
                exit 1
            fi
            ;;
    esac
fi

echo ""
print_success "Dataset validation completed successfully!"
echo ""

print_header "Checking CUDA Compatibility"
check_cuda_compatibility
echo ""

# Set up TOML configuration files
print_header "Setting Up Configuration Files"
echo ""

mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"

case $MODEL_TYPE in
    "flux")
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/flux.toml" ]; then
            print_info "flux.toml already exists in examples directory"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/flux.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/flux.toml" ]; then
            cp "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/flux.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/flux.toml"
            print_success "Copied flux.toml to examples directory"
        else
            print_warning "flux.toml not found at expected location: $NETWORK_VOLUME/runpod-lora-trainer/toml_files/flux.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/flux.toml"
        fi
        ;;
    "sdxl")
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/sdxl.toml" ]; then
            print_info "sdxl.toml already exists in examples directory"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/sdxl.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/sdxl.toml" ]; then
            cp "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/sdxl.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/sdxl.toml"
            print_success "Copied sdxl.toml to examples directory"
        else
            print_warning "sdxl.toml not found at expected location: $NETWORK_VOLUME/runpod-lora-trainer/toml_files/sdxl.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/sdxl.toml"
        fi
        ;;
    "wan13")
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/wan13_video.toml" ]; then
            print_info "wan13_video.toml already exists in examples directory"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan13_video.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan13_video.toml" ]; then
            cp "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan13_video.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan13_video.toml"
            print_success "Copied wan13_video.toml to examples directory"
        else
            print_warning "wan13_video.toml not found at expected location: $NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan13_video.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/wan13_video.toml"
        fi
        ;;
    "wan14b_t2v")
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_t2v.toml" ]; then
            print_info "wan14b_t2v.toml already exists in examples directory"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_t2v.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan14b_t2v.toml" ]; then
            cp "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan14b_t2v.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_t2v.toml"
            print_success "Copied wan14b_t2v.toml to examples directory"
        else
            print_warning "wan14b_t2v.toml not found at expected location: $NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan14b_t2v.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/wan14b_t2v.toml"
        fi
        ;;
    "wan14b_i2v")
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_i2v.toml" ]; then
            print_info "wan14b_i2v.toml already exists in examples directory"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_i2v.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan14b_i2v.toml" ]; then
            cp "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan14b_i2v.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_i2v.toml"
            print_success "Copied wan14b_i2v.toml to examples directory"
        else
            print_warning "wan14b_i2v.toml not found at expected location: $NETWORK_VOLUME/runpod-lora-trainer/toml_files/wan14b_i2v.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/wan14b_i2v.toml"
        fi
        ;;
    "qwen")
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/qwen_toml.toml" ]; then
            print_info "qwen_toml.toml already exists in examples directory"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/qwen_toml.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/qwen_toml.toml" ]; then
            cp "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/qwen_toml.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/qwen_toml.toml"
            print_success "Copied qwen_toml.toml to examples directory"
        else
            print_warning "qwen_toml.toml not found at expected location: $NETWORK_VOLUME/runpod-lora-trainer/toml_files/qwen_toml.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/qwen_toml.toml"
        fi
        ;;
    "z_image_turbo")
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/z_image_toml.toml" ]; then
            print_info "z_image_toml.toml already exists in examples directory"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/z_image_toml.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/z_image_toml.toml" ]; then
            cp "$NETWORK_VOLUME/runpod-lora-trainer/toml_files/z_image_toml.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/$LORA_NAME'|" "$NETWORK_VOLUME/diffusion_pipe/examples/z_image_toml.toml"
            print_success "Copied z_image_toml.toml to examples directory"
        else
            print_warning "z_image_toml.toml not found at expected location: $NETWORK_VOLUME/runpod-lora-trainer/toml_files/z_image_toml.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/z_image_toml.toml"
        fi
        ;;
esac

echo ""

# Update dataset.toml file with actual paths and video config
print_header "Configuring Dataset"
echo ""

DATASET_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"

if [ -f "$DATASET_TOML" ]; then
    print_info "Updating dataset.toml with actual paths..."

    # Create backup
    cp "$DATASET_TOML" "$DATASET_TOML.backup"

    # Replace $NETWORK_VOLUME with actual path in image directory
    sed -i "s|\$NETWORK_VOLUME/image_dataset_here|$NETWORK_VOLUME/image_dataset_here|g" "$DATASET_TOML" 2>/dev/null || print_warning "Failed to update image directory path in dataset.toml"

    # Replace $NETWORK_VOLUME with actual path in video directory (even if commented)
    sed -i "s|\$NETWORK_VOLUME/video_dataset_here|$NETWORK_VOLUME/video_dataset_here|g" "$DATASET_TOML" 2>/dev/null || print_warning "Failed to update video directory path in dataset.toml"

    # Uncomment video dataset section if user wants to caption videos
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Enabling video dataset in configuration..."
        # Uncomment the video directory section
        sed -i '/# \[\[directory\]\]/,/# num_repeats = 5/ s/^# //' "$DATASET_TOML" 2>/dev/null
        # Verify uncommenting worked by checking if video directory section exists uncommented
        if ! grep -q "^\[\[directory\]\]" "$DATASET_TOML" || [ -z "$(grep -A2 "^\[\[directory\]\]" "$DATASET_TOML" | grep -m1 "video_dataset_here")" ]; then
            # Check if there's a commented video section that wasn't uncommented
            if grep -q "# path = '\$NETWORK_VOLUME/video_dataset_here'" "$DATASET_TOML"; then
                print_warning "Video dataset section may not have been uncommented correctly. Please check dataset.toml manually."
            fi
        fi
    fi

    print_success "Dataset configuration updated"
else
    print_warning "dataset.toml not found at $DATASET_TOML"
fi

# Extract and display training configuration summary
print_header "Training Configuration Summary"
echo ""

# Read resolution from dataset.toml
if [ -f "$DATASET_TOML" ]; then
    RESOLUTION=$(grep "^resolutions = " "$DATASET_TOML" | sed 's/resolutions = \[\([0-9]*\)\]/\1/')
    if [ -z "$RESOLUTION" ]; then
        RESOLUTION="1024 (default)"
    fi
else
    RESOLUTION="1024 (default)"
fi

# Read training parameters from model TOML file
MODEL_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
if [ -f "$MODEL_TOML" ]; then
    EPOCHS=$(grep "^epochs = " "$MODEL_TOML" | sed 's/epochs = //')
    SAVE_EVERY=$(grep "^save_every_n_epochs = " "$MODEL_TOML" | sed 's/save_every_n_epochs = //')
    RANK=$(grep "^rank = " "$MODEL_TOML" | sed 's/rank = //')
    LR=$(grep "^lr = " "$MODEL_TOML" | sed 's/lr = //')
    OPTIMIZER_TYPE=$(grep "^type = " "$MODEL_TOML" | grep -A5 "\[optimizer\]" | grep "^type = " | sed "s/type = '//;s/'//")

    # Set defaults if not found
    [ -z "$EPOCHS" ] && EPOCHS="1000 (default)"
    [ -z "$SAVE_EVERY" ] && SAVE_EVERY="2 (default)"
    [ -z "$RANK" ] && RANK="32 (default)"
    [ -z "$LR" ] && LR="2e-5 (default)"
    [ -z "$OPTIMIZER_TYPE" ] && OPTIMIZER_TYPE="adamw_optimi (default)"
else
    # Fallback defaults if TOML file not found
    EPOCHS="1000 (default)"
    SAVE_EVERY="2 (default)"
    RANK="32 (default)"
    LR="2e-5 (default)"
    OPTIMIZER_TYPE="adamw_optimi (default)"
fi

echo -e "${BOLD}Model:${NC} $MODEL_NAME"
echo -e "${BOLD}LoRA Name:${NC} $LORA_NAME"
echo -e "${BOLD}TOML Config:${NC} examples/$TOML_FILE"
# Only show resolution as WxH if it's a number, otherwise show as-is
if [[ "$RESOLUTION" =~ ^[0-9]+$ ]]; then
    echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}x${RESOLUTION}"
else
    echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}"
fi
echo ""

echo -e "${BOLD}Training Parameters:${NC}"
echo "  📊 Epochs: $EPOCHS"
echo "  💾 Save Every: $SAVE_EVERY epochs"
echo "  🎛️  LoRA Rank: $RANK"
echo "  📈 Learning Rate: $LR"
echo "  ⚙️  Optimizer: $OPTIMIZER_TYPE"
echo ""

# Show dataset paths and repeats
if [ "$CAPTION_MODE" != "skip" ]; then
    echo -e "${BOLD}Dataset Configuration:${NC}"

    # Always show image dataset info
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        IMAGE_COUNT=$(find "$NETWORK_VOLUME/image_dataset_here" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        echo "  📷 Images: $NETWORK_VOLUME/image_dataset_here ($IMAGE_COUNT files)"
        echo "     Repeats: 1 per epoch"
    fi

    # Show video dataset info if applicable
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        VIDEO_COUNT=$(find "$NETWORK_VOLUME/video_dataset_here" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | wc -l)
        echo "  🎬 Videos: $NETWORK_VOLUME/video_dataset_here ($VIDEO_COUNT files)"
        echo "     Repeats: 5 per epoch"
    fi
else
    echo -e "${BOLD}Dataset:${NC} Using existing captions"
fi

if [ "$MODEL_TYPE" = "flux" ]; then
    echo -e "${BOLD}Hugging Face Token:${NC} Set ✓"
fi

if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    echo -e "${BOLD}Gemini API Key:${NC} Set ✓"
fi

echo ""

# Prompt user about configuration files
print_header "Training Configuration"
echo ""

print_info "Before starting training, you can modify the default training parameters in these files:"
echo ""
echo -e "${BOLD}1. Model Configuration:${NC}"
echo "   $NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
echo ""
echo -e "${BOLD}2. Dataset Configuration:${NC}"
echo "   $NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
echo ""

print_warning "These files contain important settings like:"
echo "  • Learning rate, batch size, epochs"
echo "  • Dataset paths and image/video resolutions"
echo "  • LoRA rank and other adapter settings"
echo ""

echo -e "${YELLOW}Would you like to modify these files before starting training?${NC}"
echo "1) Continue with default settings"
echo "2) Pause here - I'll modify the files manually"
echo ""

while true; do
    read -p "Enter your choice (1-2): " config_choice
    case $config_choice in
        1)
            print_success "Continuing with default training settings..."
            break
            ;;
        2)
            print_info "Training paused for manual configuration."
            echo ""
            echo -e "${BOLD}Configuration Files:${NC}"
            echo "1. Model settings: $NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
            echo "2. Dataset settings: $NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
            echo ""
            print_warning "Please modify these files as needed, then return here to continue."
            echo ""

            while true; do
                read -p "Have you finished configuring the settings? (yes/no): " config_done
                case $config_done in
                    yes|YES|y|Y)
                        print_success "Configuration completed. Reading updated settings..."
                        echo ""

                        # Re-read training parameters from updated TOML files
                        MODEL_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
                        DATASET_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"

                        # Read resolution from dataset.toml
                        if [ -f "$DATASET_TOML" ]; then
                            RESOLUTION=$(grep "^resolutions = " "$DATASET_TOML" | sed 's/resolutions = \[\([0-9]*\)\]/\1/')
                            if [ -z "$RESOLUTION" ]; then
                                RESOLUTION="1024 (default)"
                            fi
                        else
                            RESOLUTION="1024 (default)"
                        fi

                        # Read training parameters from model TOML file
                        if [ -f "$MODEL_TOML" ]; then
                            EPOCHS=$(grep "^epochs = " "$MODEL_TOML" | sed 's/epochs = //')
                            SAVE_EVERY=$(grep "^save_every_n_epochs = " "$MODEL_TOML" | sed 's/save_every_n_epochs = //')
                            RANK=$(grep "^rank = " "$MODEL_TOML" | sed 's/rank = //')
                            LR=$(grep "^lr = " "$MODEL_TOML" | sed 's/lr = //')
                            OPTIMIZER_TYPE=$(grep "^type = " "$MODEL_TOML" | grep -A5 "\[optimizer\]" | grep "^type = " | sed "s/type = '//;s/'//")

                            # Set defaults if not found
                            [ -z "$EPOCHS" ] && EPOCHS="1000 (default)"
                            [ -z "$SAVE_EVERY" ] && SAVE_EVERY="2 (default)"
                            [ -z "$RANK" ] && RANK="32 (default)"
                            [ -z "$LR" ] && LR="2e-5 (default)"
                            [ -z "$OPTIMIZER_TYPE" ] && OPTIMIZER_TYPE="adamw_optimi (default)"
                        else
                            # Fallback defaults if TOML file not found
                            EPOCHS="1000 (default)"
                            SAVE_EVERY="2 (default)"
                            RANK="32 (default)"
                            LR="2e-5 (default)"
                            OPTIMIZER_TYPE="adamw_optimi (default)"
                        fi

                        # Display updated configuration for confirmation
                        print_header "Updated Training Configuration"
                        echo ""
                        echo -e "${BOLD}Model:${NC} $MODEL_NAME"
                        echo -e "${BOLD}LoRA Name:${NC} $LORA_NAME"
                        # Only show resolution as WxH if it's a number, otherwise show as-is
                        if [[ "$RESOLUTION" =~ ^[0-9]+$ ]]; then
                            echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}x${RESOLUTION}"
                        else
                            echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}"
                        fi
                        echo ""
                        echo -e "${BOLD}Updated Training Parameters:${NC}"
                        echo "  📊 Epochs: $EPOCHS"
                        echo "  💾 Save Every: $SAVE_EVERY epochs"
                        echo "  🎛️  LoRA Rank: $RANK"
                        echo "  📈 Learning Rate: $LR"
                        echo "  ⚙️  Optimizer: $OPTIMIZER_TYPE"
                        echo ""

                        while true; do
                            read -p "Do these updated settings look correct? (yes/no): " settings_confirm
                            case $settings_confirm in
                                yes|YES|y|Y)
                                    print_success "Settings confirmed. Proceeding with training..."
                                    break 2  # Break out of both loops
                                    ;;
                                no|NO|n|N)
                                    print_info "Please modify the configuration files again."
                                    echo ""
                                    break  # Go back to configuration loop
                                    ;;
                                *)
                                    print_error "Please enter 'yes' or 'no'."
                                    ;;
                            esac
                        done
                        ;;
                    no|NO|n|N)
                        print_info "Take your time configuring the settings."
                        ;;
                    *)
                        print_error "Please enter 'yes' or 'no'."
                        ;;
                esac
            done
            break
            ;;
        *)
            print_error "Invalid choice. Please enter 1 or 2."
            ;;
    esac
done

echo ""
print_info "Configuration completed! Starting model download and setup..."
echo ""

# Model download logic - start in background
print_header "Starting Model Download"
echo ""

mkdir -p "$NETWORK_VOLUME/models"

# Initialize MODEL_DOWNLOAD_PID to ensure it's always set
MODEL_DOWNLOAD_PID=""

case $MODEL_TYPE in
    "flux")
        if [ -z "$HUGGING_FACE_TOKEN" ] || [ "$HUGGING_FACE_TOKEN" = "token_here" ]; then
            print_error "HUGGING_FACE_TOKEN is not set properly."
            exit 1
        fi

        print_info "HUGGING_FACE_TOKEN is set."
        check_disk_space "$NETWORK_VOLUME" 30 "Flux model"
        print_info "Starting Flux model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/flux"
        hf download black-forest-labs/FLUX.1-dev --local-dir "$NETWORK_VOLUME/models/flux" --repo-type model --token "$HUGGING_FACE_TOKEN" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        BACKGROUND_PIDS+=("$!")
        ;;

    "sdxl")
        check_disk_space "$NETWORK_VOLUME" 7 "SDXL model"
        print_info "Starting Base SDXL model download in background..."
        hf download timoshishi/sdXL_v10VAEFix sdXL_v10VAEFix.safetensors --local-dir "$NETWORK_VOLUME/models/" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        BACKGROUND_PIDS+=("$!")
        ;;

    "wan13")
        check_disk_space "$NETWORK_VOLUME" 5 "Wan 1.3B model"
        print_info "Starting Wan 1.3B model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B"
        hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        BACKGROUND_PIDS+=("$!")
        ;;

    "wan14b_t2v")
        check_disk_space "$NETWORK_VOLUME" 30 "Wan 14B T2V model"
        print_info "Starting Wan 14B T2V model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B"
        hf download Wan-AI/Wan2.1-T2V-14B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        BACKGROUND_PIDS+=("$!")
        ;;

    "wan14b_i2v")
        check_disk_space "$NETWORK_VOLUME" 30 "Wan 14B I2V model"
        print_info "Starting Wan 14B I2V model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P"
        hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        BACKGROUND_PIDS+=("$!")
        ;;

    "qwen")
        check_disk_space "$NETWORK_VOLUME" 15 "Qwen Image 2512 model"
        print_info "Starting Qwen Image 2512 model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Qwen-Image-2512"
        hf download Qwen/Qwen-Image-2512 --local-dir "$NETWORK_VOLUME/models/Qwen-Image-2512" > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        BACKGROUND_PIDS+=("$!")
        ;;

    "z_image_turbo")
        check_disk_space "$NETWORK_VOLUME" 15 "Z Image Turbo model"
        print_info "Starting Z Image Turbo model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/z_image"
        # Download model files using hf download and move to expected location
        (
            echo "Downloading Z Image Turbo models from HuggingFace..."
            # Download main model files (diffusion model, VAE, text encoder)
            hf download Comfy-Org/z_image_turbo --local-dir "$NETWORK_VOLUME/models/z_image_turbo_temp"
            
            echo "Moving model files to final location..."
            # Move files to the expected location
            mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$NETWORK_VOLUME/models/z_image/"
            mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/vae/ae.safetensors" "$NETWORK_VOLUME/models/z_image/"
            mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/text_encoders/qwen_3_4b.safetensors" "$NETWORK_VOLUME/models/z_image/"
            
            # Clean up temp directory
            rm -rf "$NETWORK_VOLUME/models/z_image_turbo_temp"
            
            echo "Downloading Z Image Turbo training adapter..."
            wget -q --show-progress -O "$NETWORK_VOLUME/models/z_image/zimage_turbo_training_adapter_v2.safetensors" \
                "https://huggingface.co/ostris/zimage_turbo_training_adapter/resolve/main/zimage_turbo_training_adapter_v2.safetensors"
            
            echo "Z Image Turbo model download complete!"
        ) > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        BACKGROUND_PIDS+=("$!")
        ;;
esac

echo ""

# Start captioning processes if needed
if [ "$CAPTION_MODE" != "skip" ]; then
    print_header "Starting Captioning Process"
    echo ""

    # Clear any existing subfolders in dataset directories before captioning
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Cleaning up image dataset directory..."
        # Remove any subdirectories but keep files
        find "$NETWORK_VOLUME/image_dataset_here" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        print_success "Image dataset directory cleaned"
    fi

    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Cleaning up video dataset directory..."
        # Remove any subdirectories but keep files
        find "$NETWORK_VOLUME/video_dataset_here" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        print_success "Video dataset directory cleaned"
    fi

    echo ""

    # Start image captioning in background if needed
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Starting image captioning process..."
        JOY_CAPTION_SCRIPT="$NETWORK_VOLUME/Captioning/JoyCaption/JoyCaptionRunner.sh"

        if [ -f "$JOY_CAPTION_SCRIPT" ]; then
            if [ -n "$TRIGGER_WORD" ]; then
                bash "$JOY_CAPTION_SCRIPT" --trigger-word "$TRIGGER_WORD" > "$NETWORK_VOLUME/logs/image_captioning.log" 2>&1 &
            else
                bash "$JOY_CAPTION_SCRIPT" > "$NETWORK_VOLUME/logs/image_captioning.log" 2>&1 &
            fi
            IMAGE_CAPTION_PID=$!
            BACKGROUND_PIDS+=("$!")
            print_success "Image captioning started in background (PID: $IMAGE_CAPTION_PID)"

            # Wait for image captioning with live log output
            print_info "Waiting for image captioning to complete... initial run can take 5-20 minutes."
            if ! wait_with_log "$IMAGE_CAPTION_PID" "$NETWORK_VOLUME/logs/image_captioning.log" \
                    "Image captioning" 3600 \
                    "(^\[ERROR\]|^Error:|^Traceback|Exception:|failed with exit)" \
                    "All done!"; then
                exit 1
            fi
            print_success "Image captioning completed!"
        else
            print_error "JoyCaption script not found at: $JOY_CAPTION_SCRIPT"
            exit 1
        fi
    fi

    # Start video captioning if needed
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Starting video captioning process..."
        VIDEO_CAPTION_SCRIPT="$NETWORK_VOLUME/Captioning/video_captioner.sh"

        if [ -f "$VIDEO_CAPTION_SCRIPT" ]; then
            bash "$VIDEO_CAPTION_SCRIPT" > "$NETWORK_VOLUME/logs/video_captioning.log" 2>&1 &
            VIDEO_CAPTION_PID=$!
            BACKGROUND_PIDS+=("$!")

            # Wait for video captioning with live log output
            print_info "Waiting for video captioning to complete..."
            if ! wait_with_log "$VIDEO_CAPTION_PID" "$NETWORK_VOLUME/logs/video_captioning.log" \
                    "Video captioning" 7200 \
                    "(^\[ERROR\]|^Error:|^Traceback|Exception:|failed with exit)" \
                    "video captioning complete"; then
                exit 1
            fi
            print_success "Video captioning completed successfully"
        else
            print_error "Video captioning script not found at: $VIDEO_CAPTION_SCRIPT"
            exit 1
        fi
    fi

    echo ""
fi

# Wait for model download to complete
if [ -n "$MODEL_DOWNLOAD_PID" ]; then
    print_header "Finalizing Model Download"
    echo ""
    print_info "Waiting for model download to complete..."
    echo ""
    if ! wait_with_log "$MODEL_DOWNLOAD_PID" "$NETWORK_VOLUME/logs/model_download.log" \
            "Model download" 10800 "(^ERROR|^fatal:|HTTP.*40[134]|unauthorized|Download failed|Permission denied)"; then
        exit 1
    fi
    
    # Verify model files actually exist based on MODEL_TYPE
    print_info "Verifying model download..."
    case $MODEL_TYPE in
        "flux")
            if [ ! -f "$NETWORK_VOLUME/models/flux/flux1-dev.safetensors" ] && [ ! -d "$NETWORK_VOLUME/models/flux" ]; then
                print_error "Flux model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "sdxl")
            if [ ! -f "$NETWORK_VOLUME/models/sdXL_v10VAEFix.safetensors" ]; then
                print_error "SDXL model file not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "wan13")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" 2>/dev/null)" ]; then
                print_error "Wan 1.3B model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "wan14b_t2v")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" 2>/dev/null)" ]; then
                print_error "Wan 14B T2V model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "wan14b_i2v")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" 2>/dev/null)" ]; then
                print_error "Wan 14B I2V model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "qwen")
            if [ ! -d "$NETWORK_VOLUME/models/Qwen-Image-2512" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Qwen-Image-2512" 2>/dev/null)" ]; then
                print_error "Qwen Image 2512 model files not found after download. Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
        "z_image_turbo")
            missing_files=""
            if [ ! -f "$NETWORK_VOLUME/models/z_image/z_image_turbo_bf16.safetensors" ]; then
                missing_files="$missing_files z_image_turbo_bf16.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/ae.safetensors" ]; then
                missing_files="$missing_files ae.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/qwen_3_4b.safetensors" ]; then
                missing_files="$missing_files qwen_3_4b.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/zimage_turbo_training_adapter_v2.safetensors" ]; then
                missing_files="$missing_files zimage_turbo_training_adapter_v2.safetensors"
            fi
            if [ -n "$missing_files" ]; then
                print_error "Z Image Turbo model files missing after download:$missing_files"
                print_error "Check log: $NETWORK_VOLUME/logs/model_download.log"
                exit 1
            fi
            ;;
    esac
    print_success "Model download completed and verified!"
    echo ""
fi

# Check if image captioning is still running
if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
    # Image captioning was already handled in the captioning section above
    # No need to check again here

    # Prompt user to inspect image captions
    print_header "Caption Inspection"
    echo ""
    print_info "Please manually inspect the generated captions in:"
    echo "  $NETWORK_VOLUME/image_dataset_here"
    echo ""
    print_warning "Check that the captions are accurate and appropriate for your training data."
    echo ""

    while true; do
        read -p "Have you reviewed the image captions and are ready to proceed? (yes/no): " inspect_choice
        case $inspect_choice in
            yes|YES|y|Y)
                print_success "Image captions approved. Proceeding to training..."
                break
                ;;
            no|NO|n|N)
                print_info "Please review the captions and run this script again when ready."
                exit 0
                ;;
            *)
                print_error "Please enter 'yes' or 'no'."
                ;;
        esac
    done
    echo ""
fi

# Check video captions if applicable
if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    # Video captioning was already handled in the captioning section above
    # No need to check again here

    print_header "Video Caption Inspection"
    echo ""
    print_info "Please manually inspect the generated video captions in:"
    echo "  $NETWORK_VOLUME/video_dataset_here"
    echo ""
    print_warning "Check that the video captions are accurate and appropriate for your training data."
    echo ""

    while true; do
        read -p "Have you reviewed the video captions and are ready to proceed? (yes/no): " video_inspect_choice
        case $video_inspect_choice in
            yes|YES|y|Y)
                print_success "Video captions approved. Proceeding to training..."
                break
                ;;
            no|NO|n|N)
                print_info "Please review the captions and run this script again when ready."
                exit 0
                ;;
            *)
                print_error "Please enter 'yes' or 'no'."
                ;;
        esac
    done
    echo ""
fi

# Start training
print_header "Starting Training"
echo ""

print_info "Changing to diffusion_pipe directory..."
cd "$NETWORK_VOLUME/diffusion_pipe"

print_info "Ensuring dependencies are up to date before training..."
print_info "Upgrading transformers and peft packages..."
pip install -U transformers "peft>=0.17.0" "torch==2.9.*"

echo ""

print_info "Starting LoRA training with $MODEL_NAME..."
print_info "Using configuration: examples/$TOML_FILE"
echo ""

# Add special warning for Qwen Image model initialization
if [ "$MODEL_TYPE" = "qwen" ]; then
    print_warning "⚠️  IMPORTANT: Qwen Image model initialization can take several minutes."
    print_warning "⚠️  The script may appear to hang during initialization - this is NORMAL."
    print_warning "⚠️  As long as the script doesn't exit with an error, let it run."
    echo ""
    countdown_or_enter 10 "Press Enter to continue, or waiting 10 seconds..."
    echo ""
fi

# Add special warning for Z Image Turbo model initialization
if [ "$MODEL_TYPE" = "z_image_turbo" ]; then
    print_warning "⚠️  IMPORTANT: Z Image Turbo model initialization can take several minutes."
    print_warning "⚠️  The script may appear to hang during initialization - this is NORMAL."
    print_warning "⚠️  As long as the script doesn't exit with an error, let it run."
    echo ""
    countdown_or_enter 10 "Press Enter to continue, or waiting 10 seconds..."
    echo ""
fi

print_warning "Training is starting. This may take several hours depending on your dataset size and model."
print_info "You can monitor progress in the console output below."
echo ""

# Start background watcher to copy checkpoints with LoRA name as they save
OUTPUT_DIR="$NETWORK_VOLUME/output_folder/$LORA_NAME"
LORA_WATCHER_SCRIPT="/tmp/lora_watcher.sh"
cat > "$LORA_WATCHER_SCRIPT" <<WATCHER_EOF
#!/bin/sh
OUTPUT_DIR="$OUTPUT_DIR"
LORA_NAME="$LORA_NAME"
echo "LoRA watcher started: OUTPUT_DIR=\$OUTPUT_DIR LORA_NAME=\$LORA_NAME"
while true; do
    for epoch_dir in "\$OUTPUT_DIR"/*/epoch*; do
        test -d "\$epoch_dir" || continue
        epoch_num=\$(echo "\$epoch_dir" | grep -o '[0-9]*$')
        src="\$epoch_dir/adapter_model.safetensors"
        dst="\$OUTPUT_DIR/\${LORA_NAME}_\${epoch_num}.safetensors"
        if test -f "\$src" && test ! -f "\$dst"; then
            cp "\$src" "\$dst"
            echo "[LoRA Watcher] Created \${LORA_NAME}_\${epoch_num}.safetensors"
        fi
    done
    sleep 30
done
WATCHER_EOF
chmod +x "$LORA_WATCHER_SCRIPT"
sh "$LORA_WATCHER_SCRIPT" > "$NETWORK_VOLUME/logs/lora_watcher.log" 2>&1 &
LORA_WATCHER_PID=$!
BACKGROUND_PIDS+=("$LORA_WATCHER_PID")

# Start training with the appropriate TOML file
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config "examples/$TOML_FILE"

# Stop the watcher
kill "$LORA_WATCHER_PID" 2>/dev/null || true

# Final pass to catch any checkpoints saved right before training ended
for epoch_dir in "$OUTPUT_DIR"/*/epoch*; do
    [ -d "$epoch_dir" ] || continue
    epoch_num=$(echo "$epoch_dir" | grep -o '[0-9]*$')
    src="$epoch_dir/adapter_model.safetensors"
    dst="$OUTPUT_DIR/${LORA_NAME}_${epoch_num}.safetensors"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        print_info "Created ${LORA_NAME}_${epoch_num}.safetensors"
    fi
done

# Create zip of all named LoRA files
ZIP_FILE="$OUTPUT_DIR/${LORA_NAME}.zip"
LORA_FILES=("$OUTPUT_DIR"/${LORA_NAME}_*.safetensors)
if [ -f "${LORA_FILES[0]}" ]; then
    print_info "Creating zip archive of LoRA files..."
    zip -j "$ZIP_FILE" "$OUTPUT_DIR"/${LORA_NAME}_*.safetensors
    print_success "Created $ZIP_FILE ($(du -h "$ZIP_FILE" | cut -f1))"
else
    print_warning "No renamed LoRA files found to zip."
fi

print_success "Training completed!"