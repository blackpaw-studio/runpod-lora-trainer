#!/usr/bin/env bash

# Determine which branch to clone based on environment variables
BRANCH="main"  # Default branch

if [ "$is_dev" == "true" ]; then
    BRANCH="dev"
    echo "Development mode enabled. Cloning dev branch..."
elif [ -n "$git_branch" ]; then
    BRANCH="$git_branch"
    echo "Custom branch specified: $git_branch"
else
    echo "Using default branch: main"
fi

# Clone the repository to a temporary location with the specified branch
echo "Cloning branch '$BRANCH' from repository..."
git clone --branch "$BRANCH" https://github.com/blackpaw-studio/runpod-lora-trainer.git /tmp/runpod-lora-trainer

# Check if clone was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone branch '$BRANCH'. Falling back to main branch..."
    git clone https://github.com/blackpaw-studio/runpod-lora-trainer.git /tmp/runpod-lora-trainer

    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone repository. Exiting..."
        exit 1
    fi
fi

# Install shared shell library
cp /tmp/runpod-lora-trainer/src/common.sh /usr/local/lib/runpod_common.sh

# Move start.sh to root and execute it
mv /tmp/runpod-lora-trainer/src/start.sh /
chmod +x /start.sh
bash /start.sh