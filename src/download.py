#!/usr/bin/env python3
import subprocess
import requests
import argparse
import sys

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="CivitAI model ID to download")
parser.add_argument("-t", "--token", type=str, help="CivitAI API token (if not set in environment)")
args = parser.parse_args()

# Validate model ID is numeric
import os
token = os.getenv("civitai_token", args.token)
if not args.model.isdigit():
    print("Error: model ID must be numeric.")
    sys.exit(1)

# Determine the token
if not token:
    print("Error: no token provided. Set the 'civitai_token' environment variable or use --token.")
    sys.exit(1)

# URL of the file to download
url = f"https://civitai.com/api/v1/model-versions/{args.model}"

# Perform the request
response = requests.get(url, stream=True)
if response.status_code == 200:
    download_url = f"https://civitai.com/api/download/models/{args.model}?type=Model&format=SafeTensor&token={token}"
    result = subprocess.run(["wget", download_url, "--content-disposition"], check=False)
    if result.returncode != 0:
        print("Error: wget download failed.")
        sys.exit(1)
else:
    print("Error: Failed to retrieve model metadata.")
    sys.exit(1)
