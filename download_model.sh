#!/bin/bash


if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
    exit 1
fi


# Model and file
DEST_PATH="${DEST_DIR}/${MODEL_NAME}"

# Create destination dir
mkdir -p "$DEST_DIR"

# Build URL
URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_MODEL_FILE}"

# Use wget with Bearer token
echo "Downloading from: $URL"
wget --header="Authorization: Bearer $HF_TOKEN" \
     -c "$URL" \
     -O "$DEST_PATH"

echo "Downloaded to: $DEST_PATH"
