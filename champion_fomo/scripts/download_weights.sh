#!/bin/bash
# Script to download pretrained weights from GitHub Releases

set -e

# Configuration
REPO="jbanusco/fomo25"
VERSION="${1:-latest}"  # Use first argument or 'latest'
WEIGHTS_DIR="weights"
CHECKPOINT_NAME="fomo25_mmunetvae_pretrained.ckpt"

# Create weights directory
mkdir -p "${WEIGHTS_DIR}"

# Check if weights already exist
if [ -f "${WEIGHTS_DIR}/${CHECKPOINT_NAME}" ]; then
    echo "Weights already exist at ${WEIGHTS_DIR}/${CHECKPOINT_NAME}"
    echo "Delete and re-run to download fresh copy."
    exit 0
fi

echo "Downloading pretrained weights from GitHub Releases..."
echo "Repository: ${REPO}"
echo "Version: ${VERSION}"

# Get the download URL
if [ "${VERSION}" = "latest" ]; then
    DOWNLOAD_URL="https://github.com/${REPO}/releases/latest/download/${CHECKPOINT_NAME}"
else
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${CHECKPOINT_NAME}"
fi

echo "Download URL: ${DOWNLOAD_URL}"
echo ""

# Download using curl or wget
if command -v curl &> /dev/null; then
    curl -L -o "${WEIGHTS_DIR}/${CHECKPOINT_NAME}" "${DOWNLOAD_URL}"
elif command -v wget &> /dev/null; then
    wget -O "${WEIGHTS_DIR}/${CHECKPOINT_NAME}" "${DOWNLOAD_URL}"
else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Verify download
if [ -f "${WEIGHTS_DIR}/${CHECKPOINT_NAME}" ]; then
    SIZE=$(du -h "${WEIGHTS_DIR}/${CHECKPOINT_NAME}" | cut -f1)
    echo ""
    echo "✅ Successfully downloaded weights!"
    echo "   Location: ${WEIGHTS_DIR}/${CHECKPOINT_NAME}"
    echo "   Size: ${SIZE}"
else
    echo "❌ Download failed!"
    exit 1
fi
