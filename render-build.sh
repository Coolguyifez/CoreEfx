#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# This ensures the spacy model is downloaded into the Render environment
python -m spacy download en_core_web_sm
