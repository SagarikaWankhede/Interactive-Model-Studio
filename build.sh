#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

echo "========================================="
echo "  1. BUILDING REACT FRONTEND (VITE SPA)  "
echo "========================================="
cd frontend
npm install
npm run build
cd ..

echo "========================================="
echo "  2. INSTALLING PYTHON BACKEND PACKAGES  "
echo "========================================="
python -m pip install --upgrade pip
pip install -r backend/requirements.txt -r ml_engine/requirements.txt

echo "========================================="
echo "  BUILD SUCCESSFUL — UNIFIED STUDIO READY"
echo "========================================="
