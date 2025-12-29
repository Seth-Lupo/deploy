#!/bin/bash
# Start the voice pipeline server

set -e

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default settings
PORT="${PORT:-8765}"
HOST="${HOST:-0.0.0.0}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
VOICE_REF="${VOICE_REF:-}"

echo "Starting Voice Pipeline Server..."
echo "  Host: $HOST"
echo "  Port: $PORT"

# Build command
CMD="python server.py --host $HOST --port $PORT"

if [ -n "$SYSTEM_PROMPT" ]; then
    CMD="$CMD --system-prompt \"$SYSTEM_PROMPT\""
fi

if [ -n "$VOICE_REF" ]; then
    CMD="$CMD --voice-reference $VOICE_REF"
fi

# Run server
eval $CMD
