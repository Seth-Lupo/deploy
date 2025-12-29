#!/bin/bash
# Quick deploy script for pushing code to EC2
# Usage: ./deploy.sh <EC2_HOST> [SSH_KEY]

set -e

EC2_HOST="${1:-}"
SSH_KEY="${2:-~/.ssh/id_rsa}"
REMOTE_DIR="/home/ubuntu/voice-pipeline"

if [ -z "$EC2_HOST" ]; then
    echo "Usage: ./deploy.sh <EC2_HOST> [SSH_KEY]"
    echo "Example: ./deploy.sh ec2-user@52.1.2.3 ~/.ssh/my-key.pem"
    exit 1
fi

echo "Deploying to $EC2_HOST..."

# Create remote directory
ssh -i "$SSH_KEY" "$EC2_HOST" "mkdir -p $REMOTE_DIR"

# Sync files
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    ./ "$EC2_HOST:$REMOTE_DIR/"

echo "Files synced!"

# Optionally restart server
read -p "Restart server? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh -i "$SSH_KEY" "$EC2_HOST" "cd $REMOTE_DIR && pkill -f 'python server.py' || true"
    ssh -i "$SSH_KEY" "$EC2_HOST" "cd $REMOTE_DIR && source venv/bin/activate && nohup python server.py --port 8765 > server.log 2>&1 &"
    echo "Server restarted!"
fi

echo "Deploy complete!"
echo ""
echo "To connect:"
echo "  python client.py --server ws://$EC2_HOST:8765/ws"
