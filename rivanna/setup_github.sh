#!/bin/bash
# Quick GitHub setup script for Rivanna

echo "=========================================="
echo "GitHub Setup for Rivanna"
echo "=========================================="
echo ""

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "Loading git module..."
    module load git 2>/dev/null || echo "⚠ Git module not found. Install git first."
fi

# Configure git (if not already configured)
if [ -z "$(git config --global user.name)" ]; then
    echo "Configuring Git..."
    read -p "Enter your name: " git_name
    read -p "Enter your email: " git_email
    git config --global user.name "$git_name"
    git config --global user.email "$git_email"
    echo "✓ Git configured"
else
    echo "✓ Git already configured:"
    git config --global user.name
    git config --global user.email
fi

echo ""
echo "=========================================="
echo "SSH Key Setup"
echo "=========================================="

# Check if SSH key exists
if [ -f ~/.ssh/id_ed25519.pub ]; then
    echo "✓ SSH key found: ~/.ssh/id_ed25519.pub"
    echo ""
    echo "Your public key:"
    cat ~/.ssh/id_ed25519.pub
    echo ""
    echo "Copy this key and add it to GitHub:"
    echo "  https://github.com/settings/ssh/new"
else
    echo "Generating SSH key..."
    read -p "Enter your email for SSH key: " ssh_email
    ssh-keygen -t ed25519 -C "$ssh_email" -f ~/.ssh/id_ed25519 -N ""
    echo ""
    echo "Your public key:"
    cat ~/.ssh/id_ed25519.pub
    echo ""
    echo "Copy this key and add it to GitHub:"
    echo "  https://github.com/settings/ssh/new"
fi

echo ""
read -p "Press Enter after adding the key to GitHub..."

# Test SSH connection
echo ""
echo "Testing GitHub connection..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✓ GitHub SSH connection successful!"
else
    echo "⚠ SSH connection failed. You may need to:"
    echo "  1. Add the key to GitHub"
    echo "  2. Wait a few minutes for propagation"
    echo "  3. Try: ssh -T git@github.com"
fi

echo ""
echo "=========================================="
echo "Repository Setup"
echo "=========================================="

# Check if already in a git repo
if [ -d ~/Image-Analyzer/.git ]; then
    echo "✓ Repository found at ~/Image-Analyzer"
    cd ~/Image-Analyzer
    echo "Current remote:"
    git remote -v
    echo ""
    read -p "Update remote URL? (y/N): " update_remote
    if [[ $update_remote =~ ^[Yy]$ ]]; then
        read -p "Enter GitHub username: " github_user
        read -p "Enter repository name (default: Image-Analyzer): " repo_name
        repo_name=${repo_name:-Image-Analyzer}
        git remote set-url origin git@github.com:$github_user/$repo_name.git
        echo "✓ Remote updated"
    fi
else
    read -p "Enter GitHub username: " github_user
    read -p "Enter repository name (default: Image-Analyzer): " repo_name
    repo_name=${repo_name:-Image-Analyzer}
    
    if [ -d ~/Image-Analyzer ]; then
        echo "Directory exists. Initializing git..."
        cd ~/Image-Analyzer
        git init
        git remote add origin git@github.com:$github_user/$repo_name.git
        echo "✓ Git initialized"
    else
        echo "Cloning repository..."
        cd ~
        git clone git@github.com:$github_user/$repo_name.git Image-Analyzer
        cd Image-Analyzer
        echo "✓ Repository cloned"
    fi
fi

echo ""
echo "=========================================="
echo "Pull Latest Changes"
echo "=========================================="
cd ~/Image-Analyzer
git fetch origin
git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || echo "⚠ Could not pull. Check branch name."

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To update files in the future, run:"
echo "  cd ~/Image-Analyzer && git pull origin main"
echo ""

