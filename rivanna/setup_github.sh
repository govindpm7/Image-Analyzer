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

# Check for local changes before pulling
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    echo "⚠ Local changes detected in the following files:"
    git diff --name-only HEAD
    echo ""
    echo "Options:"
    echo "  1) Stash changes, pull, then reapply (recommended)"
    echo "  2) Discard local changes and pull"
    echo "  3) Skip pull (you can commit/stash manually later)"
    echo ""
    read -p "Choose option (1-3, default: 1): " pull_choice
    pull_choice=${pull_choice:-1}
    
    case $pull_choice in
        1)
            echo "Stashing local changes..."
            STASH_MSG="Stashed before pull on $(date '+%Y-%m-%d %H:%M:%S')"
            git stash push -m "$STASH_MSG"
            echo "✓ Changes stashed"
            STASHED=true
            ;;
        2)
            read -p "Are you sure you want to discard ALL local changes? (yes/no): " confirm
            if [[ $confirm == "yes" ]]; then
                echo "Discarding local changes..."
                git reset --hard HEAD
                git clean -fd
                echo "✓ Local changes discarded"
                STASHED=false
            else
                echo "Cancelled. Skipping pull."
                STASHED=false
                skip_pull=true
            fi
            ;;
        3)
            echo "Skipping pull. You can handle changes manually."
            skip_pull=true
            STASHED=false
            ;;
        *)
            echo "Invalid option. Stashing changes..."
            git stash push -m "Stashed before pull on $(date '+%Y-%m-%d %H:%M:%S')"
            STASHED=true
            ;;
    esac
else
    STASHED=false
    skip_pull=false
fi

# Try to pull if not skipped
if [ "$skip_pull" != "true" ]; then
    if git pull origin main 2>/dev/null || git pull origin master 2>/dev/null; then
        echo "✓ Successfully pulled latest changes"
        
        # If we stashed changes, try to reapply them
        if [ "$STASHED" = "true" ]; then
            echo "Attempting to reapply stashed changes..."
            if git stash pop 2>/dev/null; then
                echo "✓ Stashed changes reapplied successfully"
            else
                echo "⚠ Conflicts detected when reapplying stashed changes"
                echo "   Your stashed changes are still available. To resolve:"
                echo "   1. Fix conflicts in the files"
                echo "   2. Run: git stash drop  (after resolving)"
                echo "   Or keep stashed: git stash apply"
            fi
        fi
    else
        echo "⚠ Could not pull. Check branch name or network connection."
        if [ "$STASHED" = "true" ]; then
            echo "   Your changes are safely stashed. Restore with: git stash pop"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To update files in the future:"
echo "  cd ~/Image-Analyzer && git pull origin main"
echo ""
echo "To avoid 'local changes would be overwritten' errors:"
echo "  1. Always commit or stash changes before pulling"
echo "  2. Or use: git stash && git pull && git stash pop"
echo "  3. Or discard changes: git reset --hard HEAD && git pull"
echo ""

