#! /usr/bin/env sh

# Mount Google Drive
python -c "from google.colab import drive; drive.mount('/content/drive')"

# Clone repository
_git_repo_url="https://github.com/NikhilKamathB/Code-Mixed-Machine-Translation.git"
cd "/content/drive/My Drive/colab"
git clone ${_git_repo_url}