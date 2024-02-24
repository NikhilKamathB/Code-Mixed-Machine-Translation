# Run these commands on colab
_git_repo_name="Code-Mixed-Machine-Translation"
_git_repo_url="https://github.com/NikhilKamathB/Code-Mixed-Machine-Translation.git"
cd "/content/drive/My Drive/colab"
git clone ${_git_repo_url}
cd ${_git_repo_name}
export __git_repo_name=${_git_repo_name}
pip install -r requirements.txt

# Train the model
cd "./scripts"
python train.py -d