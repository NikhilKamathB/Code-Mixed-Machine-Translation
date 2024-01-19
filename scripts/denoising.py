"""
    Training script for denoising model
"""
# Set the environment
import os
import sys
__GIT_REPO__ = "Code-Mixed-Machine-Translation"
try:
    # Mount the drive
    from google.colab import drive
    drive.mount('/content/drive')
    # Set wokring directory
    os.chdir(f"{os.environ['WORK_DIR']}/{__GIT_REPO__}",
             f"/content/drive/MyDrive/colab/{__GIT_REPO__}")
    sys.path.append(f"{os.environ['WORK_DIR']}/{__GIT_REPO__}")
    # Install required packages
    from scripts.utils import install_package
    install_package(["accelerate", "datasets", "tokenizers",
                    "sacrebleu", "bert_score", "evaluate", "transformers==4.36.0"])
except Exception as e:
    print("An error occured while mounting the drive: ", e)
print(f"Current working directory: {os.getcwd()}")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Add one level up directory to the path
import sys
sys.path.append("..")

# Import libraries
import torch
import random
import pandas as pd
from typing import Tuple

# Import custom modules
from src.data import *
from src.machine_translation import *
from src.data.utils import get_dataset
from src.data.preprocess import clean_text
from src.data.tokenizer import CustomBartTokenizer
from src.machine_translation.translate import translate
from src.machine_translation.net import CodeMixedModel, CodeMixedModelHGTrainer
from src.machine_translation.models.bart_conditional import BartForConditionalGeneration
from src.machine_translation.utils import get_tokenized_dataset_models, get_data_loader_models, calculate_sacrebleu_score, calculate_chrf_score, calculate_bert_score, calculate_tokens


# Load and process data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Load the dataset and clean the text
        Input: None
        Output: Tuple of train, validation, and test dataframes
    """
    # Get the dataset
    train_df, validation_df, test_df = get_dataset()
    # Clean the text
    train_df = train_df.applymap(clean_text)
    validation_df = validation_df.applymap(clean_text)
    test_df = test_df.applymap(clean_text)
    return (train_df, validation_df, test_df)

# Build or load tokenizers
def load_tokenizers(train_df: pd.DataFrame) -> Tuple[CustomBartTokenizer, CustomBartTokenizer]:
    """
        Build or load the tokenizers
        Input: train_df - training dataframe
        Output: Tuple of tokenizers for source and target languages, respectively
    """
    hi_en_bart_tokenizer = CustomBartTokenizer().build(
        data=train_df["hi_en"],
        tokenizer_style=MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_ENCODER_STYLE,
        tokenizer_bart_from_pretrained_path=MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_ENCODER_FROM_PRETRAINED
    )
    en_bart_tokenizer = CustomBartTokenizer().build(
        tokenizer_style=MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_DECODER_STYLE,
        tokenizer_bart_from_pretrained_path=MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_DECODER_FROM_PRETRAINED
    )
    return (hi_en_bart_tokenizer, en_bart_tokenizer)
