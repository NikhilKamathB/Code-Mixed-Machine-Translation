import os
import json
from enum import Enum


# Load config file
assert os.path.isfile("../config.json"), "Config file not found! Add a config.json file in ./src folder."
with open("../config.json", "r") as f:
    __config__ = json.load(f)

class STYLE(Enum):

    DEFAULT = "DEFAULT"
    APPEND = "APPEND"
    SCRATCH = "SCRATCH"

BASE_DATA_DIR = __config__["data"]["base_dir"]
PROCESSED_COLUMN_NAMES = __config__["data"]["resultant_col_name"]
HUGGINGFACE_DATA = __config__["data"]["huggingface"]
HUGGINGFACE_DATASET = __config__["data"]["huggingface"]["dataset"]
HINGLISH_TOP_DATA = __config__["data"]["hinglish_top_dataset"]
HINGLISH_TOP_DATASET = __config__["data"]["hinglish_top_dataset"]["dataset"]
LINC_DATA = __config__["data"]["linc_dataset"]
LINC_DATASET = __config__["data"]["linc_dataset"]["dataset"]
PROCESSED_DATA_BASE_DIR = __config__["data"]["processed_data"]["base_dir"]
PROCESSED_TRAIN_CSV = __config__["data"]["processed_data"]["train_data"]
PROCESSED_VALIDATION_CSV = __config__["data"]["processed_data"]["validation_data"]
PROCESSED_TEST_CSV = __config__["data"]["processed_data"]["test_data"]
MBART_TOKENIZER_BPE_TYPE = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["type"]
MBART_TOKENIZER_BPE_ADD_PREFIX_SPACE = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["add_prefix_space"]
MBART_TOKENIZER_BPE_LOWERCASE = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["lowercase"]
MBART_TOKENIZER_BPE_TRIM_OFFSETS = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["trim_offsets"]
MBART_TOKENIZER_BPE_VOCAB_FILE = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["vocab_file"]
MBART_TOKENIZER_BPE_MERGE_FILE = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["merges_file"]
MBART_TOKENIZER_BPE_DROPOUT = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["dropout"]
MBART_TOKENIZER_BPE_UNICODE_NORMALIZER = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["unicode_normalizer"]
MBART_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["continuing_subword_prefix"]
MBART_TOKENIZER_BPE_END_OF_WORD_SUFFIX = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["end_of_word_suffix"]
MBART_TOKENIZER_BPE_VOCAB_SIZE = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["vocab_size"]
MBART_TOKENIZER_BPE_MINIMUM_FREQUENCY = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["min_fequency"]
MBART_TOKENIZER_BPE_SHOW_PROGRESS = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["show_progress"]
MBART_TOKENIZER_BPE_SPECIAL_TOKENS = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["special_tokens"]
MBART_TOKENIZER_BPE_MASK_TOKEN = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["mask_token"]
MBART_TOKENIZER_BPE_LENGTH = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["length"]
MBART_TOKENIZER_BPE_SAVE_BPE = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["save_bpe"]
MBART_TOKENIZER_BPE_PREFIX_PATH = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["prefix_path"]
MBART_TOKENIZER_BPE_SAVE_PATH = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["save_path"]
MBART_TOKENIZER_BPE_LOAD_PATH = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["load_path"]
MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_FROM_PRETRAINED = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["binding"]["bart_tokenizer"]["from_pretrained"]
MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_STYLE = STYLE(__config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["binding"]["bart_tokenizer"]["style"].upper()).value
MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_ENCODER_FROM_PRETRAINED = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["binding"]["bart_tokenizer_encoder"]["from_pretrained"]
MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_ENCODER_STYLE = STYLE(__config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["binding"]["bart_tokenizer_encoder"]["style"].upper()).value
MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_DECODER_FROM_PRETRAINED = __config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["binding"]["bart_tokenizer_decoder"]["from_pretrained"]
MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_DECODER_STYLE = STYLE(__config__["mbart_code_mixed"]["tokenizer"]["byte_level_bpe_tokenizer"]["binding"]["bart_tokenizer_decoder"]["style"].upper()).value