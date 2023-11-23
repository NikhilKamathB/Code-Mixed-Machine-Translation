import os
import json


# Load config file
assert os.path.isfile("../config.json"), "Config file not found! Add a config.json file in ./src folder."
with open("../config.json", "r") as f:
    __config__ = json.load(f)

PROCESSED_COLUMN_NAMES = __config__["data"]["resultant_col_name"]
MBART_ENCODER_ADD_SPECIAL_TOKENS = __config__["mbart_code_mixed"]["dataloader"]["encoder"]["add_special_tokens"]
MBART_ENCODER_MAX_LENGTH = __config__["mbart_code_mixed"]["dataloader"]["encoder"]["max_length"]
MBART_ENCODER_RETURN_TENSORS = __config__["mbart_code_mixed"]["dataloader"]["encoder"]["return_tensors"]
MBART_ENCODER_PADDING = __config__["mbart_code_mixed"]["dataloader"]["encoder"]["padding"]
MBART_ENCODER_VERBOSE = __config__["mbart_code_mixed"]["dataloader"]["encoder"]["verbose"]
MBART_DECODER_ADD_SPECIAL_TOKENS = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["add_special_tokens"]
MBART_DECODER_MAX_LENGTH = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["max_length"]
MBART_DECODER_RETURN_TENSORS = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["return_tensors"]
MBART_DECODER_PADDING = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["padding"]
MBART_DECODER_VERBOSE = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["verbose"]
MBART_DATALOADER_TRANSLATION_MODELS = __config__["mbart_code_mixed"]["dataloader"]["translations"]
MBART_DATALOADER_TRAIN_BATCH_SIZE = __config__["mbart_code_mixed"]["dataloader"]["train_batch_size"]
MBART_DATALOADER_VALIDATION_BATCH_SIZE = __config__["mbart_code_mixed"]["dataloader"]["validation_batch_size"]
MBART_DATALOADER_TEST_BATCH_SIZE = __config__["mbart_code_mixed"]["dataloader"]["test_batch_size"]