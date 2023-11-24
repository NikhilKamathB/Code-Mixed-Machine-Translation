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
MBART_MODEL_CONDITIONAL_GENERATION_TYPE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["type"]
MBART_MODEL_CONDITIONAL_GENERATION_USE_PRETRAINED = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["use_pretrained"]
MBART_MODEL_CONDITIONAL_GENERATION_DEVICE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["device"]
MBART_MODEL_CONDITIONAL_GENERATION_EPOCHS = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["epochs"]
MBART_MODEL_CONDITIONAL_GENERATION_SAVE_MODEL = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["save_model"]
MBART_MODEL_CONDITIONAL_GENERATION_SAVE_PATH = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["save_path"]
MBART_MODEL_CONDITIONAL_GENERATION_LOAD_PATH = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["load_path"]
MBART_MODEL_CONDITIONAL_GENERATION_FROM_PRETRAINED = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["from_pretrained"]
MBART_MODEL_CONDITIONAL_GENERATION_VOCAB = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["vocab_size"]
MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["verbose"]
MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE_STEP = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["verbose_step"]
MBART_MODEL_CONDITIONAL_GENERATION_FREEZE_MODEL = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["freeze_model"]
MBART_MODEL_CONDITIONAL_GENERATION_K_RANDOM = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["k_random"]
MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_TYPE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["optimizer"]["type"]
MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_LR = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["optimizer"]["lr"]
MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_BETAS = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["optimizer"]["betas"]
MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_EPS = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["optimizer"]["eps"]
MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_WEIGHT_DECAY = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["optimizer"]["weight_decay"]
MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_CORRECT_BIAS = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["optimizer"]["correct_bias"]
MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_NO_DEPRICATION_WARNING = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["optimizer"]["no_depreciation_warning"]
MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["scheduler"]["type"]
MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_WARMUP_STEPS = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["scheduler"]["num_warmup_steps"]
MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TRAINING_STEPS = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["scheduler"]["num_training_steps"]
MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_LAST_EPOCH = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["scheduler"]["last_epoch"]