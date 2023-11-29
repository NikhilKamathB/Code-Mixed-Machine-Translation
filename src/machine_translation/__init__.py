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
MBART_ENCODER_TRUNCATION = __config__["mbart_code_mixed"]["dataloader"]["encoder"]["truncation"]
MBART_ENCODER_VERBOSE = __config__["mbart_code_mixed"]["dataloader"]["encoder"]["verbose"]
MBART_DECODER_ADD_SPECIAL_TOKENS = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["add_special_tokens"]
MBART_DECODER_MAX_LENGTH = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["max_length"]
MBART_DECODER_RETURN_TENSORS = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["return_tensors"]
MBART_DECODER_PADDING = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["padding"]
MBART_DECODER_TRUNCATION = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["truncation"]
MBART_DECODER_VERBOSE = __config__["mbart_code_mixed"]["dataloader"]["decoder"]["verbose"]
MBART_DATALOADER_TRANSLATION_MODE = __config__["mbart_code_mixed"]["dataloader"]["translation_mode"]
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
MBART_MODEL_CONDITIONAL_GENERATION_LOG_PATH = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["log_path"]
MBART_MODEL_CONDITIONAL_GENERATION_LOAD_PATH = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["load_path"]
MBART_MODEL_CONDITIONAL_GENERATION_FROM_PRETRAINED = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["from_pretrained"]
MBART_MODEL_CONDITIONAL_GENERATION_VOCAB = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["vocab_size"]
MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["verbose"]
MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE_STEP = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["verbose_step"]
MBART_MODEL_CONDITIONAL_GENERATION_FREEZE_MODEL = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["freeze_model"]
MBART_MODEL_CONDITIONAL_GENERATION_K_RANDOM = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["k_random"]
MBART_MODEL_CONDITIONAL_GENERATION_RESUME_FROM_CHECKPOINT = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["resume_from_checkpoint"]
MBART_MODEL_CONDITIONAL_GENERATION_DO_TRAIN = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["do_train"]
MBART_MODEL_CONDITIONAL_GENERATION_DO_EVAL = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["do_eval"]
MBART_MODEL_CONDITIONAL_GENERATION_DO_PREDICT = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["do_predict"]
MBART_MODEL_CONDITIONAL_GENERATION_EVALUALTION_STRATEGY = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["evaluation_strategy"]
MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_TYPE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["criterion"]["type"]
MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_WEIGHT = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["criterion"]["weight"]
MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_SIZE_AVERAGE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["criterion"]["size_average"]
MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_IGNORE_INDEX = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["criterion"]["ignore_index"]
MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_REDUCE = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["criterion"]["reduce"]
MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_REDUCTION = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["criterion"]["reduction"]
MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_LABEL_SMOOTHING = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["criterion"]["label_smoothing"]
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
MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_MAX_LENGTH = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["generate"]["max_length"]
MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_EARLY_STOPPING = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["generate"]["early_stopping"]
MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_NUM_BEAMS = __config__["mbart_code_mixed"]["model"]["bart_for_conditional_generation"]["generate"]["num_beams"]