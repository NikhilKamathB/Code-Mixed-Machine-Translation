from datetime import datetime
from transformers import BartTokenizer
from tokenizers import ByteLevelBPETokenizer
from src.data import *


class CustomBartTokenizer:

    def _build_bpe_tokenizer(self, prefix: str = "bpe_tokenizer") -> None:
        """
        Description: Builds a BPE tokenizer.
        """
        self.bpe_tokenizer = ByteLevelBPETokenizer(
            vocab_file=MBART_TOKENIZER_BPE_VOCAB_FILE,
            merges_file=MBART_TOKENIZER_BPE_MERGE_FILE,
            add_prefix_space=MBART_TOKENIZER_BPE_ADD_PREFIX_SPACE,
            lowercase=MBART_TOKENIZER_BPE_LOWERCASE,
            trim_offsets=MBART_TOKENIZER_BPE_TRIM_OFFSETS,
            dropout=MBART_TOKENIZER_BPE_DROPOUT,
            unicode_normalizer=MBART_TOKENIZER_BPE_UNICODE_NORMALIZER,
            continuing_subword_prefix=MBART_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX,
            end_of_word_suffix=MBART_TOKENIZER_BPE_END_OF_WORD_SUFFIX
        )
        self.bpe_tokenizer.train(
            file = MBART_TOKENIZER_BPE_FILES,
            vocab_size = MBART_TOKENIZER_BPE_VOCAB_SIZE,
            min_frequency = MBART_TOKENIZER_BPE_MINIMUM_FREQUENCY,
            show_progress = MBART_TOKENIZER_BPE_SHOW_PROGRESS,
            special_tokens = MBART_TOKENIZER_BPE_SPECIAL_TOKENS
        )
        self.save_path = MBART_TOKENIZER_BPE_SAVE_PATH + "_" + prefix + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.bpe_tokenizer.save_model(self.save_path)
    
    def _build_bart_tokenizer(self) -> None:
        """
        Description: Builds a BART tokenizer.
        """
        self.bart_tokenizer = BartTokenizer.from_pretrained(MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_FROM_PRETRAINED)
        if MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_STYLE == STYLE.APPEND.value:
            self.new_vocab_items = list(self.bpe_tokenizer.get_vocab().keys())
            self.bart_tokenizer.add_tokens(self.new_vocab_items)
        elif MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_STYLE == STYLE.SCRATCH.value:
            self.bart_tokenizer = BartTokenizer.from_pretrained(self.save_path)
    
    def build(self) -> BartTokenizer:
        """
        Description: Builds a BPE tokenizer and a BART tokenizer.
        """
        if MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_STYLE != STYLE.DEFAULT.value \
            and MBART_TOKENIZER_BPE_LOAD_PATH \
                and not os.path.exists(MBART_TOKENIZER_BPE_LOAD_PATH):
            self._build_bpe_tokenizer(prefix=MBART_TOKENIZER_BPE_PREFIX_PATH)
        elif MBART_TOKENIZER_BPE_LOAD_PATH:
            self.save_path = MBART_TOKENIZER_BPE_LOAD_PATH
            self.bpe_tokenizer = ByteLevelBPETokenizer.from_pretrained(self.save_path)
        self._build_bart_tokenizer()
        return self.bart_tokenizer