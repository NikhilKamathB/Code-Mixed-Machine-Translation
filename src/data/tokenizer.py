import shutil
import pandas as pd
from datetime import datetime
from transformers import BartTokenizer
from tokenizers import ByteLevelBPETokenizer
from src.data import *


class CustomBartTokenizer:

    def _build_bpe_tokenizer(self, iterator: pd.Series, prefix: str = "bpe_tokenizer") -> None:
        """
        Description: Builds a BPE tokenizer.
        Input parameters:
            - prefix: A string containing the prefix path to the BPE tokenizer to be built.
        Outputs: None.
        """
        self.bpe_tokenizer = ByteLevelBPETokenizer(
            vocab=MBART_TOKENIZER_BPE_VOCAB_FILE,
            merges=MBART_TOKENIZER_BPE_MERGE_FILE,
            add_prefix_space=MBART_TOKENIZER_BPE_ADD_PREFIX_SPACE,
            lowercase=MBART_TOKENIZER_BPE_LOWERCASE,
            trim_offsets=MBART_TOKENIZER_BPE_TRIM_OFFSETS,
            dropout=MBART_TOKENIZER_BPE_DROPOUT,
            unicode_normalizer=MBART_TOKENIZER_BPE_UNICODE_NORMALIZER,
            continuing_subword_prefix=MBART_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX,
            end_of_word_suffix=MBART_TOKENIZER_BPE_END_OF_WORD_SUFFIX
        )
        self.bpe_tokenizer.train_from_iterator(
            iterator = iterator,
            vocab_size = MBART_TOKENIZER_BPE_VOCAB_SIZE,
            min_frequency = MBART_TOKENIZER_BPE_MINIMUM_FREQUENCY,
            show_progress = MBART_TOKENIZER_BPE_SHOW_PROGRESS,
            special_tokens = MBART_TOKENIZER_BPE_SPECIAL_TOKENS,
            length = MBART_TOKENIZER_BPE_LENGTH
        )
        self.bpe_tokenizer.add_tokens([MBART_TOKENIZER_BPE_MASK_TOKEN])
        self.save_path = MBART_TOKENIZER_BPE_SAVE_PATH + "/" + prefix + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.bpe_tokenizer.save_model(self.save_path)
    
    def _build_bart_tokenizer(self, tokenizer_style: STYLE = MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_STYLE, \
                              tokenizer_bart_from_pretrained_path: str = MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_FROM_PRETRAINED) -> None:
        """
        Description: Builds a BART tokenizer.
        Input parameters:
            - tokenizer_style: A STYLE enum value.
        Outputs: None.
        """
        self.bart_tokenizer = BartTokenizer.from_pretrained(tokenizer_bart_from_pretrained_path)
        if tokenizer_style == STYLE.APPEND.value:
            self.new_vocab_items = list(self.bpe_tokenizer.get_vocab().keys())
            self.bart_tokenizer.add_tokens(self.new_vocab_items)
        elif tokenizer_style == STYLE.SCRATCH.value:
            self.bart_tokenizer = BartTokenizer.from_pretrained(self.save_path)
    
    def _clean(self) -> None:
        """
        Description: Cleans the saved BPE tokenizer.
        """
        if not MBART_TOKENIZER_BPE_SAVE_BPE:
            if os.path.exists(self.save_path):
                shutil.rmtree(self.save_path)
    
    def build(self, data: pd.Series = None, tokenizer_style: STYLE = MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_STYLE, \
              tokenizer_bpe_load_path: str = MBART_TOKENIZER_BPE_LOAD_PATH, \
                tokenizer_bpe_prefix_path: str = MBART_TOKENIZER_BPE_PREFIX_PATH, \
                    tokenizer_bart_from_pretrained_path: str = MBART_TOKENIZER_BPE_BINDING_BART_TOKENIZER_FROM_PRETRAINED) -> BartTokenizer:
        """
        Description: Builds a BPE tokenizer and a BART tokenizer.
        Input parameters:
            - data: A pandas dataframe containing the data.
            - tokenizer_style: A STYLE enum value.
            - tokenizer_bpe_load_path: A string containing the path to the BPE tokenizer to be loaded.
            - tokenizer_bpe_prefix_path: A string containing the prefix path to the BPE tokenizer to be built.
        Outputs: A BART tokenizer.
        """
        if data is None and tokenizer_style != STYLE.DEFAULT.value and not tokenizer_bpe_load_path:
            raise ValueError("`data` cannot be None when tokenizer_style is not STYLE.DEFAULT and `tokenizer_bpe_load_path` is None.")
        if (tokenizer_style != STYLE.DEFAULT.value \
            and tokenizer_bpe_load_path \
                and not os.path.exists(tokenizer_bpe_load_path)) \
                    or (tokenizer_style != STYLE.DEFAULT.value \
                        and not tokenizer_bpe_load_path):
            self._build_bpe_tokenizer(iterator=data, prefix=tokenizer_bpe_prefix_path)
            self._build_bart_tokenizer(tokenizer_style=tokenizer_style, tokenizer_bart_from_pretrained_path=tokenizer_bart_from_pretrained_path)
            self._clean()
        elif tokenizer_style == STYLE.DEFAULT.value:
            self._build_bart_tokenizer(tokenizer_style=tokenizer_style, tokenizer_bart_from_pretrained_path=tokenizer_bart_from_pretrained_path)
        elif tokenizer_bpe_load_path:
            self.bart_tokenizer = BartTokenizer.from_pretrained(tokenizer_bpe_load_path)
        return self.bart_tokenizer