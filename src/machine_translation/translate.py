import pandas as pd
from tqdm import tqdm
from typing import Union
from src.data import *
from src.machine_translation import *
from src.data.tokenizer import CustomBartTokenizer
from src.machine_translation.net import CodeMixedModelHGTrainer
from src.machine_translation.utils import get_tokenized_dataset_models


def translate(
        data: Union[pd.DataFrame, str],
        src_tgt_lang: list = PROCESSED_COLUMN_NAMES,
        encoder_tokenizer: CustomBartTokenizer = None,
        decoder_tokenizer: CustomBartTokenizer = None,
        saved_model_path: str = MBART_MODEL_CONDITIONAL_GENERATION_RESUME_FROM_CHECKPOINT,
        translation_mode: str = MBART_DATALOADER_TRANSLATION_MODE
        ) -> Union[pd.DataFrame, str]:
    '''
        This function is used to translate the text from one language to another.
    '''
    assert len(src_tgt_lang) == 2, "The length of `src_tgt_lang` must be 2 and it must be in the following format `[soruce_lang, target_lang]`."
    src_lang, tgt_lang = src_tgt_lang
    if isinstance(data, pd.DataFrame):
        assert src_lang in data.columns, f"Column `{src_lang}` not found in the dataframe."
        assert tgt_lang in data.columns, f"Column `{tgt_lang}` not found in the dataframe."
    assert encoder_tokenizer is not None, "Encoder tokenizer cannot be None."
    assert decoder_tokenizer is not None, "Decoder tokenizer cannot be None."
    model = CodeMixedModelHGTrainer(
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        verbose=False,
        inference=True
    )
    if isinstance(data, str):
        translated_string = model.infer(
            model_path=saved_model_path,
            src=data
        )
        return translated_string
    elif isinstance(data, pd.DataFrame):
        dataset = get_tokenized_dataset_models(
            test_df=data,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer
        )
        _, _, datadet = dataset[translation_mode].values()
        translated_strings = model.infer(
            model_path=saved_model_path,
            src=datadet
        )
        data.loc[:, "translations"] = translated_strings
        return data
    else:
        raise ValueError("`data` must be a pandas dataframe or a string.")