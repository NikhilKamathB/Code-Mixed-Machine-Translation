import pandas as pd
from collections import defaultdict
from transformers import BartTokenizer
from src.machine_translation import *
from src.machine_translation.data import CodeMixedDataLoader, CodeMixedTokenizedDataset


def get_data_loader_models(
        src: str = "src",
        tgt: str = "tgt",
        train_df: pd.DataFrame = None,
        validation_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        train_batch_size: int = MBART_DATALOADER_TRAIN_BATCH_SIZE,
        validation_batch_size: int = MBART_DATALOADER_VALIDATION_BATCH_SIZE,
        test_batch_size: int = MBART_DATALOADER_TEST_BATCH_SIZE,
        train_shuffle: bool = True,
        validation_shuffle: bool = False,
        test_shuffle: bool = False,
        encoder_tokenizer: BartTokenizer = None,
        encoder_add_special_tokens: bool = MBART_ENCODER_ADD_SPECIAL_TOKENS,
        encoder_max_length: int = MBART_ENCODER_MAX_LENGTH,
        encoder_return_tensors: str = MBART_ENCODER_RETURN_TENSORS,
        encoder_padding: bool = MBART_ENCODER_PADDING,
        encoder_verbose: bool = MBART_ENCODER_VERBOSE,
        decoder_tokenizer: BartTokenizer = None,
        decoder_add_special_tokens: bool = MBART_DECODER_ADD_SPECIAL_TOKENS,
        decoder_max_length: int = MBART_DECODER_MAX_LENGTH,
        decoder_return_tensors: str = MBART_DECODER_RETURN_TENSORS,
        decoder_padding: bool = MBART_DECODER_PADDING,
        decoder_verbose: bool = MBART_DECODER_VERBOSE,
        denoising_stage: bool = False,
        overfit: bool = False,
        overfit_batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False) -> dict:
    """
    Description: Returns the data loader models for all src->tgt languages mentioned in the `config.json` file.
    Input parameters:
        - src: A string containing the source language in config dict.
        - tgt: A string containing the target language in condif dict.
        - train_df: A pandas dataframe containing the training data.
        - validation_df: A pandas dataframe containing the validation data.
        - test_df: A pandas dataframe containing the test data.
        - train_batch_size: An integer containing the training batch size.
        - validation_batch_size: An integer containing the validation batch size.
        - test_batch_size: An integer containing the test batch size.
        - train_shuffle: A boolean value indicating whether to shuffle the training data.
        - validation_shuffle: A boolean value indicating whether to shuffle the validation data.
        - test_shuffle: A boolean value indicating whether to shuffle the test data.
        - encoder_tokenizer: A BART tokenizer object for the encoder.
        - encoder_add_special_tokens: A boolean value indicating whether to add special tokens to the encoder.
        - encoder_max_length: An integer containing the maximum length of the encoder.
        - encoder_return_tensors: A string containing the return tensor type for the encoder.
        - encoder_padding: A boolean value indicating whether to pad the encoder.
        - encoder_verbose: A boolean value indicating whether to print verbose logs for the encoder.
        - decoder_tokenizer: A BART tokenizer object for the decoder.
        - decoder_add_special_tokens: A boolean value indicating whether to add special tokens to the decoder.
        - decoder_max_length: An integer containing the maximum length of the decoder.
        - decoder_return_tensors: A string containing the return tensor type for the decoder.
        - decoder_padding: A boolean value indicating whether to pad the decoder.
        - decoder_verbose: A boolean value indicating whether to print verbose logs for the decoder.
        - denoising_stage: A boolean value indicating whether the data loader is being used for denoising stage.
        - overfit: A boolean value indicating whether to overfit the data loader.
        - overfit_batch_size: An integer containing the batch size for overfitting.
        - num_workers: An integer containing the number of workers for the data loader.
        - pin_memory: A boolean value indicating whether to pin memory for the data loader.
    Returns: A dictionary containing the data loader models for all translations.
    """
    data_loaders = defaultdict(dict)
    for task, translations in MBART_DATALOADER_TRANSLATION_MODELS.items():
        src, tgt = translations["src"], translations["tgt"]
        config = {
            "train_df": train_df,
            "validation_df": validation_df,
            "test_df": test_df,
            "train_batch_size": train_batch_size,
            "validation_batch_size": validation_batch_size,
            "test_batch_size": test_batch_size,
            "train_shuffle": train_shuffle,
            "validation_shuffle": validation_shuffle,
            "test_shuffle": test_shuffle,
            "encoder_tokenizer": encoder_tokenizer,
            "encoder_add_special_tokens": encoder_add_special_tokens,
            "encoder_max_length": encoder_max_length,
            "encoder_return_tensors": encoder_return_tensors,
            "encoder_padding": encoder_padding,
            "encoder_verbose": encoder_verbose,
            "decoder_tokenizer": decoder_tokenizer,
            "decoder_add_special_tokens": decoder_add_special_tokens,
            "decoder_max_length": decoder_max_length,
            "decoder_return_tensors": decoder_return_tensors,
            "decoder_padding": decoder_padding,
            "decoder_verbose": decoder_verbose,
            "denoising_stage": denoising_stage,
            "src_lang": src,
            "tgt_lang": tgt,
            "overfit": overfit,
            "overfit_batch_size": overfit_batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory
        }
        data_loader = CodeMixedDataLoader(**config)
        train_data_loader, validation_data_loader, test_data_loader = data_loader.get_data_loaders()
        data_loaders[task] = {
            "object": data_loader,
            "train": train_data_loader,
            "validation": validation_data_loader,
            "test": test_data_loader
        }
    return data_loaders

def get_tokenized_dataset(
        src: str = "src",
        tgt: str = "tgt",
        train_df: pd.DataFrame = None,
        validation_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        encoder_tokenizer: BartTokenizer = None,
        encoder_add_special_tokens: bool = MBART_ENCODER_ADD_SPECIAL_TOKENS,
        encoder_max_length: int = MBART_ENCODER_MAX_LENGTH,
        encoder_return_tensors: str = MBART_ENCODER_RETURN_TENSORS,
        encoder_padding: bool = MBART_ENCODER_PADDING,
        encoder_verbose: bool = MBART_ENCODER_VERBOSE,
        decoder_tokenizer: BartTokenizer = None,
        decoder_add_special_tokens: bool = MBART_DECODER_ADD_SPECIAL_TOKENS,
        decoder_max_length: int = MBART_DECODER_MAX_LENGTH,
        decoder_return_tensors: str = MBART_DECODER_RETURN_TENSORS,
        decoder_padding: bool = MBART_DECODER_PADDING,
        decoder_verbose: bool = MBART_DECODER_VERBOSE,
        denoising_stage: bool = False,
        overfit: bool = False,
        overfit_batch_size: int = 32) -> dict:
    """
    Description: Returns the tokenized datasets for all src->tgt languages mentioned in the `config.json` file.
    Input parameters:
        - src: A string containing the source language in config dict.
        - tgt: A string containing the target language in condif dict.
        - train_df: A pandas dataframe containing the training data.
        - validation_df: A pandas dataframe containing the validation data.
        - test_df: A pandas dataframe containing the test data.
        - encoder_tokenizer: A BART tokenizer object for the encoder.
        - encoder_add_special_tokens: A boolean value indicating whether to add special tokens to the encoder.
        - encoder_max_length: An integer containing the maximum length of the encoder.
        - encoder_return_tensors: A string containing the return tensor type for the encoder.
        - encoder_padding: A boolean value indicating whether to pad the encoder.
        - encoder_verbose: A boolean value indicating whether to print verbose logs for the encoder.
        - decoder_tokenizer: A BART tokenizer object for the decoder.
        - decoder_add_special_tokens: A boolean value indicating whether to add special tokens to the decoder.
        - decoder_max_length: An integer containing the maximum length of the decoder.
        - decoder_return_tensors: A string containing the return tensor type for the decoder.
        - decoder_padding: A boolean value indicating whether to pad the decoder.
        - decoder_verbose: A boolean value indicating whether to print verbose logs for the decoder.
        - denoising_stage: A boolean value indicating whether the data loader is being used for denoising stage.
        - overfit: A boolean value indicating whether to overfit the data loader.
        - overfit_batch_size: An integer containing the batch size for overfitting.
    Returns: A dictionary containing the data loader models for all translations.
    """
    dataset = defaultdict(dict)
    for task, translations in MBART_DATALOADER_TRANSLATION_MODELS.items():
        src, tgt = translations["src"], translations["tgt"]
        config = {
            "denoising_stage": denoising_stage,
            "src_lang": src,
            "tgt_lang": tgt,
            "encoder_tokenizer": encoder_tokenizer,
            "encoder_add_special_tokens": encoder_add_special_tokens,
            "encoder_max_length": encoder_max_length,
            "encoder_return_tensors": encoder_return_tensors,
            "encoder_padding": encoder_padding,
            "encoder_verbose": encoder_verbose,
            "decoder_tokenizer": decoder_tokenizer,
            "decoder_add_special_tokens": decoder_add_special_tokens,
            "decoder_max_length": decoder_max_length,
            "decoder_return_tensors": decoder_return_tensors,
            "decoder_padding": decoder_padding,
            "decoder_verbose": decoder_verbose,
            "overfit": overfit,
            "overfit_batch_size": overfit_batch_size,
        }
        config["data"] = train_df
        train_dataset = CodeMixedTokenizedDataset(**config)
        config["data"] = validation_df
        validation_dataset = CodeMixedTokenizedDataset(**config)
        config["data"] = test_df
        test_dataset = CodeMixedTokenizedDataset(**config)
        dataset[task] = {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        }
    return dataset