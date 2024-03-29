import random
import pandas as pd
from tqdm import tqdm
from typing import Union
from transformers import BartTokenizer
from torch.utils.data import Dataset, DataLoader
from src.machine_translation import *
from src.data.tokenizer import CustomBartTokenizer
from src.machine_translation.collator import DataCollatorForLanguageMasking, DataCollatorForLanguagePermutation


class CodeMixedDataset(Dataset):

    '''
        Custom dataset class for Code Mixed data
        Note: In denoising stage, the source and target languages are same.
    '''

    def __init__(self, data: pd.DataFrame,
                 denoising_stage: bool = False,
                 src_lang: str = "hi_en",
                 tgt_lang: str = "en",
                 overfit: bool = False,
                 overfit_size: int = 32):
        '''
            Initial definition of the dataset
            Input params:
                data: pandas dataframe containing the data
                denoising_stage: bool, if True, the dataset will be used for denoising
                src_lang: str, source language
                tgt_lang: str, target language
                overfit: bool, if True, the dataset will be overfitted on a small sample
                overfit_size: int, size of the overfit dataset
        '''
        super().__init__()
        self.data = data
        self.denoiising_stage = denoising_stage
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.overfit = overfit
        self.overfit_size = overfit_size
        if self.denoiising_stage:
            self.tgt_lang = self.src_lang

    def __len__(self):
        '''
            Returns the length of the dataset
        '''
        if self.overfit:
            return self.overfit_size
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        '''
            Returns a dict instance of the dataset
        '''
        raw_instance = self.data.iloc[index]
        instance = {
            "src": raw_instance[self.src_lang],
            "tgt": raw_instance[self.tgt_lang]
        }
        return instance


class CodeMixedTokenizedDataset(Dataset):

    '''
        Custom dataset class for Code Mixed data - tokenized version
        Note: In denoising stage, the source and target languages/tokenizers/other related attributes are same.
        The decoder gets overwritten by the encoder.
    '''

    def __init__(self, data: pd.DataFrame,
                 denoising_stage: bool = False,
                 src_lang: str = "hi_en",
                 tgt_lang: str = "en",
                 encoder_tokenizer: Union[BartTokenizer,
                                          CustomBartTokenizer] = None,
                 encoder_add_special_tokens: bool = True,
                 encoder_max_length: int = None,
                 encoder_return_tensors: str = "pt",
                 encoder_padding: Union[bool, str] = True,
                 encoder_truncation: bool = True,
                 encoder_verbose: bool = True,
                 decoder_tokenizer: Union[BartTokenizer,
                                          CustomBartTokenizer] = None,
                 decoder_add_special_tokens: bool = True,
                 decoder_max_length: int = None,
                 decoder_return_tensors: str = "pt",
                 decoder_padding: Union[bool, str] = True,
                 decoder_truncation: bool = True,
                 decoder_verbose: bool = True,
                 overfit: bool = False,
                 overfit_size: int = 32):
        '''
            Initial definition of the dataset
            Input params:
                data: pandas dataframe containing the data
                denoising_stage: bool, if True, the dataset will be used for denoising
                src_lang: str, source language
                tgt_lang: str, target language
                encoder_tokenizer: Union[BartTokenizer, CustomBartTokenizer], tokenizer for the source dataset
                encoder_add_special_tokens: bool, if True, the special tokens will be added
                encoder_max_length: int, maximum length of the sequence
                encoder_return_tensors: str, return tensors for the dataset
                encoder_padding: bool, if True, the dataset will be padded
                encoder_truncation: bool, if True, the dataset will be truncated
                encoder_verbose: bool, if True, the dataset will be printed
                decoder_tokenizer: Union[BartTokenizer, CustomBartTokenizer], tokenizer for the target dataset
                decoder_add_special_tokens: bool, if True, the special tokens will be added
                decoder_max_length: int, maximum length of the sequence
                decoder_return_tensors: str, return tensors for the dataset
                decoder_padding: bool, if True, the dataset will be padded
                decoder_truncation: bool, if True, the dataset will be truncated
                decoder_verbose: bool, if True, the dataset will be printed
                overfit: bool, if True, the dataset will be overfitted on a small sample
                overfit_size: int, size of the overfit dataset
        '''
        super().__init__()
        self.data = data
        self.denoising_stage = denoising_stage
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.encoder_tokenizer = encoder_tokenizer
        self.encoder_add_special_tokens = encoder_add_special_tokens
        self.encoder_max_length = encoder_max_length
        self.encoder_return_tensors = encoder_return_tensors
        self.encoder_padding = encoder_padding
        self.encoder_truncation = encoder_truncation
        self.encoder_verbose = encoder_verbose
        self.decoder_tokenizer = decoder_tokenizer
        self.decoder_add_special_tokens = decoder_add_special_tokens
        self.decoder_max_length = decoder_max_length
        self.decoder_return_tensors = decoder_return_tensors
        self.decoder_padding = decoder_padding
        self.decoder_truncation = decoder_truncation
        self.decoder_verbose = decoder_verbose
        self.overfit = overfit
        self.overfit_size = overfit_size
        if self.denoising_stage:
            self.tgt_lang = self.src_lang
            self.decoder_tokenizer = self.encoder_tokenizer
            self.decoder_add_special_tokens = self.encoder_add_special_tokens
            self.decoder_max_length = self.encoder_max_length
            self.decoder_return_tensors = self.encoder_return_tensors
            self.decoder_padding = self.encoder_padding
            self.decoder_truncation = self.encoder_truncation
            self.decoder_verbose = self.encoder_verbose

    def __len__(self):
        '''
            Returns the length of the dataset
        '''
        if self.overfit:
            return self.overfit_size
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        '''
            Returns a dict instance of the dataset
        '''
        raw_instance = self.data.iloc[index]
        encoder_tokenized_text = self.encoder_tokenizer(
            text=raw_instance[self.src_lang],
            max_length=self.encoder_max_length,
            add_special_tokens=self.encoder_add_special_tokens,
            return_tensors=self.encoder_return_tensors,
            padding=self.encoder_padding,
            truncation=self.encoder_truncation,
            verbose=self.encoder_verbose
        )
        decoder_tokenized_text = self.decoder_tokenizer(
            text=raw_instance[self.tgt_lang],
            max_length=self.decoder_max_length,
            add_special_tokens=self.decoder_add_special_tokens,
            return_tensors=self.decoder_return_tensors,
            padding=self.decoder_padding,
            truncation=self.decoder_truncation,
            verbose=self.decoder_verbose
        )
        labels = decoder_tokenized_text["input_ids"].flatten()
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100
        instance = {
            "input_ids": encoder_tokenized_text["input_ids"].flatten(),
            "attention_mask": encoder_tokenized_text["attention_mask"].flatten(),
            "labels": labels,
        }
        return instance

    def visualize(self) -> None:
        '''
            This function is used to visualize the dataset results.
        '''
        print('#'*100)
        print("Dataset")
        print("Number of instances: ", len(self))
        print("Denoising stage: ", self.denoising_stage)
        print("Source language: ", self.src_lang)
        print("Target language: ", self.tgt_lang)
        for k, v in self[1].items():
            print(f"Key: {k}, Value: {v}, Value shape: {v.size()}")
        print('#'*100)


class CodeMixedDataLoader(DataLoader):

    '''
        Custom dataloader class for Code Mixed data
        Note: In denoising stage, the source and target languages/tokenizers/other related attributes are same.
        The decoder gets overwritten by the encoder.
    '''

    def __init__(self,
                 train_df: pd.DataFrame = None,
                 validation_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None,
                 train_batch_size: int = 32,
                 validation_batch_size: int = 32,
                 test_batch_size: int = 32,
                 train_shuffle: bool = True,
                 validation_shuffle: bool = True,
                 test_shuffle: bool = True,
                 encoder_tokenizer: Union[BartTokenizer,
                                          CustomBartTokenizer] = None,
                 encoder_add_special_tokens: bool = True,
                 encoder_max_length: int = None,
                 encoder_return_tensors: str = "pt",
                 encoder_padding: Union[bool, str] = True,
                 encoder_truncation: bool = True,
                 encoder_verbose: bool = True,
                 decoder_tokenizer: Union[BartTokenizer,
                                          CustomBartTokenizer] = None,
                 decoder_add_special_tokens: bool = True,
                 decoder_max_length: int = None,
                 decoder_return_tensors: str = "pt",
                 decoder_padding: Union[bool, str] = True,
                 decoder_truncation: bool = True,
                 decoder_verbose: bool = True,
                 denoising_stage: bool = False,
                 src_lang: str = "hi_en",
                 tgt_lang: str = "en",
                 overfit: bool = False,
                 overfit_batch_size: int = 32,
                 num_workers: int = None,
                 pin_memory: bool = False,
                 ) -> None:
        '''
            Initial definition of the dataloader
            Input params:
                train_df: pandas dataframe containing the training data
                validation_df: pandas dataframe containing the validation data
                test_df: pandas dataframe containing the test data
                train_batch_size: int, batch size for training
                validation_batch_size: int, batch size for validation
                test_batch_size: int, batch size for testing
                train_shuffle: bool, if True, the training data will be shuffled
                validation_shuffle: bool, if True, the validation data will be shuffled
                test_shuffle: bool, if True, the test data will be shuffled
                encoder_tokenizer: Union[BartTokenizer, CustomBartTokenizer], tokenizer for the source dataset
                encoder_add_special_tokens: bool, if True, the special tokens will be added
                encoder_max_length: int, maximum length of the sequence
                encoder_return_tensors: str, return tensors for the dataset
                encoder_padding: bool, if True, the dataset will be padded
                encoder_truncation: bool, if True, the dataset will be truncated
                encoder_verbose: bool, if True, the dataset will be printed
                decoder_tokenizer: Union[BartTokenizer, CustomBartTokenizer], tokenizer for the target dataset
                decoder_add_special_tokens: bool, if True, the special tokens will be added
                decoder_max_length: int, maximum length of the sequence
                decoder_return_tensors: str, return tensors for the dataset
                decoder_padding: bool, if True, the dataset will be padded
                decoder_truncation: bool, if True, the dataset will be truncated
                decoder_verbose: bool, if True, the dataset will be printed
                denoising_stage: bool, if True, the dataset will be used for denoising
                src_lang: str, source language
                tgt_lang: str, target language
                overfit: bool, if True, the dataset will be overfitted on a small sample
                overfit_batch_size: int, batch size for overfitting
                num_workers: int, number of workers for the dataloader
                pin_memory: bool, if True, the data will be pinned to memory
        '''
        super().__init__(self)
        assert set(PROCESSED_COLUMN_NAMES).issubset(
            set(train_df.columns)), "Column names not found in train dataframe."
        assert set(PROCESSED_COLUMN_NAMES).issubset(
            set(validation_df.columns)), "Column names not found in validation dataframe."
        assert set(PROCESSED_COLUMN_NAMES).issubset(
            set(test_df.columns)), "Column names not found in test dataframe."
        assert src_lang in PROCESSED_COLUMN_NAMES, "Source language not found in column names."
        assert tgt_lang in PROCESSED_COLUMN_NAMES, "Target language not found in column names."
        assert encoder_tokenizer is not None, "Encoder tokenizer cannot be None."
        assert decoder_tokenizer is not None, "Decoder tokenizer cannot be None."
        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.validation_shuffle = validation_shuffle
        self.test_shuffle = test_shuffle
        self.encoder_tokenizer = encoder_tokenizer
        self.encoder_add_special_tokens = encoder_add_special_tokens
        self.encoder_max_length = encoder_max_length
        self.encoder_return_tensors = encoder_return_tensors
        self.encoder_padding = encoder_padding
        self.encoder_truncation = encoder_truncation
        self.encoder_verbose = encoder_verbose
        self.decoder_tokenizer = decoder_tokenizer
        self.decoder_add_special_tokens = decoder_add_special_tokens
        self.decoder_max_length = decoder_max_length
        self.decoder_return_tensors = decoder_return_tensors
        self.decoder_padding = decoder_padding
        self.decoder_truncation = decoder_truncation
        self.decoder_verbose = decoder_verbose
        self.denoising_stage = denoising_stage
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.overfit = overfit
        self.overfit_batch_size = overfit_batch_size
        self.pin_memory = pin_memory
        self.num_workers = os.cpu_count() // 2 if num_workers is None else num_workers
        if self.denoising_stage:
            self.tgt_lang = self.src_lang
            self.decoder_tokenizer = self.encoder_tokenizer
            self.decoder_add_special_tokens = self.encoder_add_special_tokens
            self.decoder_max_length = self.encoder_max_length
            self.decoder_return_tensors = self.encoder_return_tensors
            self.decoder_padding = self.encoder_padding
            self.decoder_truncation = self.encoder_truncation
            self.decoder_verbose = self.encoder_verbose
            self.mlm_data_collator = DataCollatorForLanguageMasking(
                tokenizer=self.encoder_tokenizer)
            self.plm_data_collator = DataCollatorForLanguagePermutation(
                tokenizer=self.encoder_tokenizer)
            self.data_collators = [
                self.mlm_data_collator, self.plm_data_collator]

    def custom_collate_fn(self, batch: list) -> dict:
        '''
            Returns a collated batch of data
            Input params:
                batch: list of data
            Output params:
                a collated batch of data
        '''
        batch = self.collate_fn(batch)
        encoder_tokenized_text = self.encoder_tokenizer(
            text=batch["src"],
            max_length=self.encoder_max_length,
            add_special_tokens=self.encoder_add_special_tokens,
            return_tensors=self.encoder_return_tensors,
            padding=self.encoder_padding,
            truncation=self.encoder_truncation,
            verbose=self.encoder_verbose
        )
        decoder_tokenized_text = self.decoder_tokenizer(
            text=batch["tgt"],
            max_length=self.decoder_max_length,
            add_special_tokens=self.decoder_add_special_tokens,
            return_tensors=self.decoder_return_tensors,
            padding=self.decoder_padding,
            truncation=self.decoder_truncation,
            verbose=self.decoder_verbose
        )
        batch["src_tokenized"] = encoder_tokenized_text["input_ids"]
        batch["src_attention_mask"] = encoder_tokenized_text["attention_mask"]
        batch["tgt_tokenized"] = decoder_tokenized_text["input_ids"]
        batch["tgt_attention_mask"] = decoder_tokenized_text["attention_mask"]
        if self.denoising_stage:
            data_collator = random.choice(self.data_collators)
            batch = data_collator(batch)
        return batch

    def get_data_loaders(self) -> tuple:
        '''
            Returns a dataloader for the given dataset
            Input params: None
            Output params: a tuple of dataloaders for train, val and test
        '''
        train_dataset = CodeMixedDataset(
            data=self.train_df,
            denoising_stage=self.denoising_stage,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            overfit=self.overfit,
            overfit_size=self.overfit_batch_size
        )
        validation_dataset = CodeMixedDataset(
            data=self.validation_df,
            denoising_stage=self.denoising_stage,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            overfit=self.overfit,
            overfit_size=self.overfit_batch_size
        )
        test_dataset = CodeMixedDataset(
            data=self.test_df,
            denoising_stage=self.denoising_stage,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            overfit=self.overfit,
            overfit_size=self.overfit_batch_size
        )
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.custom_collate_fn
        )
        validation_data_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=self.validation_batch_size,
            shuffle=self.validation_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.custom_collate_fn
        )
        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.test_batch_size,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.custom_collate_fn
        )
        return (train_data_loader, validation_data_loader, test_data_loader)

    def visualize(self) -> None:
        '''
            This function is used to visualize the dataset results.
            Input params: None
            Output params: None
        '''
        train_data_loader, validation_data_loader, test_data_loader = self.get_data_loaders()
        print('#'*100)
        print("Train Dataloader")
        print("Batch Size: ", self.train_batch_size)
        print("Number of batches: ", len(train_data_loader))
        batch = next(iter(train_data_loader))
        batch_src = batch["src"]
        batch_src_tokenized = batch["src_tokenized"]
        batch_src_attention_mask = batch["src_attention_mask"]
        batch_tgt = batch["tgt"]
        batch_tgt_tokenized = batch["tgt_tokenized"]
        batch_tgt_attention_mask = batch["tgt_attention_mask"]
        if self.denoising_stage and "denoising_src_tokenized" in batch.keys():
            batch_denoising_src_tokenized = batch["denoising_src_tokenized"]
            batch_denoising_labels = batch["denoising_tgt_tokenized"]
            batch_denoising_mask_labels = batch["denoising_mask_tgt_tokenized"]
            batch_denoising_mask = batch["denoising_mask"]
        print("Batch source language shape: ", batch_src_tokenized.shape)
        print("Batch source language: ", batch_src)
        print("Batch source tokens: ", batch_src_tokenized)
        print("Batch source attention mask: ", batch_src_attention_mask)
        print("Batch target language shape: ", batch_tgt_tokenized.shape)
        print("Batch target language: ", batch_tgt)
        print("Batch target tokens: ", batch_tgt_tokenized)
        print("Batch target attention mask: ", batch_tgt_attention_mask)
        if self.denoising_stage:
            print("Batch denoising source language shape: ",
                  batch_denoising_src_tokenized.shape)
            print("Batch denoising source tokens: ",
                  batch_denoising_src_tokenized)
            batch_denoising_src_decoded = self.encoder_tokenizer.batch_decode(
                batch_denoising_src_tokenized)
            print("Batch denoising source decoded: ",
                  batch_denoising_src_decoded)
            print("Batch denoising labels: ", batch_denoising_labels)
            print("Batch denoising mask labels: ", batch_denoising_mask_labels)
            print("Batch denoising mask: ", batch_denoising_mask)
        print("Validating train laoder...")
        for batch in tqdm(train_data_loader):
            _ = batch.values()
        print("Validation of train loader successful.")
        print('#'*100)
        print("Val Dataloader")
        print("Batch Size: ", self.validation_batch_size)
        print("Number of batches: ", len(validation_data_loader))
        print("Validating validation laoder...")
        for batch in tqdm(validation_data_loader):
            _ = batch.values()
        print("Validation of validation loader successful.")
        print('#'*100)
        print("Test Dataloader")
        print("Batch Size: ", self.test_batch_size)
        print("Number of batches: ", len(test_data_loader))
        print("Validating test laoder...")
        for batch in tqdm(test_data_loader):
            _ = batch.values()
        print("Validation of test loader successful.")
        print('#'*100)
