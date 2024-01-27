import copy
import torch
import random
from transformers.utils import PaddingStrategy
from typing import Union, Tuple, List, Dict, Any, Optional
from transformers import BartTokenizer, DataCollatorWithPadding, BatchEncoding
from src.data.tokenizer import CustomBartTokenizer


class DataCollatorDenoisingMixin:

    '''
        Data collator mixin for machine translation tasks.
    '''

    def __call__(self, features: Union[dict, BatchEncoding]) -> Union[dict, BatchEncoding]:
        '''
            Calls the data collator.
            Input parameters:
                - features: A dictionary/ba containing the features.
            Outputs: A dictionary containing the processed features.
        '''
        assert isinstance(features, dict) or isinstance(
            features, BatchEncoding), "`features` should be either a dictionary or a BatchEncoding object."
        _features_copy = None
        if isinstance(features, BatchEncoding):
            features = features.data
            _features_copy = copy.deepcopy(features)
        if self.return_type == "batch_encoding":
            assert self.replace, "If `return_type` is `batch_encoding`, `replace` should be `True`."
        assert "input_ids" in features or "src_tokenized" in features, "`input_ids` or `src_tokenized` should be a key in `features`."
        assert "labels" in features or "tgt_tokenized" in features, "`labels` or `tgt_tokenized` should be a key in `features`."
        input_ids, labels = "input_ids", "labels"
        if "src_tokenized" in features:
            input_ids = "src_tokenized"
        if "tgt_tokenized" in features:
            labels = "tgt_tokenized"
        processed_input_ids, processed_labels, processed_mask = self._torch_call(
            input_ids=features[input_ids])
        if self.replace:
            features[input_ids] = processed_input_ids
            features[labels] = features[labels]
            features[f"mask_{labels}"] = processed_labels
            features["mask"] = processed_mask
        else:
            features[f"denoising_{input_ids}"] = processed_input_ids
            features[f"denoising_{labels}"] = features[labels]
            features[f"denoising_mask_{labels}"] = processed_labels
            features[f"denoising_mask"] = processed_mask
        if self.return_type == "batch_encoding":
            features = BatchEncoding(features)
        # This is to handle the `forward()` of the implementer class.
        if self.hg_trainer:
            if f"mask_{labels}" in features:
                del features[f"mask_{labels}"]
            if "mask" in features:
                del features["mask"]
        return features

    def _get_special_token_mask(self, input_ids: torch.Tensor, tokenizer: Union[BartTokenizer, CustomBartTokenizer]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
            Gets the special token mask.
            Input parameters:
                input_ids: Tensor containing the input ids.
                tokenizer: A BartTokenizer or CustomBartTokenizer object.
            Outputs: A tuple containing the special token mask and the special token indices.
        '''
        self.special_tokens_mask = torch.tensor([
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ])
        self.special_tokens_indices = torch.nonzero(
            self.special_tokens_mask == 1, as_tuple=False)
        return (self.special_tokens_mask, self.special_tokens_indices)

    def get_pad_token_mask(self, input_ids: torch.Tensor, tokenizer: Union[BartTokenizer, CustomBartTokenizer]) -> torch.Tensor:
        '''
            Gets the pad token mask.
            Input parameters:
                input_ids: Tensor containing the input ids.
                tokenizer: A BartTokenizer or CustomBartTokenizer object.
            Outputs: A tensor containing the pad token mask.
        '''
        self.pad_tokens_indices = torch.nonzero(
            input_ids == tokenizer.pad_token_id, as_tuple=False)
        return self.pad_tokens_indices

    def get_window_start_indices(self, start_index: int, end_index: int, window_size: int) -> list:
        '''
            Gets valid window start indices.
            Input parameters:
                start_index: An integer containing the start index.
                end_index: An integer containing the end index.
                window_size: An integer containing the window size.
            Outputs: A list containing the window indices.
        '''
        window_indices = []
        for idx in range(start_index, end_index-window_size+1):
            window_indices.append(idx)
        return window_indices

    def group_mask_tokens(self, input_ids: torch.Tensor, lm_probability: float = 0.15, max_length: int = 5, min_window_length: int = 0) -> torch.Tensor:
        '''
            Masks a group of contiguous subject tokens.
            Input parameters:
                input_ids: Tensor containing the input ids.
                lm_probability: A float containing the probability of masking the language.
                max_length: An integer containing the maximum length for the group mask.
                min_window_length: An integer containing the minimum window length.
            Outputs: A tensor containing the masked input ids.
        '''
        group_mask_pos = torch.ones(input_ids.shape)
        fixed_tokens_indices = torch.cat(
            (self.special_tokens_indices, self.pad_tokens_indices), dim=0)
        group_mask_pos[fixed_tokens_indices[:, 0],
                       fixed_tokens_indices[:, 1]] = 0
        for row_idx in range(group_mask_pos.shape[0]):
            if random.random() < lm_probability:
                one_indices = group_mask_pos[row_idx].nonzero().squeeze()
                group_mask_pos[row_idx] = 0
                if one_indices.nelement() <= 1:
                    continue
                else:
                    start_index = one_indices[0].item()
                    end_index = one_indices[-1].item()
                    window_size = max(min_window_length, random.randint(
                        0, min(max_length, end_index - start_index + 1)))
                    if window_size == 0:
                        continue
                    valid_start_indices = self.get_window_start_indices(
                        start_index=start_index, end_index=end_index, window_size=window_size)
                    valid_end_indices = [
                        i + window_size for i in valid_start_indices]
                    valid_indices_range = list(
                        zip(valid_start_indices, valid_end_indices))
                    if not valid_indices_range or len(valid_indices_range) == 0:
                        continue
                    candidate_index = random.choice(valid_indices_range)
                    group_mask_pos[row_idx, candidate_index[0]:candidate_index[1]] = 1
            else:
                group_mask_pos[row_idx] = 0
        return group_mask_pos


class DataCollatorForLanguageMasking(DataCollatorDenoisingMixin):

    '''
        Data collator for language masking tasks.
        Example: Unit masking
            Input: "Hello, this is for masking task."
            Output (masked): "Hello, <mask> is for <mask> task."
        Example: Group masking
            Input: "Hello, this is for masking task."
            Output (masked): "Hello, <mask> <mask> <mask> masking task."
    '''

    def __init__(self, tokenizer: Union[BartTokenizer, CustomBartTokenizer], mlm: bool = True, enable_group_mask: bool = True, max_length: int = 5, mlm_probability: float = 0.15, style_switch_probability: float = 0.15, replace: bool = False, return_type: str = "dict", hg_trainer: bool = False) -> None:
        '''
            Initalizes the data collator for language masking tasks.
            Input parameters:
                tokenizer: A BartTokenizer or CustomBartTokenizer object.
                mlm: A boolean indicating whether to mask the language or not.
                enable_group_mask: A boolean indicating whether or not to mask a group of tokens.
                max_length: An integer containing the maximum length for the group mask.
                mlm_probability: A float containing the probability of masking the language.
                style_switch_probability: A float containing the probability of switching the style [style available: mask unit token | mask group of contiguous tokens]
                                          Defaults to 0.15, with main focus on unit masking.
                replace: Make changes inplace or not.
                return_type: A string containing the return type.
                hg_trainer: A boolean indicating whether the data collator is used with huggingface trainer.
        '''
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.enable_group_mask = enable_group_mask
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.style_switch_probability = style_switch_probability
        self.replace = replace
        self.return_type = return_type
        self.hg_trainer = hg_trainer

    def _mask_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''
            Masks the input ids.
            Input parameters:
                input_ids: Tensor containing the input ids.
            Outputs: A tensor containing the masked input ids.
        '''
        mask_pos = torch.where(
            (self.special_tokens_mask == 0) & (input_ids != -100), 1, 0)
        mask_pos[(torch.rand(mask_pos.shape) >= self.mlm_probability)
                 & (mask_pos != 0)] = 0
        return mask_pos

    def _process_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Processes the input ids.
            Input parameters:
                input_ids: Tensor containing the input ids.
            Outputs: A tuple containing the processed input ids, the labels and the mask.
        '''
        _, _ = self._get_special_token_mask(
            tokenizer=self.tokenizer, input_ids=input_ids)
        _ = self.get_pad_token_mask(
            tokenizer=self.tokenizer, input_ids=input_ids)
        if self.enable_group_mask and random.random() < self.style_switch_probability:
            mask = self.group_mask_tokens(
                input_ids=input_ids, lm_probability=self.mlm_probability, max_length=self.max_length)
        else:
            mask = self._mask_tokens(input_ids=input_ids)
        labels = input_ids.clone()
        labels[mask == 0] = -100
        input_ids[mask == 1] = self.tokenizer.mask_token_id
        return (input_ids, labels, mask)

    def _torch_call(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Calls the data collator for language masking tasks.
            Input parameters:
                input_ids: Tensor containing the input ids.
            Outputs: Tuple containing the processed input ids, the labels and the mask.
        '''
        if self.mlm:
            return self._process_input_ids(input_ids=input_ids)
        else:
            labels = torch.full(input_ids.shape, -100).int()
            mask = torch.zeros(input_ids.shape).int()
            return (input_ids, labels, mask)


class DataCollatorForLanguagePermutation(DataCollatorDenoisingMixin):

    '''
        Data collator for language permutation tasks.
        Example:
            Input: "Hello, this is for masking task."
            Output (masked): "Hello, for is this task."
    '''

    def __init__(self, tokenizer: Union[BartTokenizer, CustomBartTokenizer], plm: bool = True, max_length: int = 5, plm_probability: float = 0.5, min_window_length: int = 2, replace: bool = False, return_type: str = "dict", hg_trainer: bool = False) -> None:
        '''
            Initalizes the data collator for language permutation tasks.
            Input parameters:
                tokenizer: A BartTokenizer or CustomBartTokenizer object.
                plm: A boolean indicating whether to mask the language or not.
                max_length: An integer containing the maximum length for the group mask.
                plm_probability: A float containing the probability of masking the language.
                min_window_length: An integer containing the minimum window length.
                replace: Make changes inplace or not.
                return_type: A string containing the return type.
                hg_trainer: A boolean indicating whether the data collator is used with huggingface trainer.
        '''
        super().__init__()
        self.tokenizer = tokenizer
        self.plm = plm
        self.max_length = max_length
        self.plm_probability = plm_probability
        self.min_window_length = min_window_length
        self.replace = replace
        self.return_type = return_type
        self.hg_trainer = hg_trainer

    def _permute_tokens(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
            Permutes the input ids.
            Input parameters:
                input_ids: Tensor containing the input ids.
                mask: Tensor containing the mask.
            Outputs: A tensor containing the permuted input ids.
        '''
        permuted_input_ids = input_ids.clone()
        for row_idx in range(mask.shape[0]):
            if mask[row_idx].sum() == 0:
                continue
            else:
                one_indices = mask[row_idx].nonzero().squeeze()
                if one_indices.nelement() <= 1:
                    continue
                permuted_input_ids[row_idx, one_indices] = permuted_input_ids[row_idx,
                                                                              one_indices[torch.randperm(one_indices.nelement())]]
        return permuted_input_ids

    def _process_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Processes the input ids.
            Input parameters:
                input_ids: Tensor containing the input ids.
            Outputs: A tuple containing the processed input ids, the labels and the mask.
        '''
        self.special_tokens_mask, self.special_tokens_indices = self._get_special_token_mask(
            tokenizer=self.tokenizer, input_ids=input_ids)
        self.pad_tokens_indices = self.get_pad_token_mask(
            tokenizer=self.tokenizer, input_ids=input_ids)
        mask = self.group_mask_tokens(input_ids=input_ids, lm_probability=self.plm_probability,
                                      max_length=self.max_length, min_window_length=self.min_window_length)
        permuted_input_ids = self._permute_tokens(
            input_ids=input_ids, mask=mask)
        labels = input_ids.clone()
        labels[mask == 0] = -100
        return (permuted_input_ids, labels, mask)

    def _torch_call(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Calls the data collator for language masking tasks.
            Input parameters:
                input_ids: Tensor containing the input ids.
            Outputs: Tuple containing the processed input ids, the labels and the mask.
        '''
        if self.plm:
            return self._process_input_ids(input_ids=input_ids)
        else:
            labels = torch.full(input_ids.shape, -100).int()
            mask = torch.zeros(input_ids.shape).int()
            return (input_ids, labels, mask)


class DataCollator:

    '''
        Designed to be used with HuggingFace Trainer.
        All in one data collator for machine translation tasks. It supports:
            1. Padding
            2. Masking and group masking
            3. Permutation
    '''

    def __init__(self,
                 tokenizer: Union[BartTokenizer, CustomBartTokenizer],
                 mlm: bool = True,
                 plm: bool = True,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 enable_group_mask: bool = True,
                 denoising_stage: bool = False,
                 mask_max_length: int = 5,
                 permute_mask_length: int = 5,
                 padding_max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 return_tensors: str = "pt",
                 mlm_probability: float = 0.15,
                 plm_probability: float = 0.5,
                 min_window_length: int = 2,
                 style_switch_probability: float = 0.15) -> None:
        '''
            Initalizes the data collator for machine translation tasks.
            Input parameters:
                tokenizer: A BartTokenizer or CustomBartTokenizer object.
                mlm: A boolean indicating whether to mask the language or not.
                plm: A boolean indicating whether to permute the language or not.
                padding: A boolean indicating whether to pad the language or not.
                enable_group_mask: A boolean indicating whether or not to mask a group of tokens.
                denoising_stage: A boolean indicating whether or not to perform denoising.
                mask_max_length: An integer containing the maximum length for the group mask.
                permute_mask_length: An integer containing the maximum length for the permutation mask.
                padding_max_length: An integer containing the maximum length for the padding.
                pad_to_multiple_of: An integer containing the padding multiple.
                return_tensors: A string containing the return tensor type.
                mlm_probability: A float containing the probability of masking the language.
                plm_probability: A float containing the probability of permuting the language.
                min_window_length: An integer containing the minimum window length.
                style_switch_probability: A float containing the probability of switching the style [style available: mask unit token | mask group of contiguous tokens]
                                          Defaults to 0.15, with main focus on unit masking.
        '''
        self.plm = plm
        self.plm_probability = plm_probability
        self.min_window_length = min_window_length
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.padding = padding
        self.padding_max_length = padding_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.enable_group_mask = enable_group_mask
        self.denoising_stage = denoising_stage
        self.mask_max_length = mask_max_length
        self.permute_mask_length = permute_mask_length
        self.mlm_probability = mlm_probability
        self.style_switch_probability = style_switch_probability
        self.data_padding_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.padding_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        if self.denoising_stage:
            data_masking_collator = DataCollatorForLanguageMasking(
                tokenizer=self.tokenizer,
                mlm=self.mlm,
                enable_group_mask=self.enable_group_mask,
                max_length=self.mask_max_length,
                mlm_probability=self.mlm_probability,
                style_switch_probability=self.style_switch_probability,
                replace=True,
                return_type="batch_encoding",
                hg_trainer=True
            )
            data_permutation_collator = DataCollatorForLanguagePermutation(
                tokenizer=self.tokenizer,
                plm=self.plm,
                max_length=self.permute_mask_length,
                plm_probability=self.plm_probability,
                min_window_length=self.min_window_length,
                replace=True,
                return_type="batch_encoding",
                hg_trainer=True
            )
            self.denoising_data_collator = [
                data_masking_collator, data_permutation_collator]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''
            Calls the data collator.
            Input parameters:
                - features: A dictionary containing the features.
            Outputs: A dictionary containing the processed features.
        '''
        batch = self.data_padding_collator(features=features)
        if self.denoising_stage:
            denoising_collator = random.choice(self.denoising_data_collator)
            batch = denoising_collator(features=batch)
        return batch
