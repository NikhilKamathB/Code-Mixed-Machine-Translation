import torch
import random
from typing import Union, Tuple
from transformers import BartTokenizer
from src.data.tokenizer import CustomBartTokenizer


class DataCollatorMixin: 

    '''
        Data collator mixin for machine translation tasks.
    '''

    def __call__(self, features: dict) -> dict:
        '''
            Description: Calls the data collator.
            Input parameters:
                - features: A dictionary containing the features.
            Outputs: A dictionary containing the processed features.
        '''
        assert isinstance(features, dict), "`features` should be a dictionary."
        assert "input_ids" in features or "src_tokenized" in features, "`input_ids` or `src_tokenized` should be a key in `features`."
        input_ids = "input_ids"
        if "src_tokenized" in features:
            input_ids = "src_tokenized"
        processed_input_ids, processed_labels, processed_mask = self._torch_call(input_ids=features[input_ids])
        features[f"denoising_{input_ids}"] = processed_input_ids
        features[f"denoising_labels"] = processed_labels
        features[f"denoising_mask"] = processed_mask
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
        self.special_tokens_indices = torch.nonzero(self.special_tokens_mask == 1, as_tuple=False)
        return (self.special_tokens_mask, self.special_tokens_indices)

    def get_pad_token_mask(self, input_ids: torch.Tensor, tokenizer: Union[BartTokenizer, CustomBartTokenizer]) -> torch.Tensor:
        '''
            Gets the pad token mask.
            Input parameters:
                input_ids: Tensor containing the input ids.
                tokenizer: A BartTokenizer or CustomBartTokenizer object.
            Outputs: A tensor containing the pad token mask.
        '''
        self.pad_tokens_indices = torch.nonzero(input_ids == tokenizer.pad_token_id, as_tuple=False)
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
        fixed_tokens_indices = torch.cat((self.special_tokens_indices, self.pad_tokens_indices), dim=0)
        group_mask_pos[fixed_tokens_indices[:, 0], fixed_tokens_indices[:, 1]] = 0
        for row_idx in range(group_mask_pos.shape[0]):
            if random.random() < lm_probability:
                one_indices = group_mask_pos[row_idx].nonzero().squeeze()
                group_mask_pos[row_idx] = 0
                if one_indices.nelement() <= 1:
                    continue
                else:
                    start_index = one_indices[0].item()
                    end_index = one_indices[-1].item()
                    window_size = max(min_window_length, random.randint(0, min(max_length, end_index - start_index + 1)))
                    if window_size == 0:
                        continue
                    valid_start_indices = self.get_window_start_indices(start_index=start_index, end_index=end_index, window_size=window_size)
                    valid_end_indices = [i + window_size for i in valid_start_indices]
                    valid_indices_range = list(zip(valid_start_indices, valid_end_indices))
                    if not valid_indices_range or len(valid_indices_range) == 0:
                        continue
                    candidate_index = random.choice(valid_indices_range)
                    group_mask_pos[row_idx, candidate_index[0]:candidate_index[1]] = 1
            else:
                group_mask_pos[row_idx] = 0
        return group_mask_pos


class DataCollatorForLanguageMasking(DataCollatorMixin):

    '''
        Data collator for language masking tasks.
        Example: Unit masking
            Input: "Hello, this is for masking task."
            Output (masked): "Hello, <mask> is for <mask> task."
        Example: Group masking
            Input: "Hello, this is for masking task."
            Output (masked): "Hello, <mask> <mask> <mask> masking task."
    '''

    def __init__(self, tokenizer: Union[BartTokenizer, CustomBartTokenizer], mlm: bool = True, enable_group_mask: bool = True, max_length: int = 5, mlm_probability: float = 0.15, style_switch_probability: float = 0.15) -> None:
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
        '''
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.enable_group_mask = enable_group_mask
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.style_switch_probability = style_switch_probability

    def _mask_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        '''
            Masks the input ids.
            Input parameters:
                input_ids: Tensor containing the input ids.
            Outputs: A tensor containing the masked input ids.
        '''
        mask_pos = torch.where((self.special_tokens_mask == 0) & (input_ids != -100), 1, 0)
        mask_pos[(torch.rand(mask_pos.shape) >= self.mlm_probability) & (mask_pos !=0)] = 0
        return mask_pos
    
    def _process_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Processes the input ids.
            Input parameters:
                input_ids: Tensor containing the input ids.
            Outputs: A tuple containing the processed input ids, the labels and the mask.
        '''
        _, _ = self._get_special_token_mask(tokenizer=self.tokenizer, input_ids=input_ids)
        _ = self.get_pad_token_mask(tokenizer=self.tokenizer, input_ids=input_ids)
        if self.enable_group_mask and random.random() < self.style_switch_probability:
            mask = self.group_mask_tokens(input_ids=input_ids, lm_probability=self.mlm_probability, max_length=self.max_length)
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


class DataCollatorForLanguagePermutation(DataCollatorMixin):

    '''
        Data collator for language permutation tasks.
        Example:
            Input: "Hello, this is for masking task."
            Output (masked): "Hello, for is this task."
    '''

    def __init__(self, tokenizer: Union[BartTokenizer, CustomBartTokenizer], plm: bool = True, max_length: int = 5, plm_probability: float = 0.5, min_window_length: int = 2) -> None:
        '''
            Initalizes the data collator for language permutation tasks.
            Input parameters:
                tokenizer: A BartTokenizer or CustomBartTokenizer object.
                plm: A boolean indicating whether to mask the language or not.
                max_length: An integer containing the maximum length for the group mask.
                plm_probability: A float containing the probability of masking the language.
                min_window_length: An integer containing the minimum window length.
        '''
        super().__init__()
        self.tokenizer = tokenizer
        self.plm = plm
        self.max_length = max_length
        self.plm_probability = plm_probability
        self.min_window_length = min_window_length
    
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
                permuted_input_ids[row_idx, one_indices] = permuted_input_ids[row_idx, one_indices[torch.randperm(one_indices.nelement())]]
        return permuted_input_ids
    
    def _process_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Processes the input ids.
            Input parameters:
                input_ids: Tensor containing the input ids.
            Outputs: A tuple containing the processed input ids, the labels and the mask.
        '''
        self.special_tokens_mask, self.special_tokens_indices = self._get_special_token_mask(tokenizer=self.tokenizer, input_ids=input_ids)
        self.pad_tokens_indices = self.get_pad_token_mask(tokenizer=self.tokenizer, input_ids=input_ids)
        mask = self.group_mask_tokens(input_ids=input_ids, lm_probability=self.plm_probability, max_length=self.max_length, min_window_length=self.min_window_length)
        permuted_input_ids = self._permute_tokens(input_ids=input_ids, mask=mask)
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