import torch
from transformers import BartForConditionalGeneration as BartModelGen
from src.machine_translation import *


class BartForConditionalGeneration:

    def __init__(self,
                 pretrained: bool = MBART_MODEL_CONDITIONAL_GENERATION_USE_PRETRAINED,
                 pretrained_path: str = MBART_MODEL_CONDITIONAL_GENERATION_FROM_PRETRAINED,
                 device: str = MBART_MODEL_CONDITIONAL_GENERATION_DEVICE) -> None:
        '''
            Initializes the BartForConditionalGeneration model.
            Input params:
                pretrained: bool -> Whether to use a pretrained model or not.
                pretrained_path: str -> The path to the pretrained model.
                device: str -> The device to use for training.
        '''
        if pretrained:
            self.model = BartModelGen.from_pretrained(pretrained_path)
        else:
            self.model = BartModelGen()
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.model.to(self.device)

    def configure(self, embedding_size: int = MBART_MODEL_CONDITIONAL_GENERATION_VOCAB) -> None:
        '''
            Configures the model.
        '''
        self.model.config.vocab_size = embedding_size
        self.model.resize_token_embeddings(embedding_size)