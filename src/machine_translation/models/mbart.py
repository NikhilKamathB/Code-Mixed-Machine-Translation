import torch
from transformers import MBartModel


class MBart:

    def __init__(self,
                 pretrained_path: str = "facebook/bart-large",
                 device: str = "cpu",
                 max_length: int = 64,
                 num_beams: int = 4,) -> None:
        '''
            Initializes the MBART model.
            Input params:
                pretrained_path: str -> The path to the pretrained model.
                device: str -> The device to use for training.
                max_length: int -> The maximum length of the sequence.
                num_beams: int -> The number of beams to use for beam search.

        '''
        self.model = MBartModel.from_pretrained(pretrained_path)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
        self.config()
        self.model.to(self.device)

    def config(self) -> None:
        '''
            Configures the model.
        '''
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
        self.model.config.max_length = self.max_length
        self.model.config.num_beams = self.num_beams