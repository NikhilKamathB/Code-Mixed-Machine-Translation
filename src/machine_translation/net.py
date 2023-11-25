import os
import copy
import time
import torch
import random
import string
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from datasets import load_metric
from transformers import GenerationConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.data import *
from src.machine_translation import *
from src.data.tokenizer import CustomBartTokenizer
from src.machine_translation.models.bart_conditional import BartForConditionalGeneration
from transformers import TrainingArguments, Trainer



class CodeMixedModelHGTrainer:

    '''
        Main class for the Code Mixed Model training via HuggingFace Trainer
    '''

    def __init__(self,
                 train_dataset: object = None,
                 validation_dataset: object = None,
                 test_dataset: object = None,
                 use_pretrained: bool = MBART_MODEL_CONDITIONAL_GENERATION_USE_PRETRAINED,
                 from_pretrained: str = MBART_MODEL_CONDITIONAL_GENERATION_FROM_PRETRAINED,
                 save_model: bool = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_MODEL,
                 save_path_dir: str = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_PATH,
                 saved_model_path: str = MBART_MODEL_CONDITIONAL_GENERATION_LOAD_PATH,
                 model_name: str = MBART_MODEL_CONDITIONAL_GENERATION_TYPE,
                 k_random: int = MBART_MODEL_CONDITIONAL_GENERATION_K_RANDOM,
                 epochs: int = MBART_MODEL_CONDITIONAL_GENERATION_EPOCHS,
                 device: str = MBART_MODEL_CONDITIONAL_GENERATION_DEVICE,
                 verbose: bool = MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE,
                 verbose_step: int = MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE_STEP,
                 freeze: bool = MBART_MODEL_CONDITIONAL_GENERATION_FREEZE_MODEL,
                 trainable_layers: list = None,
                 train_batch_size: int = MBART_DATALOADER_TRAIN_BATCH_SIZE,
                 validation_batch_size: int = MBART_DATALOADER_VALIDATION_BATCH_SIZE,
                 test_batch_size: int = MBART_DATALOADER_TEST_BATCH_SIZE
                ) -> None:
        '''
            Initial definition of the Code Mixed Model
        '''
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.use_pretrained = use_pretrained
        self.from_pretrained = from_pretrained
        self.save_path_dir = save_path_dir
        self.saved_model_path = saved_model_path
        self.model_name = model_name
        self.epochs=epochs
        self.device = device
        self.save_model = save_model
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.freeze = freeze
        self.trainable_layers = trainable_layers
        self.optimizer = None
        self.scheduler = None
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.save_path = self.save_path_dir + \
            self.model_name + \
                ''.join(random.choices(string.ascii_uppercase + string.digits, k=k_random)) + ".pth" \
                    if self.save_path_dir is not None else None
        

        
    def _get_model(self, model_name: str = MBART_MODEL_CONDITIONAL_GENERATION_TYPE) -> object:
        '''
            Returns the model object
            Input params:
                model_name: str, name of the model to be used | default: MBart | options: MBart
        '''
        if model_name == MBART_MODEL_CONDITIONAL_GENERATION_TYPE:
            return BartForConditionalGeneration(
                pretrained=self.use_pretrained,
                pretrained_path=self.from_pretrained
            ).model
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

    def _configure_optimizer(self) -> None:
        pass

    def _configure_scheduler(self) -> None:
        pass

    def _configure_criterion(self) -> None:
        pass

    def _configure_model(self) -> None:
        pass

    def _configure_trainer(self) -> None:
        pass

    def train(self) -> None:
        pass
    
    def _configure_training_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.save_path_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.validation_batch_size
        )
    
    def _freeze_weights(self) -> None:
        '''
            Freezes the weights of the model given a list of layers to be trained, if no list is provided, all the layers are frozen
        '''
        if self.verbose: print("Freezing the model...")
        for name, param in self.model.named_parameters():
            if self.trainable_layers and name in self.trainable_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if self.trainable_layers is None:
            self.skip_train = True
    
    def _save(self, model: object) -> None:
        '''
            This function saves the model
            Input params: model - a model object
        '''
        os.makedirs(self.save_path_dir, exist_ok=True)
        model.to("cpu")
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_path)
        print(f"Model saved to -> {self.save_path}...")
        model.to(self.device)

    def _train(self) -> tuple:

        training_args = self._configure_training_arguments()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            optimizers=(self.optimizer, self.scheduler)
        )
        
        trainer.train()

        # Save the trained model
        self.model.save_pretrained(self.save_path)
        return (self.model.to("cpu"), self.best_model.to("cpu"))

    def _configure_optimizers(self) -> None:
        '''
            This function configures the optimizer and scheduler.
        '''
        if self.verbose: print("\nConfiguring optimizer and scheduler...")
        if MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_TYPE == "AdamW":
            optimizer = AdamW(
                params=self.model.parameters(),
                lr=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_LR,
                betas=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_BETAS,
                eps=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_EPS,
                weight_decay=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_WEIGHT_DECAY,
                correct_bias=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_CORRECT_BIAS,
                no_deprecation_warning=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_NO_DEPRICATION_WARNING
            )
        else:
            raise NotImplementedError(f"Optimizer {MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_TYPE} not implemented.")
        if MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE == "get_linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_WARMUP_STEPS,
                num_training_steps=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TRAINING_STEPS,
                last_epoch=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_LAST_EPOCH
            )
        else:
            raise NotImplementedError(f"Scheduler {MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE} not implemented.")
        self.optimizer = optimizer
        self.scheduler = scheduler

    def load_model(self) -> object:
        '''
            Loads the saved model
        '''
        if self.verbose:
            print("Loading model...")
        self.model = self._get_model(self.model_name)
        if self.saved_model_path is not None:
            state_dict = torch.load(self.saved_model_path, map_location=torch.device("cpu"))
            if 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
            if 'epoch' in state_dict:
                self.start_epoch = state_dict['epoch']
            if self.verbose:
                print(f"Loaded model from {self.saved_model_path}...")
        else:
            if self.verbose:
                print("No saved model to load as `saved_model_path` was not provided or is None.")
        self.model.to(self.device)
        if self.freeze:
            self._freeze_weights()
        self.best_model = copy.deepcopy(self.model)
        if self.verbose:
            print(self.model)
        return self.model
    
    def fit(self) -> tuple:
        '''
            This function fits the model
            Input params: None
            Returns: tuple of the model and the best model
        '''
        _ = self.load_model()
        self._configure_optimizers()
        return self._train()

class CodeMixedModel:

    '''
        Main class for the Code Mixed Model
    '''

    def __init__(self,
                 train_data_loader: object = None,
                 validation_data_loader: object = None,
                 test_data_loader: object = None,
                 encoder_tokenizer: CustomBartTokenizer = None,
                 decoder_tokenizer: CustomBartTokenizer = None,
                 epochs: int = MBART_MODEL_CONDITIONAL_GENERATION_EPOCHS,
                 device: str = MBART_MODEL_CONDITIONAL_GENERATION_DEVICE,
                 save_model: bool = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_MODEL,
                 save_path_dir: str = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_PATH,
                 saved_model_path: str = MBART_MODEL_CONDITIONAL_GENERATION_LOAD_PATH,
                 model_name: str = MBART_MODEL_CONDITIONAL_GENERATION_TYPE,
                 generate_config: dict = {
                        "min_length": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_MIN_LENGTH,
                        "max_length": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_MAX_LENGTH,
                        "early_stopping": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_EARLY_STOPPING,
                        "num_beams": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_NUM_BEAMS,
                        "temperature": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_TEMPERATURE,
                        "top_k": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_TOP_K,
                        "top_p": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_TOP_P
                 },
                 encoder_add_special_tokens: bool = MBART_ENCODER_ADD_SPECIAL_TOKENS,
                 encoder_max_length: int = MBART_ENCODER_MAX_LENGTH,
                 encoder_return_tensors: str = MBART_ENCODER_RETURN_TENSORS,
                 encoder_padding: bool = MBART_ENCODER_PADDING,
                 encoder_verbose: bool = MBART_ENCODER_VERBOSE,
                 decoder_add_special_tokens: bool = MBART_DECODER_ADD_SPECIAL_TOKENS,
                 decoder_max_length: int = MBART_DECODER_MAX_LENGTH,
                 decoder_return_tensors: str = MBART_DECODER_RETURN_TENSORS,
                 decoder_padding: bool = MBART_DECODER_PADDING,
                 decoder_verbose: bool = MBART_DECODER_VERBOSE,
                 verbose: bool = MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE,
                 verbose_step: int = MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE_STEP,
                 freeze: bool = MBART_MODEL_CONDITIONAL_GENERATION_FREEZE_MODEL,
                 trainable_layers: list = None,
                 k_random: int = MBART_MODEL_CONDITIONAL_GENERATION_K_RANDOM
                ) -> None:
        '''
            Initial definition of the Code Mixed Model
            Input params:
                train_data_loader: dataloader for training data
                validation_data_loader: dataloader for validation data
                test_data_loader: dataloader for test data
                encoder_tokenizer: CustomBartTokenizer object for the encoder
                decoder_tokenizer: CustomBartTokenizer object for the decoder
                epochs: int, number of epochs to train the model for
                device: str, device to train the model on
                save_model: bool, if True, the model will be saved
                save_to_gcp: bool, if True, the model will be saved to GCP
                save_path_dir: str, path to the directory where the model will be saved
                saved_model_path: str, path to the saved model (ought to be loaded)
                model_name: str, name of the model to be used | default: MBart | options: MBart
                generate_config: dict, config for the generation of the model
                encoder_add_special_tokens: bool, if True, special tokens will be added to the encoder
                encoder_max_length: int, maximum length of the encoder sequence
                encoder_return_tensors: str, return tensors for the encoder
                encoder_padding: bool, if True, the encoder will pad the sequences
                encoder_verbose: bool, if True, the encoder will be verbose
                decoder_add_special_tokens: bool, if True, special tokens will be added to the decoder
                decoder_max_length: int, maximum length of the decoder sequence
                decoder_return_tensors: str, return tensors for the decoder
                decoder_padding: bool, if True, the decoder will pad the sequences
                decoder_verbose: bool, if True, the decoder will be verbose
                verbose: bool, if True, the module will be verbose
                verbose_step: int, the step after which the module will be verbose
                freeze: bool, if True, the model will be frozen
                trainable_layers: list, list of layers to be trained, others will be frozen if `freeze` is True
                k_random: int, number of random characters to be appended to the saved model name
                google_bucket_name: str, name of the GCP bucket
                google_bucket_dir: str, directory in the GCP bucket
        '''
        super().__init__()
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.test_data_loader = test_data_loader
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.epochs = epochs
        self.device = device
        self.save_model = save_model
        self.save_path_dir = save_path_dir
        self.saved_model_path = saved_model_path
        self.model_name = model_name
        self.generate_config = generate_config
        self.encoder_add_special_tokens = encoder_add_special_tokens
        self.encoder_max_length = encoder_max_length
        self.encoder_return_tensors = encoder_return_tensors
        self.encoder_padding = encoder_padding
        self.encoder_verbose = encoder_verbose
        self.decoder_add_special_tokens = decoder_add_special_tokens
        self.decoder_max_length = decoder_max_length
        self.decoder_return_tensors = decoder_return_tensors
        self.decoder_padding = decoder_padding
        self.decoder_verbose = decoder_verbose
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.freeze = freeze
        self.trainable_layers = trainable_layers
        self.k_random = k_random
        self.skip_train = False
        self.start_epoch = 0
        self.optimizer = None
        self.scheduler = None
        self.generate_config = None
        self.train_loss = []
        self.validation_loss = []
        self.loader_features = ["src_tokenized", "src_attention_mask", "tgt_tokenized", "tgt_attention_mask"]
        self.test_model = None
        self.bleu_metric = load_metric("bleu")

    def _get_model(self, model_name: str = MBART_MODEL_CONDITIONAL_GENERATION_TYPE) -> object:
        '''
            Returns the model object
            Input params:
                model_name: str, name of the model to be used | default: MBart | options: MBart
        '''
        if model_name == MBART_MODEL_CONDITIONAL_GENERATION_TYPE:
            return BartForConditionalGeneration(
                pretrained=MBART_MODEL_CONDITIONAL_GENERATION_USE_PRETRAINED,
                pretrained_path=MBART_MODEL_CONDITIONAL_GENERATION_FROM_PRETRAINED
            ).model
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
    
    def _freeze_weights(self) -> None:
        '''
            Freezes the weights of the model given a list of layers to be trained, if no list is provided, all the layers are frozen
        '''
        if self.verbose: print("Freezing the model...")
        for name, param in self.model.named_parameters():
            if self.trainable_layers and name in self.trainable_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if self.trainable_layers is None or len(self.trainable_layers) == 0:
            self.skip_train = True
    
    def _save(self, model: object, special_msg: str) -> None:
        '''
            This function saves the model
            Input params: model - a model object, special_msg - a special message to be printed
        '''
        os.makedirs(self.save_path_dir, exist_ok=True)
        model.to("cpu")
        if special_msg is None:
            model_name = self.model_name + "_" + \
                ''.join(random.choices(string.ascii_uppercase + string.digits, k=self.k_random)) + ".pth"
        else:
            model_name = self.model_name + "_" + \
                ''.join(random.choices(string.ascii_uppercase + string.digits, k=self.k_random)) + f"_{special_msg}.pth"
        self.save_path = self.save_path_dir + '/' + model_name
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_path)
        print(f"\nModel saved to -> {self.save_path}...")
        model.to(self.device)
    
    def _configure_optimizers(self) -> None:
        '''
            This function configures the optimizer and scheduler.
        '''
        if self.verbose: print("\nConfiguring optimizer and scheduler...")
        if MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_TYPE == "AdamW":
            optimizer = AdamW(
                params=self.model.parameters(),
                lr=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_LR,
                betas=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_BETAS,
                eps=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_EPS,
                weight_decay=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_WEIGHT_DECAY,
                correct_bias=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_CORRECT_BIAS,
                no_deprecation_warning=MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_NO_DEPRICATION_WARNING
            )
        else:
            raise NotImplementedError(f"Optimizer {MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_TYPE} not implemented.")
        if MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE == "get_linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_WARMUP_STEPS,
                num_training_steps=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TRAINING_STEPS,
                last_epoch=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_LAST_EPOCH
            )
        else:
            raise NotImplementedError(f"Scheduler {MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE} not implemented.")
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def _configure_criterion(self) -> None:
        '''
            This function configures the criterion.
        '''
        if self.verbose: print("\nConfiguring criterion...")
        if MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_TYPE == "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss(
                weight=MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_WEIGHT,
                size_average=MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_SIZE_AVERAGE,
                ignore_index=MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_IGNORE_INDEX,
                reduce=MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_REDUCE,
                reduction=MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_REDUCTION,
                label_smoothing=MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_LABEL_SMOOTHING
            )
        else:
            raise NotImplementedError(f"Criterion {MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_TYPE} not implemented.")
        self.criterion = criterion

    def _plot_loss_curve(self, x_label: str = 'Epochs'):
        '''
            This function plots the loss curve
            Input params: x_label - a string representing the x label
            Returns: None
        '''
        epochs = range(1, len(self.train_loss)+1)
        plt.plot(epochs, self.train_loss, 'g', label='Training loss')
        plt.plot(epochs, self.train_loss, 'g*', label='Training loss spots')
        plt.plot(epochs, self.validation_loss, 'r', label='Validation loss')
        plt.plot(epochs, self.validation_loss, 'r*', label='Validation loss spots')
        plt.title('Training and testing Loss')
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def _compute_bleu(self, true_ids: torch.LongTensor, logits: torch.LongTensor) -> float:
        '''
            This function computes the BLEU score.
            Input params:
                true_ids: torch.LongTensor, the true ids
                logits: torch.LongTensor, the logits form the model
            Returns: float, the BLEU score
        '''
        y = true_ids.tolist()
        y_hat = torch.argmax(logits, dim=-1)
        y_hat = y_hat.tolist()
        assert len(y) == len(y_hat), "Length of true and predicted sequences not equal..."
        references, predictions = [], []
        for i in range(len(y)):
            references.append([self.decoder_tokenizer.decode(y[i], skip_special_tokens=True).split()])
            predictions.append(self.decoder_tokenizer.decode(y_hat[i], skip_special_tokens=True).split())
        return self.bleu_metric.compute(
            predictions=predictions,
            references=references
        )['bleu']
    
    def _train(self) -> tuple:
        '''
            This function trains the model
            Input params: None
            Returns: tuple of the model and the best model
        '''
        if self.verbose: print("\nTraining the model...")
        print(f"\nDEVICE - {self.device} || EPOCHS - {self.epochs} || LEARNING RATE - {self.optimizer.param_groups[0]['lr']}.")
        if self.skip_train:
            print("Skipping training as model params are all frozen...")
        else:
            for epoch in range(self.start_epoch, self.start_epoch+self.epochs):
                # Train phase
                self.model.train()
                start_epoch_time = time.time()
                if self.verbose:
                    _start_at = datetime.now().strftime('%H:%M:%S %d|%m|%Y')
                    _lr = self.optimizer.param_groups[0]['lr']
                    print(f'\nEPOCH - {epoch+1}/{self.epochs} || START AT - {_start_at} || LEARNING RATE - {_lr}\n')
                running_bleu, running_loss, step_running_loss, step_running_bleu = 0, 0, 0, 0
                start_step_time = time.time()
                for step, (batch) in enumerate(self.train_data_loader):
                    for key in batch:
                        if key in self.loader_features: batch[key] = batch[key].to(self.device)
                    out = self.model(
                        input_ids=batch["src_tokenized"],
                        attention_mask=batch["src_attention_mask"],
                        decoder_input_ids=batch["tgt_tokenized"],
                        decoder_attention_mask=batch["tgt_attention_mask"]
                    )
                    logits = out.logits
                    y, y_hat = batch["tgt_tokenized"].view(-1), logits.view(-1, logits.size(-1))
                    loss = self.criterion(y_hat, y)
                    bleu_score = self._compute_bleu(true_ids=batch["tgt_tokenized"], logits=logits)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    running_loss += loss.item()
                    running_bleu += bleu_score
                    step_running_loss += loss.item()
                    step_running_bleu += bleu_score
                    if self.verbose:
                        if (step+1) % self.verbose_step == 0:
                            print(
                                    f'\tTrain Step - {step+1}/{len(self.train_data_loader)} | ' + \
                                    f'Train Step Loss: {(step_running_loss/self.verbose_step):.5f} | ' + \
                                    f'Train BLEU Score: {(step_running_bleu/self.verbose_step):.5f} | ' + \
                                    f'Time: {(time.time() - start_step_time):.2f}s.\n'
                                )
                            step_running_loss = 0  
                            step_running_bleu = 0 
                            start_step_time = time.time()
                            if self.save_model:
                                self._save(model=self.model, special_msg=f"epoch_{str(epoch+1)}_step_{str(step+1)}")
                self.train_loss.append(running_loss/len(self.train_data_loader))
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || TRAIN-LOSS - {(running_loss/len(self.train_data_loader)):.5f} || BLEU SCORE - {bleu_score:.5f} || TIME ELAPSED - {(time.time() - start_epoch_time):.2f}s.\n')
                # Validation Phase
                start_epoch_validation_time = time.time()
                self.model.eval()
                running_validation_bleu, running_validation_loss = 0, 0
                with torch.no_grad():
                    for _, (validation_batch) in enumerate(self.validation_data_loader):
                        for key in validation_batch:
                            if key in self.loader_features: validation_batch[key] = validation_batch[key].to(self.device)
                        out = self.model(
                            input_ids=validation_batch["src_tokenized"],
                            attention_mask=validation_batch["src_attention_mask"],
                            decoder_input_ids=validation_batch["tgt_tokenized"],
                            decoder_attention_mask=validation_batch["tgt_attention_mask"]
                        )
                        logits = out.logits
                        y, y_hat = validation_batch["tgt_tokenized"].view(-1), logits.view(-1, logits.size(-1))
                        loss = self.criterion(y_hat, y)
                        bleu_score = self._compute_bleu(true_ids=validation_batch["tgt_tokenized"], logits=logits)
                        running_validation_loss += loss.item()
                        running_validation_bleu += bleu_score
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || VAL-LOSS - {(running_validation_loss/len(self.validation_data_loader)):.5f} || BLEU SCORE - {(running_validation_bleu/len(self.validation_data_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_validation_time):.2f}s.\n')
                self.validation_loss.append(running_validation_loss/len(self.validation_data_loader))
                if self.best_test_loss > running_validation_loss:
                    self.best_test_loss = running_validation_loss
                    self.best_model = copy.deepcopy(self.model)
                self.scheduler.step()
                if self.save_model:
                    self._save(model=self.model, special_msg=f"epoch_{str(epoch+1)}")
            if self.verbose:
                self.plot_loss_curve()
        if self.save_model:
            self._save(model=self.best_model)
        return (self.model.to("cpu"), self.best_model.to("cpu"))
    
    def _configure_generate_config(self) -> None:
        '''
            This function configures the generate config.
        '''
        self.generate_config = GenerationConfig(**self.generate_config)

    def _generate(self, input_ids: torch.LongTensor, model: BartForConditionalGeneration = None) -> str:
        '''
            This function generates the translation.
            Input params:
                input_ids: torch.LongTensor, the input ids
                model: BartForConditionalGeneration, the model to be used for inference
            Returns: str, the translated sentence
        '''
        if model is None:
            model = self.test_model
        return model.generate(
            input_ids=input_ids,
            generation_config=self.generate_config
        )

    def _test(self) -> tuple:
        '''
            This function tests the model
            Input params: None
            Returns: tuple of the model and the best model
        '''
        assert self.test_model is not None, "Model not loaded, set `test_model`..."
        if self.verbose: print("\nTesting the model...")
        start_test_time = time.time()
        self.test_model = self.test_model.to(self.device)
        test_loss, test_bleu_Score = 0, 0
        for _, (test_batch) in tqdm(enumerate(self.test_data_loader)):
            for key in test_batch:
                if key in self.loader_features: test_batch[key] = test_batch[key].to(self.device)
            out = self.test_model(
                input_ids=test_batch["src_tokenized"],
                attention_mask=test_batch["src_attention_mask"],
                decoder_input_ids=test_batch["tgt_tokenized"],
                decoder_attention_mask=test_batch["tgt_attention_mask"]
            )
            logits = out.logits
            y, y_hat = test_batch["tgt_tokenized"].view(-1), logits.view(-1, logits.size(-1))
            loss = self.criterion(y_hat, y)
            test_loss += loss.item()
            bleu_score = self._compute_bleu(true_ids=test_batch["tgt_tokenized"], logits=logits)
            test_bleu_score += bleu_score
        print(f'\nTEST-LOSS - {(test_loss/len(self.test_data_loader)):.5f} || BLEU SCORE - {(test_bleu_Score/len(self.test_data_loader)):.5f} || TIME ELAPSED - {(time.time() - start_test_time):.2f}s.\n')
        
    def _infer(self, src: str,
               model: BartForConditionalGeneration = None,
               device: str = "cpu") -> str:
        '''
            This function infers the translation
            Input params:
                src: str, the source sentence
                model: BartForConditionalGeneration, the model to be used for inference
                device: str, the device to be used for inference
            Returns: str, the translated sentence
        '''
        if model is None:
            model = self.test_model
        assert model is not None, "Model not found, set `test_model` or provide a parameter `model` to this function..."
        model = model.to(device)
        src_tokenized = self.encoder_tokenizer(
            src,
            add_special_tokens=self.encoder_add_special_tokens, 
            max_length=self.encoder_max_length, 
            return_tensors=self.encoder_return_tensors, 
            padding=self.encoder_padding, 
            verbose=self.encoder_verbose
        )
        output = self._generate(input_ids=src_tokenized["input_ids"].to(device))
        output_str = self.decoder_tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        return output_str

    def load_model(self) -> object:
        '''
            Loads the saved model
        '''
        if self.verbose: print("Loading model...")
        self.model = self._get_model(self.model_name)
        if self.saved_model_path is not None:
            state_dict = torch.load(self.saved_model_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.start_epoch = state_dict['epoch']
            if self.verbose: print(f"Loaded model from {self.saved_model_path}...")
        else:
            if self.verbose: print("No saved model to load as `saved_model_path` was not provided in the `__init__()`...")
        if self.freeze: self._freeze_weights()
        self.best_model = copy.deepcopy(self.model)
        if self.verbose: print(self.model)
        return self.model

    def fit(self) -> tuple:
        '''
            This function fits the model
            Input params: None
            Returns: tuple of the model and the best model
        '''
        _ = self.load_model()
        self._configure_optimizers()
        self._configure_criterion()
        return self._train()
    
    def test(self, use_best: bool = True) -> None:
        '''
            This function tests the model
            Input params:
                use_best: bool, whether to use the best model or not
            Returns: None
        '''
        if use_best:
            self.test_model = copy.deepcopy(self.best_model)
        else:
            self.test_model = copy.deepcopy(self.model)
        if self.generate_config is None: self._configure_generate_config()
        self._test()
    
    def infer(self, src: list,
              model: BartForConditionalGeneration = None,
              device: str = "cpu") -> list:
        '''
            This function infers the translation
            Input params:
                src: list, list of strings
                model: BartForConditionalGeneration, the model to be used for inference
                device: str, the device to be used for inference
            Returns: str, the translated sentence
        '''
        if model is None:
            model = self.test_model
        if model is None:
            model = self.test_model = self.load_model()
        if self.verbose: print("Inferring...")
        if self.generate_config is None: self._configure_generate_config()
        res = []
        for s in src:
            res.append((s, self._infer(
                src=s,
                model=model,
                device=device
            )))
        return res