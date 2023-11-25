import os
import copy
import time
import torch
import random
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.machine_translation import *
from src.machine_translation.models.bart_conditional import BartForConditionalGeneration


class CodeMixedModelHGTrainer:

    '''
        Main class for the Code Mixed Model training via HuggingFace Trainer
    '''

    def __init__(self,
                 use_pretrained: bool = MBART_MODEL_CONDITIONAL_GENERATION_USE_PRETRAINED,
                 from_pretrained: str = MBART_MODEL_CONDITIONAL_GENERATION_FROM_PRETRAINED,
                 save_path_dir: str = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_PATH,
                 model_name: str = MBART_MODEL_CONDITIONAL_GENERATION_TYPE,
                 k_random: int = MBART_MODEL_CONDITIONAL_GENERATION_K_RANDOM
                ) -> None:
        '''
            Initial definition of the Code Mixed Model
        '''
        super().__init__()
        self.use_pretrained = use_pretrained
        self.from_pretrained = from_pretrained
        self.save_path_dir = save_path_dir
        self.model_name = model_name
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
    
    def _configure_training_arguments(self) -> None:
        pass
    


class CodeMixedModel:

    '''
        Main class for the Code Mixed Model
    '''

    def __init__(self,
                 train_data_loader: object = None,
                 validation_data_loader: object = None,
                 test_data_loader: object = None,
                 epochs: int = MBART_MODEL_CONDITIONAL_GENERATION_EPOCHS,
                 device: str = MBART_MODEL_CONDITIONAL_GENERATION_DEVICE,
                 save_model: bool = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_MODEL,
                 save_path_dir: str = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_PATH,
                 saved_model_path: str = MBART_MODEL_CONDITIONAL_GENERATION_LOAD_PATH,
                 model_name: str = MBART_MODEL_CONDITIONAL_GENERATION_TYPE,
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
                epochs: int, number of epochs to train the model for
                device: str, device to train the model on
                save_model: bool, if True, the model will be saved
                save_path_dir: str, path to the directory where the model will be saved
                saved_model_path: str, path to the saved model (ought to be loaded)
                model_name: str, name of the model to be used | default: MBart | options: MBart
                verbose: bool, if True, the module will be verbose
                verbose_step: int, the step after which the module will be verbose
                freeze: bool, if True, the model will be frozen
                trainable_layers: list, list of layers to be trained, others will be frozen if `freeze` is True
                k_random: int, number of random characters to be appended to the saved model name
        '''
        super().__init__()
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.test_data_loader = test_data_loader
        self.epochs = epochs
        self.device = device
        self.save_model = save_model
        self.save_path_dir = save_path_dir
        self.saved_model_path = saved_model_path
        self.model_name = model_name
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.freeze = freeze
        self.trainable_layers = trainable_layers
        self.skip_train = False
        self.start_epoch = 0
        self.optimizer = None
        self.scheduler = None
        self.train_loss = []
        self.validation_loss = []
        self.loader_features = ["src_tokenized", "src_attention_mask", "tgt_tokenized", "tgt_attention_mask"]
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
            return (None, None)
        else:
            for epoch in range(self.start_epoch, self.start_epoch+self.epochs):
                # Train phase
                self.model.train()
                start_epoch_time = time.time()
                if self.verbose:
                    _start_at = datetime.now().strftime('%H:%M:%S %d|%m|%Y')
                    _lr = self.optimizer.param_groups[0]['lr']
                    print(f'\nEPOCH - {epoch+1}/{self.epochs} || START AT - {_start_at} || LEARNING RATE - {_lr}\n')
                running_loss, step_running_loss = 0, 0
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
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    running_loss += loss.item()
                    step_running_loss += loss.item()
                    if self.verbose:
                        if (step+1) % self.verbose_step == 0:
                            print(
                                    f'\tTrain Step - {step+1}/{len(self.train_data_loader)} | ' + \
                                    f'Train Step Loss: {(step_running_loss/self.verbose_step):.5f} | ' + \
                                    f'Time: {(time.time() - start_step_time):.2f}s.\n'
                                )
                            step_running_loss = 0   
                            start_step_time = time.time()
                self.train_loss.append(running_loss/len(self.train_data_loader))
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || TRAIN-LOSS - {(running_loss/len(self.train_data_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_time):.2f}s.\n')
                # Validation Phase
                start_epoch_validation_time = time.time()
                self.model.eval()
                running_validation_loss = 0
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
                        running_validation_loss += loss.item()
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || VAL-LOSS - {(running_validation_loss/len(self.validation_data_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_validation_time):.2f}s.\n')
                self.validation_loss.append(running_validation_loss/len(self.validation_data_loader))
                if self.best_test_loss > running_validation_loss:
                    self.best_test_loss = running_validation_loss
                    self.best_model = copy.deepcopy(self.model)
                self.scheduler.step()
            if self.verbose:
                self.plot_loss_curve()
            if self.save_model:
                self._save(model=self.best_model)
            return (self.model.to("cpu"), self.best_model.to("cpu"))
    
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