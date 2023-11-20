import os
import copy
import time
import torch
import random
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
try:
    from .models.mbart import MBart
except ImportError:
    print("Following alternate import for net.py")
    from models.mbart import MBart


class CodeMixedModel:

    '''
        Main class for the Code Mixed Model
    '''

    def __init__(self,
                 train_dataloader: object = None,
                 val_dataloader: object = None,
                 test_dataloader: object = None,
                 criterion: object = None,
                 epochs: int = 10,
                 device: str = "cpu",
                 save_model: bool = False,
                 save_path_dir: str = None,
                 save_model_name: str = None,
                 saved_model_path: str = None,
                 model_name: str = "MBart",
                 verbose: bool = True,
                 verbose_step: int = 100,
                 freeze: bool = False,
                 trainable_layers: list = None,
                 k_random: int = 7,
                 label: str = "label"
                ) -> None:
        '''
            Initial definition of the Code Mixed Model
            Input params:
                train_dataloader: dataloader for training data
                val_dataloader: dataloader for validation data
                test_dataloader: dataloader for test data
                criterion: loss function
                epochs: int, number of epochs to train the model for
                device: str, device to train the model on
                save_model: bool, if True, the model will be saved
                save_path_dir: str, path to the directory where the model will be saved
                save_model_name: str, name of the model to be saved
                saved_model_path: str, path to the saved model (ought to be loaded)
                model_name: str, name of the model to be used | default: MBart | options: MBart
                verbose: bool, if True, the module will be verbose
                verbose_step: int, the step after which the module will be verbose
                freeze: bool, if True, the model will be frozen
                trainable_layers: list, list of layers to be trained, others will be frozen if `freeze` is True
                k_random: int, number of random characters to be appended to the saved model name
                label: str, name of the label column
        '''
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.save_model = save_model
        self.save_path_dir = save_path_dir
        self.save_model_name = save_model_name
        self.saved_model_path = saved_model_path
        self.model_name = model_name
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.freeze = freeze
        self.trainable_layers = trainable_layers
        self.label = label
        self.skip_train = False
        self.start_epoch = 0
        self.optimizer = None
        self.scheduler = None
        self.train_loss = []
        self.val_loss = []
        self.save_path = self.save_path_dir + \
            self.model_name + \
                ''.join(random.choices(string.ascii_uppercase + string.digits, k=k_random)) + ".pth" \
                    if self.save_path_dir is not None else None
        _ = self.load_model()
        if self.verbose: print(self.model)

    def get_model(self, model_name: str = 'MBart') -> object:
        '''
            Returns the model object
            Input params:
                model_name: str, name of the model to be used | default: MBart | options: MBart
        '''
        if model_name == 'MBart':
            return MBart().model
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
        
    def load_model(self) -> object:
        '''
            Loads the saved model
        '''
        self.model = self.get_model(self.model_name)
        if self.saved_model_path is not None:
            state_dict = torch.load(self.saved_model_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.start_epoch = state_dict['epoch']
            if self.verbose: print(f"Loaded model from {self.saved_model_path}...")
        else:
            if self.verbose: print("No saved model to load as `saved_model_path` was not provided...")
        if self.freeze_model: self.freeze_weights()
        self.best_model = copy.deepcopy(self.model)
        return self.model
    
    def freeze_weights(self) -> None:
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
    
    def save(self, model: object) -> None:
        '''
            This function saves the model
            Input params: model - a model object
            Returns: None.
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
    
    def configure_optimizers(self,
                             optimizer: str = "AdamW",
                             scheduler: str = "ReduceLROnPlateau",
                             lr: int = 5e-5,
                             betas: tuple = (0.9, 0.999),
                             eps: float = 1e-8,
                             weight_decay: float = 0.01,
                             mode: str = "min",
                             patience: int = 10,) -> None:
        '''
            This function configures the optimizer and scheduler.
            Input params:
                optimizer - a string representing the optimizer | default: AdamW | options: AdamW
                scheduler - a string representing the scheduler | default: ReduceLROnPlateau | options: ReduceLROnPlateau
                lr - a float representing the learning rate
                betas - a tuple representing the betas for the optimizer
                eps - a float representing the epsilon for the optimizer
                weight_decay - a float representing the weight decay for the optimizer
                mode - a string representing the mode for the scheduler
                patience - an integer representing the patience for the scheduler
            Returns: None
        '''
        if optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented")
        if scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=mode,
                patience=patience,
            )
        else:
            raise NotImplementedError(f"Scheduler {scheduler} not implemented")
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def plot_loss_curve(self, x_label: str = 'Epochs'):
        '''
            This function plots the loss curve
            Input params: x_label - a string representing the x label
            Returns: None
        '''
        epochs = range(1, len(self.train_loss)+1)
        plt.plot(epochs, self.train_loss, 'g', label='Training loss')
        plt.plot(epochs, self.train_loss, 'g*', label='Training loss spots')
        plt.plot(epochs, self.val_loss, 'r', label='Validation loss')
        plt.plot(epochs, self.val_loss, 'r*', label='Validation loss spots')
        plt.title('Training and testing Loss')
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def train(self) -> tuple:
        '''
            This function trains the model
            Input params: None
            Returns: tuple of the model and the best model
        '''
        print(f"\nDEVICE - {self.device} || EPOCHS - {self.epochs} || LEARNING RATE - {self.optimizer.param_groups[0]['lr']}.\n")
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
                for step, (batch) in enumerate(self.train_loader):
                    for key in batch:
                        batch[key] = batch[key].to(self.device)
                    y_hat = self.model(**batch)
                    loss = self.criterion(y_hat, batch[self.label])
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    running_loss += loss.item()
                    step_running_loss += loss.item()
                    if self.verbose:
                        if (step+1) % self.verbose_step == 0:
                            print(
                                    f'\tTrain Step - {step+1}/{len(self.train_loader)} | ' + \
                                    f'Train Step Loss: {(step_running_loss/self.verbose_step):.5f} | ' + \
                                    f'Time: {(time.time() - start_step_time):.2f}s.\n'
                                )
                            step_running_loss = 0   
                            start_step_time = time.time()
                self.train_loss.append(running_loss/len(self.train_loader))
                self.scheduler.step(running_loss/len(self.train_loader))
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || TRAIN-LOSS - {(running_loss/len(self.train_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_time):.2f}s.\n')
                # Validation Phase
                start_epoch_val_time = time.time()
                self.model.eval()
                running_val_loss = 0
                with torch.no_grad():
                    for _, (val_batch) in enumerate(self.val_loader):
                        for key in val_batch:
                            val_batch[key] = val_batch[key].to(self.device)
                        y_hat = self.model(**val_batch)
                        loss = self.criterion(y_hat, val_batch[self.label])
                        running_val_loss += loss.item()
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || VAL-LOSS - {(running_val_loss/len(self.val_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_val_time):.2f}s.\n')
                self.val_loss.append(running_val_loss/len(self.val_loader))
                if self.best_test_loss > running_val_loss:
                    self.best_test_loss = running_val_loss
                    self.best_model = copy.deepcopy(self.model)
            if self.verbose:
                self.plot_loss_curve()
            if self.save_model:
                self.save(model=self.best_model)
            return (self.model.to("cpu"), self.best_model.to("cpu"))
    
    def test(self) -> None:
        '''
            This function tests the model
            Input params: None
            Returns: None
        '''
        print("Testing model...")
        self.best_model.eval()
        self.best_model.to(self.device)
        running_test_loss = 0
        start_epoch_test_time = time.time()
        with torch.no_grad():
            for _, (test_batch) in tqdm(enumerate(self.test_loader)):
                for key in test_batch:
                    test_batch[key] = test_batch[key].to(self.device)
                y_hat = self.best_model(**test_batch)
                loss = self.criterion(y_hat, test_batch[self.label])
                running_test_loss += loss.item()
        print(f'\n TEST-LOSS - {(running_test_loss/len(self.test_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_test_time):.2f}s.\n')