import os
import copy
import time
import torch
import random
import string
import numpy as np
from tqdm import tqdm
from typing import Union
from datetime import datetime
import matplotlib.pyplot as plt
from datasets import load_metric
from torch.utils.data import Dataset, DataLoader
from transformers import GenerationConfig, TrainingArguments, Trainer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.data import *
from src.machine_translation import *
from src.data.tokenizer import CustomBartTokenizer
from src.machine_translation.models.bart_conditional import BartForConditionalGeneration
from src.machine_translation.callback import GCPCallback
from src.machine_translation.collator import DataCollator


class CodeMixedModelHGTrainer:

    '''
        Main class for the Code Mixed Model training via HuggingFace Trainer
    '''

    def __init__(self,
                 train_dataset: Dataset = None,
                 validation_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 use_pretrained: bool = MBART_MODEL_CONDITIONAL_GENERATION_USE_PRETRAINED,
                 from_pretrained: str = MBART_MODEL_CONDITIONAL_GENERATION_FROM_PRETRAINED,
                 save_to_gcp: bool = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_TO_GCP,
                 save_path_dir: str = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_PATH,
                 cloud_save_path: str = MBART_MODEL_CONDITIONAL_GENERATION_GCP_SAVE_PATH,
                 save_steps: int = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_STEPS,
                 epochs: int = MBART_MODEL_CONDITIONAL_GENERATION_EPOCHS,
                 eval_steps: int = MBART_MODEL_CONDITIONAL_GENERATION_EVAL_STEPS,
                 device: str = MBART_MODEL_CONDITIONAL_GENERATION_DEVICE,
                 verbose: bool = MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE,
                 verbose_step: int = MBART_MODEL_CONDITIONAL_GENERATION_VERBOSE_STEP,
                 freeze: bool = MBART_MODEL_CONDITIONAL_GENERATION_FREEZE_MODEL,
                 trainable_layers: list = None,
                 train_batch_size: int = MBART_DATALOADER_TRAIN_BATCH_SIZE,
                 validation_batch_size: int = MBART_DATALOADER_VALIDATION_BATCH_SIZE,
                 test_batch_size: int = MBART_DATALOADER_TEST_BATCH_SIZE,
                 resume_from_checkpoint: str = MBART_MODEL_CONDITIONAL_GENERATION_RESUME_FROM_CHECKPOINT,
                 do_train: bool = MBART_MODEL_CONDITIONAL_GENERATION_DO_TRAIN,
                 do_eval: bool = MBART_MODEL_CONDITIONAL_GENERATION_DO_EVAL,
                 do_predict: bool = MBART_MODEL_CONDITIONAL_GENERATION_DO_PREDICT,
                 evaluation_strategy: str = MBART_MODEL_CONDITIONAL_GENERATION_EVALUALTION_STRATEGY,
                 log_path: str = MBART_MODEL_CONDITIONAL_GENERATION_LOG_PATH,
                 log_steps: int = MBART_MODEL_CONDITIONAL_GENERATION_LOG_STEPS,
                 max_length: int = MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_MAX_LENGTH,
                 early_stopping: bool = MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_EARLY_STOPPING,
                 num_beams: int = MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_NUM_BEAMS,
                 denoising_stage: bool = False,
                 encoder_tokenizer: CustomBartTokenizer = None,
                 decoder_tokenizer: CustomBartTokenizer = None,
                 encoder_add_special_tokens: bool = MBART_ENCODER_ADD_SPECIAL_TOKENS,
                 encoder_max_length: int = MBART_ENCODER_MAX_LENGTH,
                 encoder_return_tensors: str = MBART_ENCODER_RETURN_TENSORS,
                 encoder_padding: Union[bool, str] = MBART_ENCODER_PADDING,
                 encoder_mlm: bool = MBART_ENCODER_MLM,
                 encoder_mlm_probability: float = MBART_ENCODER_MLM_PROBABILITY,
                 encoder_enable_group_mask: bool = MBART_ENCODER_ENABLE_GROUP_MASK,
                 encoder_mask_max_length: int = MBART_ENCODER_MASK_MAX_LENGTH,
                 encoder_plm: bool = MBART_ENCODER_PLM,
                 encoder_plm_probability: float = MBART_ENCODER_PLM_PROBABILITY,
                 encoder_plm_max_length: int = MBART_ENCODER_PLM_MAX_LENGTH,
                 encoder_plm_min_window_length: int = MBART_ENCODER_PLM_MIN_WINDOW_LENGTH,
                 encoder_style_switch_probability: float = MBART_ENCODER_STYLE_SWITCH_PROBABILITY,
                 bert_lang: str = "en",
                 inference: bool = False,
                 clear_local_storage_on_cloud_save: bool = MBART_MODEL_CONDITIONAL_GENERATION_CLEAR_LOCAL_STORAGE_ON_CLOUD_SAVE
                 ) -> None:
        '''
            Initial definition of the Code Mixed Model using HuggingFace trainer.
            Input params:
                - train_dataset: Dataset, the training dataset
                - validation_dataset: Dataset, the validation dataset
                - test_dataset: Dataset, the test dataset
                - use_pretrained: bool, whether to use a pretrained model or not
                - from_pretrained: str, the path to the pretrained model
                - save_to_gcp: bool, whether to save the model to GCP or not
                - save_path_dir: str, the path to save the model
                - cloud_save_path: str, the path to save the model on cloud
                - save_steps: int, the steps after which the model will be saved
                - epochs: int, the number of epochs to train the model
                - eval_steps: int, the steps after which the model will be evaluated
                - verbose: bool, whether to print the logs or not
                - verbose_step: int, the step after which the logs will be printed
                - freeze: bool, whether to freeze the model or not
                - trainable_layers: list, the list of layers to be trained
                - train_batch_size: int, the batch size for training
                - validation_batch_size: int, the batch size for validation
                - test_batch_size: int, the batch size for testing
                - resume_from_checkpoint: str, the path to the checkpoint to resume training from
                - do_train: bool, whether to train the model or not
                - do_eval: bool, whether to evaluate the model or not
                - do_predict: bool, whether to predict the model or not
                - evaluation_strategy: str, the evaluation strategy to be used
                - log_path: str, the path to save the logs
                - log_steps: int, the steps after which the logs will be saved
                - max_length: int, the maximum length of the generated sequence
                - early_stopping: bool, whether to use early stopping or not
                - num_beams: int, the number of beams to be used for generation
                - denoising_stage: bool, whether to use denoising stage or not
                - encoder_tokenizer: CustomBartTokenizer, the tokenizer for the encoder
                - decoder_tokenizer: CustomBartTokenizer, the tokenizer for the decoder
                - encoder_add_special_tokens: bool, whether to add special tokens to the encoder or not
                - encoder_max_length: int, the maximum length of the encoder sequence
                - encoder_return_tensors: str, the return tensors for the encoder
                - encoder_padding: Union[bool, str], the padding for the encoder
                - encoder_mlm: bool, whether to use MLM for the encoder or not
                - encoder_mlm_probability: float, the probability for the MLM
                - encoder_enable_group_mask: bool, whether to enable group mask for the encoder or not
                - encoder_mask_max_length: int, the maximum length for the mask
                - encoder_plm: bool, whether to use PLM for the encoder or not
                - encoder_plm_probability: float, the probability for the PLM
                - encoder_plm_max_length: int, the maximum length for the PLM
                - encoder_plm_min_window_length: int, the minimum window length for the PLM
                - encoder_style_switch_probability: float, the probability for the style switch
                - bert_lang: str, the language for the bert model
                - inference: bool, inference mode or not
                - clear_local_storage_on_cloud_save: bool, whether to clear the local storage on cloud save or not
        '''
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.use_pretrained = use_pretrained
        self.from_pretrained = from_pretrained
        self.save_to_gcp = save_to_gcp
        dt_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_path_dir = save_path_dir + "_" + dt_now
        self.cloud_save_path = cloud_save_path
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.save_steps = save_steps
        self.device = device
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.freeze = freeze
        self.trainable_layers = trainable_layers
        self.optimizer = None
        self.scheduler = None
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.resume_from_checkpoint = resume_from_checkpoint
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_predict = do_predict
        self.evalualtion_strategy = evaluation_strategy
        self.log_path = log_path + "_" + dt_now if log_path is not None else None
        self.log_steps = log_steps
        self.max_length = max_length
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.denoising_stage = denoising_stage
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_add_special_tokens = encoder_add_special_tokens
        self.encoder_max_length = encoder_max_length
        self.encoder_return_tensors = encoder_return_tensors
        self.encoder_padding = encoder_padding
        self.encoder_mlm = encoder_mlm
        self.encoder_mlm_probability = encoder_mlm_probability
        self.encoder_enable_group_mask = encoder_enable_group_mask
        self.encoder_mask_max_length = encoder_mask_max_length
        self.encoder_plm = encoder_plm
        self.encoder_plm_probability = encoder_plm_probability
        self.encoder_plm_max_length = encoder_plm_max_length
        self.encoder_plm_min_window_length = encoder_plm_min_window_length
        self.encoder_style_switch_probability = encoder_style_switch_probability
        self.bert_lang = bert_lang
        self.inference = inference
        self.clear_local_storage_on_cloud_save = clear_local_storage_on_cloud_save
        self.sacrebleu_score_metric = load_metric("sacrebleu")
        self.chrf_score_metric = load_metric("chrf")
        self.bertscore_metric = load_metric("bertscore")
        self._configure()

    def _get_model(self,
                   pretrained: bool = MBART_MODEL_CONDITIONAL_GENERATION_USE_PRETRAINED,
                   pretrained_path: str = MBART_MODEL_CONDITIONAL_GENERATION_FROM_PRETRAINED,
                   model_name: str = MBART_MODEL_CONDITIONAL_GENERATION_TYPE) -> object:
        '''
            Returns the model object
            Input params:
                pretrained: bool, whether to use a pretrained model or not
                pretrained_path: str, the path to the pretrained model
                model_name: str, name of the model to be used | default: MBart | options: MBart
        '''
        if self.verbose:
            print("Getting the model...")
        if model_name == MBART_MODEL_CONDITIONAL_GENERATION_TYPE:
            model_obj = BartForConditionalGeneration(
                pretrained=pretrained,
                pretrained_path=pretrained_path
            )
            model_obj.configure(embedding_size=len(self.encoder_tokenizer))
            return model_obj.model
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

    def _freeze_weights(self) -> None:
        '''
            Freezes the weights of the model given a list of layers to be trained, if no list is provided, all the layers are frozen
        '''
        if self.verbose:
            print("\nFreezing the model...")
        for name, param in self.model.named_parameters():
            if self.trainable_layers and name in self.trainable_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _configure_collator(self) -> None:
        '''
            This function configures the collator
        '''
        if self.verbose:
            print("\nConfiguring collator...")
        self.data_collator = DataCollator(
            tokenizer=self.encoder_tokenizer,
            denoising_stage=self.denoising_stage,
            mlm=self.encoder_mlm,
            plm=self.encoder_plm,
            padding=self.encoder_padding,
            enable_group_mask=self.encoder_enable_group_mask,
            mask_max_length=self.encoder_mask_max_length,
            permute_mask_length=self.encoder_plm_max_length,
            padding_max_length=self.encoder_max_length,
            return_tensors=self.encoder_return_tensors,
            mlm_probability=self.encoder_mlm_probability,
            plm_probability=self.encoder_plm_probability,
            min_window_length=self.encoder_plm_min_window_length,
            style_switch_probability=self.encoder_style_switch_probability
        )

    def _configure_optimizers(self) -> None:
        '''
            This function configures the optimizer and scheduler.
        '''
        if self.verbose:
            print("\nConfiguring optimizer and scheduler...")
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
            raise NotImplementedError(
                f"Optimizer {MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_TYPE} not implemented.")
        if MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE == "get_linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_WARMUP_STEPS,
                num_training_steps=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TRAINING_STEPS,
                last_epoch=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_LAST_EPOCH
            )
        else:
            raise NotImplementedError(
                f"Scheduler {MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE} not implemented.")
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _get_eval_data_loader(self, eval_dataset: Dataset) -> DataLoader:
        '''
            This function returns the eval data loader
            Input params:
                eval_dataset: Dataset, the eval dataset
        '''
        return self.trainer.get_eval_dataloader(eval_dataset=eval_dataset)

    def _get_test_data_loader(self, test_dataset: Dataset) -> DataLoader:
        '''
            This function returns the test data loader
            Input params:
                test_dataset: Dataset, the test dataset
        '''
        return self.trainer.get_test_dataloader(test_dataset=test_dataset)

    def _generate(self, input_ids: torch.LongTensor, model: BartForConditionalGeneration = None) -> str:
        '''
            This function generates the translation.
            Input params:
                input_ids: torch.LongTensor, the input ids
                model: BartForConditionalGeneration, the model to be used for inference
            Returns: str, the translated sentence
        '''
        if model is None:
            model = self.model
        return model.generate(
            input_ids=input_ids,
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping
        )

    def _configure_training_arguments(self) -> TrainingArguments:
        '''
            This function configures the training arguments
        '''
        if self.verbose:
            print("\nConfiguring training arguments...")
        if not self.inference:
            os.makedirs(self.save_path_dir, exist_ok=True)
        if not self.inference:
            if self.log_path is not None:
                os.makedirs(self.log_path, exist_ok=True)
        args = TrainingArguments(
            output_dir=self.save_path_dir,
            do_train=self.do_train,
            do_eval=self.do_eval,
            do_predict=self.do_predict,
            evaluation_strategy=self.evalualtion_strategy,
            num_train_epochs=self.epochs,
            logging_dir=self.log_path,
            eval_steps=self.eval_steps,
            logging_steps=self.log_steps,
            save_steps=self.save_steps
        )
        args.set_dataloader(train_batch_size=self.train_batch_size,
                            eval_batch_size=self.validation_batch_size)
        args.set_testing(batch_size=self.test_batch_size)
        return args

    def _configure_trainer(self) -> None:
        '''
            This function configures the trainer
        '''
        if self.verbose:
            print("\nConfiguring trainer...")
        self.training_args = self._configure_training_arguments()
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            optimizers=(self.optimizer, self.scheduler),
            data_collator=self.data_collator,
            callbacks=[GCPCallback(
                save_to_gcp=self.save_to_gcp,
                clear_local_storage=self.clear_local_storage_on_cloud_save,
                destination_path=self.cloud_save_path,
                verbose=self.verbose
            )]
        )

    def _configure(self) -> None:
        '''
            This function configures the model
        '''
        if self.denoising_stage:
            self.decoder_tokenizer = self.encoder_tokenizer
        self.model = self._get_model()
        if self.freeze:
            self._freeze_weights()
        self._configure_optimizers()
        self._configure_collator()
        self._configure_trainer()

    def _compute_metrics(self, logits: torch.LongTensor, generations: torch.LongTensor, labels: torch.LongTensor) -> dict:
        '''
            This function computes the metrics
            Input params:
                logits: torch.LongTensor, the logits (computed by the model, by label forcing method)
                generations: torch.LongTensor, the generations (generated the model using the `self.generate_config`)
                labels: torch.LongTensor, the true labels
            Returns: dict, the metrics
        '''
        logits = torch.argmax(logits, dim=-1).tolist()
        logits = self.decoder_tokenizer.batch_decode(
            logits, skip_special_tokens=True)
        labels[labels == -100] = self.decoder_tokenizer.eos_token_id
        labels = self.decoder_tokenizer.batch_decode(
            labels, skip_special_tokens=True)
        generations = self.decoder_tokenizer.batch_decode(
            generations, skip_special_tokens=True)
        assert len(generations) == len(labels) == len(
            logits), "generations, labels and logits must have the same length"
        labels_list = [[label] for label in labels]
        sacrebleu_score = self.sacrebleu_score_metric.compute(
            predictions=logits, references=labels_list)
        chrf_score = self.chrf_score_metric.compute(
            predictions=logits, references=labels_list)
        bert_score = self.bertscore_metric.compute(
            predictions=generations, references=labels, lang=self.bert_lang)
        return {
            "sacrebleu_score": {
                "sacrebleu": sacrebleu_score["score"]
            },
            "chrf_score": {
                "chrf": chrf_score["score"]
            },
            "bert_score": {
                "precision": np.mean(np.array(bert_score["precision"])),
                "recall": np.mean(np.array(bert_score["recall"])),
                "f1": np.mean(np.array(bert_score["f1"]))
            }
        }

    def _train(self) -> None:
        '''
            This function trains the model
        '''
        checkpoint = None
        self.model.train()
        if self.resume_from_checkpoint is not None:
            checkpoint = self.resume_from_checkpoint
        elif self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.train_dataset)
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def _evaluate(self, eval_dataset: Dataset = None) -> None:
        '''
            This function evaluates the model
            Input params:
                eval_dataset: Dataset, the eval dataset
        '''
        if eval_dataset is None:
            eval_dataset = self.validation_dataset
        metrics = self.trainer.evaluate()
        metrics = {}
        metrics["validation_samples"] = len(eval_dataset)
        sacrebleu_score, chrf_score, precision, recall, f1 = 0, 0, 0, 0, 0
        self.model.to(self.device)
        self.model.eval()
        for _, batch in enumerate(tqdm(self._get_eval_data_loader(eval_dataset=eval_dataset), desc="Evaluting validation data")):
            with torch.no_grad():
                out = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device)
                )
                logits = out.logits
            generations = self.model.generate(batch["input_ids"].to(
                self.device), max_length=self.max_length, num_beams=self.num_beams, early_stopping=self.early_stopping)
            labels = copy.deepcopy(batch["labels"])
            scores = self._compute_metrics(
                logits=logits, generations=generations, labels=labels)
            sacrebleu_score += scores["sacrebleu_score"]["sacrebleu"]
            chrf_score += scores["chrf_score"]["chrf"]
            precision += scores["bert_score"]["precision"]
            recall += scores["bert_score"]["recall"]
            f1 += scores["bert_score"]["f1"]
        sacrebleu_score /= len(self._get_eval_data_loader(eval_dataset=eval_dataset))
        chrf_score /= len(self._get_eval_data_loader(eval_dataset=eval_dataset))
        precision /= len(self._get_eval_data_loader(eval_dataset=eval_dataset))
        recall /= len(self._get_eval_data_loader(eval_dataset=eval_dataset))
        f1 /= len(self._get_eval_data_loader(eval_dataset=eval_dataset))
        metrics["sacrebleu_score"] = sacrebleu_score
        metrics["chrf_score"] = chrf_score
        metrics["bert_score_precision"] = precision
        metrics["bert_score_recall"] = recall
        metrics["bert_score_f1"] = f1
        self.trainer.log_metrics("validation", metrics)
        self.trainer.save_metrics("validation", metrics)

    def _predict(self, test_dataset: Dataset = None, model: BartForConditionalGeneration = None) -> None:
        '''
            This function predicts on test data
            Input params:
                test_dataset: Dataset, the test dataset
                model: BartForConditionalGeneration, the model to be used for inference
        '''
        if test_dataset is None:
            test_dataset = self.test_dataset
        if model is None:
            model = self.model
        sacrebleu_score, chrf_score, precision, recall, f1 = 0, 0, 0, 0, 0
        model.to(self.device)
        model.eval()
        for _, batch in enumerate(tqdm(self._get_test_data_loader(test_dataset=test_dataset), desc="Evaluating test data")):
            with torch.no_grad():
                out = model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device)
                )
                logits = out.logits
            generations = model.generate(batch["input_ids"].to(
                self.device), max_length=self.max_length, num_beams=self.num_beams, early_stopping=self.early_stopping)
            labels = copy.deepcopy(batch["labels"])
            scores = self._compute_metrics(
                logits=logits, generations=generations, labels=labels)
            sacrebleu_score += scores["sacrebleu_score"]["sacrebleu"]
            chrf_score += scores["chrf_score"]["chrf"]
            precision += scores["bert_score"]["precision"]
            recall += scores["bert_score"]["recall"]
            f1 += scores["bert_score"]["f1"]
        sacrebleu_score /= len(self._get_test_data_loader(test_dataset=test_dataset))
        chrf_score /= len(self._get_test_data_loader(test_dataset=test_dataset))
        precision /= len(self._get_test_data_loader(test_dataset=test_dataset))
        recall /= len(self._get_test_data_loader(test_dataset=test_dataset))
        f1 /= len(self._get_test_data_loader(test_dataset=test_dataset))
        print(
            f"SacreBLEU score: {sacrebleu_score}\nCHRF score: {chrf_score}\nBERT score - Precision: {precision}\nBERT score - Recall: {recall}\nBERT score - F1: {f1}")

    def _fit(self) -> tuple:
        '''
            This function fits the model
            Input params: None
            Returns: tuple of the model and the best model
        '''
        if self.training_args.do_train:
            if self.verbose:
                print("Training the model...")
            self._train()
        else:
            print("Skipping training as `do_train` is False...")

    def _validate(self) -> tuple:
        '''
            This function evaluates the model
            Input params: None
            Returns: tuple of the model and the best model
        '''
        if self.training_args.do_eval:
            if self.verbose:
                print("\nValidating the model...")
            self._evaluate()
        else:
            print("Skipping evaluation as `do_eval` is False...")

    def fit(self, skip_training: bool = False, skip_validation: bool = False) -> tuple:
        '''
            This function fits the model
            Input params:
                skip_training: bool, whether to skip training or not
                skip_validation: bool, whether to skip validation or not
            Returns: tuple of the model and the best model
        '''
        if not skip_training:
            self._fit()
        if not skip_validation:
            if skip_training:
                self.model = BartForConditionalGeneration(
                    pretrained=True,
                    pretrained_path=MBART_MODEL_CONDITIONAL_GENERATION_RESUME_FROM_CHECKPOINT
                ).model
            self._validate()

    def predict(self, model_path: str = None) -> tuple:
        '''
            This function predicts the model
            Input params:
                model_path: str, path to the model to be loaded
            Returns: tuple of the model and the best model
        '''
        if self.training_args.do_predict:
            if model_path is not None:
                print("Loading the model...")
                model = BartForConditionalGeneration(
                    pretrained=True,
                    pretrained_path=model_path).model
            else:
                model = self.model
            if self.verbose:
                print("\nGenerating translations...\n")
            self._predict(model=model)
        else:
            print("Skipping prediction as `do_predict` is False...")

    def infer(self, model_path: str = None, src: Union[str, Dataset] = "Hi! tum kaisi ho?", need_print: bool = True) -> Union[str, list]:
        '''
            This function infers the model
            Input params:
                model_path: str, path to the model to be loaded
                src: str, the source language
            Returns: translated sentence
        '''
        if model_path is not None:
            if self.verbose and need_print:
                print("Loading the model...")
            model = BartForConditionalGeneration(
                pretrained=True,
                pretrained_path=model_path).model
        else:
            model = self.model
        if self.verbose and need_print:
            print("\nGenerating translation for the source string...\n")
        model.to(self.device)
        model.eval()
        if isinstance(src, str):
            src_tokenized = self.encoder_tokenizer(
                src,
                add_special_tokens=self.encoder_add_special_tokens,
                max_length=self.encoder_max_length,
                return_tensors=self.encoder_return_tensors,
                padding=self.encoder_padding,
                verbose=False
            )
            translation_ids = model.generate(
                src_tokenized["input_ids"], max_length=50, num_beams=5, early_stopping=True)
            if translation_ids.shape[1] > 1:  # More than just EOS token
                translation = self.decoder_tokenizer.decode(
                    translation_ids[0], skip_special_tokens=True)
            else:
                translation = ""
            if self.verbose and need_print:
                print("Source string: ", src)
            if self.verbose and need_print:
                print("Translated string: ", translation_ids)
            return translation
        elif isinstance(src, Dataset):
            res = []
            for _, batch in enumerate(tqdm(self._get_test_data_loader(test_dataset=src), desc="Getting translation for the data")):
                generation_ids = model.generate(batch["input_ids"].to(
                    self.device), max_length=self.max_length, num_beams=self.num_beams, early_stopping=self.early_stopping)
                generations = self.decoder_tokenizer.batch_decode(
                    generation_ids, skip_special_tokens=True)
                res.extend(generations)
            return res


# [Obsolete]
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
                 save_to_gcp: bool = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_TO_GCP,
                 save_path_dir: str = MBART_MODEL_CONDITIONAL_GENERATION_SAVE_PATH,
                 saved_model_path: str = MBART_MODEL_CONDITIONAL_GENERATION_LOAD_PATH,
                 model_name: str = MBART_MODEL_CONDITIONAL_GENERATION_TYPE,
                 generate_config: dict = {
                     "max_length": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_MAX_LENGTH,
                     "early_stopping": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_EARLY_STOPPING,
                     "num_beams": MBART_MODEL_CONDITIONAL_GENERATION_GENERATE_NUM_BEAMS
                 },
                 encoder_add_special_tokens: bool = MBART_ENCODER_ADD_SPECIAL_TOKENS,
                 encoder_max_length: int = MBART_ENCODER_MAX_LENGTH,
                 encoder_return_tensors: str = MBART_ENCODER_RETURN_TENSORS,
                 encoder_padding: Union[bool, str] = MBART_ENCODER_PADDING,
                 encoder_verbose: bool = MBART_ENCODER_VERBOSE,
                 decoder_add_special_tokens: bool = MBART_DECODER_ADD_SPECIAL_TOKENS,
                 decoder_max_length: int = MBART_DECODER_MAX_LENGTH,
                 decoder_return_tensors: str = MBART_DECODER_RETURN_TENSORS,
                 decoder_padding: Union[bool, str] = MBART_DECODER_PADDING,
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
        self.save_to_gcp = save_to_gcp
        self.save_path_dir = save_path_dir
        self.saved_model_path = saved_model_path
        self.model_name = model_name
        self.generate_config = generate_config
        self.max_length = generate_config["max_length"]
        self.early_stopping = generate_config["early_stopping"]
        self.num_beams = generate_config["num_beams"]
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
        self.loader_features = [
            "src_tokenized", "src_attention_mask", "tgt_tokenized", "tgt_attention_mask"]
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
        if self.verbose:
            print("Freezing the model...")
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
                ''.join(random.choices(string.ascii_uppercase +
                        string.digits, k=self.k_random)) + ".pth"
        else:
            model_name = self.model_name + "_" + \
                ''.join(random.choices(string.ascii_uppercase +
                        string.digits, k=self.k_random)) + f"_{special_msg}.pth"
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
        if self.verbose:
            print("\nConfiguring optimizer and scheduler...")
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
            raise NotImplementedError(
                f"Optimizer {MBART_MODEL_CONDITIONAL_GENERATION_OPTIMIZER_TYPE} not implemented.")
        if MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE == "get_linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_WARMUP_STEPS,
                num_training_steps=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TRAINING_STEPS,
                last_epoch=MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_LAST_EPOCH
            )
        else:
            raise NotImplementedError(
                f"Scheduler {MBART_MODEL_CONDITIONAL_GENERATION_SCHEDULER_TYPE} not implemented.")
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _configure_criterion(self) -> None:
        '''
            This function configures the criterion.
        '''
        if self.verbose:
            print("\nConfiguring criterion...")
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
            raise NotImplementedError(
                f"Criterion {MBART_MODEL_CONDITIONAL_GENERATION_CRITERION_TYPE} not implemented.")
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
        plt.plot(epochs, self.validation_loss, 'r*',
                 label='Validation loss spots')
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
        assert len(y) == len(
            y_hat), "Length of true and predicted sequences not equal..."
        references, predictions = [], []
        for i in range(len(y)):
            references.append([self.decoder_tokenizer.decode(
                y[i], skip_special_tokens=True).split()])
            predictions.append(self.decoder_tokenizer.decode(
                y_hat[i], skip_special_tokens=True).split())
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
        if self.verbose:
            print("\nTraining the model...")
        print(
            f"\nDEVICE - {self.device} || EPOCHS - {self.epochs} || LEARNING RATE - {self.optimizer.param_groups[0]['lr']}.")
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
                    print(
                        f'\nEPOCH - {epoch+1}/{self.epochs} || START AT - {_start_at} || LEARNING RATE - {_lr}\n')
                running_bleu, running_loss, step_running_loss, step_running_bleu = 0, 0, 0, 0
                start_step_time = time.time()
                for step, (batch) in enumerate(self.train_data_loader):
                    for key in batch:
                        if key in self.loader_features:
                            batch[key] = batch[key].to(self.device)
                    out = self.model(
                        input_ids=batch["src_tokenized"],
                        attention_mask=batch["src_attention_mask"],
                        decoder_input_ids=batch["tgt_tokenized"],
                        decoder_attention_mask=batch["tgt_attention_mask"]
                    )
                    logits = out.logits
                    y, y_hat = batch["tgt_tokenized"].view(
                        -1), logits.view(-1, logits.size(-1))
                    loss = self.criterion(y_hat, y)
                    bleu_score = self._compute_bleu(
                        true_ids=batch["tgt_tokenized"], logits=logits)
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
                                f'\tTrain Step - {step+1}/{len(self.train_data_loader)} | ' +
                                f'Train Step Loss: {(step_running_loss/self.verbose_step):.5f} | ' +
                                f'Train BLEU Score: {(step_running_bleu/self.verbose_step):.5f} | ' +
                                f'Time: {(time.time() - start_step_time):.2f}s.\n'
                            )
                            step_running_loss = 0
                            step_running_bleu = 0
                            start_step_time = time.time()
                            if self.save_model:
                                self._save(
                                    model=self.model, special_msg=f"epoch_{str(epoch+1)}_step_{str(step+1)}")
                self.train_loss.append(
                    running_loss/len(self.train_data_loader))
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || TRAIN-LOSS - {(running_loss/len(self.train_data_loader)):.5f} || BLEU SCORE - {bleu_score:.5f} || TIME ELAPSED - {(time.time() - start_epoch_time):.2f}s.\n')
                # Validation Phase
                start_epoch_validation_time = time.time()
                self.model.eval()
                running_validation_bleu, running_validation_loss = 0, 0
                with torch.no_grad():
                    for _, (validation_batch) in enumerate(self.validation_data_loader):
                        for key in validation_batch:
                            if key in self.loader_features:
                                validation_batch[key] = validation_batch[key].to(
                                    self.device)
                        out = self.model(
                            input_ids=validation_batch["src_tokenized"],
                            attention_mask=validation_batch["src_attention_mask"],
                            decoder_input_ids=validation_batch["tgt_tokenized"],
                            decoder_attention_mask=validation_batch["tgt_attention_mask"]
                        )
                        logits = out.logits
                        y, y_hat = validation_batch["tgt_tokenized"].view(
                            -1), logits.view(-1, logits.size(-1))
                        loss = self.criterion(y_hat, y)
                        bleu_score = self._compute_bleu(
                            true_ids=validation_batch["tgt_tokenized"], logits=logits)
                        running_validation_loss += loss.item()
                        running_validation_bleu += bleu_score
                if self.verbose:
                    print(f'\tEPOCH - {epoch+1}/{self.epochs} || VAL-LOSS - {(running_validation_loss/len(self.validation_data_loader)):.5f} || BLEU SCORE - {(running_validation_bleu/len(self.validation_data_loader)):.5f} || TIME ELAPSED - {(time.time() - start_epoch_validation_time):.2f}s.\n')
                self.validation_loss.append(
                    running_validation_loss/len(self.validation_data_loader))
                if self.best_test_loss > running_validation_loss:
                    self.best_test_loss = running_validation_loss
                    self.best_model = copy.deepcopy(self.model)
                self.scheduler.step()
                if self.save_model:
                    self._save(model=self.model,
                               special_msg=f"epoch_{str(epoch+1)}")
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
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping
        )

    def _test(self) -> tuple:
        '''
            This function tests the model
            Input params: None
            Returns: tuple of the model and the best model
        '''
        assert self.test_model is not None, "Model not loaded, set `test_model`..."
        if self.verbose:
            print("\nTesting the model...")
        start_test_time = time.time()
        self.test_model = self.test_model.to(self.device)
        test_loss, test_bleu_Score = 0, 0
        for _, (test_batch) in tqdm(enumerate(self.test_data_loader)):
            for key in test_batch:
                if key in self.loader_features:
                    test_batch[key] = test_batch[key].to(self.device)
            out = self.test_model(
                input_ids=test_batch["src_tokenized"],
                attention_mask=test_batch["src_attention_mask"],
                decoder_input_ids=test_batch["tgt_tokenized"],
                decoder_attention_mask=test_batch["tgt_attention_mask"]
            )
            logits = out.logits
            y, y_hat = test_batch["tgt_tokenized"].view(
                -1), logits.view(-1, logits.size(-1))
            loss = self.criterion(y_hat, y)
            test_loss += loss.item()
            bleu_score = self._compute_bleu(
                true_ids=test_batch["tgt_tokenized"], logits=logits)
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
        output = self._generate(
            input_ids=src_tokenized["input_ids"].to(device))
        output_str = self.decoder_tokenizer.decode(
            output[0].tolist(), skip_special_tokens=True)
        return output_str

    def load_model(self) -> object:
        '''
            Loads the saved model
        '''
        if self.verbose:
            print("Loading model...")
        self.model = self._get_model(self.model_name)
        if self.saved_model_path is not None:
            state_dict = torch.load(
                self.saved_model_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.start_epoch = state_dict['epoch']
            if self.verbose:
                print(f"Loaded model from {self.saved_model_path}...")
        else:
            if self.verbose:
                print(
                    "No saved model to load as `saved_model_path` was not provided in the `__init__()`...")
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
        if self.generate_config is None:
            self._configure_generate_config()
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
        if self.verbose:
            print("Inferring...")
        if self.generate_config is None:
            self._configure_generate_config()
        res = []
        for s in src:
            res.append((s, self._infer(
                src=s,
                model=model,
                device=device
            )))
        return res
