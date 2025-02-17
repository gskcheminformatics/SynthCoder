# Taken from SimpleTransforers (Nov 2023), but modified a lot. 
# https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/config/model_args.py
#
# The module is responsible for taking care of the model arguments. 
# Modify the arg values listed here to set the default values. 
#

import json
import os
import logging
from dataclasses import dataclass, field
from typing import Union, NoReturn
from regex import Regex
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)

@logged()
def get_special_tokens() -> list[str]:
    """
    Returns special tokens as a list of strings.

    Returns:
    ========
    List. Special tokens.  
    """
    return ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]



@dataclass
class ModelArgs:
    """ 
    Class for general arguments for all models.      
    """

    adafactor_beta1: float = None
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_eps: tuple = field(default_factory=lambda: (1e-30, 1e-3))
    adafactor_relative_step: bool = True
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = True
    adam_amsgrad: bool=False
    adam_betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    adam_epsilon: float = 1e-8
    adversarial_epsilon: float = 0.05
    adversarial_loss_alpha: float = 0.5
    adversarial_training: bool = False
    adversarial_training_probability: float = 1.0
    batch_size_eval: int = 4
    batch_size_train: int = 4
    block_size: int = -1
    cache_dir: str = "cache_dir/"
    confidence_learning: bool = False
    config: dict = field(default_factory=dict)
    config_name: str = None
    cosine_schedule_num_cycles: float = 0.5
    cross_attention_max_num_cmpds: int = 20 
    cross_attention_number_of_descriptors: int = None
    cross_attention_use_extended_descript_network: bool = False
    custom_layer_parameters: list = field(default_factory=list)
    # # E.g. 
    # [{"layer": 10, "lr": 1e-3,},
    # {"layer": 0, "lr": 1e-5,},]
    custom_parameter_groups: list = field(default_factory=list)
    # E.g. 
    # [{
    #     "params": ["classifier.weight", "bert.encoder.layer.10.output.dense.weight"],
    #     "lr": 1e-2,
    # }]
    cyclic_lr_scheduler_base_lr: float=1e-12
    cyclic_lr_scheduler_cycle_momentum: bool=False  # if True, it will not work with AdamW optimiser
    cyclic_lr_scheduler_gamma: float=1.0
    cyclic_lr_scheduler_mode: str="exp_range"  # can be triangular, triangular2, exp_range.
    cyclic_lr_scheduler_ratio_size_down: float=None
    cyclic_lr_scheduler_ratio_size_up: float=0.06
    dataloader_num_workers: int = 0
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = False
    dataloader_shuffle: bool = True
    detect_anomaly: bool = False  # Setting this to True has a very significant (bad) effect on the performance
    deterministic: bool = False
    early_stopping_delta: float = 0.0
    early_stopping_divergence_threshold: float = None
    early_stopping_metric: str = "valid_loss"  # Set to None to disable early stopping
    early_stopping_mode: str = "min" # can be "min" or "max"
    early_stopping_patience: int = 3
    early_stopping_threshold: float = None
    gradient_accumulation_steps: int = 1
    ignore_mismatched_sizes: bool = False
    learning_rate: float = 2e-5
    learning_rate_logging_interval: str = "step"  # can also be epoch; to disable logging set to None
    limit_val_batches: Union[int, float] = None
    log_every_n_steps: int = 50 
    logger_dir: str = "logs/"
    log_on_step: bool = True
    log_on_epoch: bool = True
    manual_seed: int = 42
    max_seq_length: int = 1024
    model_name: str = None
    model_type: str = None
    not_saved_args: list = field(default_factory=list)
    num_train_epochs: int = None
    num_train_steps: int = -1
    optimizer: str = "AdamW"
    overfit_batches: Union[float, int] = 0  # When set to 0, no overfitting will be done. 
    pad_to_multiple_of: int = 4
    pl_gradient_clip_algorithm: str = None  # be "value" or "norm", for gradient clipping via PyTorch Lightning; set to None to have no clipping
    pl_gradient_clip_val: float = None  # max magnitude of the gradient, for gradient clipping via PyTorch Lightning; set to None to have no clipping
    polynomial_decay_schedule_lr_end: float = 1e-7
    polynomial_decay_schedule_power: float = 1.0
    precision: Union[str, int] = "32-true"
    profiler: str=None  # can also be "simple"
    save_last_checkpoint: bool = True
    save_top_k_checkpoints: int = 3
    scheduler: str = "linear_schedule_with_warmup"  # Can be: "constant_schedule", "constant_schedule_with_warmup", "linear_schedule_with_warmup", "cosine_schedule_with_warmup", "cosine_with_hard_restarts_schedule_with_warmup", "polynomial_decay_schedule_with_warmup", "cyclic_lr_scheduler"
    stochastic_wght_avging_anneal_epochs: int = 10
    stochastic_wght_avging_epoch_start: Union[int, float] = 0.8
    stochastic_wght_avging_lrs: Union[float, list[float]] = 2e-5 # set to None to disable the stochastic weight averaging.
    stochastic_wght_avging_strategy: str = None # can be "cos" or "linear" 
    tokenizer_add_metaspace: bool = False
    tokenizer_descriptors_padding_value: int = -1 
    tokenizer_do_lower_case: bool = False
    tokenizer_enrichment_vocab_size: int = None
    tokenizer_min_frequency: int = 2
    tokenizer_molecule_joiner: str = "^"
    tokenizer_molecule_separator: str = "."
    tokenizer_name: str = None
    tokenizer_output_dir: str = "tokenizer_outputs/"
    tokenizer_pattern_split: Union[str, list[str]] = None
    tokenizer_punctuation_split: bool = False
    tokenizer_reaction_arrow_symbol: str = ">>"
    tokenizer_special_tokens: list = field(default_factory=get_special_tokens)
    tokenizer_split_behavior: Union[str, list[str]] = "isolated"
    tokenizer_strip_accents: bool = True
    tokenizer_suffix_CharBPETokenizer: str = "</w>"
    tokenizer_trainer_class_name: str = "BertWordPieceTokenizer"
    tokenizer_unicode_normaliser: str = "nfd"
    tokenizer_unk_token: str = "[UNK]"
    tokenizer_vocab_size: int = None
    tokenizer_whitespace_split: bool = True
    track_grad_norm: Union[float, int, str] = 2 # Set to None to disable the gradient norm tracking. 
    train_custom_parameters_only: bool = False
    warmup_ratio: float = 0.06  # take a look at this paper: How to Train BERT with an Academic Budget by Izsak et al. (https://arxiv.org/pdf/2104.07705.pdf)
    warmup_steps: int = None  # takes precedence over the warmup_ratio if different than 0 
    weight_decay: float = 0.0

    @logged()
    def update_from_dict(self, new_values: dict) -> Union[None, NoReturn]:
        """
        Updates the class named attributes with the entries from the provided dictionary. 
        
        Parameters:
        ===========
        new_values: Dict. A dictionary where the keys are the attribute names,
        and the values are the values to set for these attributes.

        Returns:
        ========
        None
        """
        logger.debug(None)
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    @logged()
    def get_args_for_saving(self) -> dict:
        """
        Returns a dictionary of the class arguments to be saved.  
                
        Returns:
        ========
        args_for_saving: Dict. 
        """
        logger.debug(None)
        args_for_saving = {
            key: value
            for key, value in vars(self).items()  # changed asdict() to vars() to account for new object variables assigned after initialisation 
            if key not in self.not_saved_args
        }

        return args_for_saving

    @logged()
    def save(self, output_dir: str) -> None:
        """
        Dumps the class arguments as model_args.json file into a specified directory.

        Parameters:
        ===========
        output_dir: Str. The directory under which the generated file will be saved. 

        Returns:
        ========
        None
        """
        logger.debug(None)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            json.dump(args_dict, f, indent=4)

    @logged()
    def load(self, input_dir: str) -> None:
        """
        Loads and updates the class arguments using model_args.json file under the specifed directory.
        
        Parameters:
        ===========
        input_dir: Str. The directory from which the model_args.json file will be read. 

        Returns:
        ========
        None
        """
        logger.debug(None)
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)


@dataclass
class LanguageModelingArgs(ModelArgs):
    """
    Model args for a LanguageModelingModel (this includes the MLM tasks/models)
    """

    # dataset_type: str = "line"
    log_molecular_correctness: bool = False
    log_mlm_exact_vector_match: bool = False
    mlm: bool = True
    mlm_probability: float = 0.15  # can be between 0 and 1
    mlm_span_rxn_probability: float = 0.0  # can be between 0 and 1
    model_class: str = "LanguageModelingModel"

    @logged()
    def save(self, output_dir: str="./") -> None:
        """
        Overloads the parent's method.

        Dumps the class arguments as model_args.json file into a specified directory.

        Parameters:
        ===========
        output_dir: Str. The directory under which the generated file will be saved.

        Returns:
        ========
        None
        """
        logger.debug(None)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            json.dump(self.get_args_for_saving(), f, indent=4)

    @logged()
    def load(self, input_dir: str) -> None:
        """
        Overloads the parent's method.

        Loads and updates the class arguments using model_args.json file under the specifed directory.

        Parameters:
        ===========
        input_dir: Str. The directory under which the  `model_args.json` can be found.

        Returns:
        ========
        None
        """
        logger.debug(None)
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)
                self.update_from_dict(model_args)


@dataclass
class ClassificationArgs(ModelArgs):
    """
    Model args for a ClassificationModel 
    (this includes regression ;-b, binary and multiclass classification).
    """

    batch_size_predict: int = 4
    batch_size_test: int = 4
    class_weights: list[Union[int, float]] = field(default_factory=list)
    create_prediction_files: bool=True
    log_metrics: bool = True
    model_class: str = "ClassificationModel"
    montecarlo_dropout: bool = False
    montecarlo_dropout_num_iters: int = 100
    num_labels: int = 2
    prediction_output_dir: str = "inference_results/"
    problem_type: str = "single_label_classification"  # can be "regression", "single_label_classification", "multi_label_classification"
