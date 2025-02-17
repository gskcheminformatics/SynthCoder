# Module with a focus on data processing 

import os
import logging
from typing import Union, NoReturn, Literal
import json
# import numpy as np
# import pandas as pd

import lightning as pl
# from lightning import Trainer
# from lightning.pytorch.callbacks import Callback

import torch
# from torch import Tensor
# import torch.functional as F
from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence

from transformers import (DataCollatorForLanguageModeling, DataCollatorWithPadding)
# from transformers.data.datasets.language_modeling import LineByLineTextDataset, TextDataset
from transformers.tokenization_utils_base import BatchEncoding
# from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer  # HuggingFace
import datasets
from datasets import Features, Value, Array2D, Array3D
from datasets.arrow_dataset import Dataset  

import synthcoder_project.utilities as utils
from synthcoder_project.synthcoder_tokenizers.tokenizer_trainers import TokenizerTrainer
from  synthcoder_project import synthcoder_config 
# from synthcoder_project.custom_collators.custom_collators import SynthBertXCollatorForLanguageModeling

# TODO change it back
from synthcoder_project.custom_collators.custom_collators import SynthBertXCollatorForLanguageModeling
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)



class DataModuleMLM(pl.LightningDataModule):
    """ 
    Class responsible for input data processing and preparation.
    Can be used directly with MLM tasks.

    - Trains/Sets-up a tokenizer
    - Prepares training and validation data
    - Is used by a PyTorch Lightning trainer for model training and validation 

    Based on the Pytorch Lightning framework:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, 
                 args: object, 
                 model_tokenizer: type,
                 tokenizer_train_file: str=None,
                 train_file: str=None,
                 validation_file: str=None,
                 **kwargs):
        
        """ 
        Initialises DataModule object. 
        Sets up tokenizer. 

        Parameters:
        ===========
        args: Object. Arguments (including all setting values) for the model, training, tokenization etc. 
        model_tokenizer: Class of the Encoder tokenizer (e.g. BertTokenizer)
        tokenizer_train_file: Str. Path to text file for the tokenizer training. 
        train_file: Str. Path to file to be used when training the model.
        validation_file: Str. Path to file to be used when validating the model.
        **kwargs
        
        Returns:
        ========
        None
        """

        super().__init__()
        self.args = args
        self.model_tokenizer = model_tokenizer
        self.tokenizer_train_file = tokenizer_train_file
        self.train_file = train_file
        self.validation_file = validation_file

        # Use the specified number of dataloader workers for dataloading and encoding, or use the max numbner of CPUs minus one. 
        self.num_avail_workers = self.args.dataloader_num_workers if self.args.dataloader_num_workers != 0 else os.cpu_count()-1

        # model_args need to be fed as a dictionary, as object will cause a crash - object cannot be saved as JSON during model saving. 
        self.tokenizer, self.new_tokenizer = self.setup_tokenizer(model_args=vars(self.args), **kwargs)
        self.revise_block_size()
        self.collator = self.set_collator()

    @logged()
    def prepare_data(self) -> None:
        """ 
        Not impelmented.

        Default PyTorch Lightning function, called automatically by the Lightning framework.
        prepare_data is called from the main process. It is not recommended 
        to assign state here (e.g. self.x = y) since it is called on a single 
        process and if you assign states here then they won't be available for other processes.

        Can be used e.g. to download data. 
        """
        pass

    @logged()
    def setup(self, stage: Literal["fit", "validate"]) -> None:
        """
        Default PyTorch Lightning function, called automatically by the Lightning framework.
        Sets up training and validation datasets based on the pre-specified train and validation data files. 

        Parameters:
        ===========
        stage: Str. ["fit", "validate"]. Name of the stage.

        Returns:
        ========
        None
        """
        # See: https://stackoverflow.com/questions/73130005/when-is-stage-is-none-in-pytorch-lightning
        if self.train_file and stage == "fit":
            print("Setting up training dataset:")
            self.train_dataset = self.set_up_dataset(file_path=self.train_file)
        if self.validation_file and stage in ("fit", "validate"):
            print("Setting up validation dataset:")
            self.validation_dataset = self.set_up_dataset(file_path=self.validation_file, stage=stage)

    @logged()
    def train_dataloader(self) -> object:
        """
        Default PyTorch Lightning function, called automatically by the Lightning framework.
        Returns initalised dataloader for model training.

        Returns:
        ========
        Object. Initialised dataloader for model training.
        """
        return DataLoader(dataset=self.train_dataset, 
                            batch_size=self.args.batch_size_train, 
                            collate_fn=self.collator,
                            shuffle=self.args.dataloader_shuffle,
                            num_workers=self.num_avail_workers,
                            # worker_init_fn=L.pl_worker_init_function,
                            generator=torch.Generator().manual_seed(self.args.manual_seed), #self.torch_generator,
                            persistent_workers=self.args.dataloader_persistent_workers,
                            pin_memory=self.args.dataloader_pin_memory,
                            )    
    
    @logged()
    def val_dataloader(self) -> object:
        """
        Default PyTorch Lightning function, called automatically by the Lightning framework.
        Returns initalised dataloader for model validation.

        Returns:
        ========
        Object. Initialised dataloader for model validation.
        """
        return DataLoader(dataset=self.validation_dataset, 
                          batch_size=self.args.batch_size_eval,
                          collate_fn=self.collator,
                          num_workers=self.num_avail_workers,
                        #   worker_init_fn=L.pl_worker_init_function,
                          generator=torch.Generator().manual_seed(self.args.manual_seed),
                          persistent_workers=self.args.dataloader_persistent_workers,
                          pin_memory=self.args.dataloader_pin_memory,
                          )
                          #  shuffle does not work (also does not make sense) for validation, testing or prediction 
    
    @logged()
    def return_len_train_dataloader(self) -> int:
        """
        Returns:
        ========
        Int. Length of the training dataloader.
        """
        return len(self.train_dataloader())

    @logged()
    def set_collator(self) -> DataCollatorForLanguageModeling:
        """ 
        Sets a collator for MLM problems.
        
        Returns:
        ========
        collator: DataCollatorForLanguageModeling. Initialised collator for MLM taks. 
        """
        collator = SynthBertXCollatorForLanguageModeling(
                            tokenizer=self.tokenizer,
                            mlm=self.args.mlm,
                            mlm_probability=self.args.mlm_probability,
                            #   padding="max_length",
                            #   max_length=self.args.block_size,
                            pad_to_multiple_of=self.args.pad_to_multiple_of,
                            mlm_descriptors=self.args.cross_attention_use_extended_descript_network,
                            mlm_span_rxn_probability=self.args.mlm_span_rxn_probability,
                          )
        
        return collator

    @logged()
    def setup_tokenizer(self, **kwargs) -> Union[tuple[object, bool], NoReturn]:     
        """
        Sets up a tokenizer from pretained one, or trains a new one.
        
        Modfied SimpleTransformers method. (language_modeling_model.py)

        Returns:
        ========
        tokenizer: Object. Pre-trained or newly trained tokenizer.
        new_tokenizer: Bool. Indicates if a new tokenizer has been trained.
        """
        new_tokenizer = False

        # Try to use pre-trained tokenizer first, based on either tokenizer_name or model_name

        if self.args.model_name:
            self.args.tokenizer_name = self.args.model_name
            tokenizer = self.model_tokenizer.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir, **kwargs)
            
            logger.info("Using the pre-trained tokenizer from the pre-trained model")
            print("Using a pre-trained tokenizer")
        
        elif self.args.tokenizer_name:
            tokenizer = self.model_tokenizer.from_pretrained(self.args.tokenizer_name, cache_dir=self.args.cache_dir, **kwargs)
            
            logger.info("Using the pre-trained tokenizer from the pre-trained model")
            print("Using a pre-defined tokenizer")

        # Try to train the tokenizer from scratch
        else:
            if not self.tokenizer_train_file:
                logger.error("model_name and tokenizer_name are not specified")
                raise ValueError(
                    "model_name and tokenizer_name are not specified."
                    "You must specify tokenizer_train_file to train a Tokenizer.")
            else:
                tokenizer = self.train_tokenizer(self.tokenizer_train_file, **kwargs)  # trains a new tokenizer <- TODO this will need to be modified to allow for free choice of the tokenizer
                new_tokenizer = True

                logger.info("Training a new tokenizer")
                print("Training a new tokenizer")
        
        # Setting `Asking-to-pad-a-fast-tokenizer` to True allows to disable a speed warning 
        # about using a collator padding with a Fast tokenizer. No better solution at the time of writing this code. 
        # The tokenizer is objectively very fast as is.
        # Please see: https://github.com/huggingface/transformers/issues/22638
        # and https://discuss.huggingface.co/t/get-using-the-call-method-is-faster-warning-with-datacollatorwithpadding/23924/4
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return tokenizer, new_tokenizer
        
    @logged()
    def train_tokenizer(
        self,
        train_file: str,
        output_dir: str=None,
        use_trained_tokenizer: bool=True,
        **kwargs
        ) -> Union[object, NoReturn]:
        """
        Train a new tokenizer on `train_files`.
        Note: unfortunately you cannot generate vocabulary directly with the tokenizers from the transformer 
        library (e.g. BertTokenizerFast), so the vocabulary is generated by tokenizers from  the tokenizers library (e.g. BertWordPieceTokenizer).
     
        Parameters:
        ===========
        train_files: Str. Path to the file to be used when training the tokenizer.
        output_dir: Str. The directory where tokenizer model files will be saved. If not given, self.args.tokenizer_output_dir will be used.
        use_trained_tokenizer: Bool. Load the trained tokenizer once training completes.

        Returns:
        ========
        tokenizer: Object. Trained tokenizer. 

        """
        if not self.args.tokenizer_vocab_size:
            logger.error("Vocab size is not specified")
            raise ValueError(
                "Cannot train a new tokenizer as tokenizer_vocab_size is not specified in args dict. "
                "Either provide a tokenizer or specify vocab_size.")

        if not train_file:
            logger.error("Tokenizer train file not provided")
            raise ValueError("Cannot train the tokenizer as the tokenizer train file has not been provided.")

        if not output_dir:
            output_dir = self.args.tokenizer_output_dir
        
        # Create the directory for storing the tokenizer files if it does not exist yet.
        # Then remove any files within that directory that have .txt or .json extension.
        # This is done to make sure that no file from an old tokenizer are left behind 
        # (different tokenizers produce different files, that may cause issues during loading when mixed together). 
        utils.create_directory_and_delete_files_inside(directory=output_dir, file_extension=(".json", ".txt"))

        if self.args.model_type in synthcoder_config.SUPPORTED_MODELS:

            tokenizer_file_path_to_save = os.path.join(output_dir, "tokenizer.json")
            # print(tokenizer_file_path_to_save)

            tokenizer_trainer = TokenizerTrainer(
                    model_type=self.args.model_type,  
                    tokenizer_train_file=train_file,
                    tokenizer_file_path_to_save=tokenizer_file_path_to_save,
                    model_tokenizer=self.model_tokenizer,
                    tokenizer_trainer_class_name=self.args.tokenizer_trainer_class_name,
                    special_tokens=self.args.tokenizer_special_tokens,
                    unk_token=self.args.tokenizer_unk_token,
                    vocab_size=self.args.tokenizer_vocab_size,
                    min_frequency=self.args.tokenizer_min_frequency,
                    strip_accents=self.args.tokenizer_strip_accents,
                    do_lower_case=self.args.tokenizer_do_lower_case,
                    unicode_normalizer=self.args.tokenizer_unicode_normaliser,
                    add_metaspace=self.args.tokenizer_add_metaspace,
                    whitespace_split=self.args.tokenizer_whitespace_split,
                    punctuation_split=self.args.tokenizer_punctuation_split,
                    pattern_split=self.args.tokenizer_pattern_split,
                    split_behavior=self.args.tokenizer_split_behavior,
                    suffix_CharBPETokenizer=self.args.tokenizer_suffix_CharBPETokenizer,
                    **kwargs
                    )

            tokenizer = tokenizer_trainer.return_trained_tokenizer()
            tokenizer.save_pretrained(output_dir)  # save all the tokenizer files needed for a later use with a new Encoder model (when `.from_pretrained()` needs to be used)

        else:
            logger.error("The specified `model_type` is not suported")
            raise ValueError("The specified `model_type` is not suported"
                            f"Supported model types are: {synthcoder_config.SUPPORTED_MODELS}")

        self.args.tokenizer_name = output_dir  # tokenizer name in this case is just the directory with the relevant tokenizer files.
        return tokenizer

    @logged()
    def is_new_tokenizer(self) -> bool:
        """
        Returns:
        ========
        new_tokenizer: bool. Indicates if a new tokenizer has been trained.
        """
        return self.new_tokenizer

    @logged()
    def return_tokenizer(self) -> object:
        """
        Returns:
        ========
        tokenizer: Object. Pre-trained or newly trained tokenizer.
        """
        return self.tokenizer

    @logged()
    def set_up_dataset(self, file_path: str, stage: str=None) -> Union[Dataset, NoReturn]:
        """
        Reads a text file from file_path and creates training features.
        Encodes text for MLM, creating iput IDs and attention mask. 
        This function is just for MLM.

        Parameters:
        ===========
        file_path: Str. Path to the file to encode. 
        stage: Str. It's optional. If provided and set to "fit" or "validate", the unmodified input IDs will be copied into "original_input_ids" column

        Returns:
        ========
        Dataset. Dataset for MLM.
        """
        # if self.args.dataset_type:
        dir, file_name = os.path.split(file_path)

        dataset = datasets.load_dataset(path=dir, data_files=file_name, cache_dir=self.args.cache_dir) #, sample_by=self.args.dataset_type)
        num_proc = self.num_avail_workers if stage == "train" else 1  # force mapping to use only 1 process when validating, testing or predicting, otherwise you can get "RuntimeError: Cannot re-initialize CUDA in forked subprocess". 
        dataset = dataset.map(self.encode, batched=False, num_proc=num_proc) #self.num_avail_workers)
        dataset = dataset.remove_columns(["text"])

        # Make sure to choose the right data columns for the model:
        columns=["input_ids", "attention_mask"] #["input_ids", "token_type_ids", "attention_mask"])  
        
        if "enriching_ids" in dataset["train"].column_names:
            logger.info("Enriching IDs generated")
            print("\nEnriching IDs generated.")
            dataset = dataset.remove_columns(["enriching_classes"])
            columns.append("enriching_ids")
        
        # Copy the input_ids into a new column as these will be corrupted (on purpose) by the collator for MLM.
        # Having a copy of the original token IDs, will allow us to evaluate how well MLM works using external/additional metrics. 
        if stage in ("fit", "validate"):
            dataset = dataset.map(lambda batch: {"original_input_ids": batch["input_ids"]})
            columns.append("original_input_ids")

        # Try adding columns for cross-attention between descriptors and tokens, if descriptors are available.
        columns = self.process_descriptors_column(dataset=dataset, columns=columns)

        if "main_rxn_component_mask" in dataset["train"].column_names:
            columns.append("main_rxn_component_mask")

        # Set datatype formant and retain only the columns present in the `columns` list. 
        dataset.set_format(type='torch', columns=columns)

        print()
        print(dataset["train"].features)
        print(dataset["train"])
        print()

        return dataset["train"]  # by default any data after loading dataset are under "train" unless specified differently
        
    @logged()
    def process_descriptors_column(self, dataset: datasets.dataset_dict.DatasetDict, columns: list) -> list:
        """
        Tries to process the "descriptors" column in the provided data, if such column exists. 

        Parameters:
        ===========
        dataset: datasets.dataset_dict.DatasetDict. The dataset generated for the provided data. 
        columns: List. List of column names so far, to be used by the model. 

        Returns:
        ========
        columns: List. Updated column list.
        """

        # Try adding columns for cross-attention between descriptors and tokens, if descriptrs are available.
        if "descriptors" in dataset["train"].column_names:
            logger.info("nDescriptors processed by the tokenizer. Descriptor masks generated.")
            print("\nDescriptors processed by the tokenizer. Descriptor masks generated.")

            found_num_descriptors = int(len(dataset["train"]["descriptors"][0]) 
                                        / self.tokenizer.cross_attention_max_num_cmpds)

            if self.args.cross_attention_number_of_descriptors != found_num_descriptors:
                logger.error("Numbers of descriptors in the file and the model arguments do not match")
                raise ValueError(f"The number of descriptors specified in the model arguments is {self.args.cross_attention_number_of_descriptors=} "
                                 f"but the number of the descriptors found during tokenization is {found_num_descriptors}")
            
            columns.append("descriptors")
            columns.append("descriptors_attention_mask")
        
        else:
            if self.args.cross_attention_number_of_descriptors:
                logger.error("No descriptors column found in the provided data file")
                raise ValueError(f"The number of descriptors specified in the model arguments is {self.args.cross_attention_number_of_descriptors=} "
                                 f"but no 'descriptors' column was found during tokenization in the provided datafile.")

        return columns
    
    @logged()
    def encode(self, examples: object) -> BatchEncoding:
        """
        Encodes text using a tokenizer.

        Parameters:
        ===========
        examples: Object. Data containing text for encoding. 

        Returns:
        ========
        BatchEncoding. Object holding the output of the tokenizer, derived from a dictionary.   
        """
        ################################## TODO Check max length of encoded sequence vs desired input to the matrix!

        #TODO check the influence of the block size on the actual size of the encoded sequences
        try:
            try:
                enriching_classes = examples["enriching_classes"]
                logger.debug("Providing `enriching_classes` to tokenizer")
            except KeyError:
                enriching_classes = None

            try:
                descriptors = examples["descriptors"]
                logger.debug("Providing `descriptors` to tokenizer")
            except KeyError:
                descriptors = None

            try:
                main_components_indices = examples["main_components_indices"]
                logger.debug("Providing `main_components_indices` to tokenizer")
            except KeyError:
                main_components_indices = None

            return self.tokenizer(examples["text"], 
                                    truncation=True, 
                                    padding='max_length',
                                    max_length=self.args.block_size, 
                                    enriching_classes=enriching_classes,
                                    descriptors=descriptors,
                                    main_components_indices=main_components_indices,
                                    )
    
        except (KeyError, TypeError):
            logger.debug("Falling back onto the traditional argument set for tokenizer.")
            return self.tokenizer(examples["text"], 
                                    truncation=True, 
                                    padding='max_length',
                                    max_length=self.args.block_size, 
                                    )

    @logged()
    def revise_block_size(self) -> None:
        """
        Sets up a revised block size to account for the max length set in the tokenizer (and possibly the chosen max seq length).
            Note: block_size is the optional input sequence length after tokenization. 
            The training dataset will be truncated in block of this size for training. 
            Normally it is default to the model max input length for single sentence inputs.

        SimpleTransformers method. (language_modeling_model.py)

        Returns:
        ========
        None
        """        
        if self.args.block_size <= 0:
            self.args.block_size = min(
                self.args.max_seq_length, 
                self.tokenizer.model_max_length
                )
        else:
            self.args.block_size = min(
                self.args.block_size,
                self.tokenizer.model_max_length, # for BertTokenizer and DistilBertTokenizer that is a very large number
                self.args.max_seq_length,
                )
            
        logger.debug(f"Block size set to: {self.args.block_size}")

    @logged()
    def set_file(self, file_path: str, step: Literal["fit", "validate"]) -> None:
        """
        Set a new file for training or validation
        
        Parameters:
        ===========
        file_path: Str. New path to be set for a file
        step: Str. ["fit", "validate"]. The step for which to change the file
        
        Returns:
        ========
        None
        """        
        if step == "fit":
            self.train_file = file_path
        elif step == "validate":
            self.validation_file = file_path

    @logged()
    def set_batch_size(self, batch_size: int , step: Literal["fit", "validate"]):
        """
        Set a new batch size for training or validation.

        Parameters:
        ===========
        batch_size: Int. New batch size for a given step.
        step: Str. ["fit", "validate"]. The step for which to change the batch size.
        
        Returns:
        ========
        None
        """
        if step == "fit":
            self.args.batch_size_train = batch_size
        elif step == "validate":
            self.args.batch_size_eval = batch_size



class DataModuleForClassification(DataModuleMLM):
    """ 
    Class responsible for input data processing and preparation for classification problems
    
    In the current form it can work for regression or single-label binary/multiclass classification.
    Modification would need to be done to also accept data for multi-label classification. 

    Based on the Pytorch Lightning framework:
    https://lightning.ai/docs/pytorch/stable/data/datamodule.html

    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, 
                 args: object, 
                 model_tokenizer: type,
                 tokenizer_train_file: str=None,
                 train_file: str=None,
                 validation_file: str=None,
                 test_file: str=None,
                 predict_file: str=None,
                 **kwargs):
        
        """ 
        Initialises DataModule object. 
        Sets up tokenizer. 

        Parameters:
        ===========
        args: Object. Arguments (including all setting values) for the model, training, tokenization etc. 
        model_tokenizer: Class of the Encoder tokenizer (e.g. BertTokenizer)
        tokenizer_train_file: Str. Path to text file for the tokenizer training. 
        train_file: Str. Path to file to be used when training the model.
        validation_file: Str. Path to file to be used when validating the model.
        test_file: Str. Path to file to be used when testing the model.
        predict_file: Str. Path to file to be used for inference with the model.
        **kwargs

        Returns:
        ========
        None
        """
        
        self.test_file = test_file
        self.predict_file = predict_file

        super().__init__(args, model_tokenizer, tokenizer_train_file, train_file, 
                         validation_file, **kwargs)
        
    @logged()
    def set_file(self, file_path: str, step: Literal["fit", "validate", "test", "predict"]) -> None:
        """
        Set a new file for training/validation/testing/inference.
        
        Parameters:
        ===========
        file_path: Str. New path to be set for a file
        step: Str. ["fit", "validate", "test", "predict"]. The step for which to change the file
        
        Returns:
        ========
        None
        """        
        super().set_file(file_path, step)
        if step == "test":
            self.test_file = file_path
        elif step == "predict":
            self.predict_file = file_path
       
    @logged()
    def set_batch_size(self, batch_size: int , step: Literal["fit", "validate", "test", "predict"]):
        """
        Set a new batch size for training/validation/testing/inference.

        Parameters:
        ===========
        batch_size: Int. New batch size for a given step.
        step: Str. ["fit", "validate", "test", "predict"]. The step for which to change the batch size.
        
        Returns:
        ========
        None
        """        
        super().set_batch_size(batch_size, step)
        if step == "test":
            self.args.batch_size_test = batch_size
        elif step == "predict":
             self.args.batch_size_predict = batch_size
    
    @logged()
    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        """
        Default PyTorch Lightning function, called automatically by the Lightning framework.
        Sets up training and validation datasets based on the pre-specified train and validation data files. 

        Parameters:
        ===========
        stage: Str. ["fit", "validate", "test", "predict"]. Name of the stage.

        Returns:
        ========
        None
        """
        super().setup(stage=stage)
        if self.test_file and stage == "test":
            print("Setting up test dataset:")
            self.test_dataset = self.set_up_dataset(file_path=self.test_file)
        if self.predict_file and stage == "predict":
            print("Setting up predict dataset:")
            self.predict_dataset = self.set_up_dataset(file_path=self.predict_file, stage=stage)

    @logged()
    def test_dataloader(self) -> None:
        """
        Default PyTorch Lightning function, called automatically by the Lightning framework.
        Returns initalised dataloader for model validation.

        # Returns:
        # ========
        # Object. Initialised dataloader for model validation.
        """
        return DataLoader(dataset=self.test_dataset, 
                        batch_size=self.args.batch_size_test,
                        collate_fn=self.collator,
                        num_workers=self.num_avail_workers,
                        generator=torch.Generator().manual_seed(self.args.manual_seed),
                        pin_memory=self.args.dataloader_pin_memory,
                        )

    @logged()
    def predict_dataloader(self) -> None:
        """
        Default PyTorch Lightning function, called automatically by the Lightning framework.
        Returns initalised dataloader for model validation.

        # Returns:
        # ========
        # Object. Initialised dataloader for model validation.
        """
        return DataLoader(dataset=self.predict_dataset, 
                        batch_size=self.args.batch_size_predict,
                        collate_fn=self.collator,
                        num_workers=self.num_avail_workers,
                        generator=torch.Generator().manual_seed(self.args.manual_seed),
                        pin_memory=self.args.dataloader_pin_memory,
                        )
                    #  shuffle does not work (also does not make sense) for validation, testing or prediction 

    @logged()
    def set_up_dataset(self, file_path: str, stage: str=None) -> Dataset:
        """
        Overloads the parent's method.

        Reads a text file from file_path and creates training features.
        Encodes text for classification/regression, creating iput IDs and attention mask, and formatting labels.
        When labels are provided as integers, will set dtype of labels as long int, 
        and when as floats, it will set dtype of labels as double.    

        Parameters:
        ===========
        file_path: Str. Path to the file to encode.
        stage: Str. It's optional. If set to "predict", the `idx` column will be included and the `labels` column 
        will not be included in the resulting dataset.  

        Returns:
        ========
        Dataset. Dataset for (binary or multiclass) classification/regression.
        """           
        dir, file_name = os.path.split(file_path)
        
        dataset = datasets.load_dataset(path=dir, data_files=file_name, cache_dir=self.args.cache_dir)
        num_proc = self.num_avail_workers if stage == "train" else 1  # force mapping to use only 1 process when validating, testing or predicting, otherwise you can get "RuntimeError: Cannot re-initialize CUDA in forked subprocess". 
        dataset = dataset.map(self.encode, batched=False, num_proc=num_proc) #self.num_avail_workers)
        dataset = dataset.remove_columns(["text"])
        # Make sure to choose the right data columns for the model:
        if stage == "predict":
            columns=["idx", "input_ids", "attention_mask"]
        else:
            columns=["input_ids", "labels", "attention_mask"] #["input_ids", "token_type_ids", "attention_mask"])

        # Process enriching IDs - for enriched embedding, if these are availble.
        if "enriching_ids" in dataset["train"].column_names:
            logger.info("Enriching IDs generated")
            print("\nEnriching IDs generated.")
            dataset = dataset.remove_columns(["enriching_classes"])
            columns.append("enriching_ids")

        # Add sample weights, which are used for loss calculation in confidence learning for UST method 
        # (https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf). 
        if "sample_weights" in dataset["train"].column_names and self.args.confidence_learning:
            logger.info("Using sample weights for confidence learning")
            columns.append("sample_weights")

        # Try adding columns for cross-attention between descriptors and tokens, if descriptors are available.
        columns = self.process_descriptors_column(dataset=dataset, columns=columns)

        # Set datatype formant and retain only the columns present in the `columns` list. 
        dataset.set_format(type='torch', columns=columns)

        print()
        print(dataset["train"].features)
        print(dataset["train"])
        print()

        return dataset["train"]  # by default any data after loading dataset are under "train" unless specified differently
            
    @logged()
    def set_collator(self) -> DataCollatorWithPadding:
        """ 
        Overloads the parent's method.

        Initiates and returns a collator with padding.
        
        Returns:
        ========
        collator: DataCollatorWithPadding. Initialised collator object. 
        """         
        collator = DataCollatorWithPadding(
                            tokenizer=self.tokenizer,
                            #   padding="max_length",
                            #   max_length=self.args.block_size,
                            pad_to_multiple_of=self.args.pad_to_multiple_of
                          )
        return collator
