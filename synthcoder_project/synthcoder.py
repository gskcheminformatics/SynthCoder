# This Transformer-Encoder platform is based on the PyTorch Lightning framework:
# https://lightning.ai/docs/pytorch/stable/
# It also uses some elements inspired by (and sometimes taken directly from) the SimpleTransformers library (version from Nov 2023):
# https://simpletransformers.ai/
# https://github.com/ThilinaRajapakse/simpletransformers

import os
import copy
from pprint import pprint
from typing import Union, Literal, NoReturn
import datetime

# import numpy as np
import pandas as pd

import logging

import torch
import lightning as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
# from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.strategies import DDPStrategy

# Importing my own modules here:
from synthcoder_project import model_args
from synthcoder_project import synthcoder_config
import synthcoder_project.utilities as utils
# TODO change the name of the module back to normal 
from synthcoder_project.lightning_modules.lightning_data_module import DataModuleMLM, DataModuleForClassification
from synthcoder_project.lightning_modules.lightning_model_module import LightningSynthCoderMLM, LightningSynthCoderForClassification
from synthcoder_project.lightning_modules.lightning_callbacks import HfModelCheckpoint, CustomWriter
from synthcoder_project.setup_logger import setup_logger, logged, create_logger

setup_logger()
logger = create_logger(module_name=__name__)

class SynthCoderModeling():

    """
    Class which allows to perform training and evaluation of encoder-based (BERT-based) large-language models.

    When a new object is initialised, data and model are prepared.
    The main methods:
    fit_model() - performs training and validation.
    validate_model() - used for model validation. 
    test_model() - for model testing.
    predict() - for running inference on new examples using a trained model.
    predict_montecarlo_dropout() - for running inference on new examples using a trained model, using Monte Carlo Dropout method.

    The structure and functionality implemented here is based on Pytorch Lightning and (to some extend) SimpleTransformers.
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self,
                model_encoder: type,
                model_config: type,
                model_tokenizer: type,
                model_name: str=None,
                ckpt_path: str=None,
                task_type: Literal["mlm", "classification", "regression"]="mlm",  # please, note that "classification" also works for regression (due to how the HugginFace models work) 
                user_args: dict={},
                tokenizer_train_file: str=None,
                train_file: str=None,
                validation_file: str=None,
                test_file: str=None,
                predict_file: str=None,
                accelerator: Literal["auto", "cpu", "gpu", "tpu", "ipu", "hpu", "mps"]="auto",
                devices: Union[str, int, list]="auto",
                distributed_trainer_strategy: Literal["auto", "ddp", "ddp2", "fsdp", "deepspeed", "hpu_parallel", "hpu_single", "xla", "single_xla"]="auto",

                ddp_timeout: int=18000,
                **kwargs,) -> None:
        
        """
        Initialises SynthCoderModeling object.

        Parameters:
        ===========
        model_encoder: Class of the Encoder model architecture (e.g. BertForMaskedLM)
        model_config: Class of the Encoder configuration (e.g. BertConfig)
        model_tokenizer: Class of the Encoder tokenizer (e.g. BertTokenizer)
        model_name: <Optional>. String. Default Transformer model name or path to a directory containing Transformer model file (pytorch_model.bin).
        ckpt_path: <Optional>. String. Path to the pre-saved checkpoint .ckpt file. If provided the training will start from the checkpoint.
        task_type: <Optional>. String. ["mlm", "classification", "regression"]. The type of the task for the model to perform.
        user_args: <Optional>. Dict. User defnined arguments for the model usage (e.g. {"mlm_probability": 0.10} ). If not provided, the default settings will be used.
        tokenizer_train_file: <Optional>. Str. Path to text file for the tokenizer training. 
        train_file: <Optional>. Str. Path to file to be used when training the model.
        test_file: <Optional>. Str. Path to file to be used when testing the model.
        validation_file: <Optional>. Str. Path to file to be used when validating the model.
        predict_file: <Optional>. Str. Path to file to be used for inference with the model.       
        accelerator: <Optional>. String. ["auto", "cpu", "gpu", "tpu", "ipu", "hpu", "mps"]. Pytorch Lightning accelerator for hardware.
        devices: <Optional>. String, Int or List. Range specifing which devices to use (e.g. "1", -1, [1,3]).
        distributed_trainer_strategy: <Optional>. String. ["auto", "ddp", "ddp2", "fsdp", "deepspeed", "hpu_parallel", "hpu_single", "xla", "single_xla"]. Model distribution strategy for Pytorch Lightning.
        ddp_timeout: <Optional>. Int. You can specify a custom timeout (in seconds) for the constructor process of DDP when a "ddp" distributed trainer strategy is selected.

        **kwargs

        Returns:
        ========
        None

        """
        os.environ["TOKENIZERS_PARALLELISM"] = "true" #"false"  # this allows to avoid issues with the Rust-based tokenizer parallelism, see: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        os.environ["RUST_BACKTRACE"] = "full"

        self.model_encoder = model_encoder
        self.model_config = model_config
        self.model_tokenizer = model_tokenizer
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.task_type = task_type
        self.tokenizer_train_file = tokenizer_train_file
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.predict_file = predict_file
        self.accelerator = accelerator
        self.devices = devices
        self.distributed_trainer_strategy = distributed_trainer_strategy
        self.ddp_timeout = ddp_timeout
    
        self.args = self.update_modeling_args(user_args)
        self.set_seed(self.args.manual_seed)
        self.run_checks()
        self.data_module, self.lightning_model = self.initiate_data_and_model_modules(**kwargs)

        # If custom ddp_timeout time was selected for DDP strategy, a new DDP instance needs to be created and passed to the trainer. 
        if self.distributed_trainer_strategy == "ddp" and self.ddp_timeout != 18000:
            self.distributed_trainer_strategy = DDPStrategy(timeout=datetime.timedelta(seconds=self.ddp_timeout))

        # Initialising the PyTorch Lightning trainer
        self.trainer = pl.Trainer(max_epochs=self.args.num_train_epochs,
                            max_steps=self.args.num_train_steps,
                            accelerator=self.accelerator, 
                            devices=self.devices,
                            strategy=self.distributed_trainer_strategy,
                            accumulate_grad_batches=self.args.gradient_accumulation_steps,  # gradient accumulation
                            gradient_clip_val=self.args.pl_gradient_clip_val,  # gradient clipping
                            gradient_clip_algorithm=self.args.pl_gradient_clip_algorithm,  # gradient clipping 
                            logger=self.return_loggers(),
                            callbacks=self.return_callbacks(),
                            deterministic=self.args.deterministic,
                            log_every_n_steps=self.args.log_every_n_steps,
                            precision=self.args.precision,
                            detect_anomaly=self.args.detect_anomaly,
                            overfit_batches=self.args.overfit_batches,
                            limit_val_batches=self.args.limit_val_batches,
                            profiler=self.args.profiler,

                            # strategy='ddp_find_unused_parameters_true'
                            )
        
    @logged()
    def set_seed(self, manual_seed: int) -> None:
        """
        Updates the manual seed in the model arguments, and sets it for the model.

        Parameters:
        ===========
        manual_seed: Int. New seed to set.   

        Returns:
        ========
        None
        """
        self.args.manual_seed = manual_seed
        utils.set_manual_seed(self.args.manual_seed)
        logger.info(f"Set random seed to {self.args.manual_seed}")

    @logged(level=logging.INFO)
    def find_learning_rate(self, 
                           update_lr_with_sugestion: bool=False,
                           num_training: int=100,
                           min_lr: float=1e-8,
                           max_lr: float=1,
                           image_path: str="lr_search_results.png") -> None:
        """
        Tries to find an optimal learning rate.
        Please do not use with any pre-trained models. 

        Please read:
        https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#learning-rate-finder

        Parameters:
        ===========
        update_lr_with_sugestion: Bool. Flag to indicate if the lernig-rate of the model should be changed to the one found by the lr_finder   
        num_training: Int. The number of learning rates to test.
        min_lr: Float. Minimum learning rate to investigate
        max_lr: Float. Maximum learning rate to investigate
        image_path: Str. Path to where save the image with the search results

        Returns:
        ========
        None
        """
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(model=self.lightning_model, 
                                  datamodule=self.data_module,
                                #   attr_name="self.args.learning_rate",
                                  update_attr=update_lr_with_sugestion,
                                  num_training=num_training,
                                  min_lr=min_lr,
                                  max_lr=max_lr,)
        print("LR suggestions:", lr_finder.suggestion())
        fig = lr_finder.plot(suggest=True)
        fig.savefig(image_path)

    @logged(level=logging.INFO)
    def fit_model(self,) -> None:
        """
        Trains the model using the PyTorch Lightning Trainer.fit method.
        
        # Parameters:
        # ===========
        # new_train_file: Str. <Optional> Path to the file for which to perform the training. 
        # new_eval_file: Str. <Optional> Path to the file for which to perform the validation. 

        Returns:
        ========
        None
        """

        self.trainer.fit(model=self.lightning_model, 
                         datamodule=self.data_module, 
                         ckpt_path=self.ckpt_path)

    
    @logged(level=logging.INFO)
    def validate_model(self, 
                        ckpt_path: str=None, 
                        new_batch_size_eval: int=None, 
                        new_eval_file: str=None,
                        csv_path_for_saving: str=None) -> pd.DataFrame:
        """ 
        Runs model validation (runs for one epoch).
        Can save the collected metric results as .csv file, if desired.

        Parameters:
        ===========
        ckpt_path: Str. <Optional> Path to the pre-saved model checkpoint .ckpt file. 
            If provided, the validation will be done with the model from the checkpoint. It has to be provided when using stochastic weight averaging.
        new_batch_size_eval: Int. <Optional> The batch size to use for testing. If provided, it will overwrite the previous settings.
        new_eval_file: Str. <Optional> Path to the file for which to perform the testing. 
            If provided, it will overwrite the previous file choice.
        csv_path_for_saving: Str. <Optional> Path to the .csv file. If provided, a .csv file will be created with the test metric results. 

        Returns:
        ========
        pd.DataFrame. Metric results.
        """
        return self._run_model(step="validate",
                                ckpt_path=ckpt_path,
                                new_batch_size=new_batch_size_eval,
                                new_file_path=new_eval_file,
                                csv_path_for_saving=csv_path_for_saving)
                                # When loading from checkpoint:
                                # Important note about loading from checkpoint concerning random seeds:
                                # https://lightning.ai/forums/t/resuming-from-checkpoint-gives-different-results/3826
                                # https://lightning.ai/forums/t/resuming-training-gives-different-model-result-weights/2677

    @logged(level=logging.INFO)
    def test_model(self, 
                   ckpt_path: str=None, 
                   new_batch_size_test: int=None, 
                   new_test_file: str=None,
                   csv_path_for_saving: str=None) -> pd.DataFrame:
        """ 
        Runs model testing (runs for one epoch).
        Can save the collected metric results as .csv file, if desired.

        Parameters:
        ===========
        ckpt_path: Str. <Optional> Path to the pre-saved model checkpoint .ckpt file. 
            If provided, the testing will be done with the model from the checkpoint.  It has to be provided when using stochastic weight averaging.
        new_batch_size_test: Int. <Optional> The batch size to use for testing. If provided, it will overwrite the previous settings.
        new_test_file: Str. <Optional> Path to the file for which to perform the testing. 
            If provided, it will overwrite the previous file choice.
        csv_path_for_saving: Str. <Optional> Path to the .csv file. If provided, a .csv file will be created with the test metric results. 

        Returns:
        ========
        pd.DataFrame. Metric results.
        """
        return self._run_model(step="test",
                                ckpt_path=ckpt_path,
                                new_batch_size=new_batch_size_test,
                                new_file_path=new_test_file,
                                csv_path_for_saving=csv_path_for_saving)
                                # When loading from checkpoint:
                                # Important note about loading from checkpoint concerning random seeds:
                                # https://lightning.ai/forums/t/resuming-from-checkpoint-gives-different-results/3826
                                # https://lightning.ai/forums/t/resuming-training-gives-different-model-result-weights/2677
    
    @logged(level=logging.INFO)
    def predict(self, 
                ckpt_path: str=None, 
                new_batch_size_predict: int=None, 
                new_predict_file: str=None, 
                csv_path_for_saving: str=None) -> pd.DataFrame:
        """
        Preforms predictions/inference with the model.
        Can save the output as .csv file, if desired.

        Parameters:
        ===========
        ckpt_path: Str. <Optional> Path to the pre-saved model checkpoint .ckpt file. 
            If provided, the inference will be done with the model from the checkpoint. It has to be provided when using stochastic weight averaging.
        new_batch_size_predict: Int. <Optional> The batch size to use for inference. If provided, it will overwrite the previous settings.
        new_predict_file: Str. <Optional> Path to the file for which to perform the predictions. 
            The file needs two columns, "idx" - column with unique example indices and "text" - column with the text/reactions/compounds as model input. 
            If provided, it will overwrite the previous file choice.
        csv_path_for_saving: Str. <Optional> Path to the .csv file. If provided, a .csv file will be created with the inference results. 

        Returns:
        ========
        pd.DataFrame. Dataframe containing inference results for the example indices.
        """
        # Cereate or clean the destination folder for saving the .pt files with the inference results
        utils.create_directory_and_delete_files_inside(self.args.prediction_output_dir, file_extension=".pt")

        predictions =  self._run_model(step="predict",
                                        ckpt_path=ckpt_path,
                                        new_batch_size=new_batch_size_predict,
                                        new_file_path=new_predict_file,
                                        csv_path_for_saving=csv_path_for_saving,
                                        column_names=("idx", "prediction"))
                                        # When loading from checkpoint:
                                        # Important note about loading from checkpoint concerning random seeds:
                                        # https://lightning.ai/forums/t/resuming-from-checkpoint-gives-different-results/3826
                                        # https://lightning.ai/forums/t/resuming-training-gives-different-model-result-weights/2677
        
        # Check how many devices are in use by PyTorch. If more than 1, read the inference results from the .pt files. 
        # The results are not natively combined/pooled from multiple devices by Lightning, and as a workaround they need be read and combined from files.  
        try:
            distributed_predictions = utils.gather_prediction_results(prediction_output_dir=self.args.prediction_output_dir)
            if distributed_predictions:
            # if torch.distributed.get_world_size(group=None) > 1:
            #     logger.debug(f"{torch.distributed.get_world_size(group=None)=}; Combining prediction data from .pt files")
            #     predictions = utils.return_combined_data_from_pt_files(self.args.prediction_output_dir)
                return self._convert_and_save_model_output_to_df(results=distributed_predictions, column_names=("idx", "prediction"), csv_path_for_saving=csv_path_for_saving)
        except RuntimeError as e:
            logger.error(f"{e}")
            print(e)
        return predictions

        # return self._run_model(step="predict",
        #                         ckpt_path=ckpt_path,
        #                         new_batch_size=new_batch_size_predict,
        #                         new_file_path=new_predict_file,
        #                         csv_path_for_saving=csv_path_for_saving,
        #                         column_names=("idx", "prediction"))
           
    @logged(level=logging.INFO, message="Running Monte Carlo Dropout-based predictions")
    def predict_montecarlo_dropout(self, 
                                   ckpt_path: str=None, 
                                   new_batch_size_predict: int=None, 
                                   new_predict_file: str=None, 
                                   csv_path_for_saving: str=None,
                                   montecarlo_dropout_num_iters: int=100,
                                   hidden_dropout_prob: float=None,
                                   attention_probs_dropout_prob: float=None,
                                   return_df=True,
                                   **kwargs) -> Union[pd.DataFrame, dict[str, torch.Tensor]]:
        """
        Preforms predictions/inference with the model using Monte Carlo Dropout method.
        The method was implemented for both classification and regression.
        It makes only temporary changes to the model arguments and configuration, which 
        are not saved beyond this method. 

        By default it returns a dataframe, but can also return a dictionary containing results as tensors.
        Can save the output as .csv file, if desired.

        Parameters:
        ===========
        ckpt_path: Str. <Optional> Path to the pre-saved model checkpoint .ckpt file. 
            If provided, the inference will be done with the model from the checkpoint. It has to be provided when using stochastic weight averaging.
        new_batch_size_predict: Int. <Optional> The batch size to use for inference. If provided, it will overwrite the previous settings.
        new_predict_file: Str. <Optional> Path to the file for which to perform the predictions. 
            The file needs two columns, "idx" - column with unique example indices and "text" - column with the text/reactions/compounds as model input. 
            If provided, it will overwrite the previous file choice.
        csv_path_for_saving: Str. <Optional> Path to the .csv file. If provided, a .csv file will be created with the inference results. 
        montecarlo_dropout_num_iters: Int. <Optional> Number of Monte Carlo Dropout Iterations to perform.
        hidden_dropout_prob: Float. <Optional> The probability of dropout for the hidden layers.
        attention_probs_dropout_prob: Float. <Optional> The probablity of dropout for the attention. 
        return_df: Bool. <Optional>. If True, it returns a dataframe with the resutls. If False, it returns a dictionary where the values 
            are results in a form of tensors. 

        Returns:
        ========
        pd.DataFrame or dict[str, torch.Tensor]. Dataframe containing inference results or a dictionary where the values are results in a form of tensors. 
        """

        # Prepare configuration with only the non-null values
        config = {
            key: value
            for key, value in {
                "hidden_dropout_prob": hidden_dropout_prob,
                "attention_probs_dropout_prob": attention_probs_dropout_prob
            }.items() if value is not None
        }

        # Build the args dictionary with conditional config inclusion
        temp_args_dict = {
            "montecarlo_dropout": True,
            "montecarlo_dropout_num_iters": montecarlo_dropout_num_iters,
            **({"config": config} if config else {}),
        }

        # Deep copy and the update the existing model arguments, so that we do not modify the original arguments
        args_for_monte_carlo_drop = copy.deepcopy(self.args)
        args_for_monte_carlo_drop.update_from_dict(temp_args_dict)

        # Initialise a new model with updated temporary model arguments specific for a MCD run.
        model_montecarlo_drop = LightningSynthCoderForClassification(args=args_for_monte_carlo_drop,
                                                                     model_encoder=self.model_encoder,
                                                                     model_config=self.model_config,
                                                                     data_module=self.data_module,
                                                                     **kwargs,)

        # Cereate or clean the destination folder for saving the .pt files with the inference results
        utils.create_directory_and_delete_files_inside(self.args.prediction_output_dir, file_extension=".pt")

        # Use the standard method to run the model
        model_output = self._run_model(step="predict",
                                       ckpt_path=ckpt_path,
                                       new_batch_size=new_batch_size_predict,
                                       new_file_path=new_predict_file,
                                       csv_path_for_saving=csv_path_for_saving,
                                       column_names=("idx", "prediction"),
                                       model=model_montecarlo_drop,
                                       monte_carlo_dropout=True)
                                    # When loading from checkpoint:
                                    # Important note about loading from checkpoint concerning random seeds:
                                    # https://lightning.ai/forums/t/resuming-from-checkpoint-gives-different-results/3826
                                    # https://lightning.ai/forums/t/resuming-training-gives-different-model-result-weights/2677
        
        # Check how many devices are in use by PyTorch. If more than 1, read the inference results from the .pt files. 
        # The results are not natively combined/pooled from multiple devices by Lightning, and as a workaround they need be read and combined from files.  
        distributed_predictions = utils.gather_prediction_results(prediction_output_dir=self.args.prediction_output_dir)
        model_output = distributed_predictions if distributed_predictions else model_output

        # Recombine the MCD inference results into a dictionary for easy use 
        combined_results = {}
        for device_output in model_output:
            for key, value in device_output.items():
                combined_results[key] = combined_results.get(key, [])
                combined_results[key].append(value)

        # Reshape the tensors in the dictionary a bit
        combined_results["example_idxs"] = torch.cat(combined_results["example_idxs"], dim=-1)
        combined_results["y_T"] = torch.cat(combined_results["y_T"], dim=1)
        combined_results["y_majority_pred"] = torch.cat(combined_results["y_majority_pred"], dim=-1)
        combined_results["y_mean"] = torch.vstack(combined_results["y_mean"])
        combined_results["y_var"] = torch.vstack(combined_results["y_var"])

        # Are we returning a dataframe?
        if return_df: 
            logger.debug(f"Returning and saving a dataframe")
            # Look for the max [softmax] value and its index (so class) in the y_mean tensor for each sample. 
            max_mean_values, max_mean_idxs = combined_results["y_mean"].max(dim=-1, keepdim=True)
            # Take the variance (for the correct class) of the prediction based on the index (of the class in classification) found above 
            var_values = torch.gather(combined_results["y_var"], -1, max_mean_idxs)

            df_results = pd.DataFrame({"idx": combined_results["example_idxs"],
                                       "prediction_average": torch.flatten(max_mean_idxs),  # prediction based on the mean softmax value
                                       "prediction_majority": combined_results["y_majority_pred"],  # prediction based on the majority voting
                                       "mean_or_softmax_mean": torch.flatten(max_mean_values),
                                       "mean_var": torch.flatten(var_values)})
            if csv_path_for_saving:
                utils.save_df_as_csv(df_results, csv_path_for_saving)
            return df_results

        return combined_results

    @logged()
    def _run_model(self, 
                   step: Literal["validate", "test", "predict"], 
                   ckpt_path: str=None, 
                   new_batch_size: int=None, 
                   new_file_path: str=None, 
                   csv_path_for_saving: str=None,
                   column_names: tuple[str, str]=("metric", "score"),
                   model: object=None,
                   monte_carlo_dropout: bool=False,
                   ) -> Union[pd.DataFrame, NoReturn]:
        """
        Runs model validation/testing/inference.
        Can save the output as .csv file, if desired.

        Parameters:
        ===========
        step: Str. ["validate", "test", "predict"]. The name of the step to run.
        ckpt_path: Str. <Optional> Path to the pre-saved model checkpoint .ckpt file. 
            If provided, the inference will be done with the model from the checkpoint. It has to be provided when using stochastic weight averaging.
        new_batch_size: Int. <Optional> The batch size to use for inference. If provided, it will overwrite the previous settings.
        new_file_path: Str. <Optional> Path to the file for which to perform the predictions. 
            The file needs two columns, "idx" - column with unique example indices and "text" - column with the text/reactions/compounds as model input. 
            If provided, it will overwrite the previous file choice.
        csv_path_for_saving: Str. <Optional> Path to the .csv file. If provided, a .csv file will be created with the inference results. 
        column_names: tuple[str, str]. <Optional> Names of the columns to generate when creating the results as dataframe.
        model: Object. <Optional> Initialised lightning model object. 
        monte_carlo_dropout: Bool. <Optional> Should be set to `True` if Monte Carlo Droput inference is performed.

        Returns:
        ========
        pd.DataFrame. Dataframe containing model output.
        """
        step_specific_functions = {"validate": self.trainer.validate,
                                   "test": self.trainer.test,
                                   "predict": self.trainer.predict,
                                   }
        if step not in step_specific_functions:
            raise ValueError(f"Wrong step name provided. The possible options are {step_specific_functions.keys()}")

        print(f"\n=====================\nRunning {step}:")
        if new_batch_size:
            self.data_module.set_batch_size(batch_size=new_batch_size, step=step)
        if new_file_path:
            self.data_module.set_file(file_path=new_file_path, step=step)
        
        if not model:  # if model is not explicitly provided as an argument, use the default self. model. 
            model = self.lightning_model
        
        if ckpt_path:
            results = step_specific_functions[step](model=model, 
                                                    datamodule=self.data_module,
                                                    ckpt_path=ckpt_path)
        else:
            if self.args.stochastic_wght_avging_strategy:
                raise Exception("As you are using the stochastic weight averging, the `ckpt_path` has " 
                                "to be provided explicitly, due to the issues with the deepcopy protocol.")
            
            print("\nLoading weights for the best model:")
            results = step_specific_functions[step](model=model,
                                                    datamodule=self.data_module, # automatically auto-loads the weights for the best model from the previous run
                                                    ckpt_path="best")
        
        if monte_carlo_dropout:
            logger.debug(f"Returning results for Monte Carlo Dropout-based run")
            return results

        # Convert model output to df and, if selected, save as .csv
        df_results = self._convert_and_save_model_output_to_df(results=results, column_names=column_names, csv_path_for_saving=csv_path_for_saving)

        return df_results

    @logged()
    def _convert_and_save_model_output_to_df(self,
                                             results: list,
                                             column_names: tuple[str, str]=("metric", "score"),
                                             csv_path_for_saving: str=None,
                                             ) -> pd.DataFrame:
        """
        Converts model output to df and, if selected, saves it as .csv

        results: List. Results generated by the model.
        column_names: tuple[str, str]. <Optional> Names of the columns to generate when creating the results as dataframe.
        model: Object. <Optional> Initialised lightning model object. 
        """
        df_results = self._convert_model_output_to_df(results, column_names=column_names)
        if csv_path_for_saving:
            utils.save_df_as_csv(df_results, csv_path_for_saving)
        return df_results

    @logged()
    def _convert_model_output_to_df(self, 
                                   model_output: list[Union[list[tuple[torch.Tensor, torch.Tensor]], dict]], 
                                   column_names: tuple[str, str]) -> pd.DataFrame:
        """
        Converts model outputs (testing and inference model outputs) to Padas Dataframe.

        Parameters:
        ===========
        model_output: list[Union[list[tuple[torch.Tensor, torch.Tensor]], dict]]. Model output to convert to df.
        column_names: tuple[str, str]. Two column names to save the data under.

        Returns:
        ========
        pd.DataFrame. Input data converted to df.
        """
        results = {column_names[0]: [], column_names[1]: []}
        for device_output in model_output:
            if isinstance(device_output, dict):
                for metric, score in device_output.items():
                    results[column_names[0]].append(metric)
                    results[column_names[1]].append(score)
            elif isinstance(device_output, list):
                for output in device_output:
                    results[column_names[0]].append(output[0].item())
                    results[column_names[1]].append(output[1].item())
        return pd.DataFrame.from_dict(results)

    @logged()
    def run_checks(self) -> Union[None, NoReturn]:
        """
        Runs some checks on selected options chosen by the user and raises error if needed:
            - checks if task_type is allowed
            - checks if problem_type is not multi_label_classification

        Returns:
        ========
        None
        """
        # Make sure that the selected task type is implemented/allowed
        if self.task_type not in synthcoder_config.ALLOWED_TASK_TYPES:
            raise ValueError(f"The selected modeling taks_type is INVALID." 
                             f"Allowed tasks are: {synthcoder_config.ALLOWED_TASK_TYPES}")
        
        try:
            if (self.task_type == "regression" and 
                self.args.num_labels != 1) or (self.task_type == "classification" and self.args.num_labels < 2):
                raise ValueError(f"Your selected `task_type`: {self.task_type} and `num_labels`: {self.args.num_labels}," 
                                f"do not match. The allowed number of labels for `regression` is 1, and for `classification` is >=2.")
        except AttributeError:
            pass

        try:
            if (self.args.problem_type == "regression" and 
                self.args.num_labels != 1) or (self.args.problem_type == "single_label_classification" and self.args.num_labels < 2):
                raise ValueError(f"Your selected `problem_type`: {self.args.problem_type} and `num_labels`: {self.args.num_labels}," 
                                f"do not match. The allowed number of labels for `regression` is 1, and for `single_label_classification` is >=2.")
        except AttributeError:
            pass 

        try:
        # Make sure not to allow multi-label classification which is not implemented or tested for. 
            if self.args.problem_type == "multi_label_classification":
                raise Exception(f"multi_label_classification is not yet implemented."
                                "Select 'regression' or 'single_label_classification' instead")
        except AttributeError:
            pass

        if self.args.cross_attention_use_extended_descript_network and not self.args.cross_attention_number_of_descriptors:
            raise ValueError(f"You are trying to use the extended descriptor network: {self.args.cross_attention_use_extended_descript_network=}, " 
                             f"but the number of descriptors is set to {self.args.cross_attention_number_of_descriptors=}. "
                             f"You need to specify an int as the number of descriptors in the `cross_attention_number_of_descriptors` argument.")

    @logged()
    def initiate_data_and_model_modules(self, **kwargs) -> tuple[object, object]:
        """
        Selects and initiates the appropriate data and model processing classes, based on the task type.

        Parameters:
        ===========
        **kwargs

        Returns:
        ========
        data_module, lightning_model: Tuple of Objects. 
        """

        # Select the appropriate classes 
        if self.task_type == "mlm":
            lightning_model = LightningSynthCoderMLM
            data_module = DataModuleMLM(args=self.args, 
                                    model_tokenizer=self.model_tokenizer,
                                    tokenizer_train_file=self.tokenizer_train_file,
                                    train_file=self.train_file,
                                    validation_file=self.validation_file,
                                    **kwargs,)
            
        elif self.task_type in ("classification", "regression"):
            lightning_model = LightningSynthCoderForClassification
            data_module = DataModuleForClassification(args=self.args,
                                    model_tokenizer=self.model_tokenizer,
                                    tokenizer_train_file=self.tokenizer_train_file,
                                    train_file=self.train_file,
                                    validation_file=self.validation_file,
                                    test_file=self.test_file,
                                    predict_file=self.predict_file,
                                    **kwargs,)
        
        lightning_model = lightning_model(args=self.args,
                                            model_encoder=self.model_encoder,
                                            model_config=self.model_config,
                                            data_module=data_module,
                                            **kwargs,)
        
        return data_module, lightning_model

    @logged()
    def update_modeling_args(self, user_args: object) -> object:
        """ 
        Updates modeling args and combines them with predefined arguments for the selected model type

        Parameters:
        ===========
        user_args: Object. Arguments (including all setting values) for the model, training, tokenization etc. 
        
        Returns:
        ========
        args: Object. Updated args. 
        """
        # Load predefined arguments
        if self.task_type == "mlm":
            model_arg_object = model_args.LanguageModelingArgs
        elif self.task_type in ("classification", "regression"):
            model_arg_object = model_args.ClassificationArgs
        logger.debug(f"Task type: {self.task_type=}")
        args = utils.load_model_args(self.model_name, model_arg_object)
        
        # Merging the predefined args with the user args
        if isinstance(user_args, dict):
            args.update_from_dict(user_args)
        # If user args names are the same as the predefined model object args, use the user args  
        elif isinstance(user_args, model_arg_object):
            args = user_args

        # Overwrite the model_name if it is provided directly to the SynthCoderModeling object
        args.model_name = utils.overwrite_with_not_null(args.model_name, self.model_name)

        # Indicate in the args if mlm is run. 
        args.mlm = True if self.task_type == "mlm" else False

        # Save names of the encoder, confing and tokenizer classes as new arguments in args
        args.class_name_model_encoder = self.model_encoder.__name__
        args.class_name_model_config = self.model_config.__name__
        args.class_name_model_tokenizer = self.model_tokenizer.__name__
        
        logger.debug(f"\n{args.class_name_model_encoder=}\n{args.class_name_model_config=}\n{args.class_name_model_tokenizer=}")

        return args

    @logged()
    def return_callbacks(self) -> list[object]:
        """
        Initalises callbacks for the PyTorch Lightning trainer.

        Returns:
        ========
        List of objects. A list of initialised callback objects.
        """
        callbacks_to_return = []

        # Callback for monitoring/logging metrics from the optimizer 
        if self.args.learning_rate_logging_interval:
            logger.debug("Learning rate monitor callback")
            lr_monitor = LearningRateMonitor(logging_interval=self.args.learning_rate_logging_interval, 
                                                log_momentum=True, 
                                                log_weight_decay=True)
            callbacks_to_return.append(lr_monitor)
        
        # Callback for early stopping based on the selected metric
        if self.args.early_stopping_metric:
            logger.debug("Early stopping metric callback")
            early_stopping = EarlyStopping(monitor=self.args.early_stopping_metric,
                                        mode=self.args.early_stopping_mode,
                                        patience=self.args.early_stopping_patience,  # It will wait x validation epochs (patience) for no improvement of the monitored metric before stopping the training 
                                        min_delta=self.args.early_stopping_delta,
                                        stopping_threshold=self.args.early_stopping_threshold,
                                        divergence_threshold=self.args.early_stopping_divergence_threshold) 
            callbacks_to_return.append(early_stopping)

        # Callback for stochastic averaging
        if self.args.stochastic_wght_avging_strategy:
            logger.debug("Stochastic weight avergaing callback")
            stoch_weight_avging = StochasticWeightAveraging(swa_lrs=self.args.stochastic_wght_avging_lrs,
                                                            annealing_strategy=self.args.stochastic_wght_avging_strategy,
                                                            swa_epoch_start=self.args.stochastic_wght_avging_epoch_start,
                                                            annealing_epochs=self.args.stochastic_wght_avging_anneal_epochs,
                                                            device=None)  # device=None will infer the device from the pl module 
            callbacks_to_return.append(stoch_weight_avging)


        # Callback for writing .pt files for predictions during inference
        if self.task_type in ("classification", "regression") and self.args.create_prediction_files:
            logger.debug("Custom writer callback")
            pred_writer = CustomWriter(output_dir=self.args.prediction_output_dir, 
                                       write_interval="epoch")
            callbacks_to_return.append(pred_writer)


        # Callback for saving model checkpoints
        # Checkpoints are saved together with the logs
        hf_model_checkpoint = HfModelCheckpoint(monitor=self.args.early_stopping_metric,  # using the same metric to evaluate which models are the best as for early stopping 
                                                mode=self.args.early_stopping_mode,
                                                save_top_k=self.args.save_top_k_checkpoints,
                                                every_n_epochs=1,
                                                enable_version_counter=True,
                                                save_last=self.args.save_last_checkpoint,)
        callbacks_to_return.append(hf_model_checkpoint)

        return callbacks_to_return

    @logged()
    def return_loggers(self) -> Union[list[object], list]:
        """
        Initalises loggers for the PyTorch Lightning trainer.

        Returns:
        ========
        List of objects or empty list. A list of initialised logger objects or an empty list if `logger_dir` not provided
        """
        if not self.args.logger_dir:
            return []

        # TesorBoard and CSV loggers cannot save to the same directory, as that causes issues with the folder version naming!!! Thus, argument "name" specifing the logger is passed. 
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.args.logger_dir, name="TensorBoard_logs")#, name=..., version="05122023") # this is also where model checkpoints are saved
        csv_logger = pl_loggers.CSVLogger(save_dir=self.args.logger_dir, name="CSV_logs")  #, name=..., version="05122023"

        return [tb_logger, csv_logger]

    @logged()
    def return_tokenizer(self) -> object:
        """
        Returns tokenizer used by the data module.

        Returns:
        ========
        Object
        """
        return self.data_module.return_tokenizer()

    @logged()
    def return_logging_dir(self) -> str:
        """
        Returns the current logging directory of the model.

        Returns:
        ========
        Str. Current logging directory for the model.  
        """
        return self.trainer.log_dir
    
    @logged()
    def print_args(self) -> None:
        pprint(vars(self.args))

    @logged()
    def print_model_config(self) -> None:
        pprint(self.lightning_model.return_model_config())


