# UST for SynthCoder

from typing import Union, Literal, NoReturn
import os 
import logging
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from synthcoder_project.synthcoder import SynthCoderModeling
import synthcoder_project.utilities as utils
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)


class SynthCoderModelingUST():

    """
    Class which allows to perform training encoder-based (BERT-based) large-language models using UST technique.

    When a new object is initialised, data and model are prepared.
    This class can be only used for model training via semi-superised approach. 
    
    The main methods:
    # fit_model() - performs training and validation.
    # validate_model() - used for model validation. 
    # test_model() - for model testing.
    # predict() - for running inference on new examples using a trained model.

    The structure and functionality implemented here is based on Pytorch Lightning and (to some extend) SimpleTransformers.

    See: https://statics.teams.cdn.office.net/evergreen-assets/safelinks/1/atp-safelinks.html
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
                unlabeled_file: str=None,
                accelerator: Literal["auto", "cpu", "gpu", "tpu", "ipu", "hpu", "mps"]="auto",
                devices: Union[str, int, list]="auto",
                distributed_trainer_strategy: Literal["auto", "ddp", "ddp2", "fsdp", "deepspeed", "hpu_parallel", "hpu_single", "xla", "single_xla"]="auto",
                ddp_timeout: int=18000,
                temp_dir: str="./temp/",
                **kwargs,) -> None:
        
        """
        Initialises SynthCoderModelingUST object.

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
        unlabeled_file: <Optional>. Str. Path to file with unlabelled examples used for smi-supervised training.
        accelerator: <Optional>. String. ["auto", "cpu", "gpu", "tpu", "ipu", "hpu", "mps"]. Pytorch Lightning accelerator for hardware.
        devices: <Optional>. String, Int or List. Range specifing which devices to use (e.g. "1", -1, [1,3]).
        distributed_trainer_strategy: <Optional>. String. ["auto", "ddp", "ddp2", "fsdp", "deepspeed", "hpu_parallel", "hpu_single", "xla", "single_xla"]. Model distribution strategy for Pytorch Lightning.
        ddp_timeout: <Optional>. Int. You can specify a custom timeout (in seconds) for the constructor process of DDP when a "ddp" distributed trainer strategy is selected.
        temp_dir: <Optional>. String. Directory of the folder to store the temporary files.  

        **kwargs

        Returns:
        ========
        None
        """

        self.model_encoder=model_encoder
        self.model_config=model_config
        self.model_tokenizer=model_tokenizer
        self.model_name=model_name
        self.ckpt_path=ckpt_path
        self.task_type=task_type
        self.user_args=user_args
        self.tokenizer_train_file=tokenizer_train_file
        self.train_file=train_file
        self.validation_file=validation_file
        self.test_file=test_file
        self.predict_file=predict_file
        self.unlabeled_file = unlabeled_file
        self.accelerator=accelerator
        self.devices=devices
        self.distributed_trainer_strategy=distributed_trainer_strategy
        self.ddp_timeout=ddp_timeout
        self.temp_dir=temp_dir
        self.kwargs = kwargs

        self.bald_funcs = {"easy": self.sample_by_bald_easiness, 
                           "easy_mod": self.sample_by_bald_easiness_mod,
                           "difficult": self.sample_by_bald_difficulty, 
                           "easy_class": self.sample_by_bald_class_easiness, 
                           "easy_class_mod": self.sample_by_bald_class_easiness_mod, 
                           "difficult_class": self.sample_by_bald_class_difficulty,
                           "uniform": self.sample_uniformly}


        # We just want to save only the best model weights, so we update the relevant user arguments for the model initalisation
        self.user_args.update({"save_top_k_checkpoints": 1,
                               "save_last_checkpoint": False})

        self.initialise_model()

        if not self.model_name and not self.ckpt_path:
            logger.warning("Neither `model_name` nor `ckpt_path` were provided. Ensure this is intentional.")
            input("You did not provide `model_name` or `ckpt_path`. Are you sure that this is correct?\nPress `ENTER` to continue.")
                
    @logged()
    def initialise_model(self) -> None:
        """
        Initialises SynthCoderModeling.
        
        Returns:
        ========
        None
        """
        logger.debug(None)

        self.model = SynthCoderModeling(model_encoder=self.model_encoder,
                                        model_config=self.model_config,
                                        model_tokenizer=self.model_tokenizer,
                                        model_name=self.model_name,
                                        ckpt_path=self.ckpt_path,
                                        task_type=self.task_type, 
                                        user_args=self.user_args,
                                        tokenizer_train_file=self.tokenizer_train_file,
                                        train_file=self.train_file,
                                        validation_file=self.validation_file,
                                        test_file=self.test_file,
                                        predict_file=self.predict_file,
                                        accelerator=self.accelerator,
                                        devices=self.devices,
                                        distributed_trainer_strategy=self.distributed_trainer_strategy,
                                        ddp_timeout=self.ddp_timeout,
                                        **self.kwargs)

    @logged()
    def fit_model(self) -> None:
        """
        Trains the model using the PyTorch Lightning Trainer.fit method.
        
        Returns:
        ========
        None
        """
        self.model.fit_model()

    @logged()
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
        logger.debug(None)

        return self.model.validate_model(ckpt_path=ckpt_path,
                                         new_batch_size_eval=new_batch_size_eval,
                                         new_eval_file=new_eval_file,
                                         csv_path_for_saving=csv_path_for_saving)
    
    @logged()
    def fit_ust(self,
                new_unlabeled_file: str=None,
                number_of_iterations: int=5,
                montecarlo_dropout_num_iters: int=100,
                hidden_dropout_prob: float=0.3, 
                attention_probs_dropout_prob: float=0.3,
                ust_method: Literal["easy", "easy_mod", "difficult", "easy_class", "easy_class_mod", "difficult_class", "uniform"]="easy_class",
                num_samples: int=2048, # number of pseudo-labeled examples to sample
                confidence_learning: bool=True,
                alpha: float=0.1,
                sample_weight_ground_truth: float=None,
                new_temp_dir: str=None,
                keep_temp_data: bool=True,
                validate_after_each_iter: bool=False,
                csv_path_for_saving_metrics: str=None,
                ) -> Union[None, pd.DataFrame]:
        
        """
        Preforms training of the model using the UST method, first only with labeled examples and then with a combination of labeled 
        and pseudo-labeled examples. The method uses MCD for confidence estimation about the generated pseudo-labels. 
        
        If `confidence_learning` is enabled, it converts the variance for predictions obtained through MCD, into sample weights, 
        which are then used for biasing the loss function. 
        The function offers different sampling methods. `"uniform"` method selects uniformly in a random manner, whereas, the remaining 
        methods are based on the BALD aquisition function, and bias selection towards easy (`"easy"` and `"easy_mod"`) or difficult 
        (`"difficult"`) samples. The BALD-based methods can be used with pseudo-label class cosideration to keep the label balance in 
        the selection ("easy_class", "easy_class_mod" and "difficult_class").  
        
        
        Parameters:
        ===========
        new_unlabeled_file: <Optional> String. Path to the file with unlabeled examples for which to perform the UST training. 
            The file needs two columns, "idx" - column with unique example indices and "text" - column with the text/reactions/compounds as model input. 
            If provided, it will overwrite the previous file choice. 
        number_of_iterations: <Optional> Int. Number of iterations of the model retraining with the pseudo-labeled examples.  
        montecarlo_dropout_num_iters: <Optional> Int. Number of the Monte Carlo Dropout iterations within each semi-supervised iteraction cycle.
        hidden_dropout_prob: <Optional> Float. The probability of dropout for the hidden layers. Only changes the model config for inference with MCD, otherwise has no other effect. 
        attention_probs_dropout_prob: <Optional> Float. The probablity of dropout for the attention. Only changes the model config for inference with MCD, otherwise has no other effect.
        ust_method: <Optional>. String. ["easy", "easy_mod", "difficult", "easy_class", "easy_class_mod", "difficult_class", "uniform"]. The sampling method for selecting the pseudo-labeled examples.
        num_samples: <Optional>. Int. The of pseudo-labeled examples to sample per iteration.
        confidence_learning: <Optional>. Bool. If `Ture`, it will convert the variance for pseudo-labels predictions to sample weights used during loss claculations.
        alpha: <Optional>. Float. Scaling factor used in conversion of variance of pseudo-labels to sample weights. 
        sample_weight_ground_truth: <Optional>. Float. If provided, it will use this value as a weight for samples with real labels. Otherwise, it will calculate its own weiths for these samples.  
        new_temp_dir: <Optional> String. New path to the folder for storing temporary files. If provided, it will overwrite the previously provided directory. 
        keep_temp_data: <Optional> Bool. If `True`, it keeps all the generated temporary files, otherwise it deletes them. 
        validate_after_each_iter: <Optional> Bool. If `True` it will run a validation on the validation data set after each re-training iteration to collect validation metrics.
        csv_path_for_saving_metrics: <Optional> String. Path to the .csv file. If provided, a .csv file will be created with the validation metric results for the tested iterations. 


        Returns:
        ========
        None or pd.Dataframe. If `validate_after_each_iter` is set to `True` or `csv_path_for_saving_metrics` is provided, the function will return a dataframe with validation 
            metrics for all UST iterations 
        """
        logger.debug(f"Running fit_ust with {number_of_iterations} iterations.")
    
        self.metric_dfs = []
        
        # Should we overwrite the unlabeled file and the temporary directory with newly provided values?
        if new_unlabeled_file:
            self.unlabeled_file = new_unlabeled_file
        if new_temp_dir:
            self.temp_dir = new_temp_dir

        # Read the labeled and unlabelled data into dataframes
        df_unlabeled_data = self.file_to_df(self.unlabeled_file)
        df_labeled_data = self.file_to_df(self.train_file)

        # Create the temporary folder if it does not yet exit, if it exits remove any files from it
        utils.create_directory_and_delete_files_inside(directory=self.temp_dir, file_extension=(".csv", ".tsv", ".xlsx"))
        # os.makedirs(self.temp_dir, exist_ok=True)
        
        # Train the model only with the labeled (ground truth) data. Done once in the whole process. 
        self.fit_model()
        # Are we running a validation to collect statistics for the current iteration? 
        if validate_after_each_iter or csv_path_for_saving_metrics:
            df_metrics = self._save_validation_as_df(iteration=0, csv_path_for_saving_metrics=csv_path_for_saving_metrics)

        # Run iterations of model retraining with labeled & pseudo-labeled examples
        for iteration in range(1, number_of_iterations+1):

            print("\n", f" UST Iteration {iteration} ".center(100, "="), "\n")
            logger.debug(f"UST Iteration {iteration}")

            # Generate predictions (pseudo-labels) and other statistics with 
            # the Monte Carlo Dropout inference for the unlabeled examples 
            mcd_results = self.model.predict_montecarlo_dropout(new_predict_file=self.unlabeled_file,
                                                                montecarlo_dropout_num_iters=montecarlo_dropout_num_iters,
                                                                hidden_dropout_prob=hidden_dropout_prob,
                                                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                                return_df=False)
            
            example_idxs, y_var, y_T, y_pred = self._process_mcd_results(mcd_results)

            # If the number of the unlabeled samples to use is not provided or larger than the number of examples, 
            # use the number of samples equal to the number of all unlabeled examples 
            # if not num_samples or (num_samples > len(example_idxs)):
            #     num_samples = len(example_idxs)
            num_samples = min(num_samples, len(example_idxs))

            # Select the sampling function to use, and sample the pseudo-labeled examples, generating example indicies to use  
            bald_func = self.bald_funcs[ust_method]
            indices = bald_func(num_samples=num_samples, y_T=y_T, y=y_pred, num_classes=self.model.args.num_labels)
            
            # Prepare sampled pseudo-labeled dataframe
            df_sampled = self._prepare_sampled_dataframe(df_unlabeled_data, example_idxs, indices, y_pred, y_var, confidence_learning, alpha)
            if sample_weight_ground_truth:
                df_labeled_data["sample_weights"] = sample_weight_ground_truth  # add sample weights to the ground truth labels 
            else:
                df_labeled_data["sample_weights"] = -np.log(1e-10)*alpha  # add sample weights to the ground truth labels 

            # Merging the data with the pseudo labels with the original training data with real labels (this is not done in the original UST code, but probably should be). 
            df_retraining = pd.concat([df_sampled, df_labeled_data]).sample(frac=1)

            # If this is a classification, make sure to update the class weights, based on the new label distribution. 
            if self.model.args.problem_type != "regression":
                logger.debug("Updating class weights")
                class_weights = utils.calc_class_weights_from_df(df_retraining)
                self.user_args.update({"class_weights": class_weights})

            # Define and generate a new training file, containing the new pseudo-labeled examples
            self.train_file = os.path.join(self.temp_dir, f"pseudo_samples_iter-{iteration}.csv")
            df_retraining.to_csv(self.train_file, index=False, sep=("\t" if not self.unlabeled_file.endswith(".csv") else ","))

            # Find the checkpoint for the latest model, and provide the trained model files as a new model to use
            model_ckpt_dir = os.path.join(self.model.return_logging_dir(), "checkpoints")
            ckpt_name = [file for file in os.listdir(model_ckpt_dir) if file.endswith(".ckpt.dir")][0]     
            self.model_name = os.path.join(model_ckpt_dir, ckpt_name)

            # Are we using the confidence_learning approach?
            self.user_args["confidence_learning"] = confidence_learning                

            # Initialise the model and fit it with the new new labeled + pseudo-labeled data 
            self.initialise_model()
            self.fit_model()

            # Are we running a validation to collect statistics for the current iteration? 
            if validate_after_each_iter or csv_path_for_saving_metrics:
                df_metrics = self._save_validation_as_df(iteration=iteration, csv_path_for_saving_metrics=csv_path_for_saving_metrics)

        # Clean up temporary files if not keeping them
        if not keep_temp_data:
            logger.debug(f"Removing data from {self.temp_dir}")
            utils.remove_tree_directory(self.temp_dir)

        # If any dataframes with validation metrics were collected, return the combined dataframe
        if self.metric_dfs:
            return df_metrics

    @logged()
    def _process_mcd_results(self, mcd_results: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes Monte Carlo Dropout results to extract necessary components for training and evaluation.       

        Parameters:
        ===========
        mcd_results: dict. A dictionary containing results from Monte Carlo Dropout, including:
            - "example_idxs": torch.Tensor containing sample indices.
            - "y_var": torch.Tensor containing variances of the predictions.
            - "y_T": torch.Tensor containing softmax values for predictions from all MCD models.
            - "y_majority_pred": torch.Tensor containing pseudo-labels derived from voting by MCD models (for classification tasks).
            - "y_mean": torch.Tensor containing mean prediction results (for regression tasks).       

        Returns:
        ========
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]. A tuple containing:
            - example_idxs: Indices of the examples.
            - y_var: Variance associated with the predictions.
            - y_T: Softmax values for all predictions.
            - y_pred: Pseudo-labels (for classification) or mean values (for regression).
        """
        example_idxs = mcd_results["example_idxs"]  # IDs/Idxs for the samples provided by the user in the file
        y_var = mcd_results["y_var"][:, 0]  # variance for the predictions
        y_T = mcd_results["y_T"]  # softmax values for all predictions generated by all MCD models 
        if self.model.args.problem_type != "regression":
            y_pred = mcd_results["y_majority_pred"]  # pseudo-labels; result of the voting by different MCD models 
        else:
            y_pred = mcd_results["y_mean"]  # pseudo-values; mean results of regression over different MCD models
            y_pred = torch.squeeze(mcd_results["y_mean"]) # reshape the tensor to remove the last dimension of shape 1 
        
        return example_idxs, y_var, y_T, y_pred

    @logged()
    def _prepare_sampled_dataframe(self, 
                                   df_unlabeled_data: pd.DataFrame,
                                   example_idxs: torch.Tensor,
                                   indices: torch.Tensor,
                                   y_pred: torch.Tensor,
                                   y_var: torch.Tensor,
                                   confidence_learning: bool, 
                                   alpha: float) -> pd.DataFrame:
        """
        Prepares a DataFrame from sampled pseudo-labeled examples with optional confidence learning adjustments.

        Parameters:
        ===========
        df_unlabeled_data: pd.DataFrame. DataFrame containing unlabeled data with an 'idx' column.
        example_idxs: torch.Tensor. Indices of the examples, used to map sampled examples.
        indices: torch.Tensor. Indices of the sampled examples for pseudo-labeling.
        y_pred: torch.Tensor. Pseudo-labels or predicted values for the sampled examples.
        y_var: torch.Tensor. Variance of predictions used for confidence learning if applicable.
        confidence_learning: bool. Flag to determine if confidence learning is performed, modifying sample weights.
        alpha: float. Scaling factor used in converting variances to confidence sample weights.    

        Returns:
        ========
        pd.DataFrame. DataFrame containing sampled pseudo-labeled examples, potentially modified by confidence learning.
        """
        df_sampled = df_unlabeled_data[df_unlabeled_data["idx"].isin(example_idxs[indices].tolist())].copy(deep=True)  # copy only the rows of the data based on the sampled indices
        df_sampled["inference_index"] = df_sampled["idx"].apply(lambda idx: example_idxs.tolist().index(idx))  # add information about the indices used by the sampling method 
        df_sampled["labels"] = df_sampled["inference_index"].apply(lambda idx: y_pred.tolist()[idx]) 
        if confidence_learning:
            logger.debug("Using confidence learning")
            x_confidence = -torch.log(y_var+1e-10)*alpha  # converting prediction variance to confidence/sample weights 
            df_sampled["sample_weights"] = df_sampled["inference_index"].apply(lambda idx: x_confidence.tolist()[idx])

        # Drop columns unnecessary for model training 
        return df_sampled.drop(columns=["idx", "inference_index"])

    @logged()
    def _save_validation_as_df(self, iteration: int, csv_path_for_saving_metrics: str=None) -> pd.DataFrame:
        """
        Runs model validation and returns the validation results as Pandas dataframe. 
        Optionally can save the combined results from all iterations as .csv file.  

        Parameters:
        ===========
        iteration: Int. The number of the current iteration. The data will be labeled with the iteration number.
        csv_path_for_saving_metrics. <Optional> String. Path to the .csv file. If provided, a .csv file will be created with the validation metric results for the tested iterations.

        Returns:
        ========
        df_metrics: pd.DataFrame. Dataframe with the validation results for the current iteration.
        """
        df_validation_metrics = self.validate_model().set_index('metric').rename_axis(None).T.reset_index(drop=True)
        df_validation_metrics["iteration"] = iteration
        df_validation_metrics["model_logging_dir"] = self.model.return_logging_dir()
        self.metric_dfs.append(df_validation_metrics) 
        # Concatane the datarames 
        df_metrics = pd.concat(self.metric_dfs)
        if csv_path_for_saving_metrics:
            # Save the comabined dataframe to a file
            df_metrics.to_csv(csv_path_for_saving_metrics, index=False)
        
        return df_metrics

    @classmethod
    @logged(name=__name__)
    def file_to_df(cls, file_path: str) -> pd.DataFrame:
        """
        Converts .csv, .tsv and .xlsx files into a Pandas data frame. 

        Parameters:
        ===========
        file_path: Str. Path to the file. 

        Returns:
        ========
        df: pd.DataFrame.
        """
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".tsv"):
            df = pd.read_csv(file_path, sep="\t")
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise AttributeError("The file with the unlabelled examples can be only provided as .csv, .tsv or .xlsx")
        return df

    @classmethod
    @logged(name=__name__)
    def get_BALD_acquisition(cls, y_T: torch.Tensor) -> torch.Tensor:
        """
        Returns BALD aquisition results.

        Equation 6 from: https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf
        Translated to PyTorch from: https://github.com/microsoft/UST/blob/main/sampler.py

        Parameters:
        ===========
        y_T: torch.Tensor. Softmax outputs of a model(s), where a prediction for each sample was done >1. 

        Returns:
        ========
        torch.Tensor.
        """
        expected_entropy = - torch.mean(torch.sum(y_T * torch.log(y_T + 1e-10), dim=-1), dim=0)
        expected_p = torch.mean(y_T, dim=0)
        entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + 1e-10), dim=-1)
        return (entropy_expected_p - expected_entropy)

    
    @classmethod
    @logged(name=__name__)
    def sample_by_bald_difficulty(cls, num_samples: int, y_T: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sample examples with a focus on difficult examples using BALD approach. No consideration for the class/label balance.

        Based on equation 7 from: https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf
        Translated to PyTorch from: https://github.com/microsoft/UST/blob/main/sampler.py

        Parameters:
        ===========
        num_samples: Int. The number of items to sample.
        y_T: torch.Tensor. Softmax outputs of a model(s), where a prediction for each sample was done >1. 
        **kwargs

        Returns:
        ========
        torch.Tensor. Array indicies of the selected samples.
        """
        BALD_acq = cls.get_BALD_acquisition(y_T)
        p_norm = torch.clamp(BALD_acq, min=0)  # Equivalent to np.maximum
        p_norm = p_norm / torch.sum(p_norm)

        indices = torch.multinomial(p_norm, num_samples, replacement=False)  # Equivalent to np.random.choice
        return indices

    @classmethod
    @logged(name=__name__)
    def sample_by_bald_easiness(cls, num_samples: int, y_T: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sample examples with a focus on easy examples using BALD approach. No consideration for the class/label balance.

        Based on equation 8 from: https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf
        Translated to PyTorch from: https://github.com/microsoft/UST/blob/main/sampler.py

        Parameters:
        ===========
        num_samples: Int. The number of items to sample.
        y_T: torch.Tensor. Softmax outputs of a model(s), where a prediction for each sample was done >1. 
        **kwargs

        Returns:
        ========
        torch.Tensor. Array indicies of the selected samples.
        """
        BALD_acq = cls.get_BALD_acquisition(y_T)
        p_norm = torch.clamp(1. - BALD_acq, min=0) / torch.sum(1. - BALD_acq)
        p_norm = p_norm / torch.sum(p_norm)

        indices = torch.multinomial(p_norm, num_samples, replacement=False)  # Equivalent to np.random.choice
        return indices

    @classmethod
    @logged(name=__name__)
    def sample_by_bald_easiness_mod(cls, num_samples: int, y_T: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sample examples with a focus on easy examples using BALD approach. No consideration for the class/label balance.

        Modified method, as the original equation results in negligible differences between easy and difficult samples.
        To be tested if it performs better than the original method. Should we use softmax?.

        Different from Equation 8: https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf
        Translated to PyTorch from: https://github.com/microsoft/UST/blob/main/sampler.py

        Parameters:
        ===========
        num_samples: Int. The number of items to sample.
        y_T: torch.Tensor. Softmax outputs of a model(s), where a prediction for each sample was done >1. 
        **kwargs

        Returns:
        ========
        torch.Tensor. Array indicies of the selected samples.
        """
        BALD_acq = cls.get_BALD_acquisition(y_T)
        p_norm = torch.clamp(BALD_acq, min=1e-10)  # Equivalent to np.maximum
        p_norm = 1 / (p_norm / torch.sum(p_norm))
        p_norm = p_norm / torch.sum(p_norm)

        indices = torch.multinomial(p_norm, num_samples, replacement=False)  # Equivalent to np.random.choice
        return indices

    @classmethod
    @logged(name=__name__)
    def sample_by_bald_class_difficulty(cls, num_samples: int, y_T: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Sample examples with a focus on difficult examples using BALD approach. No consideration for the class/label balance.

        Based on equation 7 from but with consideration for classes: https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf
        Translated to PyTorch from: https://github.com/microsoft/UST/blob/main/sampler.py

        Parameters:
        ===========
        num_samples: Int. The number of items to sample.
        y_T: torch.Tensor. Softmax outputs of a model(s), where a prediction for each sample was done >1. 
        y: torch.Tensor. Predicted labels for the examples.
        num_classes: Int.  Number of classes.

        Returns:
        ========
        torch.Tensor. Array indicies of the selected samples.
        """
        BALD_acq = cls.get_BALD_acquisition(y_T)
        samples_per_class = num_samples // num_classes

        indices = []
        for label in range(num_classes):

            # Check if there are no exmples for the current label, just skip the 
            # selection for this label, as there is nothing to select from 
            if torch.sum(y == label) < 1:
                continue

            mask_map = y!=label
            p_norm = torch.clamp(BALD_acq, min=0)
            p_norm[mask_map] = 0 
            p_norm = p_norm / torch.sum(p_norm)

            if len(y[y==label]) < samples_per_class:
                replace = True
            else:
                replace = False

            indices.append(torch.multinomial(p_norm, samples_per_class, replacement=replace))  # Equivalent to np.random.choice

        return torch.cat(indices, dim=-1)

    @classmethod
    @logged(name=__name__)
    def sample_by_bald_class_easiness(cls, num_samples: int, y_T: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Sample examples with a focus on easy examples using BALD approach. No consideration for the class/label balance.

        Based on equation 8 from but with consideration for classes: https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf
        Translated to PyTorch from: https://github.com/microsoft/UST/blob/main/sampler.py

        Parameters:
        ===========
        num_samples: Int. The number of items to sample.
        y_T: torch.Tensor. Softmax outputs of a model(s), where a prediction for each sample was done >1. 
        y: torch.Tensor. Predicted labels for the examples.
        num_classes: Int.  Number of classes.

        Returns:
        ========
        torch.Tensor. Array indicies of the selected samples.
        """
        BALD_acq = cls.get_BALD_acquisition(y_T)
        BALD_acq = (1. - BALD_acq) / torch.sum(1. - BALD_acq)
        samples_per_class = num_samples // num_classes
        
        indices = []
        for label in range(num_classes):

            # Check if there are no exmples for the current label, just skip the 
            # selection for this label, as there is nothing to select from 
            if torch.sum(y == label) < 1:
                continue

            mask_map = y!=label
            p_norm = torch.clamp(BALD_acq, min=0)
            p_norm[mask_map] = 0
            p_norm = p_norm / torch.sum(p_norm)

            if len(y[y==label]) < samples_per_class:
                replace = True
            else:
                replace = False

            indices.append(torch.multinomial(p_norm, samples_per_class, replacement=replace))  # Equivalent to np.random.choice

        return torch.cat(indices, dim=-1)
    
    @classmethod
    @logged(name=__name__)
    def sample_by_bald_class_easiness_mod(cls, num_samples: int, y_T: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Sample examples with a focus on easy examples using BALD approach. No consideration for the class/label balance.
        
        Modified method, as the original equation results in negligible differences between easy and difficult samples.
        To be tested if it performs better than the original method. Should we use softmax?.

        Based on: https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf
        Translated to PyTorch from: https://github.com/microsoft/UST/blob/main/sampler.py

        Parameters:
        ===========
        num_samples: Int. The number of items to sample.
        y_T: torch.Tensor. Softmax outputs of a model(s), where a prediction for each sample was done >1. 
        y: torch.Tensor. Predicted labels for the examples.
        num_classes: Int.  Number of classes.

        Returns:
        ========
        torch.Tensor. Array indicies of the selected samples.
        """
        BALD_acq = cls.get_BALD_acquisition(y_T)
        samples_per_class = num_samples // num_classes
        
        indices = []
        for label in range(num_classes):

            # Check if there are no exmples for the current label, just skip the 
            # selection for this label, as there is nothing to select from 
            if torch.sum(y == label) < 1:
                continue

            mask_map = y!=label
            p_norm = torch.clamp(BALD_acq, min=1e-10)  # Equivalent to np.maximum
            
            p_norm[mask_map] = 1e-10
            p_norm = 1 / (p_norm / torch.sum(p_norm))
            p_norm = p_norm / torch.sum(p_norm)
            
            p_norm[mask_map] = 0  # just to make sure that the samples with the wrong labels are never chosen

            if len(y[y==label]) < samples_per_class:
                replace = True
            else:
                replace = False

            indices.append(torch.multinomial(p_norm, samples_per_class, replacement=replace))  # Equivalent to np.random.choice

        return torch.cat(indices, dim=-1)
    
    @classmethod
    @logged(name=__name__)
    def sample_uniformly(cls, num_samples: int, y_T: torch.Tensor=None, y: torch.Tensor=None, num_classes: int=None) -> torch.Tensor:
        """
        Sample examples uniformly. No consideration for the class balance.
        
        Parameters:
        ===========
        num_samples: Int. The number of items to sample.
        y_T: torch.Tensor. - NOT USED! - Softmax outputs of a model(s), where a prediction for each sample was done >1. 
        y: torch.Tensor. Predicted labels for the examples.
        num_classes: Int. - NOT USED! - Number of classes.

        Returns:
        ========
        torch.Tensor. Array indicies of the selected samples.
        """        
        return torch.randperm(len(y))[:num_samples]
