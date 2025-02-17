# General testing of the platform functionality for model initalisation, training, validation, testing and inference.

import pandas as pd
from pandas.testing import assert_frame_equal
from transformers import (DistilBertForMaskedLM,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer, 
                          DistilBertConfig)
from synthcoder_project.synthcoder import SynthCoderModeling
from synthcoder_project.synthcoder_tokenizers.reaction_tokenizers import EnrichedBertTokenizerFast
import synthcoder_project.utilities as utils 


CHECKPOINT_MLM = "pytest_files/logs_mlm/TensorBoard_logs/version_0/checkpoints/last.ckpt"
CHECKPOINT_REGRESS = "pytest_files/logs_regression/TensorBoard_logs/version_0/checkpoints/last.ckpt"
CHECKPOINT_CLASSIF = "pytest_files/logs_classification/TensorBoard_logs/version_0/checkpoints/last.ckpt"
LOGGER_DIR_MLM = "pytest_files/logs_mlm"
LOGGER_DIR_REG = "pytest_files/logs_regression"
LOGGER_DIR_CLASS = "pytest_files/logs_classification"


mlm_args = {"model_type": "distilbert",
            "tokenizer_vocab_size": 256,
            "num_train_epochs": 1,
            "batch_size_train": 2,
            "batch_size_eval": 2,
            "precision": "16-mixed",
            "logger_dir": LOGGER_DIR_MLM,
            "deterministic": True,
            "manual_seed": 42,
            "tokenizer_name": "pytest_files/vocabulary", #/vocab.txt",
            }


regression_args = {"num_train_epochs": 1,
                   "num_labels": 1,
                   "problem_type": "regression",
                   "logger_dir": LOGGER_DIR_REG,
                   "deterministic": True,
                   "manual_seed": 42,
                   }


classification_args = {"num_train_epochs": 1,
                       "num_labels": 2,
                       "problem_type": "single_label_classification",
                       "logger_dir": LOGGER_DIR_CLASS,
                       "deterministic": True,
                       "manual_seed": 42,
                       }


def return_mlm_model():
        mlm_model = SynthCoderModeling(model_encoder=DistilBertForMaskedLM,
                                        model_config=DistilBertConfig,
                                        model_tokenizer=EnrichedBertTokenizerFast,#DistilBertTokenizer,
                                        task_type="mlm",
                                        tokenizer_train_file="pytest_files/input_files/photoredox_train.txt",
                                        train_file="pytest_files/input_files/photoredox_train.txt",
                                        validation_file="pytest_files/input_files/photoredox_validate.txt",
                                        accelerator="auto",
                                        distributed_trainer_strategy="auto",
                                        devices=1,
                                        user_args=mlm_args,
                                        )
        return mlm_model


def return_regression_model():
    regression_model = SynthCoderModeling(model_encoder=DistilBertForSequenceClassification,
                                        model_config=DistilBertConfig,
                                        model_tokenizer=EnrichedBertTokenizerFast,#DistilBertTokenizer,
                                        task_type="regression",
                                        train_file="pytest_files/input_files/photoredox_train.csv",
                                        validation_file="pytest_files/input_files/photoredox_validate.csv",
                                        test_file="pytest_files/input_files/photoredox_test.csv",
                                        predict_file="pytest_files/input_files/photoredox_inference.csv",
                                        accelerator="auto",
                                        distributed_trainer_strategy="auto",
                                        devices=1,
                                        user_args=regression_args, 
                                        model_name="pytest_files/logs_mlm/TensorBoard_logs/version_0/checkpoints/last.ckpt.dir",
                                        )
    return regression_model


def return_classif_model():
    classif_model = SynthCoderModeling(model_encoder=DistilBertForSequenceClassification,
                                    model_config=DistilBertConfig,
                                    model_tokenizer=EnrichedBertTokenizerFast,#DistilBertTokenizer,
                                    task_type="classification",
                                    train_file="pytest_files/input_files/photoredox_train_classif.csv",
                                    validation_file="pytest_files/input_files/photoredox_validate_classif.csv",
                                    test_file="pytest_files/input_files/photoredox_test_classif.csv",
                                    predict_file="pytest_files/input_files/photoredox_inference.csv",
                                    accelerator="auto",
                                    distributed_trainer_strategy="auto",
                                    devices=1,
                                    user_args=classification_args, 
                                    model_name="pytest_files/logs_mlm/TensorBoard_logs/version_0/checkpoints/last.ckpt.dir",
                                    )
    return classif_model


def assert_equal_results(funct, ckpt_path, control_file):
    results = funct(ckpt_path=ckpt_path)
    control_results = pd.read_csv(control_file) 
    assert_frame_equal(results, control_results, check_exact=False, rtol=0.2) #atol=0.1) 


class TestGeneralPlatformFuncs:

    # Test MLM model:
    # ==================================================
    def test_mlm_init(self):
        utils.remove_tree_directory(LOGGER_DIR_MLM)
        return_mlm_model()


    def test_mlm_fit(self):
        mlm_model = return_mlm_model()
        mlm_model.fit_model()


    def test_mlm_validation(self):
        model = return_mlm_model()
        # assert_equal_results(funct=model.validate_model,
        #                      ckpt_path=CHECKPOINT_MLM,
        #                      control_file="pytest_files/control_csv_files/validation_results_mlm.csv")


    # Test regression model:
    # ==================================================
    def test_regression_init(self):
        utils.remove_tree_directory(LOGGER_DIR_REG)
        return_regression_model() 


    def test_regress_fit(self):
        regress_model = return_regression_model()
        regress_model.fit_model()


    def test_regress_validation(self):
        model = return_regression_model()
        # assert_equal_results(funct=model.validate_model,
        #                      ckpt_path=CHECKPOINT_REGRESS,
        #                      control_file="pytest_files/control_csv_files/validation_results_regress.csv")


    def test_regress_testing(self):
        model = return_regression_model()
        # assert_equal_results(funct=model.test_model,
        #                      ckpt_path=CHECKPOINT_REGRESS,
        #                      control_file="pytest_files/control_csv_files/testing_results_regress.csv")


    def test_regress_inference(self):
        model = return_regression_model()
        # assert_equal_results(funct=model.predict,
        #                      ckpt_path=CHECKPOINT_REGRESS,
        #                      control_file="pytest_files/control_csv_files/inference_results_regress.csv")
    


    # Test classification model:
    # ==================================================
    
    def test_classification_init(self):
        utils.remove_tree_directory(LOGGER_DIR_CLASS)       
        return_classif_model()


    def test_classif_fit(self):
        classif_model = return_classif_model()
        classif_model.fit_model()


    def test_classif_validation(self):
        model = return_classif_model()
        # assert_equal_results(funct=model.validate_model,
        #                      ckpt_path=CHECKPOINT_CLASSIF,
        #                      control_file="pytest_files/control_csv_files/validation_results_classif.csv")


    def test_classif_testing(self):
        model = return_classif_model()
        # assert_equal_results(funct=model.test_model,
        #                      ckpt_path=CHECKPOINT_CLASSIF,
        #                      control_file="pytest_files/control_csv_files/testing_results_classif.csv")


    def test_classif_inference(self):
        model = return_classif_model()
        # assert_equal_results(funct=model.predict,
        #                      ckpt_path=CHECKPOINT_CLASSIF,
        #                      control_file="pytest_files/control_csv_files/inference_results_classif.csv")
    



    # @pytest.mark.parametrize("funct, ckpt_path, control_file", [
    #     (return_mlm_model().validate_model, CHECKPOINT_MLM, "pytest_files/control_csv_files/validation_results_mlm.csv"),
    #     (return_regression_model().validate_model,CHECKPOINT_REGRESS,"pytest_files/control_csv_files/validation_results_regress.csv"),
    #     (return_classif_model().validate_model,CHECKPOINT_CLASSIF,"pytest_files/control_csv_files/testing_results_classif.csv"),
    #     (return_regression_model().test_model,CHECKPOINT_REGRESS,"pytest_files/control_csv_files/testing_results_regress.csv"),
    #     (return_classif_model().test_model,CHECKPOINT_CLASSIF,"pytest_files/control_csv_files/testing_results_classif.csv"),
    #     (return_regression_model().predict,CHECKPOINT_REGRESS,"pytest_files/control_csv_files/inference_results_regress.csv"),
    #     (return_classif_model().predict,CHECKPOINT_CLASSIF,"pytest_files/control_csv_files/inference_results_classif.csv"),
    # ])
    # def test_equal_results(self, funct, ckpt_path, control_file):
    #     results = funct(ckpt_path=ckpt_path)
    #     control_results = pd.read_csv(control_file) 
    #     assert_frame_equal(results, control_results, check_exact=False, atol=0.01) 
