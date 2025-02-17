# Code example for SynthCoder use: Fine-Tuning

# Import DistilBERT model architecture, tokenizer and model config from HuggingFace 
from transformers import (DistilBertTokenizerFast, 
                          DistilBertConfig)
from synthcoder_project.synthcoder import SynthCoderModeling
from transformers import DistilBertForSequenceClassification

regression_args = {"num_labels": 1,
                   "problem_type": "regression",
                   "logger_dir": "fine_tuning_logs",
                   }

regression_model = SynthCoderModeling(model_encoder=DistilBertForSequenceClassification,
                                      model_config=DistilBertConfig,
                                      model_tokenizer=DistilBertTokenizerFast,
                                      task_type="regression",
                                      train_file="data/data_examples/photoredox_train.csv",
                                      validation_file="data/data_examples/photoredox_validate.csv",
                                      test_file=None,
                                      predict_file=None,
                                      accelerator="gpu",
                                      distributed_trainer_strategy="ddp",
                                      user_args=regression_args, 
                                      model_name=<path to the pre-trained model .dir>  # e.g. "pretraining_logs/TensorBoard_logs/version_0/checkpoints/epoch=29-step=330.ckpt.dir",
                                      )
# Fine-tune model
regression_model.fit_model()

# Run validation
regression_model.validate_model(csv_path_for_saving="./_results/validation_results.csv")

# Run testing
regression_model.test_model(new_test_file="data/data_examples/photoredox_test.csv",
                            csv_path_for_saving="./_results/test_results.csv")

# Perform inference
regression_model.predict(new_predict_file="data/data_examples/photoredox_inference.csv",
                         csv_path_for_saving="./_results/inference_results.csv")