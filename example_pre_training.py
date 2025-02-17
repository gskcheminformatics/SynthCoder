# Code example for SynthCoder use: Pre-training

# Import DistilBERT model architecture, tokenizer and model config from HuggingFace 
from transformers import (DistilBertForMaskedLM,
                          DistilBertTokenizerFast, 
                          DistilBertConfig)
from synthcoder_project.synthcoder import SynthCoderModeling


custom_args = {"model_type": "distilbert",
               "tokenizer_vocab_size": 8192,
               "manual_seed": 42,
               "pad_to_multiple_of": 32,
               "num_train_epochs": 30,
               "batch_size_train": 4,
               "batch_size_eval": 32,
               "gradient_accumulation_steps": 8,
               "dataloader_num_workers": 1,
               "precision": "16-mixed",
               "logger_dir": "pretraining_logs",
               "log_every_n_steps": 1,
               "early_stopping_patience": 5,
               "scheduler": "cyclic_lr_scheduler",
               "cyclic_lr_scheduler_mode": "triangular",
               "cyclic_lr_scheduler_ratio_size_down": 0.27,
               "tokenizer_punctuation_split": True,
               }

mlm_model = SynthCoderModeling(model_encoder=DistilBertForMaskedLM,
                               model_config=DistilBertConfig,
                               model_tokenizer=DistilBertTokenizerFast,
                               task_type="mlm",
                               tokenizer_train_file="data/data_examples/photoredox_train.txt",
                               train_file="data/data_examples/photoredox_train.txt",
                               validation_file="data/data_examples/photoredox_validate.txt",
                               accelerator="auto",
                               distributed_trainer_strategy="ddp",
                               user_args=custom_args,
                               )

# Pre-train the model
mlm_model.fit_model()

