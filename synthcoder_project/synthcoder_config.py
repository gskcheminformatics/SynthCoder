import logging
from tokenizers import (normalizers,
                        SentencePieceBPETokenizer, 
                        SentencePieceUnigramTokenizer,
                        BertWordPieceTokenizer,
                        CharBPETokenizer,
                        ByteLevelBPETokenizer)

import synthcoder_project.encoders.bert_poolers as poolers

# Platform logger config
ACTION_LOGGER_DIR = "log_of_platform_actions"
ACTION_LOGGER_LEVEL = logging.INFO

# General SynthCoder config
ALLOWED_TASK_TYPES  = ("mlm", "classification", "regression")
SUPPORTED_MODELS = ("bert", "distilbert")
ALLOWED_TOKENIZER_TRAINERS = {"SentencePieceBPETokenizer": SentencePieceBPETokenizer,
                              "SentencePieceUnigramTokenizer": SentencePieceUnigramTokenizer,
                              "BertWordPieceTokenizer": BertWordPieceTokenizer,
                              "CharBPETokenizer": CharBPETokenizer,
                              "ByteLevelBPETokenizer": ByteLevelBPETokenizer}

# Avaialble unicode normalizers for the tokenizer normalizers.
NORMALIZERS = {"nfc": normalizers.NFD,
               "nfd": normalizers.NFKD, 
               "nfkc": normalizers.NFC, 
               "nfkd": normalizers.NFKC}

# The regex pattern use by the BEE model for splitting reactions into tokens (SMILES)
BEE_TOKENIZATION_REGEX = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\^|\%[0-9]{2}|[0-9])"

# Available poolers for fine-tuning (classification/regression)
AVAILABLE_POOLERS = {"default": poolers.VanillaBertPooler,
                     "conv": poolers.Conv1DPooler,
                     "conv_v2": poolers.Conv1DPooler_v2,
                     "mean": poolers.MeanPooler,
                     "max": poolers.MaxPooler,
                     "mean_max": poolers.MeanMaxPooler,
                     "concat": poolers.ConcatPooler, 
                     "lstm": poolers.LSTMPooler}