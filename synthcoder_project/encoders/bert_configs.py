# from transformers import BertConfig
from transformers.configuration_utils import PretrainedConfig
from typing import Literal


class EnrichedBertConfig(PretrainedConfig):
    r"""

    This is a modified BertConfig class, with added options for networks with enriched embeddings which can use different poolers. 

    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFBertModel`]. It is used to
    instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BERT
    [bert-base-uncased](https://huggingface.co/bert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Original Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Added Args:
        mlm (`bool`, *optional*, defaults to `True`)
            !!! This flag is set automatically by the SynthCoder platform, and should not be set by the user. !!!
            Flag indicating if the MLM task is run by the model. 
        cross_attention_encoder_input (`bool`, *optional*, defaults to `False`):
            !!! This flag is set automatically by the SynthCoder platform, and should not be set by the user. !!!
            Flag indicating if the cross-attention between tokens and descripts should be run. 
        cross_attention_use_extended_descript_network(`bool`, *optional*, defaults to `False`):
            !!! This flag is set automatically by the SynthCoder platform, and should not be set by the user. !!!
            Flag to indicate if the extended cross attention network processing tokens and descripts should be run.
            The extended network performs double MLM task during pretraining, on both the masked tokens and descriptor vectors.
        delta_descriptors_loss(`float`, *optional*, defaults to `100.0`):
            Coefficient (δ) used in the extended descriptor network during loss calculation in the pretraining. 
            Loss = CrossEntropyLoss(MLM Tokens) + δ*MSELoss(MLM Descriptors) 
            It adjusts the contribution of the descriptor-based MLM to the total loss.

        tokenizer_enrichment_vocab_size(`int`, *optional*, defaults to `32`):
            Size of the enriched vocabulary. Defines the number of different tokens that can be represented by the
            `enriching_ids`.
 

        pooler_type (`str`, *optional*, defaults to `"default"`):
            Allows to select the pooler method for processing the hidden states. 
            Allowed poolers are `"default"`, `"conv"`, `"conv_v2"`, `"mean"`, `"max"`, `"mean_max"`, `"concat"`, `"lstm"`.
            Please see: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
        conv_kernel_size (`int`, *optional*, defaults to `4`):
            Only applicable if `pooler_type` is set to `"conv"`. Sets the kernel size for convolutions. 
        conv_padding_size (`int`, *optional*, defaults to `1`):
            Only applicable if `pooler_type` is set to `"conv"`. Sets the padding size for convolutions. 
        conv_stride (`int`, *optional*, defaults to `2`):
            Only applicable if `pooler_type` is set to `"conv"`. Sets the stride for convolutions. 
        conv1_out_channels (`int`, *optional*, defaults to `128`):
            Only applicable if `pooler_type` is set to `"conv"`. Sets the number of output channels for the first convolution. 
        conv2_out_channels (`int`, *optional*, defaults to `32`):
            Only applicable if `pooler_type` is set to `"conv"`. Sets the number of output channels for the second convolution. 
        pool_kernel_size (`int`, *optional*, defaults to `2`):
            Only applicable if `pooler_type` is set to `"conv"`. Sets the kenerl size for the convolutional pooler.  
        conv_pool_method (`str`, *optional*, defaults to `"max"`):
            Only applicable if `pooler_type` is set to `"conv"`. Allowed options are `"max"`, `"avg"`.
            Determines the method used for pooling after convolutions.
        concat_num_hidden_layers (`int`, *optional*, defaults to `4`):
            Only applicable if `pooler_type` is set to `"concat"`. The number of last hidden states 
            (embeddings generated by the encoder untis) to concatanate and used for classification/regression.
        lstm_embedding_type (`str`, *optional*, defaults to `"default"`):
            Only applicable if `pooler_type` is set to `"lstm"`. Indicates type of embeddings / hidden states processing, 
            that should be used for feeding the LSTM cell, used for pooling. Options: "default", "conv", "mean", "max", "mean_max".
        hiddendim_lstm (`int`, *optional*, defaults to `1024`):
            Only applicable if `pooler_type` is set to `"lstm"`. The hidden size of the LSTM cell. 
            This number should be normally larger than the number of features fed into the LSTM cell.  


    Examples:

    ```python
    >>> from transformers import BertModel
    >>> from synthcoder_project.encoders.bert_configs import EnrichedBertConfig

    >>> # Initializing a BERT bert-base-uncased style configuration
    >>> configuration = EnrichedBertConfig()

    >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
    >>> model = BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bert"

    def __init__(
        self,

        # original arguments
        vocab_size: int=30522,
        hidden_size: int=768,
        num_hidden_layers: int=12,
        num_attention_heads: int=12,
        intermediate_size: int=3072,
        hidden_act: str="gelu",
        hidden_dropout_prob: float=0.1,
        attention_probs_dropout_prob: float=0.1,
        max_position_embeddings: int=512,
        type_vocab_size: int=2,
        initializer_range: float=0.02,
        layer_norm_eps: float=1e-12,
        pad_token_id: int=0,
        position_embedding_type: str="absolute",
        use_cache: bool=True,
        classifier_dropout: bool=None,

        # added custom arguments to control the encoder cross attention
        mlm: bool=True,
        cross_attention_encoder_input: bool=False,  # do not modify yourself, SynthCoder will set it to the correct value itself when needed. 
        cross_attention_use_extended_descript_network: bool=False, # do not modify yourself, SynthCoder will set it to the correct value itself when needed.
        delta_descriptors_loss: float=100.0,
        # cross_attention_number_of_descriptors: int=None, # do not modify yourself, SynthCoder will set it to the correct value itself. 
        tokenizer_enrichment_vocab_size: int=32,

        # added custom arguments to control the encoder pooler
        pooler_type: Literal["default", "conv", "conv_v2", "mean", "max", "mean_max", "concat", "lstm"]="default",
        conv_kernel_size: int=4,
        conv_padding_size: int=1,
        conv_stride: int=2,
        conv1_out_channels: int=128,
        conv2_out_channels: int=32,
        pool_kernel_size: int=2,
        conv_pool_method: Literal["max", "avg"]="max",
        concat_num_hidden_layers: int=4,
        lstm_embedding_type: Literal["default", "conv", "mean", "max", "mean_max"]="default",
        hiddendim_lstm: int=1024,  # this number should be normally larger than the number of features, so in this case largber than the hidden_size  
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        self.mlm=mlm
        self.cross_attention_encoder_input=cross_attention_encoder_input
        self.cross_attention_use_extended_descript_network=cross_attention_use_extended_descript_network
        self.delta_descriptors_loss=delta_descriptors_loss

        self.tokenizer_enrichment_vocab_size=tokenizer_enrichment_vocab_size

        self.pooler_type=pooler_type
        self.conv_kernel_size=conv_kernel_size
        self.conv_padding_size=conv_padding_size
        self.conv_stride=conv_stride
        self.conv1_out_channels=conv1_out_channels
        self.conv2_out_channels=conv2_out_channels
        self.pool_kernel_size=pool_kernel_size
        self.conv_pool_method=conv_pool_method
        self.concat_num_hidden_layers=concat_num_hidden_layers
        self.lstm_embedding_type=lstm_embedding_type
        self.hiddendim_lstm=hiddendim_lstm

