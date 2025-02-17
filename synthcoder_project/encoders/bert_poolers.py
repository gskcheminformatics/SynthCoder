# Poolers to be used with enriched BERT architecture. 
# Please take a look at: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently


from transformers.utils import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
# import torch.nn.functional as F
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)


class VanillaBertPooler(nn.Module):
    """
    Based on the original HuggingFace BertPooler. The forward method returns logits. 
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object):
        """
        Initialises pooler with all layer needed for linear NN and classification.

        Parameters:
        ===========
        config: Object. Configuration of the network.
        
        Returns:
        ========
        None
        """

        print("Initialising VanillaBertPooler")
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier_X = nn.Linear(config.hidden_size, config.num_labels)

    @logged()
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The method taken hidden states, runs NN and returns calculated logits. 
        Generates logits.

        Parameters:
        ===========
        hidden_states: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 

        """
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = self.return_cls_embeddings(hidden_states)
        logits = self.generate_logits_with_linear_nn(first_token_tensor)
        return logits

    @logged()
    def generate_logits_with_linear_nn(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        The method takes the pooled output (first token embeddings), runs NN and returns logits. 

        Parameters:
        ===========
        hidden_states: torch.Tensor.
        
        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 
        
        """
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_X(pooled_output)
        return logits

    @staticmethod
    @logged(name=__name__, message="Running return_cls_embeddings")
    def return_cls_embeddings(hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns embeddings for the first token - classification token, from the hidden states. 
        The by extracting the embeddings for the first token, the tensor size changes 
        [batch_size, num_tokens, hidden_size] -> [batch_size, hidden_size].
        
        Parameters:
        ===========
        hidden_states: torch.Tensor.

        ** kwargs

        Returns:
        ========
        torch.Tensor. Embeddings corrsponding to the first token (cls token). 
        """
        return hidden_states[:, 0]


class Conv1DPooler(nn.Module):
    """
    Convolutional 1D pooler for hidden states with fully connected layer.

    Take a look at: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object):
        """
        Initialises object with all layers needed for NN and classification. 

        Parameters:
        ===========
        config: Object. Configuration of the network.
        
        Returns:
        ========
        None
        """
        
        print("Initialising Conv1DPooler")
        super().__init__()

        # Calculate the number of expected output embedding dimentions 
        # This is needed to figure out the size of the fully connected layer 
        conv1_dim = self.calculate_output_dim(input_dimensions=config.block_size, filter_size=config.conv_kernel_size,
                                              padding_size=config.conv_padding_size, stride_size=config.conv_stride)
        
        conv1_pooled_dim = self.calculate_output_dim(input_dimensions=conv1_dim, filter_size=config.pool_kernel_size,
                                                     stride_size=config.pool_kernel_size) # for pooling, the stride is the same as the kernel size, unless specified otherwise
        
        conv2_dim = self.calculate_output_dim(input_dimensions=conv1_pooled_dim, filter_size=config.conv_kernel_size,
                                              padding_size=config.conv_padding_size, stride_size=config.conv_stride)
        
        conv2_pooled_dim = self.calculate_output_dim(input_dimensions=conv2_dim, filter_size=config.pool_kernel_size,
                                                     stride_size=config.pool_kernel_size) # for pooling, the stride is the same as the kernel size, unless specified otherwise
        logger.debug(f"The calculated output dimensions:\n{conv1_dim=}\n{conv1_pooled_dim=}\n{conv2_dim=}\n{conv2_pooled_dim=}")

        # Convolution layers and conv. pooler
        self.conv1 = nn.Conv1d(in_channels=config.hidden_size, 
                               out_channels=config.conv1_out_channels,
                               kernel_size=config.conv_kernel_size, 
                               stride=config.conv_stride, 
                               padding=config.conv_padding_size)
        
        self.conv2 = nn.Conv1d(in_channels=config.conv1_out_channels, 
                               out_channels=config.conv2_out_channels,
                               kernel_size=config.conv_kernel_size, 
                               stride=config.conv_stride, 
                               padding=config.conv_padding_size)
        
        pool_methods = {"max": nn.MaxPool1d, "avg": nn.AvgPool1d}
        self.pool = pool_methods[config.conv_pool_method](kernel_size=config.pool_kernel_size)

        # Dropout and activation functions
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.activation_relu = nn.ReLU()
        self.activation_tanh = nn.Tanh()

        # Fully connected layer
        self.intermediate_linear_size = 128
        self.linear = nn.Linear(config.conv2_out_channels * conv2_pooled_dim, self.intermediate_linear_size)
        
        # If the pooler is not convolutional, do not initialise the last linear layer for classification
        if config.pooler_type == "conv":
            self.classifier = nn.Linear(self.intermediate_linear_size, config.num_labels) 
            logger.debug(f"Convolutional pooler. Initialised linear classification layer")

    @logged()
    def calculate_output_dim(self, input_dimensions: int, filter_size: int=2,
                             padding_size: int=0, stride_size: int=1) -> int:
        """
        Calculates the number of dimensions achieved after convolution 
        
        Parameters:
        ===========
        input_dimensions: Int. Number of the input dimensions for the convolution. 
        filter_size: Int. Size of the kernel. 
        padding_size: Int. Size of the padding. 
        stride_size: Int. Size of the stride. 
        
        Returns:
        ========
        resulting_dim: Int. Resulting number of dimensions after convolution with the provided settings. 
        """
        with torch.no_grad():
            resulting_dim = ((input_dimensions - filter_size + 2* padding_size) / stride_size) + 1
            return int(resulting_dim)

    @logged()
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The method taken hidden states, runs NN and returns calculated logits. 
        Generates logits.

        Parameters:
        ===========
        hidden_states: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 

        """        
        flat_embeddings = self.run_network(hidden_states)
        flat_embeddings = self.dropout(flat_embeddings)
        logits = self.classifier(flat_embeddings)

        return logits
    
    @logged()
    def run_network(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The method takes hidden states, runs NN and returns embeddings after one linear layer 
        and after passing the results through the activation function.
        The resulting tensor has size [batch_size, self.intermediate_linear_size].

        Parameters:
        ===========
        hidden_states: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        flat_embeddings: torch.Tensor. Calculated logits. 

        """        
        permuted_last_hidden_state = hidden_states.permute(0, 2, 1)  # [batch, sequence, dim] -> [batch, dim, sequence]

        # Convolution
        conv1_embeddings = self.activation_relu(self.conv1(permuted_last_hidden_state))     
        conv1_pooled = self.pool(conv1_embeddings)        
        conv2_embeddings = self.activation_relu(self.conv2(conv1_pooled))
        conv2_pooled = self.pool(conv2_embeddings)

        # Fully connected layer
        flat_embeddings = conv2_pooled.view(conv2_pooled.shape[0], -1) # flatten the conv features
        flat_embeddings = self.activation_tanh(self.linear(flat_embeddings))

        return flat_embeddings


class Conv1DPooler_v2(nn.Module):
    """
    Very simple version of convolutional 1D pooler for hidden states; does not have a fully connected layer.

    Take a look at: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object):
        """
        Initialises object with all layer needed for NN. 

        Parameters:
        ===========
        config: Object. Configuration of the network.
        
        Returns:
        ========
        None
        """
        
        print("Initialising Conv1DPooler_v2")
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels=config.hidden_size, out_channels=256, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=256, out_channels=config.num_labels, kernel_size=2, padding=1)

        self.activation = nn.ReLU()

    @logged()
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        The method taken hidden states, runs convolutions and returns calculated logits. 
        Generates logits.

        Parameters:
        ===========
        hidden_states: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 

        """
        permuted_last_hidden_state = hidden_states.permute(0, 2, 1)
        cnn_embeddings = self.activation(self.cnn1(permuted_last_hidden_state))
        cnn_embeddings = self.cnn2(cnn_embeddings)
        logits, _ = torch.max(cnn_embeddings, 2)

        return logits
    

class MeanPooler(VanillaBertPooler):
    """
    Pooler based on mean hidden state values for tokens. Averaging over tokens. 

    Take a look at: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object):
        """
        Initialises object.

        Parameters:
        ===========
        config: Object. Configuration of the network.
        
        Returns:
        ========
        None
        """

        print("Initialising MeanPooler")
        super().__init__(config=config)

    @logged()
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates average hidden states outputs for tokens. All hidden states for all sequence tokens are avergaed into one hidden state representation. 
        The mean embeddings are then sent through a linear classification network with dropout.
        Generates logits.

        Parameters:
        ===========
        hidden_states: torch.Tensor.
        attention_mask: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 

        """
        mean_embeddings = self.return_mean_embeddings(hidden_states, attention_mask)
        logits = self.generate_logits_with_linear_nn(mean_embeddings)
        return logits

    
    @staticmethod
    @logged(name=__name__, message="Running return_mean_embeddings")
    def return_mean_embeddings(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns mean embeddings, taking into consideration the attention mask. 
        The mean is calculated across all tokens.
        The size of the resulting tensor is [batch_size, hidden_size].
        
        Parameters:
        ===========
        hidden_states: torch.Tensor.
        attention_mask: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        mean_embeddings: torch.Tensor.  
        
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()  # change dimensions so they match the hidden states 
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)  # "masking" the hidden states based on the attention_mask and then summing hidden states for each token position
        sum_mask = input_mask_expanded.sum(1)  # summing so that dimensions match sum_embeddings; the sum will match the number of tokens needed for averaging
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # make sure that you do not divide by zero; this clamps all elements in input into the range with the minimum value of 1e-9
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooler(VanillaBertPooler):
    """
    Pooler that creates tensor with max values in the dim vector, taken from all the sequence tokens.
    
    Take a look at: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object):
        """
        Initialises object.

        Parameters:
        ===========
        config: Object. Configuration of the network.
        
        Returns:
        ========
        None
        """

        print("Initialising MaxPooler")
        super().__init__(config=config)

    @logged()
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates max hidden states outputs for tokens. All hidden states for all sequence tokens are considred and max values are taken
        into one hidden state representation. 
        The max embeddings are then sent through a linear classification network with dropout.
        Generates logits.

        Parameters:
        ===========
        hidden_states: torch.Tensor.
        attention_mask: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 

        """
        max_embeddings = self.return_max_embeddings(hidden_states, attention_mask)
        logits = self.generate_logits_with_linear_nn(max_embeddings)
        return logits

    
    @staticmethod
    @logged(name=__name__, message="Running return_max_embeddings")
    def return_max_embeddings(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns mean embeddings, taking into consideration the attention mask. 
        The mean is calculated across all tokens.
        The size of the resulting tensor is [batch_size, hidden_size].    
        
        Parameters:
        ===========
        hidden_states: torch.Tensor.
        attention_mask: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        mean_embeddings: torch.Tensor.  
        
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()  # change dimensions so they match the hidden states 
        hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(hidden_states, 1)[0]
        return max_embeddings


class MeanMaxPooler(MeanPooler, MaxPooler):
    """
    Pooler that creates tensor with combined mean and max values in the dim vector, taken from all the sequence tokens.
    
    Take a look at: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object):
        """
        Initialises the object. 
        Redefines the dense layer of the parent classes to account for a larger tensor size.
        
        Parameters:
        ===========
        config: Object. Configuration of the network.
        
        Returns:
        ========
        None  
        """

        print("Initialising MeanMaxPooler")
        super().__init__(config=config)
        self.dense = nn.Linear(config.hidden_size *2, config.hidden_size)

    @logged()
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates mean and max hidden states outputs for tokens. 
        All hidden states for all sequence tokens are considred and mean and max values are taken into one hidden state representation. 
        The combined mean & max embeddings are then sent through a linear classification network with dropout.
        Generates logits.

        Parameters:
        ===========
        hidden_states: torch.Tensor.
        attention_mask: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 

        """
        mean_max_embeddings = self.return_mean_max_embeddings(hidden_states, attention_mask)
        logits = self.generate_logits_with_linear_nn(mean_max_embeddings)
        return logits
    
    
    @staticmethod
    @logged(name=__name__, message="Running return_mean_max_embeddings")
    def return_mean_max_embeddings(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns combined mean and max embeddings, taking into consideration the attention mask. 
        The mean and max is calculated across all tokens, and then concatanated together.
        The size of the resulting tensor is [batch_size, hidden_size*2].
        
        Parameters:
        ===========
        hidden_states: torch.Tensor.
        attention_mask: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        mean_max_embeddings: torch.Tensor.  
        
        """      
        mean_embeddings = MeanPooler.return_mean_embeddings(hidden_states=hidden_states, attention_mask=attention_mask)
        max_embeddings = MaxPooler.return_max_embeddings(hidden_states=hidden_states, attention_mask=attention_mask)
        mean_max_embeddings = torch.cat((mean_embeddings, max_embeddings), 1)  # combining mean and max tensors together, e.g.: torch.cat(([16, 256], [16,256]), 1) -> [16,512]  
        return mean_max_embeddings
    

class ConcatPooler(VanillaBertPooler):
    """
    Pooler which concatanates the CLS token embeddings from the X last layers/encoder units.
    The concatanation is passed through linear NN to perform classification.

    Take a look at: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """
    
    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object):
        """
        Initialises the object. 
        Redefines the dense layer of the parent class to account for a larger tensor size.

        Parameters:
        ===========
        config: Object. Configuration of the network.
        
        Returns:
        ========
        None  
        """

        print("Initialising ConcatPooler")
        super().__init__(config)
        self.num_layers_to_use = min(config.concat_num_hidden_layers, config.num_hidden_layers)  # choose the smaller of the options
        self.dense = nn.Linear(config.hidden_size * self.num_layers_to_use, config.hidden_size)

    @logged()
    def forward(self, hidden_states: torch.Tensor, hidden_states_all_layers: tuple[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Combines embeddings for the first token (cls token) from the indicated number of last layers/encoder units.
        The created concatanated tensor is then sent through a linear classification network with dropout.
        Generates logits.
        
        Parameters:
        ===========
        hidden_states: torch.Tensor. - Not Used!
        hidden_states_all_layers: tuple[torch.Tensor]. Hidden states for all layers/encoders. 
        
        **kwargs

        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 

        """
        # Concatenate the last tensor dimension for the last x number of `layers`
        hidden_states_to_concat = tuple([hidden_states_all_layers[-layer_i] for layer_i in range(1, self.num_layers_to_use+1)])
        concat_embeddigns = torch.cat(hidden_states_to_concat, -1)  # concat the last dimension of the tensor between tensors from different layers -> [batch, sequence, dim] , so dims are concatanated

        # Select tensors only for the first token. So sequence[0] in [batch, sequence, dim] 
        first_token_tensors = concat_embeddigns[:, 0]  
        logits = self.generate_logits_with_linear_nn(first_token_tensors)
        return logits


class LSTMPooler(VanillaBertPooler):
    """
    Uses embeddings from all hidden layers/encoder units to feed into LSTM network. 
    It can used standard CLS embeddings, mean, max, mean & max, and convolutional 'embeddings'. 

    Take a look at: https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object):
        """
        Initialises the necessary network(s).
        Redifines the dense and classification layers inherited from te parent class.

        Parameters:
        ===========
        config: Object. Configuration of the network.
        
        Returns:
        ========
        None  
        """

        print("Initialising LSTMPooler")
        super().__init__(config)
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers

        # Define the available options for embeddings
        embedding_extraction_options = {
            "default": VanillaBertPooler.return_cls_embeddings,
            "mean": MeanPooler.return_mean_embeddings,
            "max": MaxPooler.return_max_embeddings,
            "mean_max": MeanMaxPooler.return_mean_max_embeddings,
        }

        # Select the embedding extractor
        if config.lstm_embedding_type != "conv":
            self.embedding_extractor = embedding_extraction_options[config.lstm_embedding_type]


        # Define the convolutional option for embedding extraction
        if config.lstm_embedding_type == "conv":
            logger.debug("Initialising Conv layers for the LSTM")

            # each layer should have its own convolutional layer:
            conv_list = [Conv1DPooler(config=config) for _ in range(config.num_hidden_layers)]
            self.conv_modules = nn.ModuleList(conv_list)

            # update hidden size, to make it appropriate for the conv network output
            self.hidden_size = self.conv_network.intermediate_linear_size  
            
            # self.conv_network = Conv1DPooler(config=config) # initialise object
            # embedding_extraction_options["conv"] = self.conv_network.run_network
        
        # Double the hidden size for mean & max embeddings
        elif config.lstm_embedding_type == "mean_max":
            self.hidden_size = config.hidden_size *2  
            logger.debug(f"Doubling the hidden size for the `mean_max` LSTM embeddings to {self.hidden_size=}")
        else:
            self.hidden_size = config.hidden_size

        # Define the LSTM network with a dropout  
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=config.hiddendim_lstm, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(config.hidden_dropout_prob)

        # Redefinining the linear layers for classificaton, to account for different number of inputs
        self.dense = nn.Linear(config.hiddendim_lstm, config.hiddendim_lstm)
        self.classifier_X = nn.Linear(config.hiddendim_lstm, config.num_labels)

    @logged()
    def forward(self, hidden_states: torch.Tensor, hidden_states_all_layers: tuple[torch.Tensor], attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Combines embeddings for from all hidden layers/encoder units, using the selected method (e.g. mean, max or convolutionally encodded embeddings).
        Runs LSTM cell on the embeddings from the different layers.
        Generates logits.
        
        Parameters:
        ===========
        hidden_states: torch.Tensor. - Not Used!
        hidden_states_all_layers: tuple[torch.Tensor]. Hidden states for all layers/encoders. 
        attention_mask: torch.Tensor.
        
        **kwargs

        Returns:
        ========
        logits: torch.Tensor. Calculated logits. 

        """
        # In case of convolution input for LSTM
        if self.config.lstm_embedding_type == "conv":
            logger.debug("Hidden states calculated for Conv layer")
            hidden_states = torch.stack(
                [
                    self.conv_modules[layer_i-1].run_network(hidden_states=hidden_states_all_layers[layer_i], attention_mask=attention_mask).squeeze() 
                    for layer_i in range(1, self.num_hidden_layers+1)
                    ], 
                dim=-1,
                )

        else:
            hidden_states = torch.stack(
                # MeanPooler.return_mean_embeddings(hidden_states=hidden_states_all_layers[layer_i], attention_mask=attention_mask)
                [
                    self.embedding_extractor(hidden_states=hidden_states_all_layers[layer_i], attention_mask=attention_mask).squeeze() 
                    for layer_i in range(1, self.num_hidden_layers+1)
                    ], 
                dim=-1,
                )

        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        lstm_embeddings, _ = self.lstm(hidden_states, None)
        lstm_embeddings = self.dropout_lstm(lstm_embeddings[:, -1, :])

        logits = self.generate_logits_with_linear_nn(lstm_embeddings)
        return logits
