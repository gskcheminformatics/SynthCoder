from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import random
import logging
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import (Mapping, _torch_collate_batch)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import synthcoder_project.utilities as utils
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)


class SynthBertXCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    This class has exactly the same functionality as the parent class `DataCollatorForLanguageModeling`
    but it can additionally generate a MLM mask for descriptors and mask tokens corresponding to one specific molecule in a reaction.

    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerBase,
                 mlm: bool = True,
                 mlm_probability: float = 0.15,
                 pad_to_multiple_of: Optional[int] = None,
                 tf_experimental_compile: bool = False,
                 return_tensors: str = "pt",
                 mlm_descriptors: bool = False,
                 mlm_span_rxn_probability: float = 0.0,
                 ) -> None:
        
        """
        Custom child class of HuggingFace's `DataCollatorForLanguageModeling`.
        Remember that dynamic masking will be applied (new mask will be generated for each epoch for each example). 

        Added custom parameters:
        ========================
        mlm_descriptors: <Optional> Bool. If `True`, it will prepare `descriptors` and `labels_descriptors` for descriptor based, douple task MLM. 
        mlm_span_rxn_probability: <Optional> Float. Probability of performing span masking of tokens for one of the main reaction components. 
            When 0.0 no span masking will be performed, when 1.0, all (reaction) examples will be masked using the span approach. If e.g. this 
            arguent is set to 0.3, 30% of the examples will be masked following the span masking approach (all tokens of one of the main reaction 
            components - selected at random - will be masked, and no other modifications to the remaining tokens will be made), whereas, the remaining 
            70% of the examples will be masked using the convensional Language Model masking approach (on the selected tokens with `self.mlm_probability` 
            gives 80% MASK, 10% random, 10% original).  

        Returns:
        ========
        None
        """
        
        super().__init__(tokenizer=tokenizer,
                         mlm=mlm,
                         mlm_probability=mlm_probability,
                         pad_to_multiple_of=pad_to_multiple_of,
                         tf_experimental_compile=tf_experimental_compile,
                         return_tensors=return_tensors,)
        
        self.mlm_descriptors = mlm_descriptors
        self.mlm_span_rxn_probability = mlm_span_rxn_probability

    @logged()
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:

            if batch.get("main_rxn_component_mask", None) is not None: 
                # This is only for when information about the main reaction components is provided.
                # With specified probability for a given example, all tokens of one of the main reaction components are completely masked,
                # otherwise, the standard masking protocol is performed.
                # This is to force the network to predict all tokens of one of the main 
                # reaction components (reactant/product) based on the other reaction components.   
                logger.debug(f"Mask for main reaction component tokens provided. Performing SpanRxn MLM masking with probability {self.mlm_span_rxn_probability=}")

                batch["input_ids"], batch["labels"] = self.torch_mixed_mask_tokens(inputs=batch["input_ids"],
                                                                                   main_rxn_component_mask=batch["main_rxn_component_mask"],
                                                                                   special_tokens_mask=special_tokens_mask,
                                                                                   mlm_span_rxn_probability=self.mlm_span_rxn_probability,
                                                                                   )
                # Remove `main_rxn_component_mask` as it's no longer needed.
                del batch["main_rxn_component_mask"]

            else:  # This is the default masking approach for MLM.
                logger.debug("Using standard MLM masking approach")

                batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )  

        else:
            logger.debug("No MLM masking is performed")

            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        
        if self.mlm_descriptors:  # Prepare masked descriptor tensors with the corresponding labels. 
            logger.debug("Masking descriptor data")

            batch["descriptors"], batch["labels_descriptors"] = self.torch_mask_descriptors(
                batch["descriptors"], batch["descriptors_attention_mask"]
            )

        return batch
   
    @logged()
    def torch_mask_descriptors(self, inputs: Any, descriptors_attention_mask: Any) -> Tuple[Any, Any]:
        """
        Prepares masked descriptor inputs and the corresponding labels for masked language modeling.
        Only one of the descriptor vectors is masked (descriptors corresponding to one molecule).
        Descriptor tensors are "unflattened" from "1D" to "2D" (to max num compounds x number of descriptors).
        Non-padded rows in the tensors are found. One of the rows (descriptor vector for one of the molecules) in modified `inputs`
        is selected at random and replaced with a pre-determined numerical padding value `tokenizer_descriptors_padding_value`.  
        Values corresponding to the non-masked descriptors in the label tensor are replaced with np.inf.  

        The tensors of both the modified input and the labels are converted from "2D" to "1D", otherwise they cannot be processed 
        properly by the used libraries.

        Returns masked descriptor input tensor and the corresponding labels.

        Parameters:
        ===========
        inputs:  Any. The input data.
        descriptors_attention_mask: Any. The attention mask for descriptors.
        
        Returns:
        ========
        inputs, labels: Tuple[Any, Any]: The masked inputs and the corresponding labels.
        """
        import torch
        import random
        import numpy as np

        # "Unflatten" the tensor
        inputs = inputs.view(-1, self.tokenizer.cross_attention_max_num_cmpds, self.tokenizer.cross_attention_number_of_descriptors)

        descriptor_labels = inputs.clone()
        descriptors_labels_mask = inputs.clone()

        # Figure out which positions are not "padding"
        non_zero_mask_elements = torch.nonzero(descriptors_attention_mask, as_tuple=True) 
        # Count the number of non-pading rows in each reaction string input
        counts = torch.nn.functional.one_hot(non_zero_mask_elements[0]).sum(dim=0)

        # Create a mask where one of the non-padding vector of descriptors will be selected at random to be masked
        j = torch.arange(self.tokenizer.cross_attention_number_of_descriptors).long()
        for num_in_batch, i in enumerate(counts):
            random_position = random.randint(0, i-1) # select one of the non-padding descriptor vectors for masking for MLM
            descriptors_labels_mask[num_in_batch, random_position, j] = np.inf # temporarily replace values in the mask with inf
        
        # Convert the inf and non-inf values in the mask to bool. 
        descriptors_labels_mask = descriptors_labels_mask.isinf() #.logical_not()
        # Replace the real selected desriptor values in the input with a padding/masking value 
        inputs[descriptors_labels_mask] = self.tokenizer.tokenizer_descriptors_padding_value #-1
        # Replace the non-masked descriptors with inf. 
        descriptor_labels[~descriptors_labels_mask] = np.inf

        # Flatten the tensors again
        inputs = torch.flatten(inputs)
        descriptor_labels = torch.flatten(descriptor_labels)

        return inputs, descriptor_labels

    @logged()
    def torch_mask_main_rxn_tokens(self, inputs: Any, main_rxn_component_mask: Any) -> Tuple[Any, Any]:
        """
        Prepares masked token inputs and the corresponding labels for masked language modeling, where all masked tokens 
        correspond to only one compound - one of the main reaction components. The masking of the input and labels is 
        done based on the provided `main_rxn_component_mask` tensor, indicating token positions of the main reaction components. 
        This function allows for a dynamic masking.
        
        Parameters:
        ===========
        inputs: Any. The input data.
        main_rxn_component_mask: Any: The mask for the main reaction components.
        
        Returns:
        ========
        inputs, labels: Tuple[Any, Any]: The masked inputs and the corresponding labels.
        """
        import torch
        
        labels = inputs.clone()

        # Check if the input tensors are empty
        if labels.numel() == 0 or main_rxn_component_mask.numel() == 0:
            # If inputs or mask is empty, return the inputs and labels unchanged
            return inputs, labels
        
        selected_cmpd_idxs = []
        for vector in main_rxn_component_mask:
            # Extract unique compound indices (where each index marks a separate main component compound), excluding value `-1` at index 0
            unique_cmpd_idxs = torch.unique(vector)[1:].flatten()

            # Randomly select an index from unique compound indices
            random_index = torch.randint(0, len(unique_cmpd_idxs), (1,))
            selected_cmpd_idxs.append(unique_cmpd_idxs[random_index])

        selected_cmpd_idxs = torch.stack(selected_cmpd_idxs) # Convert list to a tensor
        assert selected_cmpd_idxs.size()[0] == main_rxn_component_mask.size()[0] # Ensure sizes match

        # Broadcast tensor `selected_cmpd_idxs`` to the shape of tensor `main_rxn_component_mask` and compare
        main_rxn_component_mask = main_rxn_component_mask == selected_cmpd_idxs

        # Labels for all but masked tokens will have value of -100.
        labels[~main_rxn_component_mask] = -100  # We only compute loss on masked tokens

        # Mask the selected tokens for one compound (one of the main reaction components)
        inputs[main_rxn_component_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

    @logged()
    def torch_mixed_mask_tokens(self, 
                                inputs: Any, 
                                main_rxn_component_mask: Any, 
                                special_tokens_mask: Optional[Any]=None, 
                                mlm_span_rxn_probability: float=0.0) -> Tuple[Any, Any]:
        """
        This function prepares masked tokens inputs and labels for masked language modeling. It is meant to be used with reaction-based data.
        It uses two methods for masking:
        1. self.torch_mask_main_rxn_tokens: Used for a percentage of input examples specified by mlm_span_rxn_probability.
        2. self.torch_mask_tokens: Used for the remaining examples.
        
        Parameters:
        ===========
        inputs:  Any. The input data.
        main_rxn_component_mask: Any. The mask for the main reaction component.
        special_tokens_mask: <Optional> Any. The mask for the special tokens. Default is None.
        mlm_span_rxn_probability: <Optional> Float. The probability of a row being processed by torch_mask_main_rxn_tokens. 
        
        Returns:
        ========
        inputs, labels: Tuple[Any, Any]: The masked inputs and the corresponding labels. The resulting tensors keep the order of the input tensors.
        """

        import torch

        # Generate a mask to decide which rows will be processed by which function. The mask for the function selection is a tensor of booleans with the same number of elements as the shape of the input in the first dimension.
        function_mask = torch.bernoulli(torch.full((inputs.shape[0],), mlm_span_rxn_probability)).bool()

        # Indices of examples for span masking and standard masking
        main_indices = torch.where(function_mask)[0]
        mixed_indices = torch.where(~function_mask)[0]

        # Process selected rows by torch_mask_main_rxn_tokens
        main_inputs = inputs[main_indices].clone()
        main_mask = main_rxn_component_mask[main_indices].clone()
        main_inputs, main_labels = self.torch_mask_main_rxn_tokens(main_inputs, main_mask)

        # Process remaining rows by the standard torch_mask_tokens method
        mixed_inputs = inputs[mixed_indices].clone()

        # If there are no more vectors to mask, just return the current inputs and labels, otherwise, perform the standard MLM masking on the remaining vectors.  
        if mixed_inputs.shape[0] == 0:
            logger.debug("No vectors to mask for the standard MLM approach")
            return main_inputs, main_labels
        
        elif main_inputs.shape[0] == 0:
            logger.debug("No vectors to mask for the SpanRxn MLM approach")

        # Mask remaining examples using torch_mask_tokens
        mixed_special_tokens_mask = special_tokens_mask[mixed_indices].clone() if special_tokens_mask is not None else None
        mixed_inputs, mixed_labels = self.torch_mask_tokens(mixed_inputs, mixed_special_tokens_mask)

        # Initialize empty tensors for original order
        final_inputs = torch.empty_like(inputs)
        final_labels = torch.empty_like(inputs)

        # Place the masked results back into their original positions
        final_inputs[main_indices] = main_inputs
        final_labels[main_indices] = main_labels
        final_inputs[mixed_indices] = mixed_inputs
        final_labels[mixed_indices] = mixed_labels

        return final_inputs, final_labels
