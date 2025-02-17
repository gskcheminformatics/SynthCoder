import json
import random
from typing import Optional, Tuple, Union, List, Dict #, TYPE_CHECKING, Any, Dict, List, NamedTuple, Sequence
import logging
import torch
from transformers import __version__, BertTokenizerFast
from transformers.tokenization_utils_base import (BatchEncoding, TruncationStrategy)
from transformers.utils import PaddingStrategy

import synthcoder_project.utilities as utils
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

class EnrichedBertTokenizerFast(BertTokenizerFast):
    r"""
    Class inheriting from HuggingFace BertTokenizerFast.
    Allows to add an extra tokenization layer (enriching_ids) for chemical reactions as proposed in the BERT Enriched Embedding model - DOI: 10.1186/s13321-023-00685-0. 
    Converts `enriching_classes` provided as kwarg into `enriching_ids`.

    Construct a "fast" BERT tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Example:

    For reaction: N(C)C.O1CCCC1.c1(C=O)csc(Cl)n1>>c1(N(C)C)nc(C=O)cs1
    And the following kwargs:
        enriching_classes="[1, 4, 2]",
        molecule_separator=".", 
        reaction_arrow_symbol= ">>",
    
    We get:
    Tokens: ['[CLS]', 'N', '(', 'C', ')', 'C', '.', 'O', '1', 'C', 'C', 'C', 'C', '1', '.', 'c', '1', '(', 'C', '=', 'O', ')', 'c', 's', 'c', '(', 'Cl', ')', 'n', '1', '>>', 'c', '1', '(', 'N', '(', 'C', ')', 'C', ')', 'n', 'c', '(', 'C', '=', 'O', ')', 'c', 's', '1', '[SEP]']
    Token IDs:
    {'input_ids': [2, 25, 6, 21, 7, 21, 10, 26, 11, 21, 21, 21, 21, 11, 10, 32, 11, 6, 21, 17, 26, 7, 32, 38, 32, 6, 58, 7, 35, 11, 62, 32, 11, 6, 25, 6, 21, 7, 21, 7, 35, 32, 6, 21, 17, 26, 7, 32, 38, 11, 3], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    'enriching_ids': [0, 5, 5, 5, 5, 5, 1, 8, 8, 8, 8, 8, 8, 8, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0]}

    
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.

    Optional kwargs:
        model_args: Dict. 
            Dictionary containing neural network model arguments.
        
        enriching_classes: Str.
        
        descriptors: Str.

    """

    @logged()
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # Assign numerical enrichment IDs the constant reaction elements/symbols and special tokens.
        self.enrichment_special_token_id = 0
        self.enrichment_molecular_separator_id = 1
        self.enrichment_reaction_arrow_symbol_id = 2
        self.enrichment_reaction_product_id = 3

        # We need to know all special token IDs but without the ID for the unknown ([UNK]) token. 
        self.special_token_ids_no_unk = self.all_special_ids
        self.special_token_ids_no_unk.remove(self.unk_token_id)

        # Is `model_args` passed as a key word argument? If so, extract some values from it.
        model_args = kwargs.get("model_args", None)
        if model_args:
            logger.debug("Provided model arguments to the reaction tokenizer")
            self.molecule_separator=model_args.get("tokenizer_molecule_separator")
            self.reaction_arrow_symbol=model_args.get("tokenizer_reaction_arrow_symbol")

            # Find the token IDs (can be potentially more than one) corresponding 
            # to the molecuar separator symbol and the reaction arrow symbol.
            self.reaction_arrow_symbol_ids = self.find_token_ids(word=self.reaction_arrow_symbol)
            self.molecular_separator_ids = self.find_token_ids(word=self.molecule_separator)

            self.cross_attention_max_num_cmpds=model_args.get("cross_attention_max_num_cmpds")
            self.cross_attention_number_of_descriptors=model_args.get("cross_attention_number_of_descriptors")
            self.tokenizer_descriptors_padding_value=model_args.get("tokenizer_descriptors_padding_value")

    @logged()
    def _batch_encode_plus(self,  
                            batch_text_or_text_pairs: Union[
                                List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
                            ],
                            add_special_tokens: bool = True,
                            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
                            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
                            max_length: Optional[int] = None,
                            stride: int = 0,
                            is_split_into_words: bool = False,
                            pad_to_multiple_of: Optional[int] = None,
                            return_tensors: Optional[str] = None,
                            return_token_type_ids: Optional[bool] = None,
                            return_attention_mask: Optional[bool] = None,
                            return_overflowing_tokens: bool = False,
                            return_special_tokens_mask: bool = False,
                            return_offsets_mapping: bool = False,
                            return_length: bool = False,
                            verbose: bool = True,
                            **kwargs):
        """
        Extends the original method used by BertTokenizerFast.
        Allows to add an additional enrichment token layer to the standard token id layers. 
        """
        batch_encoding = super()._batch_encode_plus( batch_text_or_text_pairs,
            is_split_into_words=is_split_into_words,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            )
        
        if kwargs.get("enriching_classes", None):
            batch_encoding = self.add_enrichment_token_layer(batch_encoding, **kwargs)

        if kwargs.get("descriptors", None):
            batch_encoding = self.add_descriptor_data(batch_encoding, **kwargs)

        if kwargs.get("main_components_indices", None): 
            batch_encoding = self.add_main_rxn_components_mask(batch_encoding, **kwargs)
        
        return batch_encoding

    @logged()
    def find_token_ids(self, word: str) -> list:
        """
        Takes a word (a string) and tokenizes it into the corresponding token IDs 
        ([CLS] and [SEP] token IDs are NOT included in the returned list). 

        Parameters:
        ===========
        word: Str. The input word that should be converted into token IDs. 

        Returns:
        ========
        token_ids. List. List of token IDs.   
        """
        token_ids = self.encode(word)
        token_ids = [id for id in token_ids if id not in self.special_token_ids_no_unk]
        return token_ids

    @logged()
    def add_descriptor_data(self, sanitized_tokens: BatchEncoding, **kwargs) -> BatchEncoding:
        """
        The function takes the string which contains lists and converts it to a tensor.
        It then generates a mask for the descriptor tensor. 
        The tensors are padded to the correct size using the tokenizer. 
        The descriptor tensor is flattened. Batch encodings are generated.

        Parameters:
        ===========
        examples: Object. Data for encoding/processing. 

        Returns:
        ========
        BatchEncoding. Object holding   
        """
        descriptors = kwargs.get("descriptors", None)
        descriptors = utils.convert_string_list_to_tensor(descriptors)
        descriptor_mask = torch.ones(descriptors.size()[0])

        # # The information about the last dimension of the descriptor tensor is saved as one of the object arguments 
        # # and later should be sent to the model as part of the model config. This infomration is needed to recreate 
        # # the 2D tensor that will be flattened here and initiate the right NN layer size in the model 
        # number_of_descriptors = descriptors.size()[-1]

        # Pad the tensors to the correct size with zeros.
        descriptors =  self._adjust_descriptor_data_size(descriptors, padding_value=-1)
        descriptor_mask = self._adjust_descriptor_data_size(descriptor_mask, padding_value=0)

        # The descriptors need to be changed/resized from nested 2D tensor to 1D tensor.
        # Otherwise the tensor will give an error when processed by the tokenizer in prepration for the model.
        # The 2D tensor will need to be rebuilt by the collator and then again by the NN model.   
        descriptors = torch.flatten(descriptors)

        # Update the tokenizer's sanitized_tokens and model_input_names
        padded_tensors = {"descriptors": descriptors, "descriptors_attention_mask": descriptor_mask}
        sanitized_tokens.update(padded_tensors)

        for input_name in padded_tensors.keys():
            if input_name not in self.model_input_names:
                self.model_input_names.append(input_name)

        return sanitized_tokens

    @logged()
    def add_enrichment_token_layer(self, sanitized_tokens:  BatchEncoding, **kwargs) -> BatchEncoding:
        """
        Adds `enriching_ids` to the different generated ID types after tokenization.
        The `enrichin_ids` are gnerated based on the provided `enriching_classes`, 
        using tokens separating molecules from each other, as well as left and right side of teh chemical equation. 

        Parameters:
        ===========
        sanitized_tokens: BatchEncoding. Different token IDs after sanitazation. 
        **kwargs:
        
        Required kwargs:
        enriching_classes: Str of a list or List. Classes for conversion to `enriching_ids`. One class per molecule. E.g. "[2, 2, 6, 7, 1, 0, 3, 5]"

        Returns:
        ========
        sanitized_tokens: BatchEncoding. Sanitized, different token ids with added `enriching_ids`. 
        """
        enriching_classes = kwargs.get("enriching_classes", None)
        if isinstance(enriching_classes, str):
            enriching_classes = json.loads(enriching_classes)

        input_ids = sanitized_tokens.get("input_ids", [])

        enriching_ids = []
        for entry_ids in input_ids:
            entry_enriching_ids = []
            compound_idx = 0
            for id in entry_ids:
                if id in self.special_token_ids_no_unk:
                    entry_enriching_ids.append(self.enrichment_special_token_id)

                # elif id == molecular_separator_id:
                elif id in self.molecular_separator_ids:

                    entry_enriching_ids.append(self.enrichment_molecular_separator_id)
                    compound_idx += 1

                # elif id == reaction_arrow_symbol_id:
                elif id in self.reaction_arrow_symbol_ids:
                    entry_enriching_ids.append(self.enrichment_reaction_arrow_symbol_id)
                    compound_idx = -1

                elif compound_idx == -1:
                    entry_enriching_ids.append(self.enrichment_reaction_product_id)
                else:
                    entry_enriching_ids.append(enriching_classes[compound_idx]+4)  # +4 to account for the other ids assigned to special characters and the product
            enriching_ids.append(entry_enriching_ids)

        name_new_ids = "enriching_ids"

        sanitized_tokens[name_new_ids] = enriching_ids
        if name_new_ids not in self.model_input_names:
            self.model_input_names.append(name_new_ids)

        return sanitized_tokens
    
    @logged()
    def add_main_rxn_components_mask(self, sanitized_tokens:  BatchEncoding, **kwargs) -> BatchEncoding:
        """
        Creates a mask for reaction components and adds it to the returned `sanitized_tokens`.
        The main reaction components have their tokens assigned a mask of the corresponding compound index number in the scheme, 
        whereas, all other tokens are assigned a mask of -1. 

        Parameters:
        ===========
        sanitized_tokens: BatchEncoding. Different token IDs after sanitazation. 
        **kwargs
            Required kwargs:
            main_components_indices: Str of a list or List. `main_components_indices` should contain the compound indices of the main reaction components. 
        
        Returns:
        ========
        sanitized_tokens: BatchEncoding. Sanitized, different token ids with added `main_rxn_component_mask`. 
        """
        main_components_indices = kwargs.get("main_components_indices", None)

        if isinstance(main_components_indices, str):
            main_components_indices = json.loads(main_components_indices)  # convert a string of a list to a list

        # idx_selected_component = random.choice(main_components_indices)  # randomly select an index of the main reaction component      
        input_ids = sanitized_tokens.get("input_ids", [])

        mask = []
        for entry_ids in input_ids:
            mask_ids = []
            compound_idx = 0
            arrow_symbol = False

            for id in entry_ids:
                if id in self.special_token_ids_no_unk:  # special tokens are not masked 
                    mask_ids.append(-1)
                elif id in self.molecular_separator_ids:  # if there is a molecular separator symbol, advance the molecule count by 1  
                    mask_ids.append(-1)
                    compound_idx += 1
                elif id in self.reaction_arrow_symbol_ids:
                    mask_ids.append(-1)
                    if not arrow_symbol:  # this is in case there are more than one arrow symbols one after the other
                        compound_idx += 1
                        arrow_symbol = True

                elif compound_idx in main_components_indices:
                    # Assign 1 to the selected main reaction component in the mask
                    mask_ids.append(compound_idx)  
                else:
                    # For any other compound tokens assign 0 in the mask
                    mask_ids.append(-1) 
                
            mask.append(mask_ids)

        name_new_ids = "main_rxn_component_mask"

        # Update sanitized tokens with the generated mask, and add the name of the created mask data to the model input names 
        sanitized_tokens[name_new_ids] = mask
        if name_new_ids not in self.model_input_names:
            self.model_input_names.append(name_new_ids)

        return sanitized_tokens

    @logged()
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Overwrites the original method for PreTrainedTokenizerBase class.
        Adds an option for padding for the enrichment token IDs, original input IDs and main reaction component mask. 

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                
                ######### Added for chemistry #########
                if "enriching_ids" in encoded_inputs:
                    logger.debug("Padding enriching IDs")
                    encoded_inputs["enriching_ids"] = (
                        encoded_inputs["enriching_ids"] + [self.enrichment_special_token_id] * difference
                    )

                if "original_input_ids" in encoded_inputs:
                    logger.debug("Padding original input IDs")
                    encoded_inputs["original_input_ids"] = (
                        encoded_inputs["original_input_ids"] + [self.pad_token_id] * difference
                    )

                if "main_rxn_component_mask" in encoded_inputs:
                        logger.debug("Padding main rxn component mask")
                        encoded_inputs["main_rxn_component_mask"] = (
                        encoded_inputs["main_rxn_component_mask"] + [0] * difference
                    )
                #######################################

                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference

            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs["token_type_ids"]

                ######### Added for chemistry #########
                if "enriching_ids" in encoded_inputs:
                    logger.debug("Padding enriching IDs")
                    encoded_inputs["enriching_ids"] = [self.enrichment_special_token_id] * difference + encoded_inputs["enriching_ids"]

                if "original_input_ids" in encoded_inputs:
                    logger.debug("Padding original input IDs")
                    encoded_inputs["original_input_ids"] = [self.pad_token_id] * difference + encoded_inputs["original_input_ids"]

                if "main_rxn_component_mask" in encoded_inputs:
                    logger.debug("Padding main rxn component mask")
                    encoded_inputs["main_rxn_component_mask"] = [0] * difference + encoded_inputs["main_rxn_component_mask"]
                #######################################

                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        
        return encoded_inputs

    @logged()
    def _adjust_descriptor_data_size(self, input_tensor: torch.Tensor, padding_value: Union[int, float]) -> torch.Tensor:
        """
        Changes the size of the input tensor to the size in dimension 0 defined by `self.cross_attention_max_num_cmpds`.

        The tensors that are smaller in dimension 0 than the size indicated in `self.cross_attention_max_num_cmpds`, are padded with zeros.
        The tensor that is larger than `self.cross_attention_max_num_cmpds` in dimension 0 will be truncated.
        If the tensor size in dimension 0 is equal to `self.cross_attention_max_num_cmpds`, no changes are made.

        Tensor of size [n, m, z] will be transformed to size [`self.cross_attention_max_num_cmpds`, m, z].

        Parameters:
        ===========
        input_tensor: torch.Tensor. Input tensor whose size should be adjusted/padded. 
        paddin_value: int or float. The value to be used for padding.
        
        Returns:
        ========
        torch.Tensor. Resulting tensor.

        """
        size_difference = self.cross_attention_max_num_cmpds - input_tensor.size()[0]

        if size_difference > 0: # we need to add padding in the first dimension.

            # Padding with specifiec value
            padding_tensor = torch.full((size_difference, *input_tensor.size()[1:]), padding_value)

            output_tensor = torch.cat((input_tensor, padding_tensor), dim=0)
        elif size_difference < 0: # we need to truncate data in the first dimension.           
            if input_tensor.dim() >= 2: # if there are 2 or more dimensions in the tensor
                output_tensor = input_tensor[:self.cross_attention_max_num_cmpds, :]
            else: 
                output_tensor = input_tensor[:self.cross_attention_max_num_cmpds]
            
        else: # size matches the max number of compounds/rows.
            output_tensor = input_tensor

        return output_tensor
