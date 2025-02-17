import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.bert.modeling_bert import (BertEmbeddings,
                                                    BertModel,
                                                    BertEncoder,
                                                    BertPooler,
                                                    BertOnlyMLMHead,
                                                    BertPreTrainedModel,
                                                    BertSelfOutput,
                                                    BertAttention,
                                                    BertSelfAttention,
                                                    BertLayer,
                                                    BertIntermediate,
                                                    BertOutput,
                                                    BertEncoder,
                                                    BertPredictionHeadTransform,
                                                    )
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging, ModelOutput
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)


@dataclass
class BaseModelOutputForSynthBertX(BaseModelOutputWithPastAndCrossAttentions):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.

        # TODO add docs for last_hidden_state_descriptors
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state_descriptors: Optional[Tuple[torch.FloatTensor]] = None


class CrossAttentDescriptorsPseudoEmbeddings(nn.Module):

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        super().__init__()

        self.cross_attention_number_of_descriptors = config.cross_attention_number_of_descriptors   
        self.cross_attention_max_num_cmpds = config.cross_attention_max_num_cmpds

        self.descriptor_pseudoembeddings = nn.Linear(self.cross_attention_number_of_descriptors, config.hidden_size)
        self.descriptor_position_embeddings = nn.Embedding(self.cross_attention_max_num_cmpds, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.register_buffer(
            "descriptor_position_ids", torch.arange(self.cross_attention_max_num_cmpds).expand((1, -1)), persistent=False
        )

    @logged()
    def forward(self, 
                descriptors: torch.FloatTensor = None,
                ) -> torch.Tensor:
        
        # resize the tensor with descriptors to have a vector of descriptors per molecule
        descriptors = descriptors.view(-1, self.cross_attention_max_num_cmpds, self.cross_attention_number_of_descriptors)
        descriptor_position_ids = self.descriptor_position_ids[:, :self.cross_attention_max_num_cmpds]

        pseudoembeddings = self.descriptor_pseudoembeddings(descriptors)

        if self.position_embedding_type == "absolute":
            pos_embeddings = self.descriptor_position_embeddings(descriptor_position_ids)
            pseudoembeddings += pos_embeddings

        pseudoembeddings = self.LayerNorm(pseudoembeddings)
        pseudoembeddings = self.dropout(pseudoembeddings)
        
        return pseudoembeddings


class BertCrossAttentionEncoder(BertEncoder):
    
    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        super().__init__(config)
        
        if config.is_decoder:
            raise ValueError(f"This model arrangement cannot be used as a decoder, " 
                             f"but the configuration was set to {config.is_decoder=}!!!")


        # rewrite the module list, to have the 
        module_list = [BertCrossAttentionLayer(config), ] + [BertLayer(config) for _ in range(config.num_hidden_layers-1)]
        self.layer = nn.ModuleList(module_list)

    @logged()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,

        # Added to facilitate the cross-attention for chemistry
        descriptors_hidden_states: Optional[torch.Tensor] = None,
        descriptors_attention_mask: Optional[torch.Tensor] = None,
        raw_descriptors_attention_mask: Optional[torch.Tensor] = None,
        raw_attention_mask: Optional[torch.Tensor] = None,

    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None


            # # Make sure that only the first layer runs cross-attention
            # if i != 0:
            #     encoder_hidden_states = None
            #     encoder_attention_mask = None


            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                
                layer_kwargs = {"hidden_states": hidden_states, "attention_mask": attention_mask, 
                                "head_mask": layer_head_mask, "encoder_hidden_states": encoder_hidden_states,
                                "encoder_attention_mask": encoder_attention_mask}
                if i == 0:  # For the first layer add descriptor information, to run the cross-attention between the descriptors and tokens
                    layer_kwargs.update({"hidden_states2": descriptors_hidden_states,
                                         "attention_mask2": descriptors_attention_mask, 
                                         "raw_attention_mask2": raw_descriptors_attention_mask,
                                         "raw_attention_mask": raw_attention_mask})

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    **layer_kwargs
                )
            else:
                layer_kwargs = {"hidden_states": hidden_states, "attention_mask": attention_mask, 
                                "head_mask": layer_head_mask, "encoder_hidden_states": encoder_hidden_states,
                                "encoder_attention_mask": encoder_attention_mask, "past_key_value": past_key_value,
                                "output_attentions": output_attentions}
                
                if i == 0:  # For the first layer add descriptor information, to run the cross-attention between the descriptors and tokens
                    layer_kwargs.update({"hidden_states2": descriptors_hidden_states,
                                         "attention_mask2": descriptors_attention_mask, 
                                         "raw_attention_mask2": raw_descriptors_attention_mask,
                                         "raw_attention_mask": raw_attention_mask})

                layer_outputs = layer_module(
                    **layer_kwargs
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    # all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            # cross_attentions=all_cross_attentions,
        )


class BertCrossAttentionLayer(BertLayer, nn.Module):

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        nn.Module.__init__(self)  # we only want to initiate nn.Module but not BertLayer - we only want methods of BertLayer

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.crossattention = BertCrossAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    @logged()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,

        # Added to facilitate the cross-attention for chemistry
        hidden_states2: Optional[torch.Tensor] = None,
        attention_mask2: Optional[torch.Tensor] = None,
        raw_attention_mask2: Optional[torch.Tensor] = None,
        raw_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        
        cross_attention_outputs = self.crossattention(
                hidden_states,
                attention_mask,
                head_mask,
                # encoder_hidden_states,
                # encoder_attention_mask,
                # cross_attn_past_key_value,
                output_attentions,

                hidden_states2,
                attention_mask2,
                raw_attention_mask2,
                raw_attention_mask,
            )

        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs

        return outputs



class BertCrossAttention(BertAttention, nn.Module):

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config, position_embedding_type=None):
        nn.Module.__init__(self)
        # super().__init__(config, position_embedding_type)

        self.cross_attention_heads =  BertCrossAttentionHeads(config, position_embedding_type)
        self.cross_attention_heads2 =  BertCrossAttentionHeads(config, position_embedding_type)

        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    @logged()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        # encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,

        # Added to facilitate the cross-attention for chemistry

        hidden_states2: Optional[torch.Tensor] = None,
        attention_mask2: Optional[torch.Tensor] = None,
        raw_attention_mask2: Optional[torch.Tensor] = None,
        raw_attention_mask: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor]:
        
        # ########################### Single Cross-Attention ############################ 
        # self_outputs = self.cross_attention_heads(
        #     input_for_query=hidden_states,
        #     attention_mask=attention_mask2,
        #     head_mask=head_mask,
        #     output_attentions=False,

        #     input_for_key_and_value=hidden_states2,
        #     attention_mask2=raw_attention_mask,
        # )
        # logger.info("Initialised Single Cross-Attention")
        # ###############################################################################


        ########################### Double Cross-Attention ############################ 
        self_outputs_cross_attention = self.cross_attention_heads(
            input_for_query=hidden_states,
            attention_mask=attention_mask2,
            head_mask=head_mask,
            output_attentions=False,

            input_for_key_and_value=hidden_states2,
            attention_mask2=raw_attention_mask,
        )

        # TODO check what happens when you swap the inputs to the query and key&value
        self_outputs = self.cross_attention_heads2(
            input_for_query=self_outputs_cross_attention[0],
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,

            input_for_key_and_value=hidden_states,
            # descriptors_attention_mask,
        )

        logger.info("Initialised Double Cross-Attention")
        ###############################################################################

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    

class BertCrossAttentionHeads(BertSelfAttention):
    
    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

    @logged()
    def forward(
        self,
        input_for_query: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        # encoder_hidden_states: Optional[torch.FloatTensor] = None, # for cross_attention
        # encoder_attention_mask: Optional[torch.FloatTensor] = None, # for cross_attention
        # past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,

        # Added to facilitate the cross-attention for chemistry
        input_for_key_and_value: Optional[torch.Tensor] = None,
        attention_mask2: Optional[torch.Tensor] = None,
        # descriptors_attention_mask: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor]:
        
        key_layer = self.transpose_for_scores(self.key(input_for_key_and_value))
        query_layer = self.transpose_for_scores(self.query(input_for_query)) 
        value_layer = self.transpose_for_scores(self.value(input_for_key_and_value))
        # attention_mask = encoder_attention_mask  # should we keep it????

        # # If this is instantiated as a cross-attention module, the keys
        # # and values come from an encoder; the attention mask needs to be
        # # such that the encoder's padding tokens are not attended to.
        # is_cross_attention = encoder_hidden_states is not None

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask which is precomputed for all layers in BertModel/EnrichedEmbeddingBertModel forward() function.
            attention_scores = attention_scores + attention_mask 


        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    
        if attention_mask2 is not None:
            # print("descriptors_attention_mask.size()", attention_mask2.size())
            attention_probs = attention_probs * attention_mask2

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # Concatanating heads together in the last dimension. 
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # print(context_layer.size())

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    

class SynthBertXEncoder(BertEncoder):

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        super().__init__(config)
        
        if config.is_decoder:
            raise ValueError(f"This model arrangement cannot be used as a decoder, " 
                             f"but the configuration was set to {config.is_decoder=}!!!")


        # rewrite the module list, to have the 
        module_list = [BertCrossAttentionLayer(config), ] + [BertLayer(config) for _ in range(config.num_hidden_layers-2)] + [BertCrossAttentionLayer(config), ]
        self.layer = nn.ModuleList(module_list)
         # This is for the "descriptors' path"
        self.cross_attention_layer_first = BertCrossAttentionLayer(config)
        if config.mlm:
            self.cross_attention_layer_last = BertCrossAttentionLayer(config)

    @logged()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,

        # Added to facilitate the cross-attention for chemistry
        descriptors_hidden_states: Optional[torch.Tensor] = None,
        descriptors_attention_mask: Optional[torch.Tensor] = None,
        raw_descriptors_attention_mask: Optional[torch.Tensor] = None,
        raw_attention_mask: Optional[torch.Tensor] = None,

    ) -> Union[Tuple[torch.Tensor], BaseModelOutputForSynthBertX]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):


            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                
                layer_kwargs = {"hidden_states": hidden_states, "attention_mask": attention_mask, 
                                "head_mask": layer_head_mask, "encoder_hidden_states": encoder_hidden_states,
                                "encoder_attention_mask": encoder_attention_mask}
                
                if i == 0 or i == len(self.layer)-1: # For the first or last layer add descriptor information, to run the cross-attention between the descriptors and tokens
                    layer_kwargs.update({"hidden_states2": descriptors_hidden_states,
                                         "attention_mask2": descriptors_attention_mask, 
                                         "raw_attention_mask2": raw_descriptors_attention_mask,
                                         "raw_attention_mask": raw_attention_mask})

                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module),
                                                                  **layer_kwargs
                                                                  )
                branch_layer_outputs = None

                # This is the "descriptors' path"
                if i == 0 or (i == len(self.layer)-1 and self.config.mlm): # For the first and the very last layer (last layer only if MLM)
                    
                    if i == 0:
                        cross_attention_layer = self.cross_attention_layer_first
                    else:
                        cross_attention_layer = self.cross_attention_layer_last
                    
                    layer_kwargs.update({"hidden_states": descriptors_hidden_states, 
                                         "attention_mask":descriptors_attention_mask,
                                         "head_mask": layer_head_mask, 
                                         "hidden_states2": hidden_states,
                                         "attention_mask2": attention_mask, 
                                         "raw_attention_mask2": raw_attention_mask,
                                         "raw_attention_mask": raw_descriptors_attention_mask,}) 
                    
                    branch_layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(cross_attention_layer),
                        **layer_kwargs
                    )


            else:
                layer_kwargs = {"hidden_states": hidden_states, "attention_mask": attention_mask, 
                                "head_mask": layer_head_mask, "encoder_hidden_states": encoder_hidden_states,
                                "encoder_attention_mask": encoder_attention_mask, "past_key_value": past_key_value,
                                "output_attentions": output_attentions}
                
                if i == 0 or i == len(self.layer)-1:  # For the first and last layer add descriptor information, to run the cross-attention between the descriptors and tokens
                    layer_kwargs.update({"hidden_states2": descriptors_hidden_states,
                                         "attention_mask2": descriptors_attention_mask, 
                                         "raw_attention_mask2": raw_descriptors_attention_mask,
                                         "raw_attention_mask": raw_attention_mask})

                layer_outputs = layer_module(**layer_kwargs)
                branch_layer_outputs = None

                if i == 0 or (i == len(self.layer)-1 and self.config.mlm): # For the first and the very last layer (last layer only if MLM)
                    # print("Running 2nd branch for descriptor cross-attention")
                    
                    if i == 0:
                        cross_attention_layer = self.cross_attention_layer_first
                    else:
                        cross_attention_layer = self.cross_attention_layer_last
                    
                    layer_kwargs.update({"hidden_states": descriptors_hidden_states, 
                                         "attention_mask":descriptors_attention_mask,
                                         "head_mask": layer_head_mask, 
                                         "hidden_states2": hidden_states,
                                         "attention_mask2": attention_mask, 
                                         "raw_attention_mask2": raw_attention_mask,
                                         "raw_attention_mask": raw_descriptors_attention_mask,}) 
                    
                    branch_layer_outputs = cross_attention_layer(**layer_kwargs)
                    # print(branch_layer_outputs)


            hidden_states = layer_outputs[0]
            if branch_layer_outputs:
                descriptors_hidden_states = branch_layer_outputs[0]
                # print("descriptors_hidden_states", descriptors_hidden_states)
                # print("descriptors_hidden_states.size()", descriptors_hidden_states.size())

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # Output attentions only for the token branch of calculations
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # if self.config.add_cross_attention:
                #     all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Output hidden states only for the token branch of calculations
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    descriptors_hidden_states,
                    # all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputForSynthBertX(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            last_hidden_state_descriptors=descriptors_hidden_states,
            # cross_attentions=all_cross_attentions,
        )


class SynthBertXEncoder_v2(BertEncoder):

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        super().__init__(config)
        
        if config.is_decoder:
            raise ValueError(f"This model arrangement cannot be used as a decoder, " 
                             f"but the configuration was set to {config.is_decoder=}!!!")


        # rewrite the module list, to have the 
        module_list = [BertCrossAttentionLayer(config), ] + [BertLayer(config) for _ in range(config.num_hidden_layers-1)]

        self.layer = nn.ModuleList(module_list)
        if config.mlm:
            self.cross_attention_layer_first = BertCrossAttentionLayer(config)
            self.cross_attention_layer_last = BertCrossAttentionLayer(config)

    @logged()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,

        # Added to facilitate the cross-attention for chemistry
        descriptors_hidden_states: Optional[torch.Tensor] = None,
        descriptors_attention_mask: Optional[torch.Tensor] = None,
        raw_descriptors_attention_mask: Optional[torch.Tensor] = None,
        raw_attention_mask: Optional[torch.Tensor] = None,

    ) -> Union[Tuple[torch.Tensor], BaseModelOutputForSynthBertX]:
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):


            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None


            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward
                
                layer_kwargs = {"hidden_states": hidden_states, "attention_mask": attention_mask, 
                                "head_mask": layer_head_mask, "encoder_hidden_states": encoder_hidden_states,
                                "encoder_attention_mask": encoder_attention_mask}
        

                if i == 0: # For the first layer add descriptor information, to run the cross-attention between the descriptors and tokens
                    layer_kwargs.update({"hidden_states2": descriptors_hidden_states,
                                         "attention_mask2": descriptors_attention_mask, 
                                         "raw_attention_mask2": raw_descriptors_attention_mask,
                                         "raw_attention_mask": raw_attention_mask})

                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module),
                                                                  **layer_kwargs
                                                                  )
                branch_layer_outputs = None

                if (i == 0 or i == len(self.layer)-1) and self.config.mlm: # For the first and the very last layer (but only if MLM)

                    if i == 0:
                        tokens_hidden_states = hidden_states
                        cross_attention_layer = self.cross_attention_layer_first
                    else:
                        tokens_hidden_states = layer_outputs[0]
                        cross_attention_layer = self.cross_attention_layer_last

                    
                    layer_kwargs.update({"hidden_states": descriptors_hidden_states, 
                                         "attention_mask":descriptors_attention_mask,
                                         "head_mask": layer_head_mask, 
                                         "hidden_states2": tokens_hidden_states,
                                         "attention_mask2": attention_mask, 
                                         "raw_attention_mask2": raw_attention_mask,
                                         "raw_attention_mask": raw_descriptors_attention_mask,}) 
                    
                    branch_layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(cross_attention_layer),
                        **layer_kwargs
                    )


            else:
                layer_kwargs = {"hidden_states": hidden_states, "attention_mask": attention_mask, 
                                "head_mask": layer_head_mask, "encoder_hidden_states": encoder_hidden_states,
                                "encoder_attention_mask": encoder_attention_mask, "past_key_value": past_key_value,
                                "output_attentions": output_attentions}

                if i == 0:  # For the first layer add descriptor information, to run the cross-attention between the descriptors and tokens
                    layer_kwargs.update({"hidden_states2": descriptors_hidden_states,
                                         "attention_mask2": descriptors_attention_mask, 
                                         "raw_attention_mask2": raw_descriptors_attention_mask,
                                         "raw_attention_mask": raw_attention_mask})


                layer_outputs = layer_module(**layer_kwargs)
                branch_layer_outputs = None

                if (i == 0 or i == len(self.layer)-1) and self.config.mlm: # For the first and the very last layer (but only if MLM)

                    if i == 0:
                        tokens_hidden_states = hidden_states
                        cross_attention_layer = self.cross_attention_layer_first
                    else:
                        tokens_hidden_states = layer_outputs[0]
                        cross_attention_layer = self.cross_attention_layer_last

                    # print("Running 2nd branch for descriptor cross-attention")
                    layer_kwargs.update({"hidden_states": descriptors_hidden_states, 
                                         "attention_mask":descriptors_attention_mask,
                                         "head_mask": layer_head_mask, 
                                         "hidden_states2": tokens_hidden_states,
                                         "attention_mask2": attention_mask, 
                                         "raw_attention_mask2": raw_attention_mask,
                                         "raw_attention_mask": raw_descriptors_attention_mask,}) 
                    
                    branch_layer_outputs = cross_attention_layer(**layer_kwargs)


            hidden_states = layer_outputs[0]
            if branch_layer_outputs:
                descriptors_hidden_states = branch_layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # Output attentions only for the token branch of calculations
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # Output hidden states only for the token branch of calculations
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    descriptors_hidden_states,
                    # all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputForSynthBertX(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            last_hidden_state_descriptors=descriptors_hidden_states,
            # cross_attentions=all_cross_attentions,
        )


class SynthBertXDescriptorLMPredictionHead(nn.Module):

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.cross_attention_number_of_descriptors, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.cross_attention_number_of_descriptors))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    @logged()
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
