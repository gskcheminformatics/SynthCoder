# All models are based on/inheriting from the HuggingFace transformers models.

from transformers import (DistilBertForMaskedLM,
                          DistilBertForSequenceClassification,
                          BertForMaskedLM,
                          BertForSequenceClassification,
                          )
from transformers.models.bert.modeling_bert import (BertEmbeddings,
                                                    BertModel,
                                                    BertEncoder,
                                                    BertPooler,
                                                    BertOnlyMLMHead,
                                                    BertPreTrainedModel,
                                                    )
from transformers.modeling_outputs import (MaskedLMOutput, 
                                           BaseModelOutputWithPoolingAndCrossAttentions,
                                           SequenceClassifierOutput)
from transformers.utils import logging, ModelOutput
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
# import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses  import dataclass

# import synthcoder_project.encoders.bert_poolers as poolers
from  synthcoder_project.synthcoder_config import AVAILABLE_POOLERS

from synthcoder_project.encoders.bert_cross_attention import BertCrossAttentionEncoder, CrossAttentDescriptorsPseudoEmbeddings, SynthBertXEncoder, SynthBertXEncoder_v2, SynthBertXDescriptorLMPredictionHead
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)


@dataclass
class SynthBertXOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
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
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.

        #TODO complete docs
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state_descriptors: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SynthBertXMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        #TODO complete docs
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_tokens_mlm: Optional[torch.FloatTensor] = None
    loss_descriptors_mlm: Optional[torch.FloatTensor] = None


class SynthBertXForMaskedLM(BertForMaskedLM):

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # self.bert = EnrichedEmbeddingBertModel(config, add_pooling_layer=False)
        self.bert = AdversarialEnrichedEmbeddingBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        if config.cross_attention_use_extended_descript_network:
            self.descriptor_regress = SynthBertXDescriptorLMPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @logged()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        enriching_ids: Optional[torch.Tensor] = None,  #added for chemistry
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # Added to facilitate the cross-attention for chemistry
        descriptors: Optional[torch.Tensor] = None,
        descriptors_attention_mask: Optional[torch.Tensor] = None,
        labels_descriptors: Optional[torch.Tensor] = None,

        # Adversarial parameters
        adversarial_embedding_output: Optional[torch.Tensor] = None,
        adversarial_descriptors_hidden_states: Optional[torch.Tensor] = None,

    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Run the BERT-based model to generate outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            enriching_ids=enriching_ids,  # added for chemistry
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            # Added to facilitate the cross-attention for chemistry
            descriptors=descriptors,
            descriptors_attention_mask=descriptors_attention_mask,

            # Adversarial parameters
            adversarial_embedding_output=adversarial_embedding_output,
            adversarial_descriptors_hidden_states=adversarial_descriptors_hidden_states,
        )

        # Acess the last hidden state genrated by the model
        sequence_output = outputs[0]
        # Run the classification
        prediction_scores = self.cls(sequence_output)

        ######### Calculate the Masked Language loss for "word" tokens #########
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            logger.debug("Calculating loss for MLM task for 'word' tokens")
        ########################################################################

        ############## Calculate the loss for masked descriptors ###############
        masked_descriptor_loss = 0
        if labels_descriptors is not None and self.config.cross_attention_use_extended_descript_network:
            descriptors_output = outputs[-1] 
            descriptors_prediction_scores = self.descriptor_regress(descriptors_output)

            # "Unflatten the descriptor labels"
            labels_descriptors = labels_descriptors.view(-1, self.config.cross_attention_max_num_cmpds, self.config.cross_attention_number_of_descriptors)
            labels_descriptors_mask = labels_descriptors.isinf().logical_not()
            labels_descriptors = labels_descriptors[torch.all(labels_descriptors_mask, dim=-1)]
            descriptors_prediction_scores = descriptors_prediction_scores[torch.all(labels_descriptors_mask, dim=-1)]

            masked_descriptor_loss_fct =  MSELoss()
            if self.config.cross_attention_number_of_descriptors == 1:
                masked_descriptor_loss = masked_descriptor_loss_fct(descriptors_prediction_scores.squeeze(), labels_descriptors.squeeze())
            else:
                masked_descriptor_loss = masked_descriptor_loss_fct(descriptors_prediction_scores, labels_descriptors)
            logger.debug("Calculating loss for MLM task for descriptors")
            
        ########################################################################

        # Calculate the final total loss
        if masked_lm_loss:
            total_loss = masked_lm_loss + self.config.delta_descriptors_loss * masked_descriptor_loss
        else:
            total_loss = None
    
        # Return the outputs
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output + (masked_lm_loss, masked_descriptor_loss)) if total_loss is not None else output

        return SynthBertXMaskedLMOutput(
            loss=total_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_tokens_mlm=masked_lm_loss,
            loss_descriptors_mlm=masked_descriptor_loss,
        )


class EnrichedEmbeddingBertForMaskedLM(BertForMaskedLM):
    """
    To be decomissioned and replaced by SynthBertXForMaskedLM!!!
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # self.bert = EnrichedEmbeddingBertModel(config, add_pooling_layer=False)
        self.bert = AdversarialEnrichedEmbeddingBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @logged()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        enriching_ids: Optional[torch.Tensor] = None,  #added for chemistry
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # Added to facilitate the cross-attention for chemistry
        descriptors: Optional[torch.Tensor] = None,
        descriptors_attention_mask: Optional[torch.Tensor] = None,

        # Adversarial parameters
        adversarial_embedding_output: Optional[torch.Tensor] = None,
        adversarial_descriptors_hidden_states: Optional[torch.Tensor] = None,

    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""

        # TODO update the description of `encoder_hidden_states` and `encoder_attention_mask` 

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            enriching_ids=enriching_ids,  # added for chemistry
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            # Added to facilitate the cross-attention for chemistry
            descriptors=descriptors,
            descriptors_attention_mask=descriptors_attention_mask,

            # Adversarial parameters
            adversarial_embedding_output=adversarial_embedding_output,
            adversarial_descriptors_hidden_states=adversarial_descriptors_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            logger.debug("Calculating loss for MLM task for 'word' tokens")

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EnrichedEmbeddingBertForSequenceClassification(BertForSequenceClassification, BertPreTrainedModel):

    """
    Modified class in respect to the original BertForSequenceClassification:
        - It uses EnrichedEmbeddingBertModel class as BERT model
        - It does not process the outputs of pooler anymore - it does not have an explicit classifier, as the pooler itself includes the classification layer. 

        # TODO update the description of `encoder_hidden_states` and `encoder_attention_mask` 
     
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config):
        super(BertPreTrainedModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.bert = EnrichedEmbeddingBertModel(config)
        self.bert = AdversarialEnrichedEmbeddingBertModel(config)


        # Initialize weights and apply final processing
        self.post_init()

    @logged()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        enriching_ids: Optional[torch.Tensor] = None,  # added for chemsitry
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # Added to facilitate the cross-attention for chemistry
        descriptors: Optional[torch.Tensor] = None,
        descriptors_attention_mask: Optional[torch.Tensor] = None,

        # Adversarial parameters
        adversarial_embedding_output: Optional[torch.Tensor] = None,
        adversarial_descriptors_hidden_states: Optional[torch.Tensor] = None,

    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""

        # TODO update the description of `encoder_hidden_states` and `encoder_attention_mask` 

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            enriching_ids=enriching_ids,  # added for chemistry
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            # Added to facilitate the cross-attention for chemistry
            descriptors=descriptors,
            descriptors_attention_mask=descriptors_attention_mask,

            # Adversarial parameters
            adversarial_embedding_output=adversarial_embedding_output,
            adversarial_descriptors_hidden_states=adversarial_descriptors_hidden_states,
        )

        logits = outputs[1]  # outputs[1] is the output of the pooler

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AdversarialEnrichedEmbeddingBertModel(BertModel, BertPreTrainedModel):
    """
    Highly customised EmbeddingBertModel.
    Class with an ability for adversarial training done done on the embeddings, through SynthCoderModeling. 
    The class also allows for cross-attention based training using secondary input of numeical descriptors.
    It also allows for processing of 'enriching_ids' that are processed and added to the embeddings. 
    
    It performs embedding, runs encoder stack and then it optionally runs a pooler/classification. 
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config, add_pooling_layer=True):
        # super().__init__(config)
        BertPreTrainedModel.__init__(self, config)
        self.config = config
        self.embeddings = EnrichedBertEmbeddings(config)  # This is a custom layer. 
        
        # These variables are needed for optional adversarial training
        self.embedding_output = None
        self.descriptors_hidden_states = None

        ###### Determine which encoder stack should be used ###### 
        try:
            self.cross_attention_encoder_input = config.cross_attention_encoder_input
        except AttributeError as e:
            self.cross_attention_encoder_input = False

        if self.cross_attention_encoder_input:
            print("Using cross-attention input encoder variant\n")
            logger.info("Using cross-attention input encoder variant")
            if config.cross_attention_use_extended_descript_network:
                print(f"Using extended descriptor network")
                self.encoder = SynthBertXEncoder(config)
                # self.encoder = SynthBertXEncoder_v2(config)
            else:
                self.encoder = BertCrossAttentionEncoder(config)
            
            # We also need to initialise the pseudo-embedder for the descriptors/tensors 
            # that will be used in the custom cross-attention 
            self.cross_attention_embeddings = CrossAttentDescriptorsPseudoEmbeddings(config)
        else:
            self.encoder = BertEncoder(config)
        print("Using Encoder Class:", self.encoder.__class__.__name__)
        logger.info(f"Using Encoder Class: {self.encoder.__class__.__name__}")

        ########### Allow to choose the desired pooler ###########
        try:
            print("Pooler Type:", config.pooler_type)
            self.pooler = AVAILABLE_POOLERS[config.pooler_type](config) if add_pooling_layer else None
        except AttributeError as e:
            print(e)
            self.pooler = AVAILABLE_POOLERS["default"](config) if add_pooling_layer else None
            
        print("Using Pooler Class:", self.pooler.__class__.__name__, "\n")
        logger.info(f"Using Pooler Class: {self.pooler.__class__.__name__}")

        ##########################################################

        # Initialize weights and apply final processing
        self.post_init()

    @logged()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        enriching_ids: Optional[torch.Tensor] = None,  # added for chemistry
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # Parameters added to facilitate the cross-attention for chemistry
        descriptors: Optional[torch.Tensor] = None,
        descriptors_attention_mask: Optional[torch.Tensor] = None,

        # Adversarial parameters
        adversarial_embedding_output: Optional[torch.Tensor] = None,
        adversarial_descriptors_hidden_states: Optional[torch.Tensor] = None,


    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""

        # TODO update the description of `encoder_hidden_states` and `encoder_attention_mask` 

        # encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        #     Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
        #     the model is configured as a decoder.
        # encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
        #     Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
        #     the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

        #     - 1 for tokens that are **not masked**,
        #     - 0 for tokens that are **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        

        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        
        #  ConcatPooler and LSTMPooler require hidden_states
        if self.pooler.__class__.__name__ in ("ConcatPooler", "LSTMPooler"):
            output_hidden_states = True  

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            logger.error("You cannot specify both input_ids and inputs_embeds at the same time")
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            logger.error("You have to specify either input_ids or inputs_embeds")
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        #####################  Added for chemistry  ####################
        if enriching_ids is None:
            if hasattr(self.embeddings, "enriching_ids"):
                buffered_enriching_ids = self.embeddings.enriching_ids[:, :seq_length]
                buffered_enriching_ids_expanded = buffered_enriching_ids.expand(batch_size, seq_length)
                enriching_ids = buffered_enriching_ids_expanded
            else:
                logger.debug("Setting `enriching_ids` to a tensor of 0s")
                enriching_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        ################################################################


        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask=attention_mask, input_shape=input_shape)  # This converts 1s and 0s to max and min values for later masking with softmax
        

        if descriptors_attention_mask is not None:

            # For cross-attention with descriptors we need to use the raw/unprocessed attention mask
            attention_mask = attention_mask.view(-1, 1, attention_mask.size()[-1], 1)
            raw_attention_mask = attention_mask.expand(-1, -1, -1, descriptors_attention_mask.size()[-1])

            extended_descriptors_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask=descriptors_attention_mask, 
                                                                                                 input_shape=(descriptors_attention_mask.size(-2), descriptors_attention_mask.size(-1)))  # This converts 1s and 0s to max and min values for later masking with softmax
            # We also need raw/unprocessed attention mask for descriptors
            descriptors_attention_mask = descriptors_attention_mask.view(-1, 1, descriptors_attention_mask.size()[-1], 1)
            raw_descriptors_attention_mask = descriptors_attention_mask.expand(-1, -1, -1, input_shape[-1])


        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        ######## Calculate Embeddings ########
        # If adversarial_embedding_output is not provided, calculate embedding_output
        if not isinstance(adversarial_embedding_output, torch.Tensor):
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                enriching_ids=enriching_ids,  # added for chemistry
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
            logger.debug("Calculating new `self.embedding_output` based on the provided inputs")
        else:
            embedding_output = adversarial_embedding_output
            logger.debug("Model provided with `adversarial_embedding_output` - using this tensor as `self.embedding_output`.")
        self.embedding_output = embedding_output  # this is necessary for adversarial training through SynthCoder 
        #######################################

        # Define kwargs to be fed to the encoder
        encoder_kwargs = {
            "hidden_states": embedding_output,
            "attention_mask": extended_attention_mask,
            "head_mask": head_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_extended_attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }

        if self.cross_attention_encoder_input:
            logger.debug(f"Calculating embeddings for descriptor input")

            ######## Calculate Embeddings for Molecular Descriptors ########
            # If adversarial_descriptors_hidden_states is not provided, calculate descriptors_hidden_states
            if not isinstance(adversarial_descriptors_hidden_states, torch.Tensor):
                descriptors_hidden_states = self.cross_attention_embeddings(descriptors=descriptors)
            else:
                descriptors_hidden_states = adversarial_descriptors_hidden_states
            self.descriptors_hidden_states = descriptors_hidden_states   # this is necessary for adversarial training through SynthCoder 
            ###############################################################

            # Update the kwargs that will be fed to the encoder with the data necessary for the cross-attention with descriptors
            encoder_kwargs.update({"descriptors_hidden_states": descriptors_hidden_states,
                                   "descriptors_attention_mask": extended_descriptors_attention_mask,
                                   "raw_descriptors_attention_mask": raw_descriptors_attention_mask,
                                   "raw_attention_mask": raw_attention_mask,})
            
        # Run the encoder
        encoder_outputs = self.encoder(**encoder_kwargs)
        sequence_output = encoder_outputs[0]  # This is the last hidden state of the encoder

        # Pooler performs classification. `pooler_output` is logits after classification. 
        pooled_output = self.pooler(hidden_states=sequence_output, attention_mask=attention_mask, hidden_states_all_layers=encoder_outputs.hidden_states) if self.pooler is not None else None


        ############### Return the outputs ###############
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        if self.config.cross_attention_use_extended_descript_network:
            logger.debug(f"Returning initialised `SynthBertXOutputWithPoolingAndCrossAttentions` object")
            
            return SynthBertXOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
                # last_hidden_state_descriptors=encoder_outputs[-1],
                last_hidden_state_descriptors=encoder_outputs.last_hidden_state_descriptors,
            )

        logger.debug(f"Returning initialised `BaseModelOutputWithPoolingAndCrossAttentions` object")
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        ##################################################


class EnrichedBertEmbeddings(BertEmbeddings):
    """
    Construct the embeddings from word, position and token_type embeddings.
    Allows to add enriching_ids that are processed and added to the resulting embeddings.
    """
    
    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self, config: object) -> None:
        """
        Initialisation method inherited from `BertEmbeddings`. 
        It executes `__init__()` of `BertEmbeddings`, and then adds nn.Embedding layer to accomate the enriching ids. 
        Also, performs buffer registration for enriching_ids (registed as all zeros).

        Parameters:
        ===========
        config: Object. Model configuration. 

        Returns:
        ========
        None
        """

        super().__init__(config)
        self.enriching_embeddings = nn.Embedding(config.tokenizer_enrichment_vocab_size, config.hidden_size)
        self.register_buffer(
            "enriching_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    @logged()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        enriching_ids: Optional[torch.FloatTensor] = None,  # This is added for chemistry
        past_key_values_length: int = 0,
        ) -> torch.Tensor:
        """
        This method overwrites `forward()` function of `BertEmbeddings`.
        All operations are the same as in the original method, but
        embeddings of enriching_ids are added to the final embeddings.
        If `enriching_ids` are not provided, their embeddings are "calculated" for zeros.

        Parameters:
        ===========
        input_ids: torch.LongTensor. IDs for the input tokens.
        token_type_ids: torch.LongTensor. IDs indicating the token type. 
        position_ids: torch.LongTensor. IDs indicating token position.
        inputs_embeds: torch.FloatTensor. Precalculated embeddings for the token IDs.
        enriching_ids: torch.FloatTensor. IDs for the enriching IDs - Extra IDs added for chemistry purposes for reactions (but can be used also for other purposes).
        past_key_values_length: Int.  

        Returns:
        ========
        embeddings. torch.Tensor. All embedding layers for token ids + type ids + enriching ids and also position ids if selected.
        """


        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # Same comment as above, but for enriching_ids
        #####################  Added for chemistry  ####################
        if enriching_ids is None:  # This bit is added for chemistry
            if hasattr(self, "enriching_ids"):
                buffered_enriching_ids = self.enriching_ids[:, :seq_length]
                buffered_enriching_ids_expanded = buffered_enriching_ids.expand(input_shape[0], seq_length)
                enriching_ids = buffered_enriching_ids_expanded
            else:
                logger.debug("Setting `enriching_ids` to a tensor of 0s")
                enriching_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        ################################################################

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        enrich_embeddings = self.enriching_embeddings(enriching_ids)  # This bit is added for chemistry

        embeddings = inputs_embeds + token_type_embeddings + enrich_embeddings  # This bit is modified for chemistry
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)

            embeddings += position_embeddings
        
        # embeddings is of size: [batch_size, num_words, hidden_size]

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
