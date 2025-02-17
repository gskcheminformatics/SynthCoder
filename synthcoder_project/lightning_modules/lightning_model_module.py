# Module with a focus on model preparation

from typing import Union, Any, Callable, NoReturn, Literal
from functools import partial
import math
import re
import random
import numpy as np
import logging 
import lightning as pl
from lightning.pytorch.utilities import grad_norm

from torchmetrics.regression import (
    ExplainedVariance,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,  # can also be used for calculating rmse
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
    )
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    F1Score,
    MatthewsCorrCoef,
    CohenKappa,
    MultilabelRankingLoss,
    )


import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.tokenization_utils_base import BatchEncoding
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from torch_optimizer import Adafactor

import synthcoder_project.utilities as utils
from synthcoder_project.utilities import convert_batch_content_to_tensors
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)


class LightningSynthCoderMLM(pl.LightningModule):
    """
    Class based on Pytorch Lightning with the key 
    training/validation/testing functionalty implemented for an encoder.

    Can be used directly for MLM tasks.

    Based on the Pytorch Lightning framework:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """
    
    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self,
                 args: object, 
                 model_encoder: type, 
                 model_config: type, 
                 data_module: object,
                 **kwargs) -> None:
        
        """ 
        Initialises LightningSynthCoder object. 

        Parameters:
        ===========
        args: Object. Arguments (including all setting values) for the model, training, tokenization etc. 
        model_encoder: Class. Encoder model architecture (e.g. BertForMaskedLM)
        model_config: Class. Encoder configuration (e.g. BertConfig)
        data_module: Object. Initialised PyTorch Lightning data preparation module, responsible for tokenizer preparations.
        **kwargs
        
        Returns:
        ========
        None
        """
        
        super().__init__()
        self.args = args
        self.model_encoder = model_encoder
        self.model_config = model_config
        self.data_module = data_module

        self.tokenizer = self.data_module.return_tokenizer()
        self.config = self.define_config_for_current_model(**kwargs)
        self.update_model_config()

        # Loading pretrained model requires the model config to be already set up properly
        self.model = self.try_loading_pretrained_model(**kwargs)
        self.optimizer_grouped_parameters = self.setup_optimizer_grouped_parameters()
        self.learning_rate = self.args.learning_rate  # this is necessary as the name 'learning_rate' is recognised by Lightning during learning rate optimisation
        self.save_hyperparameters(ignore=["model_encoder", "model_config", "data_module"])

        self.reconstructed_mlm_ids = []  # list to collected the reconstructed inputs with the model predicted masked tokens

    @logged()
    def on_fit_start(self) -> None:
        """
        Default PyTorch Lightning function. Called automatically by the Lightning framework.
        Performs actions at the beginning of the model fit:
            - Reseeds pseudo-random number generators in: pytorch, numpy, python.random etc (this is done on the current device).
        
        Returns:
        ========
        None
        """
        logger.debug(None)

        utils.set_manual_seed(self.args.manual_seed)

    @logged()
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> torch.Tensor:
        """
        Default PyTorch Lightning function. Called automatically by the Lightning 
        framework during model training/fitting. Defines the training loop.
        Logs and returns loss for a given batch
        
        Parameters:
        ===========
        batch: BatchEncoding. Batch of encoded data for trining.
        batch_idx: Int. Index of the batch. Not used here. 
        
        Returns:
        ========
        loss: torch.Tensor. Calculated loss for a batch 
        """        

        batch_size = batch.get("input_ids").size()[0]

        model_output = self.model(**batch)
        loss = model_output[0]


        # Recalculate loss for adversarial training
        if self.args.adversarial_training:
            logger.debug("Calculating adversarial loss")
            loss = self._calculate_adversarial_loss(loss=loss,
                                                    model=self.model,
                                                    inputs=batch,
                                                    p=self.args.adversarial_training_probability)

        self.log('train_loss', 
                 value=loss, 
                 on_epoch=self.args.log_on_epoch, 
                 on_step=self.args.log_on_step, 
                 sync_dist=True, 
                 logger=True,
                 batch_size=batch_size,)
        
        if self.args.cross_attention_use_extended_descript_network:
            logger.debug("Calculating loss for tokens and descriptors MLM")
            self.log_token_and_descriptor_losses(model_output=model_output, 
                                                stage_name="train", 
                                                batch_size=batch_size)

        return loss

    @logged()
    def validation_step(self, batch: BatchEncoding, batch_idx: int) -> None:
        """
        Default PyTorch Lightning function. Called automatically by the Lightning 
        framework during model training/fitting.
        Logs and returns loss for a given batch
        
        Parameters:
        ===========
        batch: BatchEncoding. Batch of encoded data for trining.
        batch_idx: Int. Index of the batch. Not used here. 
        
        Returns:
        ========
        None
        """

        batch_size = batch.get("input_ids").size()[0]

        if self.args.log_molecular_correctness and batch["original_input_ids"].size() != batch["input_ids"].size():
            raise IndexError(f"The tensor sizes of `original_input_ids` {batch['original_input_ids'].size()}"
                             f" and `original_input_ids` {batch['input_ids'].size()} are different."
                             f" Please make sure that you are using the correct chemistry dedicated tokenizer," 
                             f" or alternatively set argument `log_molecular_correctness` to `False`.")


        # Remove "original_input_ids" from the batch
        original_ids = batch.pop("original_input_ids")

        # Run the model and collect the output
        model_output = self.model(**batch)

        # Extract the loss, logits and labels
        loss = model_output[0]
        logits = model_output[1]
        labels = batch["labels"]


        # loss = self.model(**batch).loss
        self.log('valid_loss', 
                 value=loss, 
                 on_epoch=self.args.log_on_epoch, 
                 on_step=self.args.log_on_step, 
                 sync_dist=True, 
                 logger=True,
                 batch_size=batch_size,)


        # Are we logging the correctness of molecules or percentage of correct 
        # predictions for a single molecule in reaction during MLM? 
        if self.args.log_molecular_correctness or self.args.log_mlm_exact_vector_match:
            
            # Generate the reconstructed by the MLM model input IDs
            reconstructed_ids = self.reconstruct_mlm_ids(logits=logits, 
                                                         original_ids=original_ids, 
                                                         labels=labels)
            
            if self.args.log_molecular_correctness:
                logger.debug("Calculating molecular correctness")
                self.reconstructed_mlm_ids.append(reconstructed_ids)
            if self.args.log_mlm_exact_vector_match:
                logger.debug("Calculating MLM exact vector match")
                self.log_match_between_tensors(tensor_a=reconstructed_ids,
                                               tensor_b=original_ids,
                                               stage_name="valid")

        # Are we also logging the loss for descriptors?
        if self.args.cross_attention_use_extended_descript_network:
            logger.debug("Log loss for both the tokens and descriptors")
            self.log_token_and_descriptor_losses(model_output=model_output, 
                                                 stage_name="valid", 
                                                 batch_size=batch_size)
    
    @logged()
    def _calculate_adversarial_loss(self, 
                                    loss: torch.Tensor, 
                                    model: object, 
                                    inputs: BatchEncoding, 
                                    p: float=1, 
                                    loss_fct: object=None, 
                                    # loss_fct_no_reduction: object=None,
                                    ) -> torch.Tensor:
        """
        Calculates adversarial loss for the model and the provided input following the equation:

        J(θ, x, y) = α𝐽(θ, x, y) + (1-α)𝐽(θ, x.embeddings + x.embeddings ϵ sign(∇ₓembeddings 𝐽(θ, x, y)), y)
        
        based on the fast gradient sign method, where 𝐽 is a regular loss function used in the training (e.g. cross entropy) 
        for model parameters θ, model input x and model output y. ∇ₓembeddings is gradient calculated for x.embeddings input 
        embeddings (e.g. word embeddings) used by the transformer. ϵ is a scaling parameter. Higher ϵ causes more added noise 
        to the embeddings. The adversarial training is performed with the probability equal `p`, and if not run it just returns the original loss. 

        The above equation is a modified adversarial loss function developed specifically for compatibility with transformer-encoder 
        embeddings. 

        This function is not compatible with the default Huggingface models, and only appropriately modified encoder models can be used.


        Take a look at: 
        https://doi.org/10.48550/arXiv.1412.6572
        and: 
        https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

        
        Parameters:
        ===========
        loss: torch.Tensor. Loss calculated for the model with the unmodified inputs.
        model: Object. NN model that is being trained. 
        inputs: BatchEncoding. Inputs (unmodified) to the model.
        p: Float. Propability of performing the adversarial training. 1.0 is 100% of chance running the adversarial training, 0.0 is 0% 
        loss_fct: Object|None. Optional loss function (e.g. cross entropy).
        # loss_fct_no_reduction: Object|None. Optional loss function (e.g. cross entropy) which does not perform reduction.

        Returns:
        ========
        torch.Tensor. Calculated adversarial loss J(θ, x, y) [see the equation above], or the original loss if the adversarial 
        training is not performed. 
        """

        # Decide if we should perform the adversarial training with the propability `p`
        if not utils.return_True_or_False(probability_of_True=p):
            return loss
        
        # Zero the gradients
        opt = self.optimizers()
        opt.zero_grad()
        
        # The non-leaf nodes do not reatin gradients after passes though the model, 
        # so make these tensors reatain the gradients.
        model.bert.embedding_output.retain_grad()
        try:  # `descriptors_hidden_states` may not be used by the model
            model.bert.descriptors_hidden_states.retain_grad()
        except AttributeError as e:
            logger.debug(f"{e}\n`descriptors_hidden_states` attribute not found, skipping its gradient retention.")

        # Calculate gradients using the regular loss, retain the computational 
        # graph for the second back propagation (done by Lightning with the final loss) 
        loss.backward(retain_graph=True)

        # Prepare modified/adversarial embeddings that will be fed into the model, replacing the regualar inputs 
        adversarial_embedding_output = self.generate_adversarial_inputs_fgsm(inputs=model.bert.embedding_output, 
                                                                             epsilon=self.args.adversarial_epsilon)

        adversarial_descriptors_hidden_states = self.generate_adversarial_inputs_fgsm(inputs=model.bert.descriptors_hidden_states, 
                                                                                      epsilon=self.args.adversarial_epsilon)

        # Zero gradients
        opt.zero_grad()
        model.bert.embedding_output.grad = None  # does not zero automatically, as optimizer does not keep track of it.
        try:  # `descriptors_hidden_states` may not be used by the model
            model.bert.descriptors_hidden_states.grad = None  # does not zero automatically, as optimizer does not keep track of it.
        except AttributeError as e:
            logger.debug(f"{e}\n`descriptors_hidden_states` attribute not found, skipping its gradient zeroing.")
        
        # Run the model (training run, so it contribues to the future gradient accumulation)
        model_outputs = model(**inputs, 
                              adversarial_embedding_output=adversarial_embedding_output, 
                              adversarial_descriptors_hidden_states=adversarial_descriptors_hidden_states)

        # Model outputs are always tuple in pytorch-transformers (see docs)
        # If explicit loss function is provided to this method, use it to calculate the loss for adversarial examples  
        if loss_fct:
            logits = model_outputs[1]
            labels = inputs["labels"]
            # calculate loss for adversarial examples
            adversarial_loss = loss_fct(logits=logits, labels=labels)
            logger.debug("Calculating adversarial loss with the provided loss function")
        else: # otherwise use the loss already calculated by the model
            adversarial_loss = model_outputs[0]
            logger.debug("Using the loss value already provided by the model")
        
        # Return the final adversarial loss J(θ, x, y)
        return self.args.adversarial_loss_alpha * loss + (1-self.args.adversarial_loss_alpha) * adversarial_loss
          
    @logged()
    def generate_adversarial_inputs_fgsm(self, inputs: Union[torch.Tensor, None], epsilon: float) -> Union[torch.Tensor, None]:
        """
        Uses a fast gradient sign method to modify the tensor input.
        The modification of the input is done in the gradient direction scaled by `epsilon`.
        Returns None if `inputs` is not torch.Tensor. 

        Parameters:
        ===========
        inputs: torch.Tensor|None. Output generated by the model, without the loss, but otherwise unprocessed. 
        epsilon: Float. Scaling factor for data modiciation. 
        
        Returns:
        ========
        perturbed_inputs: torch.Tensor. Tensor containing modified input values. 
        Returns None if `inputs` is not torch.Tensor. 
        """

        if not isinstance(inputs, torch.Tensor):
            logger.debug("Invalid input type for adversarial perturbation, returning None.")
            return None
        
        # Look up the calculated gradient for the data
        data_grad = inputs.grad.data
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        
        # Create the perturbed embeddings input
        inputs_detached =  inputs.clone().detach()  # make sure to clone the tensor and detach it from the computation graph, so that generation of the perturbed embeddings is not backpropagated
        perturbed_inputs = inputs_detached + (inputs_detached*epsilon*sign_data_grad) # Probably multiplication is be better for our application with transfomers than summation
        
        # Return the perturbed data
        return perturbed_inputs

    @logged()
    def log_match_between_tensors(self,
                                  tensor_a: torch.Tensor, 
                                  tensor_b: torch.Tensor,
                                  stage_name: str,
                                  log_title: str = "match_between_tensors") -> None:
        """
        Calculates and logs the percentage of exact equivalence/match between two tensors in the last tensor dimension.
        E.g. `tensor_a` is `[[0, 1], [0, 1]]` and `tensor_b` is `[[0, 1], [0, 0]]`, the match score will be 0.5,
        as vectors at index 0: `[0, 1]` are the same in both tensors, but vectors at index 1 are differnt.   

        The score range is between 0 and 1.  

        Parameters:
        ===========
        tensor_a: torch.Tensor. Tensor to comapre with `tensor_b`. The size of the tensor must be (batch_size,  block_size).
        tensor_b: torch.Tensor. Tensor to compare with `tensor_a`. The size of the tensor must be (batch_size,  block_size).
        stage_name: Str. Name of the current stage.
        log_title: Str. Title to use as a label for the data log. 

        Returns:
        ========
        None
        """

        tensor_a_size = tensor_a.size()
        tensor_b_size = tensor_b.size()
        batch_size = tensor_a_size[0]

        # Check if the tensor sizes are the same 
        if tensor_a_size != tensor_b_size:
            logger.error("Tensor match score not logged!: Tensor sizes do not match.")
            print(f"Tensor match score not logged!: Tensor sizes do not match: {tensor_a_size} vs {tensor_b_size}")
            return None

        # Find the number of the exact vector matches between the two tensors 
        num_exact_matches = 0
        for sample_idx in range(0, batch_size):  # iterate through all the samples (all sentences) 
            if torch.equal(tensor_b[sample_idx], tensor_a[sample_idx]):  # compare vector values at the same index position of the tensors
                num_exact_matches += 1

        # Log the results
        self.log(f"{stage_name}_{log_title}", 
                 value=num_exact_matches/batch_size, 
                 on_epoch=self.args.log_on_epoch, 
                 on_step=False,
                 sync_dist=True,
                 logger=True)

    @logged()
    def log_token_and_descriptor_losses(self, 
                                        model_output: object, 
                                        stage_name: str, 
                                        batch_size: int) -> None:
        """
        Logs the loss obtained only for the tokens and separately only for the descriptors.
        This function is only to be used with the model outputs which provide these two losses.
        
        Parameters:
        ===========
        model_output: Object. The output of the model. The model needs to be able to provide loss
            for token-based MLM and descriptor-based MLM. Thus this function should be used only 
            with the appropriate "chemistry" models.
        stage_name: Str. Name of the current stage.
        batch_size: Int. The size of the batch.
        
        Returns:
        ========
        None
        """

        # Loss calculated just for the tokens
        loss_tokens = model_output[-2]
        # Loss calculated just for the descriptors
        loss_descriptors = model_output[-1]

        self.log(f"{stage_name}_loss_tokens_only", 
                 value=loss_tokens, 
                 on_epoch=self.args.log_on_epoch, 
                 on_step=self.args.log_on_step, 
                 sync_dist=True, 
                 logger=True,
                 batch_size=batch_size,)
        
        self.log(f"{stage_name}_loss_descriptors_only", 
                 value=loss_descriptors, 
                 on_epoch=self.args.log_on_epoch, 
                 on_step=self.args.log_on_step, 
                 sync_dist=True, 
                 logger=True,
                 batch_size=batch_size,)

    @logged()
    def reconstruct_mlm_ids(self, 
                            logits: torch.Tensor, 
                            original_ids: torch.Tensor, 
                            labels: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original token IDs based on the MLM predictions.
        The original IDs (before corruption for the MLM  task) are merged 
        with the model predictions of the masked token IDs. This way 
        the corruption of the input caused by the collator is avoided (e.g. 
        the replacement of random tokens with different tokens).

        The IDs predicted by the MLM model replace the corresponding original token IDs, 
        resulting in reconstructed token IDs.

        Parameters:
        ===========
        logits: torch.Tensor. Logits generated by the MLM model.
        original_ids: torch.Tensor. Original IDs before corruption for MLM.
        labels: torch.Tensor. Labels given to the MLM model, where value -100 idicates token ids to ignore.
        
        Returns:
        ========
        reconstructed_ids: torch.Tensor. Reconstructed token IDs based on the model predictions.

        """

        batch_size = logits.size()[0]
        
        # Find model predicted token IDs
        logits_max_indices = logits.view(batch_size, -1, self.config.vocab_size).max(2).indices

        # Generate the map of the masking patter to see which token IDs were masked and which were not. 
        # Value -100 means no masking. 
        mask_map = labels!=-100

        # Replace the original, uncorrupted token IDs with the predicted by the model IDs (only at the masked spots) 
        reconstructed_ids = original_ids.clone()
        reconstructed_ids[mask_map] = logits_max_indices[mask_map]

        return reconstructed_ids

    @logged()
    def on_validation_epoch_end(self) -> None:
        """
        Default PyTorch Lightning function. Called automatically by the Lightning framework.
        Performs actions at the end of the validation:
            - if selected through model arguments, assesses and logs correctness of SMILES 
              and molecules generated by the model druging the MLM.
        
        Returns:
        ========
        None
        """
        if self.args.log_molecular_correctness:
            self.log_molecular_correctness()

    @logged()
    def log_molecular_correctness(self) -> None:
        """
        Assesses and logs correctness of SMILES and molecules generated by the model druging the MLM.
        Uses RDKit and ChemAxon StructureCheck-based Java program for checking the SMILES.

        The score range is between 0 and 1.  

        StructureCheck requires the following commands to be executed before starting the Python script:
            module load applications-extra java/11.0.7 jchem/22.17.5
            export CHEMAXON=/hpc/mydata/stv/ak590819/CSC_cxn_netbeans/netbeans

        Returns:
        ========
        None
        """
        # Initiate the variables for keeping track of the stats
        rdkit_num_valid_smiles = 0
        rdkit_num_valid_molecules = 0
        structurecheck_num_valid_molecules = 0
        total_num_molecules = 0
        log_structurechecker_results = True
        tokens_to_ignore = self.tokenizer.all_special_tokens + [None]

        # Iterate trough the generated/predicted token IDs during the MLM for validation set
        for batch_outputs in self.reconstructed_mlm_ids:
            batch_outputs = batch_outputs.tolist()
            for ids in batch_outputs:
                tokens = self.tokenizer.convert_ids_to_tokens(ids)
                tokens = [token for token in tokens if token not in tokens_to_ignore]
                reconstructed_input = "".join(tokens)

                # Replace tokenizer specific symbols/elements
                reconstructed_input = re.sub("##|</w>|_", "", reconstructed_input)

                # Split the reaction string into molecules
                reconstruct_reaction_molecules = re.split(rf"\{self.args.tokenizer_molecule_separator}|{self.args.tokenizer_reaction_arrow_symbol}", reconstructed_input)
                
                # Check validity of each molecule from the generated reaction string
                for molecule in reconstruct_reaction_molecules:
                    total_num_molecules += 1

                    # Replace the special symbol - molecule joiner (normally "^") indicating different components within one molecule entry with a "."
                    molecule = re.sub(f"\{self.args.tokenizer_molecule_joiner}", ".", molecule)
                    
                    rdkit_valid_smiles, rdkit_valid_molecule = utils.check_smiles_and_chemical_validity_with_rdkit(smiles=molecule)
                    rdkit_num_valid_smiles += 1 if rdkit_valid_smiles else 0
                    rdkit_num_valid_molecules += 1 if rdkit_valid_molecule else 0
                    
                    try:  # if the Java enviroment and the required tools are not properly loaded the structurechecker function will error out
                        structurecheck_valid_molecule = utils.check_chemical_validity_with_structurecheck(smiles=molecule)
                        structurecheck_num_valid_molecules += 1 if structurecheck_valid_molecule else 0
                    except TypeError:
                        log_structurechecker_results = False

        # Log the metrics:
        self.log('rdkit_valid_smiles', 
                 value=rdkit_num_valid_smiles/total_num_molecules, 
                 on_epoch=self.args.log_on_epoch, 
                 on_step=False, 
                 sync_dist=True, 
                 logger=True)
        
        self.log('rdkit_valid_molecules', 
                 value=rdkit_num_valid_molecules/total_num_molecules, 
                 on_epoch=self.args.log_on_epoch, 
                 on_step=False, 
                 sync_dist=True, 
                 logger=True)
        
        if log_structurechecker_results:
            self.log('structurecheck_valid_molecules', 
                    value=structurecheck_num_valid_molecules/total_num_molecules, 
                    on_epoch=self.args.log_on_epoch, 
                    on_step=False, 
                    sync_dist=True, 
                    logger=True)
        
        # Empty the list with the reconstructed token IDs 
        self.reconstructed_mlm_ids = []

    @logged()
    def on_before_optimizer_step(self, optimizer) -> None:
        """
        Default PyTorch Lightning function. Called automatically by the Lightning framework.
        Performs actions before stepping the optimizer:
            - Collects and logs gradient norm.
        
        Returns:
        ========
        None
        """
        if self.args.track_grad_norm:
            # Compute the 2-norm for each layer
            # If using mixed precision, the gradients are already unscaled here
            norms = grad_norm(self.model, norm_type=self.args.track_grad_norm)
            self.log_dict(norms)

    @logged()
    def configure_optimizers(self) -> Union[object, dict, NoReturn]:
        """
        Default PyTorch Lightning function for setting up the model optimizer.
        Called automatically by the Lightning framework.
        The current implementation supports AdamW and Adafactor optimizers.
        Initialises and returns an optimiser. If user specified a scheduler in the model args,
        a scheduler config is also returned.

        Returns:
        ========
        Object. Optimizer object if no scheduler is to be used. | Dict. A dictionary containing an optimiser object and a learning rate scheduler config.
        """

        # Initialise optimisers
        if self.args.optimizer == "AdamW":  # AdamW us an optimizer from PyTorch
            logger.debug("Using AdamW optimiser")
            optimizer = AdamW(
                self.optimizer_grouped_parameters,
                # lr=self.args.learning_rate,
                lr=self.learning_rate,
                eps=self.args.adam_epsilon,
                betas=self.args.adam_betas,
                amsgrad=self.args.adam_amsgrad,
            )

        # Different Adafactor optimizer used than in SimpleTransformers 
        elif self.args.optimizer == "Adafactor":  
            logger.debug("Using Adafactor optimiser")            
            optimizer = Adafactor(  # this Adafactor is an optimizer from Pytorch-Optimizer package
                params=self.optimizer_grouped_parameters,
                # lr=self.args.learning_rate,
                lr=self.learning_rate,
                eps2=self.args.adafactor_eps,
                clip_threshold=self.args.adafactor_clip_threshold,
                decay_rate=self.args.adafactor_decay_rate,
                beta1=self.args.adafactor_beta1,
                weight_decay=self.args.weight_decay,
                scale_parameter=self.args.adafactor_scale_parameter,
                relative_step=self.args.adafactor_relative_step,
                warmup_init=self.args.adafactor_warmup_init,
            )

        else:
            logger.error("Invalid optimiser selected")
            raise ValueError(
                f"{self.args.optimizer} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead."
            )

        if not self.args.scheduler:
            logger.debug("Using no scheduler")
            print("\n\nUsing no scheduler\n")
            return optimizer
        # If a scheduler has been specified by a user:  
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.return_scheduler_config(optimizer=optimizer)
            }

    @logged()
    def return_scheduler_config(self, optimizer: object) -> dict:
        """
        Returns configuration for a scheduler necessary for the configure_optimizers() method.
        Defines the scheduler, updating interval, frequency and name for the logger. 
        [A metric to monitor by the scheduler can also be added].

        Parameters:
        ===========
        optimizer: Object. The initiated optimiser to be used during the model training.

        Returns:
        ========
        Dict. Config for scheduler. 
        """
        return {# REQUIRED - The scheduler instance:
                "scheduler": self.return_scheduler(optimizer=optimizer),

                # The unit of the scheduler's step size, could be 'step' or 'epoch'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update:
                "interval": "step",  # can also be "epoch"

                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step:
                "frequency": 1,

                # # If using the `LearningRateMonitor` callback to monitor the
                # # learning rate progress, this keyword can be used to specify
                # # a custom logged name:
                # "name": "lr_scheduler_name_test_ak",
                
                # # Metric to monitor for schedulers like `ReduceLROnPlateau`:
                # "monitor": "val_loss",

                # # If set to `True`, will enforce that the value specified 'monitor'
                # # is available when the scheduler is updated, thus stopping
                # # training if not found. If set to `False`, it will only produce a warning:
                # "strict": True,
                }

    @logged()
    def return_scheduler(self, optimizer: object) -> Union[object, NoReturn]:
        """
        Returns a scheduler.
        Internally calculates the total number of steps and warm-up steps for the training. 
        The schedulers used by the function are HuggingFace objects. 

        SimpleTransformers method. (language_modeling_model.py)

        Parameters:
        ===========
        optimizer: Object. The initiated optimiser to be used during the model training.

        Returns:
        ========
        scheduler: Object.  
        """        
        if isinstance(self.args.num_train_steps, int) and self.args.num_train_steps != -1:
            self.total_step_number = self.args.num_train_steps  # use the provided number of steps when possible
        else:
            self.total_step_number = self.find_total_step_number()  # otherwise calculate the number of steps from the number of epochs

        self.warmup_steps = self.return_num_warmup_steps()

        logger.debug(f"Total number of steps: {self.total_step_number}; Warm-up steps (if applicable): {self.warmup_steps}")

        if self.args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif self.args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.warmup_steps
            )

        elif self.args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_step_number,
            )

        elif self.args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_step_number,
                num_cycles=self.args.cosine_schedule_num_cycles,
            )

        elif self.args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_step_number,
                num_cycles=self.args.cosine_schedule_num_cycles,
            )

        elif self.args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_step_number,
                lr_end=self.args.polynomial_decay_schedule_lr_end,
                power=self.args.polynomial_decay_schedule_power,
            )

        elif self.args.scheduler == "cyclic_lr_scheduler":

            step_size_up = math.ceil(self.total_step_number * self.args.cyclic_lr_scheduler_ratio_size_up)
            if self.args.cyclic_lr_scheduler_ratio_size_down:
                step_size_down = math.ceil(self.total_step_number * self.args.cyclic_lr_scheduler_ratio_size_down)
            else:
                step_size_down = None

            scheduler = CyclicLR(
                optimizer,
                base_lr=self.args.cyclic_lr_scheduler_base_lr,
                max_lr=self.args.learning_rate,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                mode=self.args.cyclic_lr_scheduler_mode, 
                cycle_momentum=self.args.cyclic_lr_scheduler_cycle_momentum,
                gamma=self.args.cyclic_lr_scheduler_gamma,
            )         

        else:
            logger.error("Invalid scheduler")
            raise ValueError(f"{self.args.scheduler} is not a valid scheduler.")

        return scheduler
    
    @logged()
    def find_total_step_number(self) -> int:
        """
        Calculates the total number of training steps.

        Returns:
        ========
        total_step_number: Int. 
        """        
        total_step_number = int(
            math.ceil(self.data_module.return_len_train_dataloader()
            / self.args.gradient_accumulation_steps)  # even in case len_train_loader/gradient_accumulation_steps = 1.6, there will be 2 steps in PyTorch Lightning! Thus, use math.ceil
            * self.args.num_train_epochs
        )

        # Account for the number of devices used for the distributed training
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(group=None)
        else:
            world_size = 1

        if world_size > 1:
            total_step_number =  math.ceil(total_step_number / world_size)

        print(f"{total_step_number=}")
        logger.info(f"Total step number: {total_step_number}")
        return total_step_number

    @logged()
    def return_num_warmup_steps(self) -> int:
        """
        Returns the number of warm-up steps for the scheduler. 
        warmup_steps overwrites warmup_ratio provided in the model arguments.

        Returns:
        ========
        Int. The number of warm-up steps  
        """
        if self.args.warmup_steps:
            return self.args.warmup_steps
        elif self.args.warmup_ratio:
            return math.ceil(self.total_step_number * self.args.warmup_ratio)
        return 0
    
    @logged()
    def setup_optimizer_grouped_parameters(self) -> list[dict]:
        """
        Sets the values of the weight decay to the appropriate parameters in the model.
        Bias and LayerNorm.weight will have no decay. 
        The generated output can be later fed into the optimiser. 

        Any named parameters specified through custom_layer_parameters with bias 
        or LayerNorm.weight in the name will have their weight_decay set to 0.0. 
        This also happens for any parameters not specified in either custom_parameter_groups 
        or in custom_layer_parameters but does not happen for parameters specified 
        through custom_parameter_groups.
        See https://simpletransformers.ai/docs/tips-and-tricks/#custom-parameter-groups for more info.

        SimpleTransformers method. (language_modeling_model.py)

        Returns:
        ========
        optimizer_grouped_parameters: List of dicts. Returns a list of dictionaries 
        specifying the appropriayte weight decay for the model parameters.

        """
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()

        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in self.model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in self.model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        return optimizer_grouped_parameters

    @logged()
    def define_config_for_current_model(self, **kwargs) -> object:
        """
        Defines the config for the current model. 
        It tries to load the config of a pretrained model or from a predefined config file (config.json).
        If predefined configuration specified in the model args cannot be loaded, 
        uses current and user defined args to create a new model config. 
        
        Modfied SimpleTransformers method. (language_modeling_model.py)
        
        Parameters:
        ===========
        **kwargs
        
        Returns:
        ========
        config: Object. Defined, initialised model config object. 

        """
        # Try loading the pretrained model config using either the provided config_name or the provided model_name
        if self.args.config_name:  # config_name <- name of pretrained config or path to a directory containing a config.json file
            config = self.model_config.from_pretrained(
                self.args.config_name, cache_dir=self.args.cache_dir, **self.args.config, **kwargs
            )
            logger.info(f"Using a pre-defined config file: {self.args.config_name}")
            print("Using predefined config file")

        elif self.args.model_name and self.args.model_name != "electra":  # model_name <- name of the model, or a path to the folder with the existing pre-saved model files 
            config = self.model_config.from_pretrained(
                self.args.model_name, cache_dir=self.args.cache_dir, **self.args.config, **kwargs
            )
            logger.info(f"Using config file from a pre-trained model: {self.args.model_name}")
            print("Using config file from a pre-trained model")

        # Otherwise use the current and user defined args to create a new model config 
        else:
            config = self.model_config(**self.args.config, **kwargs)
            logger.info(f"Creating a new config file")
            print("Creating a new config file")

        return config

    @logged()
    def update_model_config(self) -> None:
        """
        Updates the model config with new settings.

        Updates: 
            - Config with the user defined config settings
            - Vocabulary size
            - Max number of positional embeddings
            - Enrichment vocabulary size if provided
            - Adds info about the number of descriptors per molecule if available, to be used for cross attention
            - Updating `cross_attention_encoder_input` bool flag
            - Adds info about the max number of compounds per reaction for the cross-attention calculations
        
            Note: vocab_size for the model config is the maximum size of the vocabulary of the tokenizer. 
            If the model is new the model configuaration vocab_size is set to the current tokenizer vocabulary size. 
        
        Returns:
        ========
        None 
        """

        # If the tokenizer is newly set up or predefined, but an existing model is not being loaded (as this causes mismatch with the vocabulary dimentions in the network),
        # update the learned vocab_size by the tokenizer in the config. 
        if self.data_module.is_new_tokenizer() or (self.args.tokenizer_name and not self.args.model_name):
            self.config.vocab_size = len(self.tokenizer)
        logger.info(f"Vocab size for model config: {self.config.vocab_size}")
        print(f"{self.config.vocab_size=}")

        # Update the maximum size of the position embeddings of the model
        self.config.max_position_embeddings = self.args.max_seq_length

        # Add information about the enrichment vocabulary size if available
        if self.args.tokenizer_enrichment_vocab_size:
            logger.info(f"Adding information about the enrichment vocabulary size to the model config.")
            print("\nConfig Info: Adding information about the enrichment vocabulary size to the model config.\n")
            self.config.tokenizer_enrichment_vocab_size = self.args.tokenizer_enrichment_vocab_size

        # Add information about the number of descriptors per molecule (if any) for cross-attention 
        # Add information about the max number of compounds per reaction, needed for cross-attention 
        # If this info is present, set the `cross_attention_encoder_input` flag in the model config to `True`
        if self.args.cross_attention_number_of_descriptors:
            logger.info(f"Providing info about cross attention with descriptors to the model config\n{self.args.cross_attention_number_of_descriptors=}, {self.args.cross_attention_max_num_cmpds=}")
            print("\nConfig Info: Adding information about the number of descriptors to the model config."
                  "\nConfig Info: Adding information about the max number of compounds per reaction."
                  "\nConfig Info: Turning ON the encoder cross-attention.\n")
            self.config.cross_attention_number_of_descriptors = self.args.cross_attention_number_of_descriptors
            self.config.cross_attention_max_num_cmpds = self.args.cross_attention_max_num_cmpds
            self.config.cross_attention_encoder_input = True

        # Turn on the extended descritors network
        if self.args.cross_attention_use_extended_descript_network:
            logger.info(f"Turning ON the extended descriptors network")
            print("\nConfig Info: Turning ON the extended descriptors network (with descriptors MLM in pretraining)!!!.\n")
            self.config.cross_attention_use_extended_descript_network = self.args.cross_attention_use_extended_descript_network


        # Add information to the model config about the block_size
        self.config.block_size = self.args.block_size

    @logged()
    def return_model_config(self) -> object:
        """
        Returns:
        ========
        config: Object. Model config settings.
        """
        return self.config

    @logged()
    def try_loading_pretrained_model(self, **kwargs):
        """
        Tries to load the pretrained model. 
        If that's not possible, it will initiate a new model with the earlier defined model config.
        If new model is initialised, certain model dimensions are resized. 

        Modified SimpleTransformers method. (language_modeling_model.py)

        Parameters:
        ===========
        **kwargs 
        
        Returns:
        ========
        model: Object. Pretained or newly initialised model. 
        """
        # Try loading already pre-trained model
        if self.args.model_name:
            model = self.model_encoder.from_pretrained(self.args.model_name, 
                                                        config=self.config, 
                                                        cache_dir=self.args.cache_dir,
                                                        ignore_mismatched_sizes=self.args.ignore_mismatched_sizes,
                                                        **kwargs)
            logger.info(f"Loaded existing model: {self.args.model_name}")
            print("Loaded existing model")
        
        # Othwerise initialise a new model
        else:
            model = self.model_encoder(config=self.config)

            # This is done to:
            # Resize input token embeddings matrix of the model if new_num_tokens [so len(self.tokenizer)] != config.vocab_size.
            # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
            # Changes also the vocabulary size in the model config (to multiple of self.args.pad_to_multiple_of)!!!
            model_to_resize = (model.module if hasattr(model, "module") else model)
            model_to_resize.resize_token_embeddings(new_num_tokens=len(self.tokenizer), pad_to_multiple_of=self.args.pad_to_multiple_of)
            
            # guideline for pad_to_multiple_of: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            logger.debug("Initialised a new model")
            print("Initialised a new model")
        return model
    


class LightningSynthCoderForClassification(LightningSynthCoderMLM):
    """
    Class based on Pytorch Lightning with the key 
    training/validation/testing functionalty implemented for an encoder.

    Inherits from the class for MLM tasks.
    Can be used for regression, binary classification and multiclass classification tasks.

    The loss function and metrics that are collected are chosen based on the problem type 
    specified by the user in the model args. 
    
    Based on the Pytorch Lightning framework:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    @logged(level=logging.INFO, message="Initialising {cls}")
    def __init__(self,
                 args: object, 
                 model_encoder: type, 
                 model_config: type, 
                 data_module: object,
                 **kwargs) -> None:
        
        """ 
        Initialises LightningSynthCoder object. 

        Parameters:
        ===========
        args: Object. Arguments (including all setting values) for the model, training, tokenization etc. 
        model_encoder: Class. Encoder model architecture (e.g. BertForMaskedLM)
        model_config: Class. Encoder configuration (e.g. BertConfig)
        data_module: Object. Initialised PyTorch Lightning data preparation module, responsible for tokenizer preparations.
        **kwargs

        Returns:
        ========
        None
        """
        
        super().__init__(args, model_encoder, model_config, data_module, **kwargs)
        
        self.ensure_correct_num_labels()
        self._initialise_loss_functions_on_device(force_init=True)
        self.metrics_train = self.initiate_metrics()
        self.metrics_valid = self.initiate_metrics()

    @logged()
    def ensure_correct_num_labels(self) -> None:
        """
        Checks if the number of lables argument (num_lables) specified by the user is 1 or more,
        and that it is an integer as expected.   
        
        Returns:
        ========
        None
        """
        assert self.config.num_labels >= 1, "The number of labels (num_labels) must be set to at least 1"
        assert isinstance(self.config.num_labels, int), "The number of labels (num_labels) must be an integer"

    @logged()
    def _select_loss_function(self, reduction: Literal["mean", "sum", "none"] ="mean") -> partial:
        """
        Selects the loss function based on the problem type (regression, single_label_classification 
        or multi_label_classification), and passes class weights to the loss function object if weights 
        are available and applicable to the problem.

        Creates and returns a new partially initialised function that can process data logits and labels 
        tensors and calculates loss.

        Loss function selection is very loosely based on the code from modeling_distilbert.py (`forward` method) 
        in HuggingFace Transfromers.

        NOTE: The loss can be calculated by the HuggingFace models directly, but not all models will accept 
        weights for classes/labels. Calculaing the loss outside of the model also gives more flexibility, 
        and assures that the same method for the loss calculation is used independent of the model.
        
        Parameters:
        ===========
        reduction: String. ["mean", "sum", "none"]. Reduction parameter for loss calculation. Losses are averaged or summed over observations 
            for each minibatch. "none'": no reduction will be applied, "mean": the weighted mean of the output is taken, "sum": the output will be summed. 

        Returns:
        ========
        partial. New, partially initialised function for processing data logits and labels and calculating loss. 
        """
        # First, define the two functions needed to construct our loss calculating method
        @logged()
        def initalise_loss_object(loss_fct_class: type) -> object:
            """ Intialises and returns a loss function object, when possible with 
            the class weights passed as an argument. """

            if len(self.class_weights) != 0 and self.config.num_labels > 1:  # we do not want to apply any class weights if we work with a regression problem
                print(f"Using class weights with the loss function")
                return loss_fct_class(weight=self.class_weights, reduction=reduction)
            return loss_fct_class(reduction=reduction)

        @logged()
        def prepare_input_and_calc_loss(logits: torch.Tensor, 
                                        labels: torch.Tensor, 
                                        prepare_input_fct: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]], 
                                        loss_object: object) -> torch.Tensor:
            """ Reshapes tensors for logits and labels with the provided function and 
            returns the calculated loss value, using the specified loss function object. """
            
            logits, labels = prepare_input_fct(logits, labels)
            return loss_object(logits, labels)

        # Select which loss class for loss calculation is appropriate based on the specifed problem type
        # and also define the appropriate lambda function for processing the input logits and label tensors
        if self.config.problem_type == "regression":
            loss_fct_class = MSELoss
            self.prepare_input_fct = lambda logits, labels: (logits.squeeze(), labels.squeeze())
        elif self.config.problem_type == "single_label_classification":
            loss_fct_class = CrossEntropyLoss
            self.prepare_input_fct = lambda logits, labels: (logits.view(-1, self.config.num_labels), labels.view(-1) )  
        elif self.config.problem_type == "multi_label_classification":
            loss_fct_class = BCEWithLogitsLoss
            self.prepare_input_fct = lambda logits, labels: (logits, labels)

        logger.info(f"Problem type: {self.config.problem_type}; Loss function: {loss_fct_class.__name__}")
        print(f"\nUsing {loss_fct_class.__name__} loss function")

        # Initialise the loss function object  
        self.loss_object = initalise_loss_object(loss_fct_class)

        # Partially initialise the function to return, by providing it with the appropriate function for 
        # data processing and the initialised loss function object
        loss_fct = partial(prepare_input_and_calc_loss, 
                           prepare_input_fct=self.prepare_input_fct,
                           loss_object=self.loss_object) 
        return loss_fct  

    @logged()
    def _initialise_loss_functions_on_device(self, force_init: bool=False) -> None:
        """
        Initialises `self.class_weight` tensor and loss_functions (`self.loss_function` and `self.loss_function_no_reduction`) 
        on the current device, if `self.class_weight` is on cpu and the current device is different from cpu. `force_init` 
        set to `True` initalises the above regardless.

        Parameters:
        ===========
        force_init: <Optional> Bool. If `True`, forces (re)initialisation. 

        Returns:
        ========
        None
        """

        if force_init or (self.class_weights.get_device() == -1 and str(self.device) != "cpu"):
            logger.debug(f"Initialising loss function on device {self.device=}")

            self.class_weights = torch.Tensor(self.args.class_weights).to(self.device)  # it is necessary for the loss func. to change the data type to Tensor. It needs to be on the current device, otherwise it will cause a crash.
            self.loss_function = self._select_loss_function()
            self.loss_function_no_reduction = self._select_loss_function(reduction="none")
        
    @logged()
    def on_fit_start(self) -> None:
        """
        Default PyTorch Lightning function. Called automatically by the Lightning framework. 
        Performs actions at the beginning of the model fit: 
            - Reseeds pseudo-random number generators in: pytorch, numpy, python.random etc (this is done on the current device).
            - Reinitialises loss functions on the current device 
        
        Returns:
        ========
        None
        """
        super().on_fit_start()
        self._initialise_loss_functions_on_device()

    @logged()
    def on_train_start(self) -> None:
        """
        Default PyTorch Lightning function. Called at the beginning of training after sanity check:
            - Reinitialises loss functions with weights on the current device 
        """
        super().on_train_start()
        self._initialise_loss_functions_on_device()

    @logged()
    def on_test_start(self) -> None:
        """
        Default PyTorch Lightning function. Called at the beginning of testing:
            - Reinitialises loss functions with weights on the current device 
        """
        super().on_test_start()
        self._initialise_loss_functions_on_device()

    @logged()
    def on_validation_start(self) -> None:
        """
        Default PyTorch Lightning function. Called at the beginning of validation:
            - Reinitialises loss functions with weights on the current device 
        """
        super().on_validation_start()
        self._initialise_loss_functions_on_device()

    @logged()
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> torch.Tensor:
        """
        Overloads the parent's method.

        Default PyTorch Lightning function. Called automatically by the Lightning 
        framework during model training/fitting. Defines the training loop.
        Logs loss and metrics for a batch. Returns loss for a given batch.
        
        Parameters:
        ===========
        batch: BatchEncoding. Batch of encoded data for trining.
        batch_idx: Int. Index of the batch. Not used here. 
        
        Returns:
        ========
        loss: torch.Tensor. Calculated loss for a batch 
        """
        loss = self.calculate_loss_and_metrics(batch=batch,
                                                metrics=self.metrics_train,
                                                step_type="train")
        return loss
    
    @logged()
    def validation_step(self, batch: BatchEncoding, batch_idx: int, step_type="valid") -> None:
        """
        Overloads the parent's method.

        Default PyTorch Lightning function. Called automatically by the Lightning 
        framework during model training/fitting or validation.
        Logs metrics and loss for a given batch.
        
        Parameters:
        ===========
        batch: BatchEncoding. Batch of encoded data for trining.
        batch_idx: Int. Index of the batch. Not used here. 
        
        Returns:
        ========
        None
        """
        self.calculate_loss_and_metrics(batch=batch,
                                        metrics=self.metrics_valid,
                                        step_type=step_type)


    @logged()
    def test_step(self, batch: BatchEncoding, batch_idx: int) -> None:
        """
        Overloads the parent's method.

        Default PyTorch Lightning function. Called automatically by the Lightning 
        framework during model testing.
        Just calls the `self.validation_step` function.
        Logs metrics and loss for a given batch.
        
        Parameters:
        ===========
        batch: BatchEncoding. Batch of encoded data for training.
        batch_idx: Int. Index of the batch.
        
        Returns:
        ========
        None
        """
        self.validation_step(batch, batch_idx, step_type="test")

    @logged()
    def predict_step(self, 
                     batch: BatchEncoding, 
                     batch_idx: int, 
                     dataloader_idx: int=0,
                     ) -> Union[list[tuple[torch.Tensor, torch.Tensor]], dict[str, torch.Tensor]]:
        """
        Overloads the parent's method.

        Default PyTorch Lightning function. Called automatically by the Lightning 
        framework during inference with the model.
        It facilitates the Monte Carlo Dropout (MCD) method. 
        
        Parameters:
        ===========
        batch: BatchEncoding. Batch of encoded data for inference.
        batch_idx: Int. Index of the batch.
        
        Returns:
        ========
        list[tuple[torch.Tensor, torch.Tensor], ...] or dict[str, torch.Tensor]. Returns a list of tuples containing tensors, 
            where the first tensor contains the index of the sample and the second contains the model predictions for that sample.
            Alternatively, if you run a prediction with Monte Carlo Dropout enabled, a dictionary is returned, containing keys and 
            the corresponding values for: "example_idxs", "y_T" (raw [softmax, if classification] predictions for samples from all 
            MCD iterations, listed separately), "y_mean" (the mean [softmax, if classification] prediction value for the samples 
            across the MCD iterations), "y_var" (variance calculated for the [softmax, if classification] predictions across 
            different MCD iterations), "y_majority_pred" (the class prediction as a result of the voting by the MCD models or NaNs 
            if this is a regression problem). 
        """
        example_idxs = batch.pop("idx")

        # Do we run the prediction with Monte Carlo Dropout?
        if self.args.montecarlo_dropout:
            y_T, y_mean, y_var, y_majority_pred = self.run_model_with_mc_dropout(batch=batch)
            results = {"example_idxs": example_idxs, "y_T": y_T, "y_mean": y_mean, "y_var": y_var, "y_majority_pred": y_majority_pred}
            return results

        outputs = self.model(**batch)
        logits = outputs["logits"]
        predictions = self._extract_predictions(logits=logits)
        results = list(zip(example_idxs, predictions))
        return results

    @logged()
    def run_model_with_mc_dropout(self, batch: BatchEncoding) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs model predictions using Monte Carlo Dropout method.
        The dropout layers in the model are set to the training mode to keep them active.
        
        See: https://doi.org/10.48550/arXiv.1506.02142
        See: https://github.com/microsoft/UST/blob/ee1b7b26ba876e792255df57aaf89f36ba8f6019/ust.py#L38 (function `mc_dropout_evaluate`)

        Parameters:
        ===========
        batch: BatchEncoding. Batch of encoded data for inference.
        
        Returns:
        ========
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]. Returns a tuple of four tensors, corresponding to
            `y_T` (raw [softmax, if classification] predictions, for samples from all MCD iterations, listed separately), 
            `y_mean` (the mean softmax prediction value for the samples across the MCD iterations), `y_var` (variance calculated for 
            the [softmax] predictions across different MCD iterations) and `y_majority_pred` (the class prediction as a result of the 
            voting by the MCD models for that sample [if applicable, otherwise a tensor filled with NaNs]).
        """
        # Iterate through all modules and set the dropout to train mode. 
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

        # Run the Monte Carlo Dopout models T-times.        
        y_T = []
        for _ in range(self.args.montecarlo_dropout_num_iters):
            outputs = self.model(**batch)
            y_T.append(outputs["logits"])

        y_T = torch.stack(y_T) # stacked logits
        return self._process_outputs_mc_dropout(y_T=y_T)  

    @logged()
    def _process_outputs_mc_dropout(self, y_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes the logits obtained from different Monte Carlo Iterations (MCD). 
        Generates softmax values for the logits (in case of classification, otherwise returns the input `y_T` tensor), calculates 
        mean of the [softmax, if classification] predictions, calculates variance for the [softmax, if classification] values 
        from different iterations, calulates the voting result from different MCD iterations.  

        See: https://github.com/microsoft/UST/blob/ee1b7b26ba876e792255df57aaf89f36ba8f6019/ust.py#L38 (function `mc_dropout_evaluate`)
                
        Parameters:
        ===========
        y_T: torch.Tensor. Stacked raw logits of model predictions obtained from different Monte Carlo Dropout iterations.
        
        Returns:
        ========
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]. Returns a tuple of four tensors, corresponding to
            `y_T` (raw [softmax, if classification] predictions, for samples from all MCD iterations, listed separately), 
            `y_mean` (the mean softmax prediction value for the samples across the MCD iterations), `y_var` (variance calculated for 
            the [softmax] predictions across different MCD iterations) and `y_majority_pred` (the class prediction as a result of the 
            voting by the MCD models for that sample [if applicable, otherwise a tensor filled with NaNs]).
        """

        if self.config.problem_type != "regression":  # in case of classification:
            # Convert the logits to softmax outputs 
            y_T = F.softmax(y_T, dim=-1)
            # Find the class index selected by the majority of the Monte Carlo Dropout iterations.  
            y_T_argmax = torch.argmax(y_T, dim=-1)  # finds the index of the maximum value along the last dimension
            y_majority_pred, _ = torch.mode(y_T_argmax, dim=0)  # finds the mode (the value that appears most frequently) along the first dimension (`dim=0`) of the tensor. The coutn tensor is ignored (as `_`)
        else:
            # Genrate a placeholder tensor for majority class predictions, filled with NaNs 
            # this tensor will be returned in case of regression.  
            y_majority_pred = torch.full([y_T.size()[1]], np.nan)

        # Calculate mean and variance of the values from different iterations
        y_mean = torch.mean(y_T, dim=0)
        y_var = torch.var(y_T, dim=0)
            
        return y_T, y_mean, y_var, y_majority_pred

    @logged()
    def calculate_loss_and_metrics(self, batch: BatchEncoding, metrics: torch.nn.ModuleDict, step_type: str) -> torch.Tensor:
        """
        Runs the model, collects loss and calculates metrics for the provided batch of data.
        Returns and logs loss. Metrics are not returned, but they are logged. 

        Parameters:
        ===========
        batch: BatchEncoding. Encoded batch data.
        metrics: torch.nn.ModuleDict. Dict-like datatype with metric names as keys and initialised metrics as values.
        step_type: Str. The name of the step (e.g. "train", "valid") to be included as a prefix before the metic names in the logs.
        
        Returns:
        ========
        loss: torch.Tensor. Loss calculated by the model for a given batch.
        """
        loss, *model_outputs = self._calculate_loss(self.model,
                                                    inputs=batch,
                                                    loss_fct=self.loss_function,
                                                    step_type=step_type,
                                                    loss_fct_no_reduction=self.loss_function_no_reduction)

        predictions = self._extract_predictions(logits=model_outputs[0])

        # Log the calculated loss and metrics
        self.log(step_type + '_loss', 
                 value=loss, 
                 on_epoch=self.args.log_on_epoch, 
                 on_step=self.args.log_on_step, 
                 sync_dist=True, 
                 logger=True)  # on_step=True

        if self.args.log_metrics:
            self.log_metrics(preds=predictions,
                            targets=batch["labels"],
                            metrics=metrics,
                            step_type=step_type)
        return loss

    @logged()
    def _calculate_loss(self, 
                        model: object, 
                        inputs: BatchEncoding, 
                        loss_fct: object, 
                        step_type: str=None, 
                        loss_fct_no_reduction: object=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the model prediction and returns the calculated loss and all the remaining model outputs.
        It accomodates adversarial loss calculations and also loss calculation for the confidence learning in the UST method. 

        Adversarial Training: https://doi.org/10.48550/arXiv.1412.6572
        UST: https://proceedings.neurips.cc/paper_files/paper/2020/file/f23d125da1e29e34c552f448610ff25f-Paper.pdf

        Parameters:
        ===========
        model: Object. Model to be used for predictions.
        inputs: BatchEncoding. Inputs to the model.
        loss_fct: Object. Initiated loss function.
        step_type: String. Type of the step that is currently performed. 
        loss_fct_no_reduction: Object. Initiated loss function with no reduction.
        
        Returns:
        ========
        torch.Tensor, torch.Tensor. Loss and the remaining model outputs. 
        """

        if step_type == "train" and self.args.confidence_learning:
            sample_weights = inputs.pop("sample_weights")
        else:
            sample_weights = None

        outputs = model(**inputs)

        # model outputs are always tuple in pytorch-transformers (see docs)
        if loss_fct:
            logits = outputs[1]
            labels = inputs["labels"]

            # If we are running a confidence learning UST method during training then
            # we need to take into consideration the sample weights and use a loss function with no reduction
            if step_type == "train" and self.args.confidence_learning:
                loss_fct = partial(self.run_loss_funct_no_reduction,
                                   loss_fct_no_reduction=loss_fct_no_reduction,
                                   sample_weights=sample_weights)

            loss = loss_fct(logits=logits, labels=labels)

        # Recalculate loss for adversarial training, only during the training stage
        if step_type == "train" and self.args.adversarial_training:
            logger.debug("Calculating adversarial loss")
            loss = self._calculate_adversarial_loss(loss=loss,
                                                    model=model,
                                                    inputs=inputs,
                                                    p=self.args.adversarial_training_probability,
                                                    loss_fct=loss_fct, # this can be a regular loss fct or loss fct for confidence learning UST
                                                    )

        return (loss, *outputs[1:])
    
    @logged()
    def run_loss_funct_no_reduction(self,
                          logits: torch.Tensor,
                          labels: torch.Tensor,
                          loss_fct_no_reduction: object=None,
                          sample_weights: torch.Tensor=None,
                          ) -> torch.Tensor:
        
        """
        A function to calculate loss based on the sample weights. 
        Rerquires an initialised loss function object with no reduction. 
        Returns calculated mean loss. 

        Parameters:
        ===========
        logits: torch.Tensor. Logits generated by the model for examples. 
        labels: torch.Tensor. Ground truth labels for the examples. 
        loss_fct_no_reduction: Object. Initiated loss function with no reduction.
        sample_weights: torch.Tensor. Weights of examples to use for scaling. 
        
        Returns:
        ========
        torch.Tensor. Calculated mean loss. 
        """
        loss = loss_fct_no_reduction(logits=logits, labels=labels)
        loss = loss * sample_weights
        return  loss.mean()

    @logged()
    def _extract_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Extracts predictions from the model logits.
        In case of regression, it just returns the logit (dimensions changed).
        In case of single-label binary classification, it returns the index of the max logit.
        In any other case, it just returns unchanged logits.
                
        Parameters:
        ===========
        model_outputs: torch.Tensor. Output generated by the model, without the loss, but otherwise unprocessed. 
        Logits are assumed to be at index 0 in the provided model_outputs list.

        Returns:
        ========
        torch.Tensor. Tensor containing model predictions ready to be fed to a TorchMetrics metric.  TorchMetrics. 
        """
        if self.config.problem_type == "regression":
            return logits.squeeze(1)
        elif self.config.num_labels == 2 and self.config.problem_type == "single_label_classification": 
            return logits.max(1).indices
        else:
            return logits
            
    @logged()
    def update_model_config(self) -> None:
        """
        Extends the parent method.

        In addition to the updates done by the parent method it 
        updates the model config with these new settings:
            - Updates the number of labels in the dataset.
            - Updates the problem type.
            - Updates the MLM flag.
        
        Returns:
        ========
        None
        """
        super().update_model_config()
        self.config.num_labels = self.args.num_labels
        self.config.problem_type = self.args.problem_type
        self.config.mlm = False

    @logged()
    def initiate_metrics(self) -> Union[torch.nn.ModuleDict, NoReturn]:
        """
        Initialises and returns appropriate TorchMeric metrics based on the problem_type and the number of labels. 
        
        Returns:
        ========
        initiated_metrics: torch.nn.ModuleDict. A dict-like data type containing
        initialised metric objects, where the keys are the metric names, and values are the objects. 
        """
        initiated_metrics = torch.nn.ModuleDict()  # regular Python dict will not work, when the model is run on a device different than CPU 
        
        if self.config.problem_type == "regression" and self.config.num_labels == 1:  # regression problem
            metrics = [ExplainedVariance, MeanAbsoluteError, MeanAbsolutePercentageError,
                        MeanSquaredError, R2Score,]  # SpearmanCorrCoef, PearsonCorrCoef
        
            for metric in metrics:
                self.metric = metric()
                initiated_metrics[metric.__name__] = self.metric
            initiated_metrics["RMSE"] = MeanSquaredError(squared=False)


        elif self.config.problem_type == "single_label_classification" and self.config.num_labels > 1:
            metrics = [Accuracy, AUROC, F1Score, MatthewsCorrCoef, CohenKappa]
        
            for metric in metrics:
                if self.config.num_labels == 2:  # binary classification problem
                    self.metric = metric(task="binary", num_classes=self.config.num_labels)
                else: # multiclass classification problem
                    try: # try setting average to macro when possible for a metric
                        self.metric = metric(task="multiclass", average="macro", num_classes=self.config.num_labels)
                    except ValueError:
                        self.metric = metric(task="multiclass", num_classes=self.config.num_labels)
                initiated_metrics[metric.__name__] = self.metric


        # multi_label_classification is only included here for potential future developments 
        elif self.config.problem_type == "multi_label_classification" and self.config.num_labels > 1:
            metrics = [Accuracy, AUROC, F1Score, MatthewsCorrCoef, CohenKappa, 
                        MultilabelRankingLoss]
            
            for metric in metrics:
                if metric is MultilabelRankingLoss:
                    self.metric = metric(num_classes=self.config.num_labels)
                    metrics[metric.__name__] = self.metric
                    continue

                self.metric = metric(task="multilabel", average="macro", num_classes=self.config.num_labels)  # consider different average type? 
                initiated_metrics[metric.__name__] = self.metric
        
        else:
            logger.error(f"{self.config.problem_type=} and  {self.config.num_labels=} are not compatible")
            raise Exception(f"The provided {self.config.problem_type=} and {self.config.num_labels=} are not compatible! "
                            "Also, make sure that the selected problem_type name is spelled correctly.")

        return initiated_metrics

    @logged()
    def log_metrics(self, preds: torch.Tensor, targets:torch.Tensor, metrics: torch.nn.ModuleDict, step_type: str="train") -> None:
        """
        Calculates and logs the metrics for a step. 
                
        Parameters:
        ===========
        preds: torch.Tensor. Tensor containing predicted lables/values.
        targets: torch.Tensor. Tensor contining the true labels/values.
        metrics: torch.nn.ModuleDict. Dict-like object containing initialised metrics as values, and keys with the metric names. 
        step_type: Str. The name of the step (e.g. "train", "valid") to be included as a prefix before the metic names in the logs.
    
        Returns:
        ========
        None
        """
        # If there are fewer than 2 predictions, do not try to run the forward method on the metric,
        # as it will cause issues with some metrics like R2, and even when the ValueError exception
        # is caught, the metric will fail at the end of the whole epoch. 
        if len(preds) < 2:
            logger.debug("Fewer than two predictions, does not calculate metrics")
            return None

        for metric_name, metric in metrics.items():
            try:
                metric(preds, targets)

                name_for_log = step_type + "_" + metric_name
                self.log(name_for_log, 
                        metric, 
                        on_epoch=self.args.log_on_epoch,
                        on_step=self.args.log_on_step, 
                        sync_dist=True, 
                        logger=True)
            except ValueError as e:
                logger.debug(e)
                print(e) 
    
    @logged()
    def on_validation_epoch_end(self) -> None:
        """
        Default PyTorch Lightning function. Called automatically by the Lightning framework.
        Performs actions at the end of the validation:
            - None
        
        Needed as it overwrites the parent's method.
              
        Returns:
        ========
        None
        """
        pass

    @logged()
    def on_fit_end(self) -> None:
        """
        Not implemented.
        Default PyTorch Lightning function. Called automatically by the Lightning framework.
        
        Returns:
        ========
        None
        """
        pass

    @logged()
    def on_validation_end(self) -> None:
        """
        Not implemented.
        Default PyTorch Lightning function. Called automatically by the Lightning framework.
        
        Returns:
        ========
        None
        """

        # TODO construct a confusion matrix if binary classification
        # TODO construct a heatmap if multiclass classification
        # TODO construct real vs predicted value chart if regression
        pass
