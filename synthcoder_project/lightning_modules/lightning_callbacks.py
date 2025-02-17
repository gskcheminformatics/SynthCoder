# Module containing custom callback classes for the Pytorch Lightning trainer.

import logging
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, BasePredictionWriter
from fsspec.core import url_to_fs
import torch
import os
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)


class HfModelCheckpoint(ModelCheckpoint):

    """
    Extends the original callback from PyTroch Lightning allowing to save checkpoints.
    The class saves normal PyTorch Lightning checkpoints with all the original functionality but it also saves 
    model configuration, weights, special tokens map, tokenizer config, training args and vocabulary 
    as separatre files in a distinc folder.  

    Taken from https://github.com/Lightning-AI/pytorch-lightning/issues/3096
    """

    @logged()
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @logged()
    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        """
        Extends the original ModelCheckpoint method.
        Saves model configuration, weights, special tokens map, tokenizer config, training args and vocabulary
        as separate files in a distinct .dir folder.

        Parameters:
        ===========
        trainer: pl.Trainer. The Pytorch Lightning trainer object. 
        filepath: Str. Filepath for saving the checkpoint files. 
        
        Returns:
        ========
        None
        """
        super()._save_checkpoint(trainer, filepath)
        hf_save_dir = filepath+".dir"
        if trainer.is_global_zero:
            trainer.lightning_module.model.save_pretrained(hf_save_dir)
            trainer.lightning_module.tokenizer.save_pretrained(hf_save_dir)
            trainer.lightning_module.args.save(hf_save_dir)
            torch.save(trainer.lightning_module.args, os.path.join(hf_save_dir, "training_args.bin"))
    

    # https://github.com/Lightning-AI/lightning/pull/16067
    @logged()
    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        """
        Extends the original ModelCheckpoint method.
        Remove checkpoint filepath from the filesystem,
        in this case, removes the specific .dir folder.

        Parameters:
        ===========
        trainer: pl.Trainer. The Pytorch Lightning trainer object. 
        filepath: Str. Filepath for saving the checkpoint files. 
        
        Returns:
        ========
        None 
        """
        super()._remove_checkpoint(trainer, filepath)
        hf_save_dir = filepath+".dir"
        if trainer.is_global_zero:
            fs, _ = url_to_fs(hf_save_dir)
            if fs.exists(hf_save_dir):
                fs.rm(hf_save_dir, recursive=True)


class CustomWriter(BasePredictionWriter):
    """
    A class for saving inference results as files, appropiate for multi-device inference.
    
    Taken from: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html
    """

    @logged()
    def __init__(self, output_dir: str, write_interval: str) -> None:
        """
        Initiates the object.
        
        Parameters:
        ===========
        output_dir: Str.
        write_interval: Str. 

        Returns:
        ========
        None
        """
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @logged()
    def write_on_epoch_end(self, trainer: pl.Trainer, pl_module: object, predictions: list, batch_indices: list) -> None:
        """
        Writes .pt files. The first group of files (`predictions_{trainer.global_rank}.pt`) contains outputs of the model during the inference returned by the predict_step() method.
        The second group of files (`batch_indices_{trainer.global_rank}.pt`) contains the information about the batch indices for the generated predictions.  
        Appropriate for multi-device inference.

        Parameters:
        ===========
        trainer: pl.Trainer. Lightning trainer. 
        pl_module: Object. Lightning module object. - Not used here.
        predictions: List. Model predictions generated during the inference.
        batch_indices. List. Batch indices associated with the generated predictions.

        Returns:
        ========
        None
        """
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))