# Module with some utility functions

from typing import Union
import logging
import os
import random
import shutil
import regex as re
import json
import torch
import pandas as pd
import lightning as pl
from rdkit import Chem
from rdkit import RDLogger
import jpype
import jpype.imports
from transformers.tokenization_utils_base import BatchEncoding
from synthcoder_project.setup_logger import logged, create_logger

logger = create_logger(module_name=__name__)

RDLogger.DisableLog('rdApp.*')


@logged()
def load_model_args(input_dir: str, model_arg_object: object) -> object:
    """ 
    Initialises model_arg_object and then updates it with predefined args.
    The predefined args are loaded form a specified file.
 
    Parameters:
    ===========
    input_dir: Str. Path to the folder containg the file with the predefined args (model_args.json).  
    model_arg_object: Object. Args object for a specific model. 
    
    Returns:
    ========
    args: Object. Updated and initiated args object. 
    """
    args = model_arg_object()
    args.load(input_dir)
    return args


@logged()
def set_manual_seed(seed: int, workers: bool=True) -> None:
    """
    Sets seed for pseudo-random number generators in: pytorch, numpy, python.random etc.

    Parameters:
    ===========
    seed: Int. A number to be used as a seed.
    workers: Bool. If set to True, will properly configure all dataloaders 
    passed to the Trainer with a worker_init_fn. If the user already provides 
    such a function for their dataloaders, setting this argument will 
    have no influence.
    
    Returns:
    ========
    None
    """
    logger.debug(f"Setting random seed to {seed}")
    pl.seed_everything(seed=seed, workers=workers)
    # torch.use_deterministic_algorithms(deterministic)
    # torch.backends.cudnn.deterministic = deterministic
    # torch.backends.cudnn.benchmark = benchmark
    os.environ["PYTHONHASHSEED"] = str(seed)


@logged()
def overwrite_with_not_null(var_to_overwrite: any, new_var_value: any) -> any:
    """
    Overwrites var_to_overwrite if the new value is not null.

    Parameters:
    ===========
    var_to_overwrite: Any. Variable to overwrite. 
    new_var_value: Any. Variable/vale to assign to var_to_overwrite
    
    Returns:
    ========
    var_to_overwrite: Any. The variable with a new or old value, depending on the outcome of the method.  
    """
    var_to_overwrite = new_var_value if new_var_value else var_to_overwrite
    return var_to_overwrite


@logged()
def save_df_as_csv(df: pd.DataFrame, csv_path_for_saving: str) -> None:
    """
    Saves Pandas as a .csv file under at the specified path.
    If the desired directory  for the output file does not yet exist, 
    it will be created.

    Parameters:
    ===========
    df: pd.DataFrame. Dataframe to save as .csv
    csv_path_for_saving: Str. Path for the .csv file to create.
    
    Returns:
    ========
    None
    """
    logger.debug(None)
    # Create a new directory if needed
    dir_name = os.path.dirname(csv_path_for_saving)
    os.makedirs(dir_name, exist_ok=True)
    
    # Save all the predictions as a .csv file
    df.to_csv(csv_path_for_saving, index=False)


@logged()
def create_directory_and_delete_files_inside(directory: str, 
                                             file_extension: Union[str, tuple[str, ...]]) -> None:
    """
    Creates a new directory, and if the directory already exists it 
    deletes the files inside with the specified extension(s).

    Parameters:
    ===========
    directory: Str. Directory that needs to be created/cleaned up.
    file_extension: Str or Tuple of strings. Extension(s) of the files to be deleted. 
    
    Returns:
    ========
    None
    """
    os.makedirs(directory, exist_ok=True)
    filelist = [ file for file in os.listdir(directory) if file.endswith(file_extension)]
    # print(filelist)
    for file in filelist:
        # print(file)
        try:
            os.remove(os.path.join(directory, file))
        except FileNotFoundError:
            print(f"\nAttempted to delete file '{file}', but the file was not found.")


@logged()
def remove_tree_directory(directory: str) -> None:
    """
    Removes a whole directory tree.

    Parameters:
    ===========
    directory: Str. Directory that needs to be removed with all its content.
    
    Returns:
    ========
    None
    """
    try:
        shutil.rmtree(directory)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")


@logged()
def return_combined_data_from_pt_files(directory: str, regex_pattern: str="predictions_\d*.pt") -> list:
    """
    Uses PyTorch to read .pt files and combines the content from these files into one list.

    Parameters:
    ===========
    directory: Str. Directory that needs to be removed with all its content.
    regex_pattern: <Optional> Str. Regex pattern to use for detection of appropriate files in the directory.

    Returns:
    ========
    file_content: List. Content from all the PyTorch files matching the regex, put into one list. 
    """
    filelist = [file for file in os.listdir(directory) if (file.endswith(".pt") and re.findall(regex_pattern, file))]
    if not filelist:
        raise FileNotFoundError(f"No expected inference files were found in {directory}. Make sure that `create_prediction_files` is set to `True` and the correct `prediction_output_dir` is provided.")
    
    file_content = []
    for file in filelist:
        try:
            file_content += torch.load(os.path.join(directory, file))
        except FileNotFoundError:
            print(f"\nAttempted to read file '{file}', but the file was not found.")
    
    return file_content


@logged()
def check_smiles_and_chemical_validity_with_rdkit(smiles: str) -> tuple[bool, bool]:
    """
    Checks correctness of the provided SMILES string in respect to the grammar of SMILES and the chemistry.
    The checks are done via RDKit. The chemistry is considered to be valid if the SMILES can be converted 
    into a molecule after sanitisation. 

    Parameters:
    ===========
    smiles: Str. The SMILES string to evaluate.
    
    Returns:
    ========
    valid_smiles, valid_molecule: tuple[bool, bool]. Bool values indicating if the SMILES is grammatically 
                                                     correct, and if it can be interpreted as a valid molecule.
    """
    valid_smiles = False
    valid_molecule = False
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m:
        valid_smiles = True  # is the SMILES grammar correct?
    try:
        Chem.SanitizeMol(m)  # can the SMILES be converted into a molecule after sanitisation?
        valid_molecule = True  
    except:
        pass
    return valid_smiles, valid_molecule


@logged()
def check_chemical_validity_with_structurecheck(smiles: str) -> bool:
    """
    Checks correctness of the provided SMILES string using ChemAxon StructureCheck.
    Uses Java for evaluation, through jpype.

    Parameters:
    ===========
    smiles: Str. The SMILES string to evaluate.
    
    Returns:
    ========
    bool. Bool values indicating if the SMILES is correct according to the predefined rules.
    """
    jvmpath=os.getenv('JAVA_ROOT') + '/lib/server/libjvm.so'
    
    if not jpype.isJVMStarted():
        jpype.startJVM(jvmpath, "-ea", "-Djava.class.path="+os.getenv('CHEMAXON')+"/dist/chem.jar", "-Djava.util.logging.config.file="+os.getenv('CHEMAXON')+"/log.properties")
    
    # should not have a java directory where you are running it from
    import java.lang
    
    if not java.lang.Thread.isAttached():
        jpype.attachThreadToJVM()
    
    jpype.addClassPath(os.getenv('CHEMAXON')+'/dist/lib/*')
    
    StructureChecker=jpype.JClass("com.gsk.csc.chem.StructureCheckerComponent")
    md = jpype.JClass("com.gsk.csc.chem.data.MoleculeData")
    
    check=StructureChecker(jpype.java.lang.Boolean(True), jpype.java.lang.Boolean(True), jpype.JString(''))
    md=check.checkStructure(jpype.JString(smiles))

    return md.pass_flag
    # md.getcxSmiles()
    # md.getProcessingComments()


@logged()
def convert_string_list_to_tensor(string_list: str) -> torch.Tensor:
    """
    Converts a string, where the string is a list of lists containing numerical values, to a troch.Tensor.

    Parameters:
    ===========
    string_list: Str. String - list of lists, to convert into a tensor.
    
    Returns:
    ========
    torch.Tensor. Resulting tensor. 


    Example:

        Input to the function:

        "[[1,2,3], [4,5,6], [7,8,9], [10,11,12]]"
  
        
        Output of the function:

        tensor([[ 1.,  2.,  3.],
                [ 4.,  5.,  6.],
                [ 7.,  8.,  9.],
                [10., 11., 12.]])

    """
    converted_list = json.loads(string_list)
    return torch.Tensor(converted_list)


@logged()
def convert_batch_content_to_tensors(func: object) -> object:
    """
    Decorator, to make sure that all data in the batch for the model are tensors. 
    """
    def inner(batch: BatchEncoding, *args, **kwargs) -> object:

        for key, value in batch.copy().items():
            if not isinstance(value, torch.Tensor):
                batch[key] = torch.Tensor(value)

        return func(batch, *args, **kwargs)
    return inner


@logged()
def return_True_or_False(probability_of_True: float=0.5) -> bool:
    """
    Returns `True` or `False` by random. You can specify the probability of returning `True`.

    Parameters:
    ===========
    probability_of_True: <Optional> Float. Probability of returning `True`. Should be between 0 and 1.
    
    Returns:
    ========
    true_or_false: Bool. `True` or `False` boolian.  
    """
    true_or_false = random.choices([True, False], weights=(probability_of_True, 1-probability_of_True))[0]
    return true_or_false


@logged()
def calc_class_weights_from_df(df: pd.DataFrame) -> list:
    """
    Calculates weights for labels, based on the data from Pandas Dataframe. The weights are calculated to balance the occurence of labels.  

    Parameters:
    ===========
    df: pd.Dataframe. Dataframe containing column called "labels". 

    Returns:
    ========
    class_weights: List. Weights for labels. Weights are returned in a list, in order corresponding to increasingly sorted labels.  

    """
    # Clount number of normalised occurences for each label
    value_counts = df["labels"].value_counts(normalize=True).to_dict()

    # Sort the labels from smallest to largest (0, 1, 2...) and convert the correponding normalised ocurrences to a tensor
    normalised_counts = torch.Tensor(list(dict(sorted(value_counts.items())).values()))

    # "Invert" the normalised occurences - deprioritise the more often occuring ones and increase priority of the less occuring ones, so that all labels have the same balanced impact 
    class_weights = (1.0/normalised_counts)/torch.sum(1.0/normalised_counts)
    return class_weights.tolist()


@logged()
def gather_prediction_results(prediction_output_dir: str) -> Union[list, None]:
    """
    Gathers and combines model prediction results produced during distributed inference.
    This function checks if PyTorch's distributed computing is being used with more than one device. 
    If so, it collects prediction results from multiple `.pt` files corresponding to individual devices 
    and combines them into a single list of model predictions.

    Parameters:
    ===========
    prediction_output_dir: Str. The directory where `.pt` files containing individual device 
        prediction results are stored. These files are generated during distributed inference runs.

    Returns:
    ========
    list: A list containing combined model predictions from all devices involved 
        in the distributed inference process. If the world size is 1 or less, no combining is done 
        and the function will return None.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_world_size(group=None) > 1:
        logger.debug(f"{torch.distributed.get_world_size(group=None)=}; Combining prediction data from .pt files")
        model_predictions = return_combined_data_from_pt_files(prediction_output_dir)
        return model_predictions


@logged()
def calc_combined_mean_and_var(means: list, variances: list, sample_sizes: list) -> tuple:
    """
    Calculates combined mean and variance for different mean and variances collected for a number of data sets.
    
    Parameters:
    ===========
    means: List. Means for data sets which should be combined.
    variances: List. Varaiances corresponding to the provided means. 
    sampe_sizes: List. A list of the corresponsing sample sizes used to calculate a given mean and variance.
   
    Returns:
    ========
    combined_mean, combined_variance: Tuple. Calculated total mean and variance for the provided input.

    Equations:
    ==========
    Combined Mean:
        combined_mean = (sum of (sample_size_i * mean_i) for all datasets) / (total number of samples)
    Combined Variance:
        combined_variance = (sum of ((sample_size_i - 1) * variance_i) + sum of (sample_size_i * (mean_i - combined_mean)^2)) / (total number of samples - 1)
    
    where "i" is the index of each dataset.
    """

    # Make sure that the correct number of means, variances and the corresponding sample sizes was provided
    assert len(means) == len(variances) == len(sample_sizes), "All provided lists should have the same number of elements!"

    # Total number of samples across all data sets
    total_samples = sum(sample_sizes)

    # Calculate combined mean weighted by the number of corresponding samples
    temp_sum_mu = 0
    for mean, sample_size in zip(means, sample_sizes):
        temp_sum_mu += sample_size * mean
    combined_mean = temp_sum_mu / total_samples

    # Calculate combined variance
    temp_sum_variances = 0
    temp_sum_means = 0 
    for mean, variance, sample_size in zip(means, variances, sample_sizes):
        temp_sum_variances += (sample_size - 1)*variance
        temp_sum_means += sample_size*((mean-combined_mean)**2)
    combined_variance = (temp_sum_means + temp_sum_variances) / (total_samples - 1)

    return combined_mean, combined_variance