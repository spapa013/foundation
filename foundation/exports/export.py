from pathlib import Path
import numpy as np
import pandas as pd
from itertools import repeat
from typing import Union
from foundation.utils.logging import get_logger, tqdm
from foundation.exports import utils
from djutils import cache_rowproperty

logger = get_logger(__name__)

def export_readout_weights_and_locations(
    data_ids: Union[list, tuple], 
    network_id: str, 
    instance_id: str, 
    target_dir=None
):
    """
    Export readout feature weights and locations for a set of scans specified by data_id.

    Parameters:
        data_ids (list or tuple): List of data IDs for the scans to export.
        network_id (str): ID of the network.
        instance_id (str): ID of the instance.
        target_dir (os.PathLike | None): Directory to save the exported data. If None, uses the current working directory.
    Returns:
        None: The function saves the readout weights and locations to the specified target directory.
    Raises:
        AssertionError: If data_ids is not a list or tuple.
    """
    logger.info(f"Exporting readout weights and locations...")
    target_dir = utils.prepare_target_directory(target_dir)
    
    logger.info(f"Loading data...")
    assert isinstance(data_ids, (list, tuple)), "data_ids should be a list or tuple of data IDs"
    weights_arrays, location_arrays, unit_dfs, meta_dfs = [], [], [], []
    for net_id, inst_id, data_id in tqdm(zip(repeat(network_id), repeat(instance_id), data_ids), total=len(data_ids), desc="Scans"):
        model_key = {'network_id': net_id, 'instance_id': inst_id, 'data_id': data_id}
        weight_array, location_array, unit_df, meta_df = utils.get_readout_weights_and_location(**model_key)
        weights_arrays.append(weight_array)
        location_arrays.append(location_array)
        unit_dfs.append(unit_df)
        meta_dfs.append(meta_df)

    logger.info(f"Consolidating data across scans...")
    weights_array = np.concatenate(weights_arrays)  # units x (stream x features)
    location_array = np.concatenate(location_arrays)  # units x (X position, Y position)
    scan_unit_df = pd.concat(unit_dfs, ignore_index=True)
    scan_meta_df = pd.concat(meta_dfs, ignore_index=True)
    
    logger.info(f"Saving data to {target_dir}...")
    np.save(target_dir / "readout_weights.npy", weights_array)
    np.save(target_dir / "readout_locations.npy", location_array)
    scan_unit_df.to_csv(target_dir / "units.csv", index=False)
    scan_meta_df.to_csv(target_dir / "metadata.csv", index=False)


def export_performance_metrics(
    data_ids: Union[list, tuple], 
    network_id: str, 
    instance_id: str, 
    trial_filterset_id,
    videoset_id,
    perspective,
    modulation,
    burnin_frames,
    target_dir=None
):
    """
    Export performance metrics for a set of scans specified by data_ids.

    Parameters:
        data_ids (list or tuple): List of data IDs for the scans to export.
        network_id (str): ID of the network.
        instance_id (str): ID of the instance.
        trial_filterset_id (str): ID of the trial filter set.
        videoset_id (str): ID of the video set.
        perspective (str): Perspective to use for the performance metrics.
        modulation (str): Modulation to use for the performance metrics.
        burnin_frames (int): Number of burn-in frames to use for the performance metrics.
        target_dir (os.PathLike | None): Directory to save the exported data. 
            If None, uses the current working directory.
    Returns:
        None: The function saves the performance metrics to the specified target directory.
    Raises:
        AssertionError: If data_ids is not a list or tuple.
    """
    logger.info(f"Exporting performance metrics...")
    target_dir = utils.prepare_target_directory(target_dir)

    logger.info(f"Loading data...")
    assert isinstance(data_ids, (list, tuple)), "data_ids should be a list or tuple of data IDs"
    unit_dfs, meta_dfs = [], []
    for net_id, inst_id, data_id in tqdm(zip(repeat(network_id), repeat(instance_id), data_ids), total=len(data_ids), desc="Scans"):
        model_key = {'network_id': net_id, 'instance_id': inst_id, 'data_id': data_id}
        unit_df, meta_df = utils.get_model_performance(
            **model_key,
            trial_filterset_id=trial_filterset_id,
            videoset_id=videoset_id,
            perspective=perspective,
            modulation=modulation,
            burnin_frames=burnin_frames
        )
        unit_dfs.append(unit_df)
        meta_dfs.append(meta_df)

    logger.info(f"Consolidating data across scans...")
    scan_unit_df = pd.concat(unit_dfs, ignore_index=True)
    scan_meta_df = pd.concat(meta_dfs, ignore_index=True)

    logger.info(f"Saving data to {target_dir}...")
    scan_unit_df.to_csv(target_dir / "units.csv", index=False)
    scan_meta_df.to_csv(target_dir / "metadata.csv", index=False)


def export_orientation_direction_tuning(
    data_ids: Union[list, tuple], 
    network_id: str, 
    instance_id: str, 
    videoset_id: str,
    offset_id: str,
    impulse_id: str,
    precision_id: str,
    target_dir=None
):
    """
    Export orientation and direction tuning data for a set of scans specified by data_ids.

    Parameters:
        data_ids (list or tuple): List of data IDs for the scans to export.
        network_id (str): ID of the network.
        instance_id (str): ID of the instance.
        videoset_id (str): ID of the video set.
        offset_id (str): ID of the offset.
        impulse_id (str): ID of the impulse.
        precision_id (str): ID of the precision.
        target_dir (os.PathLike | None): Directory to save the exported data. 
            If None, uses the current working directory.
    
    Returns:
        None: The function saves the orientation and direction tuning data to the specified target directory.
    
    Raises:
        AssertionError: If data_ids is not a list or tuple.
    """
    logger.info(f"Exporting orientation and direction tuning data...")
    target_dir = utils.prepare_target_directory(target_dir)

    logger.info(f"Loading data...")
    assert isinstance(data_ids, (list, tuple)), "data_ids should be a list or tuple of data IDs"
    unit_dfs, meta_dfs = [], []
    for net_id, inst_id, data_id in tqdm(zip(repeat(network_id), repeat(instance_id), data_ids), total=len(data_ids), desc="Scans"):
        model_key = {'network_id': net_id, 'instance_id': inst_id, 'data_id': data_id}
        unit_df, meta_df = utils.get_orientation_direction_tuning(
            **model_key,
            videoset_id=videoset_id,
            offset_id=offset_id,
            impulse_id=impulse_id,
            precision_id=precision_id,
        )
        unit_dfs.append(unit_df)
        meta_dfs.append(meta_df)

    logger.info(f"Consolidating data across scans...")
    scan_unit_df = pd.concat(unit_dfs, ignore_index=True)
    scan_meta_df = pd.concat(meta_dfs, ignore_index=True)

    logger.info(f"Saving data to {target_dir}...")
    scan_unit_df.to_csv(target_dir / "units.csv", index=False)
    scan_meta_df.to_csv(target_dir / "metadata.csv", index=False)

def export_stimulus_and_model_responses(
    data_ids: Union[list, tuple], 
    network_id: str, 
    instance_id: str, 
    videoset_id: str,
    test: bool = False,
    target_dir=None
):
    """
    Export stimulus and model responses for a set of scans specified by data_ids.
    Parameters:
        data_ids (list or tuple): List of data IDs for the scans to export.
        network_id (str): ID of the network.
        instance_id (str): ID of the instance.
        videoset_id (str): ID of the video set.
        target_dir (os.PathLike | None): Directory to save the exported data. 
            If None, uses the current working directory.
    Returns:
        None: The function saves the stimulus and model responses to the specified target directory.
    Raises:
        AssertionError: If data_ids is not a list or tuple.
    """
    logger.info(f"Exporting stimulus videos and model responses...")
    target_dir = utils.prepare_target_directory(target_dir)

    logger.info(f"Loading data...")
    resp_arrays, unit_dfs, meta_dfs = [], [], []
    for net_id, inst_id, data_id in tqdm(zip(repeat(network_id), repeat(instance_id), data_ids), total=len(data_ids), desc="Scans"):
        model_key = {'network_id': net_id, 'instance_id': inst_id, 'data_id': data_id}
        resp_array, stim_array, unit_df, meta_df = utils.compute_model_responses(
            **model_key,
            videoset_id=videoset_id,
            test=test
        )
        resp_arrays.append(resp_array)
        unit_dfs.append(unit_df)
        meta_dfs.append(meta_df)
    
    logger.info(f"Consolidating data across scans...")
    scan_resp_array = np.concatenate(resp_arrays) # units x frames
    scan_unit_df = pd.concat(unit_dfs, ignore_index=True)
    scan_meta_df = pd.concat(meta_dfs, ignore_index=True)
    
    logger.info(f"Saving data to {target_dir}...")
    np.save(target_dir / "responses.npy", scan_resp_array)
    scan_unit_df.to_csv(target_dir / "units.csv", index=False)
    scan_meta_df.to_csv(target_dir / "metadata.csv", index=False)
    np.save(target_dir / "stimulus.npy", stim_array)  # Save the last stimulus array (they are identical across scans)


def export_training_data(data_id, target_dir=None):
    """
    Export the training dataset and metadata for a given data_id.
    
    Parameters:
        data_id (str): ID of the data used for the model.
        target_dir (os.PathLike | None): Directory to save the exported data. If None, uses the current working directory.
    
    Returns:
        tuple: (dataset_df, meta_df) where dataset_df is a DataFrame with the training dataset paths
               and meta_df is a DataFrame with metadata.
    """
    logger.info(f"Exporting training data...")
    target_dir = utils.prepare_target_directory(target_dir)

    logger.info(f"Loading data...")
    dataset_df, meta_df = utils.get_training_dataset(data_id)
    
    logger.info(f"Making destination subdirectory...")
    animal_id = meta_df["animal_id"].values.item()
    session = meta_df["session"].values.item()
    scan_idx = meta_df["scan_idx"].values.item()
    dst_dir = target_dir / f'{animal_id}_{session}_{scan_idx}'
    dst_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Destination subdirectory set to: {dst_dir}")

    logger.info(f"Saving data...")
    for col in dataset_df:
        col_dir = dst_dir / col
        col_dir.mkdir(exist_ok=True)
        for i, value in tqdm(enumerate(dataset_df[col]), desc=col, total=len(dataset_df)):
            fp = col_dir / f'trial{i}.npy'
            if hasattr(value, 'fp'):
                np.save(fp, np.load(value.fp))
            else:
                np.save(fp, value)
    
    logger.info(f"Saving metadata...")
    meta_df.to_csv(dst_dir / "metadata.csv", index=False)


def export_visual_trial_data(data_id, network_id, instance_id, trial_filterset_id, videoset_id, target_dir=None):
    """
    Export visual trial data for a given data_id, network_id, instance_id, trial_filterset_id, and videoset_id.
    
    Parameters:
        data_id (str): ID of the data used for the model.
        network_id (str): ID of the network.
        instance_id (str): ID of the instance.
        trial_filterset_id (str): ID of the trial filter set.
        videoset_id (str): ID of the video set.
        target_dir (os.PathLike | None): Directory to save the exported data. 
            If None, uses the current working directory.
    Returns:
        None: The function saves the visual trial data to the specified target directory.            
    """
    logger.info(f"Exporting visual trial data...")
    target_dir = utils.prepare_target_directory(target_dir)

    logger.info(f"Loading data...")
    data, trial_df, meta_df = utils.get_visual_trial_data(
        data_id=data_id, 
        network_id=network_id, 
        instance_id=instance_id, 
        trial_filterset_id=trial_filterset_id, 
        videoset_id=videoset_id
    )

    logger.info(f"Making destination subdirectory...")
    animal_id = meta_df["animal_id"].values.item()
    session = meta_df["session"].values.item()
    scan_idx = meta_df["scan_idx"].values.item()
    dst_dir = target_dir / f'{animal_id}_{session}_{scan_idx}'
    dst_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Destination subdirectory set to: {dst_dir}")

    logger.info(f"Saving data...")
    data_type_names = ['stimuli', 'units', 'modulations', 'perspectives']
    base_dirs = {dtn: dst_dir / dtn for dtn in data_type_names}
    
    for bd in base_dirs.values():
        bd.mkdir(exist_ok=True)
    
    with cache_rowproperty():
        for i, (_, vdf) in enumerate(tqdm(trial_df.groupby("video_id"), desc="Videos")):
            # trial ids
            trial_ids = list(vdf.trial_id)

            # trial stimuli
            stimuli = list(data.trial_stimuli(trial_ids))
            
            # all of the stimuli should be the same
            assert all(np.array_equal(s, stimuli[0]) for s in stimuli), (
                "All trials should have the same stimulus"
            )
            
            # trial units
            units = list(data.trial_units(trial_ids))

            # trial perspectives
            perspectives = list(data.trial_perspectives(trial_ids))

            # trial modulations
            modulations = list(data.trial_modulations(trial_ids))

            assert len(stimuli) == len(trial_ids) == len(units) == len(perspectives) == len(modulations)
            
            # save data
            vid_name = f'video{i}'
            
            ## create video directory for each data type
            vid_dirs = {data_type_names: base_dir / vid_name for data_type_names, base_dir in base_dirs.items()}
            for vid_dir in vid_dirs.values():
                vid_dir.mkdir(exist_ok=True)        
            
            data_to_save = {dtn: data for dtn, data in zip(data_type_names, [stimuli, units, perspectives, modulations])}
            for data_type_name, data_type in data_to_save.items():
                for j, trial_data in enumerate(data_type):
                    trial_name = f'repeat{j}.npy'
                    np.save(vid_dirs[data_type_name] / trial_name, trial_data)
    
    logger.info(f"Saving metadata...")
    meta_df.to_csv(dst_dir / "metadata.csv", index=False)