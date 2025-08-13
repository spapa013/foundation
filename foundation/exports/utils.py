from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from foundation.fnn import model
from foundation.virtual import fnn, utility, stimulus, recording, tuning
from foundation.virtual.bridge import pipe_fuse
from foundation.tuning.compute.direction import bi_von_mises, uniform
from foundation.fnn.model import Model
from foundation.utility.resize import Resize
from foundation.fnn import data
from foundation.fnn.data import Data
from foundation.stimulus.video import VideoSet, Video
from foundation.recording.compute.visual import VisualTrialSet
from foundation.utils.logging import tqdm, disable_tqdm
from fnn.model.utils import isotropic_grid_sample_2d
from djutils import merge
from foundation.utils.logging import get_logger


logger = get_logger(__name__)


def prepare_target_directory(target_dir=None):
    """
    Prepare the target directory for exporting data.
    
    Parameters:
        target_dir (os.PathLike | None): Directory to save the exported data. If None, uses the current working directory.
    
    Returns:
    """
    logger.info(f"Preparing target directory...")
    target_dir = Path(target_dir) if target_dir is not None else Path()
    target_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Target directory is: {target_dir}")
    return target_dir


def make_metadata_df(data_id, network_id=None, instance_id=None, **kwargs):
    """
    Create a DataFrame with scan metadata for the given model.
    Args:
        - data_id: ID of the data used for the model.
        - network_id: ID of the network.
        - instance_id: ID of the model instance.
    Returns:
        - meta_df: DataFrame with metadata.
    """
    if network_id is not None and instance_id is not None:
        meta_rel = model.Model * fnn.Data.VisualScan & {'network_id': network_id, 'instance_id': instance_id, 'data_id': data_id}
    else:
        meta_rel = fnn.Data.VisualScan & {'data_id': data_id}
    meta_df = pd.DataFrame(
        meta_rel.fetch(
            'animal_id',
            'session',
            'scan_idx',
            "KEY",
            as_dict=True,
        )
    )
    first_cols = ['animal_id', 'session', 'scan_idx']
    meta_df = meta_df[first_cols + [col for col in meta_df.columns if col not in first_cols]]
    for k, v in kwargs.items():
        meta_df[k] = v
    return meta_df


def get_readout_weights_and_location(network_id, instance_id, data_id):
    """
    Get readout weights and their locations for a given model.
    Args:
        - network_id: ID of the network.
        - instance_id: ID of the model instance.
        - data_id: ID of the data used for the model.
    Returns:
        - weights_array: Array of readout weights (units x (stream x features)).
        - location_array: Array of readout locations (units x 2).
        - unit_df: DataFrame with unit information.
        - meta_df: DataFrame with metadata.
    """
    model_key = {'network_id': network_id, 'instance_id': instance_id, 'data_id': data_id}
    # get model
    m = (model.Model & model_key).model(device="cpu")
    m._reset()

    # Readout weights
    readout = m.readout.feature().numpy()
    assert readout.shape[2] == 1, "more than 1 readout found!"
    readout = readout[:, :, 0, :]  # stream x unit x features
    readout = readout.transpose(1, 0, 2)  # unit x stream x features
    weights_array = readout.reshape(readout.shape[0], -1)  # unit x (stream * features)

    # Readout location
    height, width = (
        fnn.Spec.VisualSpec
        & (fnn.Data.VisualScan & {"data_id": model_key["data_id"]})
    ).fetch1("height", "width")
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    default_perspective = np.zeros(m.perspectives)[None]
    default_perspective = torch.tensor(default_perspective, dtype=torch.float)

    # default perspective is zero
    # https://github.com/cajal/fnn/blob/614d33e76fe4415f8c6d0d295b9201459874f1a4/fnn/model/networks.py#L253
    stim_grids = torch.stack(
        [
            torch.tensor(x_grid, dtype=torch.float),
            torch.tensor(y_grid, dtype=torch.float),
        ]
    )[:, None, ...]  # N, S, H, W

    # get the perspective transform of the sitmulus x, y grid
    # https://github.com/cajal/fnn/blob/614d33e76fe4415f8c6d0d295b9201459874f1a4/fnn/model/perspectives.py#L182-L183
    rmat = m.perspective.rmat(default_perspective).expand(2, -1, -1)
    rays = m.perspective.retina.rays(rmat)
    grid = m.perspective.monitor.project(rays)
    readout_grids = isotropic_grid_sample_2d(
        stim_grids, grid=grid, pad_mode="constant", pad_value=-1
    )
    readout_grids[readout_grids < 0] = np.nan

    # find the readout locations
    readout_pos = m.readout.bound(m.readout.position.mean)
    # map the readout locations to the stimulus grid
    stim_pos = (
        F.grid_sample(
            readout_grids,
            readout_pos.expand(2, -1, -1).unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    stim_x, stim_y = stim_pos
    stim_x_nrm = 2 * (stim_x / (width - 1)) - 1
    stim_y_nrm = 2 * (stim_y / (height - 1)) - 1
    location_array = np.stack([stim_x_nrm, stim_y_nrm], axis=1) # units x (X position, Y position)
    
    # get unit information
    unit_rel = (model.Model * recording.ScanUnitOrder * recording.Trace.ScanUnit * fnn.Data.VisualScan) & model_key
    unit_df = pd.DataFrame(unit_rel.fetch(
        "KEY",
        "unit_id",
        order_by="trace_order",
        as_dict=True,
    ))

    # create data df
    unit_df = unit_df[['session', 'scan_idx', 'unit_id']].copy()
    meta_df = make_metadata_df(**model_key)
    return weights_array, location_array, unit_df, meta_df


def get_model_performance(network_id, instance_id, data_id, trial_filterset_id, videoset_id, perspective, modulation, burnin_frames):
    """
    Get performance metrics for a given model.
    Args:
        - network_id: ID of the network.
        - instance_id: ID of the model instance.
        - data_id: ID of the data used for the model.
        - trial_filterset_id: ID of the trial filter set.
        - videoset_id: ID of the video set.
        - perspective: Perspective boolean.
        - modulation: Modulation boolean.
        - burnin_frames: Number of burn-in frames.
    Returns:
        - data_df: DataFrame with model unit performance metrics.
        - meta_df: DataFrame with metadata.
    """
    model_key = {'network_id': network_id, 'instance_id': instance_id, 'data_id': data_id}
    data_key = {
        **model_key, 
        "trial_filterset_id": trial_filterset_id, 
        "videoset_id": videoset_id, 
        'perspective': perspective, 
        'modulation': modulation,
        'burnin': burnin_frames
    }
    data_rel = (
        merge(
            fnn.Model,
            recording.TrialFilterSet.proj(),
            stimulus.VideoSet.proj(),
            utility.Burnin,
            utility.Bool.proj(perspective="bool"),
            utility.Bool.proj(modulation="bool"),
            utility.Correlation.CCSignal,
        )
    & data_key
)
    data_scan_spec = fnn.Data.VisualScan.proj(
                "spec_id",
                "trace_filterset_id",
                "pipe_version",
                "animal_id",
                "session",
                "scan_idx",
                "segmentation_method",
                "spike_method",
            ) * fnn.Spec.VisualSpec.proj(
                "rate_id", offset_id="offset_id_unit", resample_id="resample_id_unit"
            )
    all_unit_trace_rel = (
        data_rel
        * data_scan_spec  # data_id -> specs + scan key
        * recording.ScanUnitOrder  # scan key + trace_filterset_id -> trace_ids
        * recording.Trace.ScanUnit  # trace_id -> unit key
    )
    all_units_df = all_unit_trace_rel.fetch(format="frame").reset_index()
    # fetch cc_max
    cc_max = (
        (recording.VisualMeasure & utility.Measure.CCMax & all_unit_trace_rel)
        .fetch(format="frame")
        .reset_index()
        .rename(columns={"measure": "cc_max"})
    )
    # fetch cc_abs
    cc_abs_df = (
        ((fnn.VisualRecordingCorrelation & utility.Correlation.CCSignal) & all_unit_trace_rel)
        .fetch(format="frame")
        .reset_index()
        .rename(columns={"correlation": "cc_abs", "unit": "trace_order"})
    ) 
    merge_df = (
        all_units_df.merge(cc_abs_df, how="left", validate="one_to_one")
        .merge(cc_max, how="left", validate="one_to_one")
        .assign(cc_norm=lambda df: df.cc_abs / df.cc_max)
    )
    unit_df = merge_df[['session', 'scan_idx', 'unit_id', 'cc_abs', 'cc_max', 'cc_norm']].copy()
    meta_df = make_metadata_df(**data_key)
    return unit_df, meta_df


def get_orientation_direction_tuning(network_id, instance_id, data_id, videoset_id, offset_id, impulse_id, precision_id):
    """
    Get orientation and direction tuning for a given model.
    Args:
        - network_id: ID of the network.
        - instance_id: ID of the model instance.
        - data_id: ID of the data used for the model.   
        - tuning_key: Dictionary with tuning keys.
        - videoset_id: ID of the video set.
        - offset_id: ID of the offset.
        - impulse_id: ID of the impulse.
        - precision_id: ID of the precision.
    Returns:
        - data_df: DataFrame with orientation and direction tuning metrics.
        - meta_df: DataFrame with metadata.
    """
    model_key = {'network_id': network_id, 'instance_id': instance_id, 'data_id': data_id}
    tuning_key = dict(
        videoset_id=videoset_id,
        offset_id=offset_id,
        impulse_id=impulse_id,
        precision_id=precision_id,
    )
    data_rel = (
        tuning.GlobalOSI()
        * tuning.GlobalDSI()
        * tuning.BiVonMises()
    )
    unit_ori = pd.DataFrame((
        data_rel
        * tuning.Direction.FnnVisualDirection
        * fnn.Data.VisualScan
        & model_key
        & tuning_key
    ).fetch(as_dict=True))

    # compute OSI and DSI
    def get_bi_von_mises_model(params):
        b = partial(
            bi_von_mises,
            mu=params["mu"],
            phi=params["phi"],
            kappa=params["kappa"],
            amp=params["scale"],
        )
        u = partial(uniform, amp=params["bias"])
        def model(x):
            return b(x) + u(x)
        return model

    def computeOSI(params):
        model = get_bi_von_mises_model(params)
        # compute osi from fitted model
        pref_ori = params["mu"] % np.pi
        orth_ori = (params["mu"] + np.pi / 2) % np.pi
        r_pref_ori = (model(x=pref_ori) + model(x=pref_ori + np.pi)) / 2
        r_orth_ori = (model(x=orth_ori) + model(x=orth_ori + np.pi)) / 2
        return (r_pref_ori - r_orth_ori) / (r_pref_ori + r_orth_ori)

    def computeDSI(params):
        model = get_bi_von_mises_model(params)
        # compute dsi from fitted model
        r_pref_dir = model(x=params["mu"])
        r_null_dir = model(x=params["mu"] + np.pi)
        return 1 - (r_null_dir / r_pref_dir)

    unit_ori['OSI'] = unit_ori.apply(lambda r: computeOSI({'mu': r.mu, 'phi': r.phi, 'kappa': r.kappa, 'scale': r.scale, 'bias': r.bias}), axis=1)
    unit_ori['DSI'] = unit_ori.apply(lambda r: computeDSI({'mu': r.mu, 'phi': r.phi, 'kappa': r.kappa, 'scale': r.scale, 'bias': r.bias}), axis=1)

    unit2unit_key = pd.DataFrame(
        (
            recording.ScanUnitOrder.proj(unit="trace_order")
            * fnn.Data.VisualScan
            * recording.Trace.ScanUnit
            & model_key
        ).fetch(
            "data_id",
            "unit",
            "animal_id",
            "session",
            "scan_idx",
            "unit_id",
            as_dict=True,
        )
    )  # unit to unit_id
    unit_ori = unit_ori.merge(
        unit2unit_key, how="left", validate="one_to_one"
    )  # direction_id to unit_id
    assert (
        unit_ori.groupby(["animal_id", "session", "scan_idx", "unit_id"])
        .size()
        .max()
        == 1
    )
    unit_ori["pref_ori"] = unit_ori["mu"] % np.pi / np.pi * 180
    unit_ori["pref_dir"] = unit_ori["mu"] / np.pi * 180
    unit_ori = unit_ori.rename(
        columns={
            "unit": "trace_order",
            "global_osi": "gOSI",
            "global_dsi": "gDSI",
    })
    data_col = ['OSI', 'DSI', 'gOSI', 'gDSI', 'pref_ori', 'pref_dir']
    unit_df = unit_ori[['session', 'scan_idx', 'unit_id', *data_col]].copy()
    meta_df = make_metadata_df(**model_key, **tuning_key)
    return unit_df, meta_df


def get_stimulus_videos(data_id, videoset_id):
    """
    Get stimulus videos for a given data_id.
    Args:
        - data_id: ID of the data used for the model.
        - videoset_id: ID of the video set.
    Returns:
        - stim_array: Array of stimulus videos (frame x height x width).
        - meta_df: DataFrame with metadata.
    """
    key = {
        "data_id": data_id,
        "videoset_id": videoset_id,
    }
    data = (Data & key).link.compute
    video_ids = (VideoSet & key).members.fetch(
        "video_id", order_by="videoset_index ASC"
    )
    # get stimulus video and resample
    model_period = data.sampling_period
    assert model_period > 0, "model period must be greater than 0"
    model_offset = data.unit_offset
    assert model_offset == 0, "model offset is not 0"
    height, width = data.resolution
    resize_id = data.resize_id
    stimuli_resample = []
    current_idx = 0
    col_start_idx = np.zeros(len(video_ids), dtype=int)
    col_end_idx = np.zeros(len(video_ids), dtype=int)
    for i, vid in enumerate(tqdm(video_ids, desc="Preparing stimulus video")):
        with disable_tqdm():
            rvideo = (Resize & {"resize_id": resize_id}).link.resize(
                video=(Video & f'video_id="{vid}"').link.compute.video,
                height=height,
                width=width,
            )
            rvideo = list(rvideo.generate(period=model_period))
            stimuli_resample.append(rvideo)
            col_start_idx[i] = current_idx
            current_idx += len(rvideo)
            col_end_idx[i] = current_idx
    stim_array = np.concatenate(stimuli_resample, axis=0) # frame x height x width
    
    # prepare metadata
    meta_df = make_metadata_df(**key)
    meta_df['sampling_frequency'] = 1 / model_period
    meta_df['stimulus_height'] = height
    meta_df['stimulus_width'] = width
    return stim_array, meta_df


def compute_model_responses(network_id, instance_id, data_id, videoset_id, test=False):
    """
    Compute model responses for a given model.
    Args:
        - network_id: ID of the network.
        - instance_id: ID of the model instance.
        - data_id: ID of the data used for the model.
        - videoset_id: ID of the video set.
        - test: If True, skip the computation and return a dummy response.
    Returns:
        - resp_array: Model responses (unit x frame).
        - stim_array: Stimulus array (frame x height x width).
        - unit_df: DataFrame with unit information.
        - meta_df: DataFrame with metadata.
    """
    model_key = {'network_id': network_id, 'instance_id': instance_id, 'data_id': data_id}
    key = {**model_key, 'videoset_id': videoset_id}
    model = (Model & key).model(device="cuda")

    if not test:
        # get stimulus videos
        stim_array, stim_meta_df = get_stimulus_videos(data_id, videoset_id) # frame x height x width
        # compute model response
        r = model.generate_response(
            stimuli=tqdm(stim_array, desc="Computing model unit responses"),
        )
        resp_array = np.stack(list(r), axis=1)  # unit x frame
    else:
        stim_meta_df = make_metadata_df(**key)
        stim_array = np.zeros((10, 10, 10))  # dummy stimulus
        resp_array = np.zeros((10, 10))  # dummy response

    # get unit dataframe
    burnin_frames = (
        fnn.Model
        * fnn.Instance.Individual
        * fnn.Train.Optimize
        * fnn.Objective.NetworkLoss
        & key
    ).fetch1("burnin_frames")

    # get trace order that would sort the traces by unit_id
    unit_rel = (
        Model
        * Data
        * VideoSet
        * fnn.VisualScan
        * pipe_fuse.ScanSet.Unit
        * fnn.Data.VisualScan
        * recording.TraceSet.Member
        * recording.Trace.ScanUnit
        * recording.ScanUnits
        * recording.ScanUnitOrder
        & key
    ).fetch('session', 'scan_idx', 'unit_id', order_by="trace_order ASC", as_dict=True)
    unit_df = pd.DataFrame(unit_rel)
    meta_df = make_metadata_df(**key, burnin_frames=burnin_frames).merge(stim_meta_df)
    return resp_array, stim_array, unit_df, meta_df


def get_training_dataset(data_id):
    """
    Get the training dataset DataFrame for a given data_id.
    Args:
        - data_id: ID of the data used for the model.
    Returns:
        - dataset_df: DataFrame with the training dataset paths.  
        - meta_df: DataFrame with metadata.
    """
    data_key = {'data_id': data_id}
    data_rel = Data.VisualScan & data_key
    dataset = (data.VisualScan & data_rel.fetch1()).compute.dataset
    meta_df = make_metadata_df(**data_key)
    return dataset.df, meta_df


def get_visual_trial_data(data_id, network_id, instance_id, trial_filterset_id, videoset_id):
    """"
    Get the visual trial dataset DataFrame for a given data_id, network_id, instance_id, trial_filterset_id, and videoset_id.
    Args:
        - data_id: ID of the data used for the model.
        - network_id: ID of the network.
        - instance_id: ID of the model instance.
        - trial_filterset_id: ID of the trial filter set.
        - videoset_id: ID of the video set.
    Returns:
        - nodel_data: Object with references to the model data
        - trial_df: DataFrame with the visual trials specified by the trial_filterset_id and videoset_id.
        - meta_df: DataFrame with metadata.
    """
    model_key = dict(
        data_id=data_id,
        network_id=network_id,
        instance_id=instance_id,
    )
    data = (Data & model_key).link.compute
    visual_trial_set_key = dict(
        trialset_id=data.trialset_id,
        trial_filterset_id=trial_filterset_id,
        videoset_id=videoset_id
    )
    trial_df = (VisualTrialSet & visual_trial_set_key).df
    meta_df = make_metadata_df(**model_key, **visual_trial_set_key)
    return data, trial_df, meta_df