import os
import torch
import pandas as pd
from tqdm import tqdm
from shutil import make_archive, rmtree
from collections import OrderedDict
from foundation.exports import export as exp, utils
import logging

logger = logging.getLogger(__name__)

NETWORK_ID = "c17d459afa99a88b3e48a32fbabc21e4"
INSTANCE_ID = "6600970e9cfe7860b80a70375cb6f20c"
DATA_IDS = [
    "232ba7ad384c7b93f58842b51e7e1ef6",
    "98a58d55e28951a38cc61cfec7d63f76",
    "6dea1dbe556674b1ebd8f984edd102c0",
    "45dc6cb6ed757fb223cdfa846060a2bf",
    "c947b82486ab3d4dfd2972e21ad2ce3b",
    "54aa25585c44713e66671a63068cc5a6",
    "748c36efcdb2b94d5a7e587ca3e80005",
    "cf36b881ec9c3e3aa50986a813ea33d5",
    "fa774b612d638ec8332e0a2fbdc2ed59",
    "e75fe99564d314b39e7348e6a9793cc8",
    "71b64f25a1671f02a589b2dfd3ab0f69",
    "96c3fdd9b93a2736615d43fee8d0d037",
    "11e7be67a39d58be8a10202f654af2b3",
]

PERFORMANCE_TRIAL_FILTERSET_ID = "d00bbb175d63398818ca652391c18856"
PERFORMANCE_VIDEOSET_ID = "acb04adeca72c460a2c5849c22630b14"
PERFORMANCE_PERSPECTIVE = True
PERFORMANCE_MODULATION = True
PERFORMANCE_BURNIN_FRAMES = 10

ORIDIR_NETWORK_ID = "c17d459afa99a88b3e48a32fbabc21e4"
ORIDIR_INSTANCE_ID = "15c03d50c410911ed4937feffbfebd95"
ORIDIR_DATA_IDS = [
    'e39f2b0628b9d8da2177fb7eb7a1073a',
    'c5871a9b7433af825c4d629e74781b09',
    '82c4767aa534d1abfd65cab2fe2f5d52',
    '8c93b4c9447a8511cac6482edd3f8306',
    'c88870779ae0d9e44989353562282bb7',
    'd11ac53397285094421fdac94971a364',
    '9abced9c0805d0f3e6e4da11fd653370',
    '9d7bbf1f603a0f5727e86e2f4cf8e531',
    '552cd156049517a8403235bb1de4e2eb',
    'c6b86a36f468314af5ab34925fb47d1b',
    '4a18269b82150344df6ccf787090e878',
    '0a7a8be8de7e14c059f1fc39ba6ac933',
    '19cbcdcc98024cacd9d1421f0fd593a5'
]
ORIDIR_VIDEOSET_ID = "b504dea89dcb82dbca3608dfe460bed8"
ORIDIR_OFFSET_ID = "33dbc06858d00826c17ed7b1defa525f"
ORIDIR_IMPULSE_ID = "36877ae5679c3e1cdb3476e8a97525e3"
ORIDIR_PRECISION_ID = "a647ad04c5e3f6190dd22df2821c9121"

RESPONSES_VIDEOSET_ID = "e3dd23445aaca70cb9d0d4eb8eea95ce"


def export(target_dir=None):
    """
    Parameters
    ----------
    target_dir : os.PathLike | None
        target directory

    Returns
    -------
    str
        export file path
    """
    from foundation.fnn.model import Model
    from foundation.fnn.query.scan import VisualScanRecording

    if target_dir is None:
        target_dir = os.getcwd()

    mdir = os.path.join(target_dir, "microns")
    os.makedirs(mdir, exist_ok=False)
    assert not os.path.exists(f"{mdir}.zip"), f"{mdir}.zip already exists"

    dfs = []
    scans = []

    for i, data_id in enumerate(tqdm(DATA_IDS, desc="Scans")):

        # scan meta data
        recording = VisualScanRecording & {"data_id": data_id}

        units = recording.units
        units = units.fetch(
            "session",
            "scan_idx",
            "unit_id",
            "trace_order",
            order_by="trace_order",
            as_dict=True,
        )
        units = pd.DataFrame(units).rename(columns={"trace_order": "readout_id"})
        dfs.append(units)

        session, scan_idx = recording.key.fetch1("session", "scan_idx")
        scan = {
            "session": session,
            "scan_idx": scan_idx,
            "units": len(units),
            "data_id": data_id,
        }
        scans.append(scan)

        # scan model
        params = Model & {
            "data_id": data_id,
            "network_id": NETWORK_ID,
            "instance_id": INSTANCE_ID,
        }
        params = params.model().state_dict()

        if not i:
            torch.save(
                OrderedDict({k: v for k, v in params.items() if k.startswith("core.")}),
                os.path.join(mdir, "params_core.pt"),
            )

        torch.save(
            OrderedDict({k: v for k, v in params.items() if not k.startswith("core.")}),
            os.path.join(mdir, f"params_{session}_{scan_idx}.pt"),
        )

    units = pd.concat(dfs, ignore_index=True)
    units.to_csv(os.path.join(mdir, "units.csv"), index=False)

    scans = pd.DataFrame(scans)
    scans.to_csv(os.path.join(mdir, "scans.csv"), index=False)

    try:
        return make_archive(mdir, "zip", mdir)
    finally:
        rmtree(mdir)


def export_properties(target_dir=None, readout=False, performance=False, ori_dir_tuning=False, responses=False, responses_test=False):
    """
    Export properties of the model including readout weights, performance metrics,
    orientation and direction tuning, and model responses.
    Parameters :
        target_dir : os.PathLike | None
            Directory to save the exported properties. If None, uses the current working directory.
        readout : bool
            If True, exports readout weights and locations.
        performance : bool
            If True, exports model performance metrics.
        ori_dir_tuning : bool
            If True, exports orientation and direction tuning data.
        responses : bool
            If True, exports stimulus videos and model responses.
        responses_test=False : bool
            If True, exports dummy data for model responses.
    Returns :
        None
    """
    target_dir = utils.prepare_target_directory(target_dir)

    if readout:
        exp.export_readout_weights_and_locations(
            data_ids=DATA_IDS,
            network_id=NETWORK_ID,
            instance_id=INSTANCE_ID,
            target_dir=target_dir / "readout",
        )
    if performance:
        exp.export_performance_metrics(
            data_ids=DATA_IDS,
            network_id=NETWORK_ID,
            instance_id=INSTANCE_ID,
            trial_filterset_id=PERFORMANCE_TRIAL_FILTERSET_ID,
            videoset_id=PERFORMANCE_VIDEOSET_ID,
            perspective=PERFORMANCE_PERSPECTIVE,
            modulation=PERFORMANCE_MODULATION,
            burnin_frames=PERFORMANCE_BURNIN_FRAMES,
            target_dir=target_dir / "perfomance",
        )
    if ori_dir_tuning: 
        exp.export_orientation_direction_tuning(
            data_ids=ORIDIR_DATA_IDS,
            network_id=ORIDIR_NETWORK_ID,
            instance_id=ORIDIR_INSTANCE_ID,
            videoset_id=ORIDIR_VIDEOSET_ID,
            offset_id=ORIDIR_OFFSET_ID,
            impulse_id=ORIDIR_IMPULSE_ID,
            precision_id=ORIDIR_PRECISION_ID,
            target_dir=target_dir / "ori_dir_tuning",
        )
    if responses:
        exp.export_stimulus_and_model_responses(
            data_ids=DATA_IDS,
            network_id=NETWORK_ID,
            instance_id=INSTANCE_ID,
            videoset_id=RESPONSES_VIDEOSET_ID,
            test=responses_test,
            target_dir=target_dir / "responses",
        )
