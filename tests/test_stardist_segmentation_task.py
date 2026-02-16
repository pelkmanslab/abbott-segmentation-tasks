from pathlib import Path

import numpy as np
import pytest
from ngio import OmeZarrContainer, create_synthetic_ome_zarr
from skimage.metrics import adapted_rand_error

from abbott_segmentation_tasks.stardist_segmentation_task import (
    stardist_segmentation_task,
)
from abbott_segmentation_tasks.utils import (
    IteratorConfiguration,
    MaskingConfiguration,
    StardistChannel,
    StardistModels,
)


class MockStardistModel:
    def __init__(self, *args, **kwargs):
        pass

    def eval(self, image, **kwargs):
        masks = np.ones(image.shape[1:], dtype=np.uint32)
        return masks, None, None


def check_label_quality(
    ome_zarr: OmeZarrContainer, label_name: str, gt_name: str = "nuclei"
):
    prediction = ome_zarr.get_label(label_name).get_as_numpy(axes_order="tzyx", t=0)
    ground_truth = ome_zarr.get_label(gt_name).get_as_numpy(axes_order="tzyx", t=0)
    are, _, _ = adapted_rand_error(ground_truth, prediction)
    assert are < 0.1, f"Adapted Rand Error too high: {are}<0.1. Labels might be wrong."


def check_masked_label_quality(
    ome_zarr: OmeZarrContainer,
    label_name: str,
    gt_name: str = "nuclei",
    masking_label_name: str = "nuclei_mask",
):
    prediction = ome_zarr.get_label(label_name).get_as_numpy(axes_order="tzyx", t=0)
    ground_truth = ome_zarr.get_label(gt_name).get_as_numpy(axes_order="tzyx", t=0)
    mask = ome_zarr.get_label(masking_label_name).get_as_numpy(axes_order="tzyx", t=0)

    ground_truth = ground_truth * (mask > 0)  # Apply mask to ground truth
    are, _, _ = adapted_rand_error(ground_truth, prediction)
    assert are < 0.1, f"Adapted Rand Error too high: {are}<0.1. Labels might be wrong."


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((64, 64), "yx"),
        ((1, 64, 64), "cyx"),
        ((3, 64, 64), "cyx"),
        ((4, 64, 64), "tyx"),
        ((1, 64, 64), "zyx"),
        ((1, 1, 64, 64), "czyx"),
        ((1, 2, 64, 64), "czyx"),
        ((1, 1, 64, 64), "tzyx"),
        ((1, 3, 64, 64), "tcyx"),
        ((2, 1, 2, 64, 64), "tczyx"),
    ],
)
def test_stardist_segmentation_task(
    monkeypatch, is_github_or_fast, tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Base test for the Stardist segmentation task."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = StardistChannel(mode="label", identifiers=["DAPI_0"])

    if is_github_or_fast:
        import stardist.models

        monkeypatch.setattr(stardist.models, "StarDist2D", MockStardistModel)
        monkeypatch.setattr(stardist.models, "StarDist3D", MockStardistModel)

    if ome_zarr.is_2d:
        model_type = StardistModels.VERSATILE_FLUO_2D
    else:
        model_type = StardistModels.DEMO_3D

    stardist_segmentation_task(
        zarr_url=str(test_data_path),
        channel=channel,
        model_type=model_type,
        overwrite=False,
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((64, 64), "yx"),
        ((1, 64, 64), "cyx"),
        ((3, 64, 64), "cyx"),
        ((4, 64, 64), "tyx"),
        ((1, 64, 64), "zyx"),
        ((1, 1, 64, 64), "czyx"),
        ((1, 2, 64, 64), "czyx"),
        ((1, 1, 64, 64), "tzyx"),
        ((1, 3, 64, 64), "tcyx"),
        ((2, 1, 2, 64, 64), "tczyx"),
    ],
)
def test_stardist_segmentation_task_masked(
    monkeypatch, is_github_or_fast, tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Test the Stardist segmentation task with a masking configuration."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )
    channel = StardistChannel(mode="label", identifiers=["DAPI_0"])

    iter_config = IteratorConfiguration(
        masking=MaskingConfiguration(mode="Label Name", identifier="nuclei_mask"),
        roi_table=None,
    )

    if is_github_or_fast:
        import stardist.models

        monkeypatch.setattr(stardist.models, "StarDist2D", MockStardistModel)
        monkeypatch.setattr(stardist.models, "StarDist3D", MockStardistModel)

    if ome_zarr.is_2d:
        model_type = StardistModels.VERSATILE_FLUO_2D
    else:
        model_type = StardistModels.DEMO_3D

    stardist_segmentation_task(
        zarr_url=str(test_data_path),
        channel=channel,
        model_type=model_type,
        overwrite=False,
        iterator_configuration=iter_config,
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()
    if is_github_or_fast:
        return
    # TODO: re-enable masked label quality check, currently failing for some
    # check_masked_label_quality(
    #     ome_zarr,
    #     "DAPI_0_segmented",
    # )


def test_stardist_segmentation_task_no_mock(tmp_path: Path):
    """Base test for the Stardist segmentation task without mocking."""
    test_data_path = tmp_path / "data.zarr"
    shape = (1, 64, 64)
    axes = "cyx"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = StardistChannel(mode="label", identifiers=["DAPI_0"])

    if ome_zarr.is_2d:
        model_type = StardistModels.VERSATILE_FLUO_2D
    else:
        model_type = StardistModels.DEMO_3D

    stardist_segmentation_task(
        zarr_url=str(test_data_path),
        channel=channel,
        model_type=model_type,
        overwrite=False,
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()

    check_label_quality(ome_zarr, "DAPI_0_segmented")
