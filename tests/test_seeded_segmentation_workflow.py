from pathlib import Path

import pytest
from ngio import OmeZarrContainer, create_synthetic_ome_zarr
from skimage.metrics import adapted_rand_error

from abbott_segmentation_tasks.pre_post_process import (
    GaussianFilter,
    MedianFilter,
    PrePostProcessConfiguration,
    SizeFilter,
)
from abbott_segmentation_tasks.seeded_segmentation import (
    seeded_segmentation,
)
from abbott_segmentation_tasks.utils import (
    IteratorConfiguration,
    MaskingConfiguration,
    SeededSegmentationChannels,
)


def check_label_quality(
    ome_zarr: OmeZarrContainer, label_name: str, gt_name: str = "nuclei"
):
    if ome_zarr.is_3d:
        # Synthetic data is 2D only
        # we run 3D tests to check the API but cannot check label quality
        return
    prediction = ome_zarr.get_label(label_name).get_as_numpy(axes_order="tzyx", t=0)
    ground_truth = ome_zarr.get_label(gt_name).get_as_numpy(axes_order="tzyx", t=0)
    are, _, _ = adapted_rand_error(ground_truth, prediction)
    assert are < 0.1, f"Adapted Rand Error too high: {are}>0.1. Labels might be wrong."


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
def test_seeded_segmentation_segmentation_task(
    tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Base test for the seeded segmentation task."""
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

    channel = SeededSegmentationChannels(mode="label", identifiers=["DAPI_0"])

    pre_post = PrePostProcessConfiguration(
        pre_process=[
            GaussianFilter(sigma_xy=1.0),
            MedianFilter(size_xy=3),
        ],
        post_process=[SizeFilter(max_size=10)],
    )
    seeded_segmentation(
        zarr_url=str(test_data_path),
        channels=channel,
        label_name="nuclei_mask",
        overwrite=False,
        pre_post_process=pre_post,
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
def test_seeded_segmentation_task_masked(
    tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Test the seeded segmentation task with a masking configuration."""
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
    channel = SeededSegmentationChannels(
        mode="label",
        identifiers=["DAPI_0"],
    )

    iter_config = IteratorConfiguration(
        masking=MaskingConfiguration(mode="Label Name", identifier="nuclei_mask"),
        roi_table=None,
    )

    seeded_segmentation(
        zarr_url=str(test_data_path),
        label_name="nuclei_mask",
        channels=channel,
        overwrite=False,
        iterator_configuration=iter_config,
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()


def test_seeded_segmentation_task_no_mock(tmp_path: Path):
    """Base test for the cellpose segmentation task without mocking."""
    test_data_path = tmp_path / "data.zarr"
    shape = (1, 64, 64)
    axes = "cyx"
    channel_labels = ["DAPI_0"]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )

    channel = SeededSegmentationChannels(mode="label", identifiers=["DAPI_0"])

    seeded_segmentation(
        zarr_url=str(test_data_path),
        label_name="nuclei_mask",
        channels=channel,
        overwrite=False,
    )

    # Check that the label image was created
    assert "DAPI_0_segmented" in ome_zarr.list_labels()
