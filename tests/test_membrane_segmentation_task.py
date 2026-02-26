from pathlib import Path

import pytest
from ngio import OmeZarrContainer, create_synthetic_ome_zarr
from skimage.metrics import adapted_rand_error

from abbott_segmentation_tasks.membrane_segmentation_task import (
    membrane_segmentation_task,
)
from abbott_segmentation_tasks.utils import (
    IteratorConfiguration,
    MaskingConfiguration,
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
def test_membrane_segmentation_segmentation_task(
    is_github_or_fast, tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Base test for the membrane segmentation task."""

    if is_github_or_fast:
        pytest.skip("Skipping test in GitHub Actions.")

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

    ome_zarr.derive_label(name="cells", channels_policy="squeeze")

    membrane_segmentation_task(
        zarr_url=str(test_data_path),
        label_name="cells",
        output_label_name="membranes",
        overwrite=False,
    )

    # Check that the label image was created
    assert "membranes" in ome_zarr.list_labels()


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
def test_membrane_segmentation_task_masked(
    is_github_or_fast, tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Test the seeded segmentation task with a masking configuration."""

    if is_github_or_fast:
        pytest.skip("Skipping test in GitHub Actions.")

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
    ome_zarr.derive_label(name="cells", channels_policy="squeeze")

    iter_config = IteratorConfiguration(
        masking=MaskingConfiguration(mode="Label Name", identifier="cells"),
        roi_table=None,
    )

    membrane_segmentation_task(
        zarr_url=str(test_data_path),
        label_name="cells",
        output_label_name="membranes",
        overwrite=False,
        iterator_configuration=iter_config,
    )

    # Check that the label image was created
    assert "membranes" in ome_zarr.list_labels()


def test_membrane_segmentation_task_no_mock(is_github_or_fast, tmp_path: Path):
    """Base test for the membrane segmentation task without mocking."""

    if is_github_or_fast:
        pytest.skip("Skipping test in GitHub Actions.")

    test_data_path = tmp_path / "data.zarr"
    shape = (1, 64, 64)
    axes = "cyx"

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        overwrite=False,
        axes_names=axes,
    )

    ome_zarr.derive_label(name="cells", channels_policy="squeeze")

    membrane_segmentation_task(
        zarr_url=str(test_data_path),
        label_name="cells",
        overwrite=False,
    )

    # Check that the label image was created
    assert "membranes" in ome_zarr.list_labels()


def test_skip_if_missing_with_valid_channel(is_github_or_fast, tmp_path: Path):
    """When skip_if_missing=True and the channel exists, the task runs normally."""
    if is_github_or_fast:
        pytest.skip("Skipping test in GitHub Actions.")

    test_data_path = tmp_path / "data.zarr"
    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=(64, 64),
        channels_meta=["DAPI_0"],
        overwrite=False,
        axes_names="yx",
    )

    ome_zarr.derive_label(name="cells", channels_policy="squeeze")

    membrane_segmentation_task(
        zarr_url=str(test_data_path),
        label_name="cells",
        skip_if_missing=True,
        overwrite=False,
    )

    # Channel exists → task must have run and created the label
    assert "membranes" in ome_zarr.list_labels()


def test_skip_if_missing_with_invalid_channel(is_github_or_fast, tmp_path: Path):
    """When skip_if_missing=True and the channel is absent, the task skips silently."""
    if is_github_or_fast:
        pytest.skip("Skipping test in GitHub Actions.")

    test_data_path = tmp_path / "data.zarr"
    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=(64, 64),
        channels_meta=["DAPI_0"],
        overwrite=False,
        axes_names="yx",
    )

    result = membrane_segmentation_task(
        zarr_url=str(test_data_path),
        label_name="cells",
        skip_if_missing=True,
        overwrite=False,
    )

    # Task should return None without raising and without creating any label
    assert result is None
    assert "membranes" not in ome_zarr.list_labels()


def test_raise_if_missing_with_invalid_channel(is_github_or_fast, tmp_path: Path):
    """When skip_if_missing=False and channel is absent, a ValueError is raised."""
    if is_github_or_fast:
        pytest.skip("Skipping test in GitHub Actions.")

    test_data_path = tmp_path / "data.zarr"
    create_synthetic_ome_zarr(
        store=test_data_path,
        shape=(64, 64),
        channels_meta=["DAPI_0"],
        overwrite=False,
        axes_names="yx",
    )

    # skip_if_missing defaults to False
    with pytest.raises(ValueError, match="cells"):
        membrane_segmentation_task(
            zarr_url=str(test_data_path),
            label_name="cells",
            skip_if_missing=False,
            overwrite=False,
        )
