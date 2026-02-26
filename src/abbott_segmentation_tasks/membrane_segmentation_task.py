"""This is the Python module for my_task."""

import logging
import time

import numpy as np
from ngio import open_ome_zarr_container
from ngio.images._masked_image import MaskedLabel
from ngio.images._ome_zarr_container import OmeZarrContainer
from ngio.utils import NgioValueError
from pyclesperanto import reduce_labels_to_label_edges, select_device
from pydantic import validate_call

from abbott_segmentation_tasks.utils import (
    AnyCreateRoiTableModel,
    CreateMaskingRoiTable,
    IteratorConfiguration,
    LabelSegmentationIterator,
    MaskedLabelSegmentationIterator,
    MaskingConfiguration,
    SkipCreateMaskingRoiTable,
)

logger = logging.getLogger("cellpose_sam_task")


def segmentation_function(
    *,
    label_data: np.ndarray,
    use_gpu: bool = True,
) -> np.ndarray:
    """Wrap membrane segmentation call.

    Args:
        label_data (np.ndarray): Input label data
        use_gpu (bool): Whether to use GPU for segmentation. Defaults to True.

    Returns:
        np.ndarray: Segmented image
    """
    # Pre-processing

    device = select_device(device_type="gpu" if use_gpu else "cpu")

    masks = reduce_labels_to_label_edges(
        input_image=label_data,
        device=device,
    ).astype(np.uint32)
    # masks = np.expand_dims(masks, axis=0).astype(np.uint32)
    return masks


def load_masked_label(
    ome_zarr: OmeZarrContainer,
    label_name: str,
    masking_configuration: MaskingConfiguration,
    level_path: str | None = None,
) -> MaskedLabel:
    """Load a masked image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        label_name: The name of the label to extract membranes from.
        masking_configuration (MaskingConfiguration): Configuration for masking.
        level_path (str | None): Optional path to a specific resolution level.

    """
    if masking_configuration.mode == "Table Name":
        masking_table_name = masking_configuration.identifier
        masking_label_name = None
    else:
        masking_label_name = masking_configuration.identifier
        masking_table_name = None
    logger.info(f"Using masking with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_label = ome_zarr.get_masked_label(
        label_name=label_name,
        masking_label_name=masking_label_name,
        masking_table_name=masking_table_name,
        path=level_path,
    )
    return masked_label


def _skip_segmentation(
    channel: str, skip_if_missing: bool, ome_zarr: OmeZarrContainer
) -> bool:
    """Check wheter to skip the current task based on the channel configuration.

    If the channel selection specified in the channel parameter is not
    valid for the provided OME-Zarr image, this function checks the
    skip_if_missing attribute of the channel configuration.
    If skip_if_missing is True, the function returns True, indicating that the task
    should be skipped. If skip_if_missing is False, a ValueError is raised.

    Args:
        channel (current): The name of the selected label image.
        skip_if_missing (current): Whether to skip the task if the channel is missing.
        ome_zarr (OmeZarrContainer): The OME-Zarr container to check against.

    Returns:
        bool: True if the task should be skipped due to missing channels,
        False otherwise.

    """
    try:
        ome_zarr.get_label(name=channel)
    except NgioValueError as e:
        if skip_if_missing:
            logger.warning(
                f"Label selection {channel} is not valid for the provided "
                "image, but skip_if_missing is set to True. Skipping segmentation."
            )
            logger.debug(f"Original error message: {e}")
            return True
        else:
            raise ValueError(
                f"Label selection {channel} is not valid for the provided "
                "image. If you want to skip processing when channels are missing, "
                "set skip_if_missing to True."
            ) from e
    return False


@validate_call
def membrane_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    label_name: str = "cells",
    skip_if_missing: bool = False,
    output_label_name: str = "membranes",
    level_path: str | None = None,
    # Iteration parameters
    iterator_configuration: IteratorConfiguration | None = None,
    # Custom parameters
    use_gpu: bool = True,
    create_masking_roi_table: AnyCreateRoiTableModel = SkipCreateMaskingRoiTable(),  # noqa: B008
    overwrite: bool = True,
) -> None:
    """Reduce cell segmentation to membrane label.

    For more information, see:
        https://github.com/clEsperanto/pyclesperanto

    Args:
        zarr_url (str): URL to the OME-Zarr container
        label_name (str): Name of the label image to extract membranes from.
            Defaults to "cells".
        skip_if_missing (bool): Whether to skip the task if the specified label
            image is missing. Defaults to False, which means that a ValueError
            will beraised if the label image is not found. If set to True, a
            warning will be logged and the function will return without performing
            segmentation.
        output_label_name (str | None): Name of the resulting label image. If not
            provided, it will default to "membranes".
        level_path (str | None): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (IteratorConfiguration | None): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        use_gpu: If `False`, always use the CPU; if `True`, use the GPU if
            possible.
        create_masking_roi_table (AnyCreateRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    # Use the first of input_paths
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    # Validate that the specified channels are present in the image
    if _skip_segmentation(
        channel=label_name, skip_if_missing=skip_if_missing, ome_zarr=ome_zarr
    ):
        return None

    # Derive the label and an get it at the specified level path
    ome_zarr.derive_label(name=output_label_name, overwrite=overwrite)
    output_label = ome_zarr.get_label(name=output_label_name, path=level_path)
    logger.info(f"Derived label image: {output_label=}")

    # Set up the appropriate iterator based on the configuration
    if iterator_configuration is None:
        iterator_configuration = IteratorConfiguration()

    # Determine if we are doing 3D segmentation
    # If so we need to set the anisotropy factor
    if ome_zarr.is_3d:
        axes_order = "zyx"
        px_z, (px_y, px_x) = output_label.pixel_size.z, output_label.pixel_size.yx
        # Pixelsize must be isotropic in XY (to some extent)
        perc_diff_xy = abs(px_x - px_y) / max(px_x, px_y)
        if perc_diff_xy >= 0.01:
            logger.warning(
                f"Non-isotropic pixel size in XY detected: px_x={px_x}, px_y={px_y}"
            )
        px_xy = (px_x + px_y) / 2.0
        anisotropy = px_z / px_xy
        logger.info(
            "Anisotropy factor calculated: "
            f"(px_z={px_z} / px_xy={px_xy}) = {anisotropy}"
        )
    else:
        axes_order = "yx"
        anisotropy = None
    logger.info(f"Segmenting using {axes_order=}")

    if iterator_configuration.masking is None:
        # Create a basic SegmentationIterator without masking
        input_label = ome_zarr.get_label(name=label_name, path=level_path)
        logger.info(f"{input_label=}")
        iterator = LabelSegmentationIterator(
            input_label=input_label,
            output_label=output_label,
            axes_order=axes_order,
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        masked_label = load_masked_label(
            ome_zarr=ome_zarr,
            label_name=label_name,
            masking_configuration=iterator_configuration.masking,
            level_path=level_path,
        )
        logger.info(f"{masked_label=}")
        # A masked iterator is created instead of a basic segmentation iterator
        # This will do two major things:
        # 1) It will iterate only over the regions of interest defined by the
        #   masking table or label image
        # 2) It will only write the segmentation results within the masked regions
        iterator = MaskedLabelSegmentationIterator(
            input_label=masked_label,
            output_label=output_label,
            axes_order=axes_order,
        )
    # Make sure that if we have a time axis, we iterate over it
    # Strict=False means that if there no z axis or z is size 1, it will still work
    # If your segmentation needs requires a volume, use strict=True
    iterator = iterator.by_zyx(strict=False)
    logger.info(f"Iterator created: {iterator=}")

    if iterator_configuration.roi_table is not None:
        # If a ROI table is provided, we load it and use it to further restrict
        # the iteration to the ROIs defined in the table
        # Be aware that this is not an alternative to masking
        # but only an additional restriction
        table = ome_zarr.get_generic_roi_table(name=iterator_configuration.roi_table)
        logger.info(f"ROI table retrieved: {table=}")
        iterator = iterator.product(table)
        logger.info(f"Iterator updated with ROI table: {iterator=}")

    # Keep track of the maximum label to ensure unique across iterations
    max_label = 0
    #
    # Core processing loop
    #
    logger.info("Starting processing...")
    run_times = []
    num_rois = len(iterator.rois)
    logging_step = max(1, num_rois // 10)
    for it, (label_data, writer) in enumerate(iterator.iter_as_numpy()):
        start_time = time.time()
        label_img = segmentation_function(
            label_data=label_data,
            use_gpu=use_gpu,
        )
        # Ensure unique labels across different chunks
        label_img = np.where(label_img == 0, 0, label_img + max_label)
        max_label = max(max_label, label_img.max())
        writer(label_img)
        iteration_time = time.time() - start_time
        run_times.append(iteration_time)

        # Only log the progress every logging_step iterations
        if it % logging_step == 0 or it == num_rois - 1:
            avg_time = sum(run_times) / len(run_times)
            logger.info(
                f"Processed ROI {it + 1}/{num_rois} "
                f"(avg time per ROI: {avg_time:.2f} s)"
            )
    logger.info(f"label {output_label_name} successfully created at {zarr_url}")

    # Building a masking roi table
    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        masking_roi_table = output_label.build_masking_roi_table()
        ome_zarr.add_table(
            name=table_name, table=masking_roi_table, overwrite=overwrite
        )
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=membrane_segmentation_task)
