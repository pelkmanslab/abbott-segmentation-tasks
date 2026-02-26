"""Seeded Segmentation Fractal Task."""

import logging
import time
from typing import Optional

import numpy as np
from ngio import OmeZarrContainer, open_ome_zarr_container
from ngio.images._image import _parse_channel_selection
from ngio.images._masked_image import MaskedImage, MaskedLabel
from ngio.utils import NgioValueError
from pydantic import validate_call
from skimage.segmentation import watershed

from abbott_segmentation_tasks.pre_post_process import (
    PrePostProcessConfiguration,
    apply_post_process,
    apply_pre_process,
)
from abbott_segmentation_tasks.utils import (
    AnyCreateRoiTableModel,
    CreateMaskingRoiTable,
    IteratorConfiguration,
    MaskingConfiguration,
    SeededMaskedSegmentationIterator,
    SeededSegmentationChannels,
    SeededSegmentationIterator,
    SeededSegmentationParams,
    SkipCreateMaskingRoiTable,
)

logger = logging.getLogger(__name__)


def segmentation_function(
    *,
    image_data: np.ndarray,
    seed_label_data: np.ndarray,
    parameters: SeededSegmentationParams,
    pre_post_process: PrePostProcessConfiguration,
) -> np.ndarray:
    """Wrap Seeded segmentation call.

    Args:
        image_data (np.ndarray): Input image data
        seed_label_data (np.ndarray): Seed label image data
        parameters (SeededSegmentationParams): Advanced parameters for
            seeded watershed segmentation.
        pre_post_process (PrePostProcessConfiguration): Configuration for pre- and
            post-processing steps.

    Returns:
        np.ndarray: Segmented image
    """
    # Pre-processing
    image_data = apply_pre_process(
        image=image_data,
        pre_process_steps=pre_post_process.pre_process,
    )

    masks = watershed(
        image=image_data,
        markers=seed_label_data,
        compactness=parameters.compactness,
    )

    # Post-processing
    masks = apply_post_process(
        labels=masks,
        post_process_steps=pre_post_process.post_process,
    )
    masks = masks.astype(np.uint32)
    return masks


def load_masked_image(
    ome_zarr,
    masking_configuration: MaskingConfiguration,
    level_path: Optional[str] = None,
) -> MaskedImage:
    """Load a masked image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        masking_configuration (MaskingConfiguration): Configuration for masking.
        level_path (Optional[str]): Optional path to a specific resolution level.

    """
    if masking_configuration.mode == "Table Name":
        masking_table_name = masking_configuration.identifier
        masking_label_name = None
    else:
        masking_label_name = masking_configuration.identifier
        masking_table_name = None
    logger.info(f"Using masking with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_image = ome_zarr.get_masked_image(
        masking_label_name=masking_label_name,
        masking_table_name=masking_table_name,
        path=level_path,
    )
    return masked_image


def load_masked_label(
    ome_zarr,
    label_name: str,
    masking_configuration: MaskingConfiguration,
    level_path: Optional[str] = None,
) -> MaskedLabel:
    """Load a masked label image from an OME-Zarr based on the masking configuration.

    Args:
        ome_zarr: The OME-Zarr container.
        label_name (str): Name of the label image to load.
        masking_configuration (MaskingConfiguration): Configuration for masking.
        level_path (Optional[str]): Optional path to a specific resolution level.

    """
    if masking_configuration.mode == "Table Name":
        masking_table_name = masking_configuration.identifier
        masking_label_name = None
    else:
        masking_label_name = masking_configuration.identifier
        masking_table_name = None
    logger.info(f"Using masked label with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_label = ome_zarr.get_masked_label(
        label_name=label_name,
        masking_label_name=masking_label_name,
        masking_table_name=masking_table_name,
        path=level_path,
    )
    return masked_label


def _skip_segmentation(
    channels: SeededSegmentationChannels, ome_zarr: OmeZarrContainer
) -> bool:
    """Check wheter to skip the current task based on the channel configuration.

    If the channel selection specified in the channel parameter is not
    valid for the provided OME-Zarr image, this function checks the
    skip_if_missing attribute of the channel configuration.
    If skip_if_missing is True, the function returns True, indicating that the task
    should be skipped. If skip_if_missing is False, a ValueError is raised.

    Args:
        channels (current): The channel selection configuration.
        ome_zarr (OmeZarrContainer): The OME-Zarr container to check against.

    Returns:
        bool: True if the task should be skipped due to missing channels,
        False otherwise.

    """
    channels_list = channels.to_list()
    image = ome_zarr.get_image()
    try:
        _parse_channel_selection(image=image, channel_selection=channels.to_list())
    except NgioValueError as e:
        if channels.skip_if_missing:
            logger.warning(
                f"Channel selection {channels_list} is not valid for the provided "
                "image, but skip_if_missing is set to True. Skipping segmentation."
            )
            logger.debug(f"Original error message: {e}")
            return True
        else:
            raise ValueError(
                f"Channel selection {channels_list} is not valid for the provided "
                "image. If you want to skip processing when channels are missing, "
                "set skip_if_missing to True."
            ) from e
    return False


@validate_call
def seeded_segmentation(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    label_name: str,
    channels: SeededSegmentationChannels,
    output_label_name: Optional[str] = None,
    level_path: Optional[str] = None,
    # Iteration parameters
    iterator_configuration: Optional[IteratorConfiguration] = None,
    # Advanced segmentation parameters
    advanced_parameters: SeededSegmentationParams = SeededSegmentationParams(),  # noqa: B008
    pre_post_process: PrePostProcessConfiguration = PrePostProcessConfiguration(),  # noqa: B008
    create_masking_roi_table: AnyCreateRoiTableModel = SkipCreateMaskingRoiTable(),  # noqa: B008
    overwrite: bool = True,
) -> None:
    """Segment Cells using Seeded Segmentation.

    Args:
        zarr_url (str): URL to the OME-Zarr container
        label_name (str): Name of the seed label image to use for segmentation e.g.
            "nuclei".
        channels (SeededSegmentationChannels): Channels to use for segmentation.
            It must contain between 1 and 3 channel identifiers.
        output_label_name (Optional[str]): Name of the resulting label image.
            If not provided, it will be set to "<channel_identifier>_segmented".
        level_path (Optional[str]): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (Optional[IteratorConfiguration]): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        advanced_parameters (SeededSegmentationParams): Advanced parameters
            for Seeded segmentation.
        pre_post_process (PrePostProcessConfiguration): Configuration for pre- and
            post-processing steps.
        create_masking_roi_table (AnyCreateMaskingRoiTableModel): Configuration to
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
    if _skip_segmentation(channels=channels, ome_zarr=ome_zarr):
        return None

    if output_label_name is None:
        output_label_name = f"{channels.identifiers[0]}_segmented"

    # Derive the label and an get it at the specified level path
    ome_zarr.derive_label(name=output_label_name, overwrite=overwrite)
    label = ome_zarr.get_label(name=output_label_name, path=level_path)
    logger.info(f"Derived label image: {label=}")

    # Set up the appropriate iterator based on the configuration
    if iterator_configuration is None:
        iterator_configuration = IteratorConfiguration()

    # Determine if we are doing 3D segmentation
    if ome_zarr.is_3d:
        axes_order = "czyx"
    else:
        axes_order = "cyx"

    if iterator_configuration.masking is None:
        # Create a basic SegmentationIterator without masking
        seed_label = ome_zarr.get_label(label_name, path=level_path)
        image = ome_zarr.get_image(path=level_path)
        logger.info(f"{image=}")
        iterator = SeededSegmentationIterator(
            input_image=image,
            input_label=seed_label,
            output_label=label,
            channel_selection=channels.to_list(),
            axes_order=axes_order,
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        seed_label = load_masked_label(
            ome_zarr=ome_zarr,
            label_name=label_name,
            masking_configuration=iterator_configuration.masking,
            level_path=level_path,
        )
        masked_image = load_masked_image(
            ome_zarr=ome_zarr,
            masking_configuration=iterator_configuration.masking,
            level_path=level_path,
        )
        logger.info(f"{masked_image=}")
        # A masked iterator is created instead of a basic segmentation iterator
        # This will do two major things:
        # 1) It will iterate only over the regions of interest defined by the
        #   masking table or label image
        # 2) It will only write the segmentation results within the masked regions
        iterator = SeededMaskedSegmentationIterator(
            input_image=masked_image,
            input_label=seed_label,
            output_label=label,
            channel_selection=channels.to_list(),
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
    for it, ((image_data, seed_label_data), writer) in enumerate(
        iterator.iter_as_numpy()
    ):
        start_time = time.time()
        label_img = segmentation_function(
            image_data=image_data,
            seed_label_data=seed_label_data,
            parameters=advanced_parameters,
            pre_post_process=pre_post_process,
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

    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(
            label_name=output_label_name
        )
        masking_roi_table = label.build_masking_roi_table()
        ome_zarr.add_table(name=table_name, table=masking_roi_table)

    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=seeded_segmentation)
