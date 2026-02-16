"""Stardist Segmentation Task"""

import logging
import time
from random import uniform
from typing import Optional

import numpy as np
from ngio import open_ome_zarr_container
from ngio.experimental.iterators import MaskedSegmentationIterator, SegmentationIterator
from ngio.images._masked_image import MaskedImage
from pydantic import validate_call
from stardist.models import StarDist2D, StarDist3D

from abbott_segmentation_tasks.pre_post_process import (
    PrePostProcessConfiguration,
    apply_post_process,
    apply_pre_process,
)
from abbott_segmentation_tasks.utils import (
    AdvancedStardistParams,
    AnyCreateRoiTableModel,
    CreateMaskingRoiTable,
    IteratorConfiguration,
    MaskingConfiguration,
    SkipCreateMaskingRoiTable,
    StardistChannel,
    StardistModels,
    StardistpretrainedModel,
    normalize_stardist_channel,
)

logger = logging.getLogger(__name__)


def segmentation_function(
    *,
    image_data: np.ndarray,
    model: StardistModels,
    parameters: AdvancedStardistParams,
    do_3D: bool,
    pre_post_process: PrePostProcessConfiguration,
) -> np.ndarray:
    """Wrap Stardist segmentation call.

    Args:
        image_data (np.ndarray): Input image data
        model (StardistModels): Preloaded Stardist model.
        parameters (AdvancedStardistParams): Advanced parameters for
            Stardist segmentation.
        do_3D (bool): Whether to perform 3D segmentation.
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

    normalization = parameters.normalization
    image_data = normalize_stardist_channel(image_data, normalization)

    scale = parameters.scale
    n_tiles = parameters.n_tiles
    print(image_data.shape)

    if do_3D:
        if image_data.ndim == 4:
            # 3D image with channel dimension needs to be squeezed
            image_data = np.squeeze(image_data)
    else:
        image_data = np.squeeze(image_data)
        scale = tuple(scale[-2:])
        n_tiles = tuple(n_tiles[-2:])

    masks, _ = model.predict_instances(
        image_data,
        sparse=parameters.sparse,
        prob_thresh=parameters.prob_thresh,
        nms_thresh=parameters.nms_thresh,
        scale=scale,
        n_tiles=n_tiles,
        show_tile_progress=parameters.show_tile_progress,
        verbose=parameters.verbose,
        predict_kwargs=parameters.predict_kwargs,
        nms_kwargs=parameters.nms_kwargs,
    )

    # Post-processing
    masks = apply_post_process(
        labels=masks,
        post_process_steps=pre_post_process.post_process,
    )

    masks = np.expand_dims(masks, axis=0).astype(np.uint32)
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
    logging.info(f"Using masking with {masking_table_name=}, {masking_label_name=}")

    # Base Iterator with masking
    masked_image = ome_zarr.get_masked_image(
        masking_label_name=masking_label_name,
        masking_table_name=masking_table_name,
        path=level_path,
    )
    return masked_image


@validate_call
def stardist_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channel: StardistChannel,
    label_name: Optional[str] = None,
    level_path: Optional[str] = None,
    # Iteration parameters
    iterator_configuration: Optional[IteratorConfiguration] = None,
    # Stardist parameters
    model_type: StardistModels = StardistModels.DEMO_3D,
    custom_model: Optional[StardistpretrainedModel] = None,
    advanced_parameters: AdvancedStardistParams = AdvancedStardistParams(),  # noqa: B008
    pre_post_process: PrePostProcessConfiguration = PrePostProcessConfiguration(),  # noqa: B008
    create_masking_roi_table: AnyCreateRoiTableModel = SkipCreateMaskingRoiTable(),  # noqa: B008
    overwrite: bool = True,
) -> None:
    """Segment an image using Stardist.

    For more information, see:
        https://github.com/stardist/stardist

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channel (StardistChannel): Channel to use for segmentation.
        label_name (Optional[str]): Name of the resulting label image. If not provided,
            it will be set to "<channel_identifier>_segmented".
        level_path (Optional[str]): If the OME-Zarr has multiple resolution levels,
            the level to use can be specified here. If not provided, the highest
            resolution level will be used.
        iterator_configuration (Optional[IteratorConfiguration]): Configuration
            for the segmentation iterator. This can be used to specify masking
            and/or a ROI table.
        model_type: Parameter of `Stardist_ModelNames` class. Defines which model
            should be used. E.g. `2D_versatile_fluo`, `2D_versatile_he`,
            `2D_demo`, `3D_demo`.
        custom_model: Allows you to specify the path of
            a custom trained stardist model (takes precedence over `model_type`).
        advanced_parameters (AdvancedStardistParams): Advanced parameters
            for Stardist segmentation.
        pre_post_process (PrePostProcessConfiguration): Configuration for pre- and
            post-processing steps.
        create_masking_roi_table (AnyCreateMaskingRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    # Use the first of input_paths
    logging.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)

    logging.info(f"{ome_zarr=}")

    if label_name is None:
        label_name = f"{channel.identifiers[0]}_segmented"
    label = ome_zarr.derive_label(name=label_name, overwrite=overwrite)
    logging.info(f"Output label image: {label=}")

    if iterator_configuration is None:
        iterator_configuration = IteratorConfiguration()

    # Determine if we are doing 3D segmentation
    if ome_zarr.is_3d:
        axes_order = "czyx"
    else:
        axes_order = "cyx"

    # Set up the appropriate iterator based on the configuration
    label = ome_zarr.get_label(name=label_name, path=level_path)

    if iterator_configuration.masking is None:
        # Create a basic SegmentationIterator without masking
        image = ome_zarr.get_image(path=level_path)
        logging.info(f"{image=}")
        iterator = SegmentationIterator(
            input_image=image,
            output_label=label,
            channel_selection=channel.to_list(),
            axes_order=axes_order,
        )
    else:
        # Since masking is requested, we need to determine load a masking image
        masked_image = load_masked_image(
            ome_zarr=ome_zarr,
            masking_configuration=iterator_configuration.masking,
            level_path=level_path,
        )
        logging.info(f"{masked_image=}")
        # A masked iterator is created instead of a basic segmentation iterator
        # This will do two major things:
        # 1) It will iterate only over the regions of interest defined by the
        #   masking table or label image
        # 2) It will only write the segmentation results within the masked regions
        iterator = MaskedSegmentationIterator(
            input_image=masked_image,
            output_label=label,
            channel_selection=channel.to_list(),
            axes_order=axes_order,
        )
    # Make sure that if we have a time axis, we iterate over it
    # Strict=False means that if there no z axis or z is size 1, it will still work
    # If your segmentation needs requires a volume, use strict=True
    iterator = iterator.by_zyx(strict=False)
    logging.info(f"Iterator created: {iterator=}")

    if iterator_configuration.roi_table is not None:
        # If a ROI table is provided, we load it and use it to further restrict
        # the iteration to the ROIs defined in the table
        # Be aware that this is not an alternative to masking
        # but only an additional restriction
        table = ome_zarr.get_generic_roi_table(name=iterator_configuration.roi_table)
        logging.info(f"ROI table rßetrieved: {table=}")
        iterator = iterator.product(table)
        logging.info(f"Iterator updated with ROI table: {iterator=}")

    # Initialize Stardist model
    # Check if colab notebook instance has GPU access
    # Initialize stardist
    # Fixes #52 Startdist Model OSError
    model_loaded = False
    attempts = 0
    max_attempts = 10

    while not model_loaded and attempts < max_attempts:
        try:
            if custom_model:
                model_class = StarDist3D if ome_zarr.is_3d else StarDist2D
                model = model_class(
                    None,
                    name=custom_model.pretrained_model_name,
                    basedir=custom_model.base_fld,
                )
            else:
                model_class = StarDist3D if ome_zarr.is_3d else StarDist2D
                model = model_class.from_pretrained(model_type.value)

            if model:
                model_loaded = True
                logger.info("StarDist model loaded successfully.")
        except Exception as e:
            attempts += 1
            logger.warning(
                f"Attempt {attempts}/{max_attempts} to load StarDist model failed: {e}"
            )
            time.sleep(uniform(2, 7))

    if not model_loaded:
        raise RuntimeError("Failed to load StarDist model after multiple attempts.")

    # Keep track of the maximum label to ensure unique across iterations
    max_label = 0
    #
    # Core processing loop
    #
    logging.info("Starting processing...")
    for image_data, writer in iterator.iter_as_numpy():
        label_img = segmentation_function(
            image_data=image_data,
            model=model,
            parameters=advanced_parameters,
            do_3D=ome_zarr.is_3d,
            pre_post_process=pre_post_process,
        )
        # Ensure unique labels across different chunks
        label_img = np.where(label_img == 0, 0, label_img + max_label)
        max_label = label_img.max()
        writer(label_img)

    logging.info(f"label {label_name} successfully created at {zarr_url}")

    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        masking_roi_table = label.build_masking_roi_table()
        ome_zarr.add_table(name=table_name, table=masking_roi_table)

    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stardist_segmentation_task)
