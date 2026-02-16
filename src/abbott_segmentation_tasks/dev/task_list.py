"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Ruth Hornbachner"


DOCS_LINK = "https://github.com/pelkmanslab/abbott-segmentation-tasks"

INPUT_MODELS = [
    ("ngio", "images/_image.py", "ChannelSelectionModel"),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "MaskingConfiguration",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "IteratorConfiguration",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "SkipCreateMaskingRoiTable",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "CreateMaskingRoiTable",
    ),
    (
        "abbott_segmentation_tasks",
        "pre_post_process.py",
        "PrePostProcessConfiguration",
    ),
    (
        "abbott_segmentation_tasks",
        "pre_post_process.py",
        "GaussianFilter",
    ),
    (
        "abbott_segmentation_tasks",
        "pre_post_process.py",
        "MedianFilter",
    ),
    (
        "abbott_segmentation_tasks",
        "pre_post_process.py",
        "SizeFilter",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "SeededSegmentationParams",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "SeededSegmentationChannels",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "SeededSegmentationIterator",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "SeededMaskedSegmentationIterator",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "StardistChannel",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "NormalizationParameters",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "AdvancedStardistParams",
    ),
    (
        "abbott_segmentation_tasks",
        "utils.py",
        "StardistpretrainedModel",
    ),
]


TASK_LIST = [
    ParallelTask(
        name="Stardist Segmentation",
        executable="stardist_segmentation_task.py",
        meta={"cpus_per_task": 4, "mem": 16000, "needs_gpu": True},
        category="Segmentation",
        tags=["Instance Segmentation", "Classical segmentation", "2D", "3D"],
        docs_info="file:docs_info/stardist_segmentation_task.md",
    ),
    ParallelTask(
        name="Seeded Watershed Segmentation",
        executable="seeded_segmentation.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Segmentation",
        tags=[
            "scikit-image",
            "3D",
        ],
        docs_info="file:docs_info/seeded_segmentation.md",
    ),
]
