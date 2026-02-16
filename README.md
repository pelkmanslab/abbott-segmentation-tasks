# abbott-segmentation-tasks

Fractal task collection to run Segmentation Tasks

## Available Tasks

| Task | Description | Passing |
| --- | --- | --- |
| Stardist Segmentation | Run Segmentation with pretrained/custom Stardist Model |✓|
| Seeded Watershed Segmentation | Performs segmentation (e.g., of cells) using a label image as seeds and an intensity image (e.g., membrane stain) for boundary detection. |✓|

## Installation

To install this task package on a Fractal server, get the tar.gz release and install using Pixi.

To install this package locally:
```
git clone https://github.com/pelkmanslab/abbott-segmentation_tasks
cd abbott-segmentation_tasks
pip install -e .
```

For development:
```
git clone https://github.com/pelkmanslab/abbott-segmentation_tasks
cd abbott-segmentation_tasks
pip install -e ".[dev]" 
```
