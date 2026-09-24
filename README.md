# abbott-segmentation-tasks

## Overview
The abbott-segmentation-tasks Task Collection is intended to be used in combination with the [Fractal Analytics Platform](https://github.com/fractal-analytics-platform) maintained by the [BioVisionCenter Zurich](https://www.biovisioncenter.uzh.ch/en.html) (co-founded by the Friedrich Miescher Institute and the University of Zurich).

The tasks in abbott-segmentation-tasks are focused on extending Fractal's capabilities to segment objects (e.g., nuclei and cells) in (multiplexed) 3D image data. For pre-processing of 3D multiplexed imaging data take a look at [abbott](https://github.com/pelkmanslab/abbott/tree/main). For feature extraction from the resulting label images check out [abbott-features](https://github.com/pelkmanslab/abbott-features).

## Available Tasks
| Task | Description | Passing |
| --- | --- | --- |
| Stardist Segmentation | Segments objects using a pretrained or custom StarDist model. | ✓ |
| Seeded Watershed Segmentation | Segments objects (e.g., cells) using a label image as seeds and an intensity image (e.g., membrane stain) for boundary detection. | ✓ |

## Installation

### On a Fractal server

Download the `.tar.gz` from the latest [GitHub release](https://github.com/pelkmanslab/abbott-segmentation-tasks/releases) and install it with Fractal's pixi task collection.

### Locally

```bash
git clone https://github.com/pelkmanslab/abbott-segmentation-tasks
cd abbott-segmentation-tasks
pip install -e .
```

## Development

The development environment is managed with [pixi](https://pixi.sh):

```bash
git clone https://github.com/pelkmanslab/abbott-segmentation-tasks.git
cd abbott-segmentation-tasks
pixi run init-tasks    # install pre-commit hooks, format code, build manifest, run tests
```

Individual tasks:

```bash
pixi run -e dev create-manifest   # regenerate __FRACTAL_MANIFEST__.json
pixi run -e dev format-code       # ruff format
pixi run -e test test             # run the test suite
```
