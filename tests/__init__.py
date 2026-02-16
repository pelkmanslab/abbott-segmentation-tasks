import json
from pathlib import Path

import abbott_segmentation_tasks

PACKAGE_DIR = Path(abbott_segmentation_tasks.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
