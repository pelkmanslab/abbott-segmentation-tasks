### Purpose
- **Segments images using Stardist models**.
- Supports both **built-in Stardist models** (shipped with Stardist) and **user-trained models**.
- Accepts single channel image input for segmentation.
- Can process **arbitrary regions of interest (ROIs)**, including whole images, fields of view (FOVs), or masked outputs from prior segmentations, based on corresponding ROI tables.
- Provides access to all advanced Stardist parameters.

### Outputs
- Generates a new label image named `<channel_identifier>_segmented` if no label name is provided.
