### Purpose
- **Seeded Segmentation to retrieve e.g. Cell Segmentation**.
- Accepts label_name of label image to use as seeds (e.g. nuclei) and single channel image input that contains boundary (e.g. membrane) marker.
- Can process **arbitrary regions of interest (ROIs)**, including whole images, fields of view (FOVs), or masked outputs from prior segmentations, based on corresponding ROI tables.
- If masked segmentation should be performed, use ROI table of type masking_roi_table.

### Limitations
- This task assumes that label_name and channel image are in same zarr_url.
