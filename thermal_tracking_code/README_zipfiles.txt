## Handling Large `composite_thermals` Files

The `composite_thermals.zip` files exceed GitHub's 100 MB file size limit.
To comply with GitHub restrictions, these files are split into smaller parts:

composite_thermals_split.z01
composite_thermals_split.z02
...
composite_thermals_split.zip

### Download

1. Download all split files from the repository.
2. Make sure they are placed in the same directory.

Example:

uw_thermal_tracking_case1_cropped/
    composite_thermals_split.z01
    composite_thermals_split.z02
    ...
    composite_thermals_split.zip

---

### Reconstruct the original file

Run the following commands:

bash
zip -s 0 composite_thermals_split.zip --out composite_thermals_full.zip
unzip composite_thermals_full.zip

This will generate:

composite_thermals_full.zip
composite_thermals/

