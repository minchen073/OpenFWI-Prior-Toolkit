# Third-party attribution

The DhariwalUNet, EDM preconditioning, and network modules in
`src/openfwi_prior/vendor/` are adapted from InverseBench. Its MIT license is
reproduced in [licenses/InverseBench-MIT.txt](licenses/InverseBench-MIT.txt).
Source-level attribution is retained in the corresponding files.

The example velocity and seismic arrays are selected from OpenFWI:
*Large-Scale Multi-Structural Benchmark Datasets for Seismic Full Waveform
Inversion*, NeurIPS 2022. The [OpenFWI project](https://openfwi-lanl.github.io/)
provides dataset information. Source filenames and sample indices are in
`assets/manifest.json`.

The pretrained EDM priors were trained by the PSG-FWI authors on the respective
OpenFWI families. Example data and pretrained model assets are provided under
[CC BY-NC-SA 4.0](assets/LICENSE.md), separately from the code license.
