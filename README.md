Learning Underwater Active Perception in Simulation\
[[website]](https://roboticimaging.org/Projects/ActiveUW/) [[paper]](https://arxiv.org/abs/2504.17817)
===============
This repository contains the code of the "Learning Underwater Active Perception in Simulation" publication. Specifically, it contains the MLP model and a prediction script with two example scenarios, the conversion of the Jerlov water type data, and an example Blender scene.

### Setup

The following Python packages are required:
  * `numpy`
  * `tensorflow`
  * `scipy`
  * `matplotlib`
  * `scikit-image`

To open the Blender scene, Blender >= 4.3 is required.

### Predicting the best position and illumination
Use the `predict.py` script to find the best distance and light intensity given the current settings. The MLP model and two examples are provided.

The script loads the image `img_x.tif` and the corresponding parameters (current distance, light intensity, depth, and calibration profiles) stored as numpy arrays in `img_x_params.txt` and `img_x_kc_off.txt`. It then outputs the recommended distance and light intensity.

### Jerlov water type data conversion
Use the `wavelength2rgb.py` script to convert the wavelength-dependent absorption and scattering coefficients to RGB. The wavelengths for the 10 Jerlov water types (I, II, III, IA, IB, IC, 3C, 5C, 7C, 9C) are provided, from 300nm to 700nm.

By default the `[400, 500]` interval is used for blue, `[500, 600]` for green, and `[600, 700]` for red. They can be modified to better correspond to specific camera characteristics. The script outputs the corresponding channel-wise coefficients and plots them.

### Underwater imagery simulation in Blender
As part of this project, we updated the Blender modelling software. To adapt it to underwater environments, we replace the Henyey-Greenstein scattering phase function with the Fournier-Forand function. The latter function better accounts for large and small scattering angles and the backscatter fraction, which are crucial for physically accurate underwater image rendering.

This addition to Blender is now publicly available and was part of the official 4.3 release of the software, along with the integration of other phase functions such as the Draine and Rayleigh phase functions.

Additionally, we include the possibility to modify channel-wise densities instead of colour coefficients. At the moment, this last part is only available using the following Blender branch: https://projects.blender.org/Alexandre-Cardaillac/blender/src/branch/density_modes

This makes it possible to replicate the existing oceans and their light absorption and scattering. A sample scene `water_types.blend` is included in the repository and includes the major ocean types based on the inherent optical properties of Jerlov water types using the conversion presented above. The ten water types are included, from coastal waters to oceanic waters, with various levels of turbidity.

## Citation
If you find our work useful, please cite the following paper:
```
@misc{cardaillac2025learning,
  title={Learning Underwater Active Perception in Simulation},
  author={Alexandre Cardaillac and Donald G. Dansereau},
  year={2025},
  eprint={2504.17817},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2504.17817},
}
```
