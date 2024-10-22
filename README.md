# Meerkat Vital Sign Monitoring

Dataset and Analysis Pipeline to estimate vital signs of neonates from RGB-D data

## Description

Pipeline elements can be ran directly from main.ipynb.
Dataset is structured by patients containing one folder for each of the files.
On running a pipeline, the subject is selected from a list of available subjects.
Empty folder indicate unavailable/invalid signals.

## Getting Started

* Dataset can be downloaded from Apollo: https://doi.org/10.17863/CAM.111417
* Database path in main.ipynb needs to be changed to the corresponding location

### Dependencies

* Python (version 3.11.5)
* NumPy (version 1.25.2)
* SciPy (version 1.11.2)
* scikit-learn (version 1.3.1)
* simdkalman (1.0.4)
* ventiliser (version 1.0.0)
* Matplotlib (version 3.8.0)
* seaborn (version 0.13.0) 




## Authors

* Silas Ruhrberg Estevez
* sr933@cam.ac.uk


## License

This project is licensed under the CC-BY License

## Acknowledgments
* Alex Grafton, Lynn Thomson and Kathryn Beardsall for dataset collection
* Alex Grafton, Joana Warnecke and Joan Lasenby for supervision
* Rosetrees Trust for funding of the clinical study




