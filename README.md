# An Input Representation for Convolutional Neural Networks in Exoplanet Transit Detection

## About this repo
The binning by quantile method is proposed to preprocess TESS light curves, where quantiles in each bin are calculated and placed in channels of the arrays. And a neural network model is trained to detect signals of exoplanet with TESS 2-min cadence data.

## List of python files for this project
[data_preprocess](data_preprocess/)
- [fold_bin_quantile.py](data_preprocess/fold_bin_quantile.py) preprocessing the light curves for the training-validation-test dataset.
- [fold_bin_quantile_sectors.py](data_preprocess/fold_bin_quantile_sectors.py) preprocessing the light curve of any sector and generating the model inputs representation .

[models](models/)
- [models.py](models/models.py) neural network architecture
- [train.py](models/train.py) model training and referring 
- [infer_sectors.py](models/infer_sectors.py) inference and prediction of any sector

## Preparing the data for the neural network
Run ```python data_preprocess/fold_bin_quantile.py``` to get 'training_data.npz' in the 'model_input/' folder.

Run ```python data_preprocess/fold_bin_quantile_sectors.py``` to get 'sectorXX.npz' in the 'model_input/data_q' folder.

## Model training and referring
For example, run
```
cd models/
sh run_training_part1.sh
sh run_training_part2.sh
sh run_infer.sh
```
to train ten models for the exoplanet *vetting* task with data augmentation method inversion (*i*).

![图片](https://user-images.githubusercontent.com/79409336/133979691-bc18c9ff-ce72-473c-97e1-638d396c6b58.png)
