# An Input Representation for Convolutional Neural Networks in Exoplanet Transit Detection

## About this repo
The binning by quantile method is proposed to preprocess TESS light curves, where quantiles in each bin are calculated and placed in channels of the arrays. And a neural network model is trained to detect signals of exoplanet with TESS 2-min cadence data.

## List of python files for this project
[data_preprocess](data_preprocess/)
- [fold_bin_quantile.py](data_preprocess/fold_bin_quantile.py) downloading, detrending the light curves for the training-validation-test dataset.
- [fold_bin_quantile_sectors.py](data_preprocess/fold_bin_quantile_sectors.py) downloading, detrending the light curves for the TOI dataset.

[models](models/)
- [models.py](models/models.py) neural network architecture
- [train_model.py](models/train_model.py) model training and referring 
- [infer_sectors.py](models/infer_sectors.py) averaging the cross validation results for the test set and show some results.

## Preparing the data for the neural network
Run ```python data_preprocess/fold_bin_quantile.py``` to get 'train.npz', 'val.npz' and 'test.npz' in the 'model_input/data_q' folder.

Run ```python data_preprocess/fold_bin_quantile_sectors.py``` to get 'sectorXX.npz' in the 'model_input/data_q' folder.

## Model training and referring
For example, run
```
cd models/
sh run_training.sh
```
to train ten models for the exoplanet *vetting* task with data augmentation method inversion (*i*).
