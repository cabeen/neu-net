# neU-net

This package provides tools for U-net segmentation of biomedical imaging data.  This is a typical implementation of the U-net architecture in [PyTorch](https://pytorch.org/) with additional functionality for running experiments and deploying models to run in imaging studies.  The name is an abbreviation of "neural U-net" since it was developed for a variety of neuroimaging tasks.

## Design

The general approach taken by the package can be outlined as follows:
* Train a the model with a first batch of data for some number of epoches
* Validate the resulting models with second batch of data to determine the optimal model
* Test the optimal model with a third batch to estimate the accuracy in practical usage

There are additional features to know about:
* Creating mosaic plots for quickly viewing 3D volumes
* Handling multi-channel data
* Data augmentation by shifting, flipping, etc.
* Multi-slice learning of 2D images

You can learn more about the architecture by checking out the [original MICCAI paper](https://arxiv.org/abs/1505.04597) or [this nice lecture](https://youtu.be/azM57JuQpQI).  This work was inspired by the approach in [this paper](https://biorxiv.org/cgi/content/short/2020.11.17.385898v1) and [repo](https://github.com/HumanBrainED/NHP-BrainExtraction).

## Requirements

This package is implemented in Python, and you can find the required packages in the `reqs.txt` file. I recommend you create a virtual environment, using either [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Python venv](https://docs.python.org/3/tutorial/venv.html).

## Acknowledgements

Author: Ryan Cabeen, cabeen@gmail.com

This work was made possible in part by the CZI Imaging Scientist Award Program, under grant number 2020-225670 from the Chan Zuckerberg Initiative DAF, an advised fund of Silicon Valley Community Foundation.
