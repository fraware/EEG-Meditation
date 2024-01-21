# Alpha Waves Dataset

Repository with basic scripts for using the Alpha Waves Dataset developed at GIPSA-lab [1]. The dataset files and their documentation are all available at

[https://zenodo.org/record/2348892](https://zenodo.org/record/2348892#.XBdqNs9Ki3I)

The code of this repository was developed in **Python 3.9 and 3.10** using MNE-Python [2, 3] as a tool for the EEG processing.

The package can be downloaded with pip:

```
pip install alphawaves
```

Alternatively, you might want to clone the GitHub repository on your local computer, and install the package by running:

```
python setup.py develop
```

All dependencies are listed in `requirements.txt` for your interest.

Then, to ensure that your code finds the right scripts, open a Python shell and type:

```
import alphawaves
```

Note that you might want to create a _virtual environment_ before doing all these installations, e.g.:

```
conda create -n eegalpha python=3.9
```

# References

[3] Gramfort et al. "MEG and EEG data analysis with MNE-Python" [DOI](https://doi.org/10.3389/fnins.2013.00267)
