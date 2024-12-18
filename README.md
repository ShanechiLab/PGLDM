# PGLDM
Python implementation of PGLDM (prioritized generalized-linear dynamical modeling). PGLDM is a multi-step analytical subspace identification algorithm for learning a generalized-linear dynamical model that explicitly models shared and private dynamics between two generalized-linear time-series.

More information about the algorithm can be found in our manuscript below.

## Publication
[Oganesian, L. L., Sani, O. G., Shanechi, M. M. Spectral Learning of Shared Dynamics Between Generalized-Linear Processes. In Advances in Neural Information Processing Systems 2024. (accepted)](https://openreview.net/forum?id=DupvYqqlAG)

## Installation 
We recommend using python3.11 to execute the code. If testing in a Unix or Unix-like operating system, we also recommend using a virtual environment.

```
python3 -m venv venv

source venv/bin/activate

python -m pip install -r requirements.txt
```

Add the following file venv/lib/python3.11/site-packages/pgldm.pth with contents
```
/path/to/code/directory
```

This will ensure that your virtual environment can find the relevant source code. 

In order to model Bernoulli observations, our codebase uses the latest version of [bestLDS](https://github.com/irisstone/bestLDS) to perform the appropriate moment conversions. You can follow a similar procedure to setup this dependency:
1. Clone the bestLDS repository to your desired location.
2. Add a venv/lib/python3.11/site-packages/bestLDS.pth file with the following contexts
```
/path/to/code/directory/bestLDS
```

To use your virtual environment in a jupyter notebook environment, we recommend using ipykernel. For example:

```
python -m ipykernel install --user --name=pgldm_demo
```

Run the demo notebook locally using jupyter notebook. You may need to change the kernel to point towards the correct environment "pgldm_demo".

## PGLDM Tutorial
We provide a [tutorial notebook](https://github.com/ShanechiLab/PGLDM/blob/master/PGLDM_Demo.ipynb) demonstrating usage of PGLDM across several different generalized-linear time-series observation pairs.

### Sample data for the tutorial
Sample data can be found in [this Google Drive folder](https://drive.google.com/drive/folders/1-b0tmYm6W4VGOV8j6LYAlBq13sSTiiR7?usp=sharing). The notebook expects the data to be stored in the base directory as ```/path/to/PGLDM/sample_data/```

## Troubleshooting

Please feel free to email or file GitHub issues if you have any questions, find any bugs, or run into problems getting setup.

## Licence
Copyright (c) 2024 University of Southern California  <br />
See full notice in [LICENSE.md](LICENSE.md)  <br />
Lucine L. Oganesian, Omid G. Sani, and Maryam M. Shanechi  <br />
Shanechi Lab, University of Southern California
