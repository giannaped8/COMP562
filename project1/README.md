# Installation
## 1. Set up the Python Environment
We recommend to use Conda for managing your Python enviroment. Please download and install [Anaconda](https://www.anaconda.com/) or its minimal version [Miniconda](https://docs.conda.io/en/latest/miniconda.html). After successfully installing Conda, run the following commands to create a new Conda environment and activate it:

```
conda create --name comp462 python=3.9 numpy
conda activate comp462
```

## 2. Install the [PyBullet](https://pybullet.org/wordpress/) module

In your active Conda environment, run:

```
pip install pybullet --upgrade
```

Then, you can try to import the PyBullet module by ```import pybullet``` to check if the installation is successful.



# Saved Output
I saved the output of my Task 4 experiments in the task4_output.txt and ask4_relocation_results.csv files. 
I copied any useful information from these files into my report, so the files themselves are basically just raw data.
