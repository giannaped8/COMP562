# Installation
## 1. Set up the Python Environment
We recommend to use Conda for managing your Python enviroment. Please download and install [Anaconda](https://www.anaconda.com/) or its minimal version [Miniconda](https://docs.conda.io/en/latest/miniconda.html). After successfully installing Conda, run the following commands to create a new Conda environment and activate it:

```
conda create --name comp562 python=3.9 numpy
conda activate comp562
```

## 2. Install the [PyBullet](https://pybullet.org/wordpress/) module

In your active Conda environment, run:

```
pip install pybullet --upgrade
```

Then, you can try to import the PyBullet module by ```import pybullet``` to check if the installation is successful.


# 3. To avoid abundance of printed warnings
```
Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2) enabled only processors has been deprecated. Intel oneAPI Math Kernel Library 2025.0 will require Intel(R) Advanced Vector Extensions (Intel(R) AVX) instructions.
```
If the following lines appear in the terminal when running the tasks, they can overpower the program output, 
and make it unable to decipher where the printed results of the code are

To run and evaluate each task, use the following corresponding terminal command
```
python -u main.py --task 1 2>&1 | awk '!/Intel MKL WARNING/ && !/Intel oneAPI Math Kernel Library 2025.0/'
python -u main.py --task 2 2>&1 | awk '!/Intel MKL WARNING/ && !/Intel oneAPI Math Kernel Library 2025.0/'
python -u main.py --task 3 2>&1 | awk '!/Intel MKL WARNING/ && !/Intel oneAPI Math Kernel Library 2025.0/'
python -u main.py --task 4 2>&1 | awk '!/Intel MKL WARNING/ && !/Intel oneAPI Math Kernel Library 2025.0/'
```

# 4. Saved Output
I saved the output of my Task 4 experiments in the task4_output.txt and ask4_relocation_results.csv files. 
I copied any useful information from these files into my report, so the files themselves are basically just raw data.
