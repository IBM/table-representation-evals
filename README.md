# table-representation-evals
This project provides a benchmark suite for evaluating the abilities of various models to perform tasks over tabular data, such as finding similar tables/columns/rows or clustering entities.

## Section 1: Benchmark Tasks

The benchmark currently includes the following task:

### Task 1: Row Similarity Search

**Description:** Given an input table and a row, the goal is to find the most similar row from the input table to the given row.

**Approaches:** To solve the tasks, approaches can provide row embeddings of a given table (implement the row_embedding_component) or provide a ranked list of the most similar rows (implement the row_similarity_search_component).
Make sure that you set the *run_similarity_search_based_on* parameter in the experiment config accordingly.


## Section 2: Installation

1) Checkout this repository

2) Create a conda environment and activate it (use lower python version if required by the approaches you are using). If you don't have conda installed yet, follow the installation instructions here: https://github.com/conda-forge/miniforge. 

```
conda create -n <environment-name> python=<3.13> 
```

3) Install packages from requirements.txt

```
pip install -r requirements.txt
```

4) Install the main benchmark evaluation project by running

```
pip install -e .
````

5) Install the approaches project by running

```
pip install -e approaches/.
```

6) Download the datasets and place them where you like, you will point to their location in the yaml configuration of each experiment that you run
 
## Section 3: How to add your approach

Please implement all code needed to run your approach in the /approaches folder, here are the necessary steps:

1) In the folder approaches/benchmark_approaches_src, create a copy of the <approach_name> folder, and rename it accordingly to the name of your approach

2) Open the approach.py file in your approach folder and rename the class

3) Import your class in the approaches/benchmark_approaches_src/__init__.py file

4) Make a copy of the _approach_name.yaml file in the approaches/configs/approach folder and fill in the name of your approach, as well as the foldername and classname that you set in step 1

5) Implement the functions in approach.py as well as in the components you need to run the benchmark (see description in Section 1 of this README). Please delete all the component files that you do not implement (if you plan to implement them later, just copy them from the template folder again).

    Hydra will automatically save a log file in the results folder (see documentation [here](https://hydra.cc/docs/tutorials/basic/running_your_app/logging/)). For printing, therefore please use the logging functions instead of print(). You can set the level to DEBUG in your experiment yaml config if needed.

## Section 4: How to run the benchmark

The benchmark is run one approach at a time, but you can configure several hyperaparameters that you want to try out. Some approaches require you to run their setup.sh script in the respective approach folder to install all necessary libraries before running the benchmark. 

1) In approaches/configs/experiment, make a copy of <experiment>.yaml and rename it. Set the *benchmark_datasets_dir:* parameter to the filepath where you saved the datasets. Then set all parameters for the approach you want to evaluate. 

2) In the commandline, from the embedding_benchmarks folder, run the following command, replacing <experiment_name> with the filename of the experiment yaml file you created in the previous step:

```
sh run_benchmark.sh <experiment_name>
```

3) Your results will be saved in the benchmark_results folder

## Section 5: Overview of the Repository
```
Tabular Benchmark Evaluation Framework
├── approaches                                       # approaches to be evaluated on the benchmark
│   ├── configs
│   │   ├── approach
│   │   │   └── <tabular_embedding_approach>.yaml    # hydra config per approach
│   │   └── experiment
│   │       └── <experiment>.yaml                    # run bemchmark on multiple configurations of an approach
│   └── benchmark_approaches_src
│       ├── __init__.py                              # import every class
│       └── <tabular_embedding_approach>             # implement interfaces per approach
│           ├── approach.py                          # the main class for the approach
│           └── <task-specific>_component.py.        # several component files for different ways to approach tasks
├── benchmark_src
│   ├── config                                       # hydra config files
│   ├── approach_interfaces                          # interfaces for the approaches and all components
│   ├── tasks                                        # task-specific code 
│   └── utils                                        # compute metrics, gather results, etc. 
└── results                                          # results folder
    └── <approach_name>
        └── <specific_parameters>
            └── <task_name>
                └── <dataset_name>
```
