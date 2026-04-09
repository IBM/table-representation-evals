# TEmBed - Tabular Embedding Test Bed
This project provides a comprehensive test suite for evaluating tabular emebddings across four representation levels: cell, row, column and table using a diverse collection of tasks and datasets.

## Section 1: Benchmark Tasks

The benchmark currently includes the following task:

### 1) Tasks on Row Level

#### Row Similarity Search
**Description:** Given an input table and a row, the goal is to find the most similar row from the input table to the given row.

**Approaches:** To solve the tasks, approaches can provide row embeddings of a given table (implement the `row_embedding_component`) or provide a ranked list of the most similar rows (implement the `row_similarity_search_component`).  
Make sure that you set the *run_similarity_search_based_on* parameter in the experiment config accordingly.

#### Triplet-Based Evaluation
**Description:** Given a triplet of rows (anchor, positive, negative), the goal is to evaluate whether the anchor row is more similar to the positive row than to the negative row.

**Approaches:** Approaches should provide row embeddings (via the `row_embedding_component`).

#### Tabular Prediction
**Description:** Use row embeddings as features for downstream supervised tasks such as classification or regression.

**Approaches:** Approaches must provide row embeddings (via the `row_embedding_component`) or implement the `predictive_ml_component`. 



### 2) Tasks on Column Level

#### Column Similarity Search
**Description:** Given a query column, the goal is to retrieve and rank the most semantically similar columns from a data lake.

**Approaches:** Approaches must provide column embeddings (implement the `column_embedding_component`).


### 3) Tasks on Cell Level

#### Cell Level Semantic Retrieval
**Description:** Given a query cell, the goal is to retrieve the top-k most semantically similar cells across a collection of tables.

**Approaches:** Approaches must provide cell embeddings (implement the `cell_embedding_component`).

### 4) Tasks on Table Level

#### Table Retrieval
**Description:** Given a query table, the goal is to retrieve and rank the most semantically similar tables from a collection.

**Approaches:** Approaches must provide cell embeddings (implement the `table_embedding_component`).

## Section 2: Installation

1) Checkout this repository

2) Make a copy of the  `setup_benchmark.sh.template` script and rename it to `setup_benchmark.sh`. 

3) Adapt the parameters in `setup_benchmark.sh` to your needs, depending on which embedding approaches you want to use (you can re-run the script to install further approaches). 

Note: The benchmark uses Hugging Face Transformers which will cache models in the default location:
- Linux/Mac: `~/.cache/huggingface`
- Windows: `C:\Users\username\.cache\huggingface`

To use a different cache location, you can set the `HF_HOME` environment variable:
```bash
export HF_HOME="/path/to/your/preferred/cache"
```

4) To complete the installation, run 

```
bash setup_benchmark.sh
```

 
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

1) In approaches/configs/experiment, make a copy of <experiment>.yaml and rename it and fill the required fields.

2) In the commandline, from the embedding_benchmarks folder, run the following command, replacing <experiment_name> with the filename of the experiment yaml file you created in the previous step and the name of the conda environment:

```
bash run_benchmark.sh <experiment_name> benchmark_env
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
