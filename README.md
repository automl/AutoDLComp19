# AutoDLComp19
AutoDL Competition 2019


## Installation

Activate a conda python3.5 environment then run
```bash
bash install/requirements_gcc.sh
pip install -r install/requirements.txt
bash install/requirements_torch_cuda100.sh
bash install/install_winner_cv.sh
bash install/install_winner_speech.sh
bash install/install_just.sh        # Optional command runner
bash install/install_precommit.sh   # Developer dependency
```


## Usage


### Running locally

To run the competition evaluation locally run
```bash
python -m src.competition.run_local_test \
    --dataset_dir DATASET_DIR \
    --code_dir src \
    --model_config_name CONFIG.yaml \
    --experiment_group EXPERIMENT_GROUP \
    --experiment_name EXPERIMENT_NAME \
    --time_budget 1200 \
```

CONFIG corresponds to one of the names of the general configs in `src/configs/`. If this argument is ommited, `src/configs/default.yaml` is used.

You can use `--time_budget_approx <LOWER_TIME>` and `--time_budget <ACTUAL_TIME>` to simulate cutting a run with budget `<ACTUAL_TIME>` after `<LOWER_TIME>` seconds.

If you want to overwrite the output dir (for repeated local testing for example), supply the `--overwrite` flag.

### Do not run pre-commit hooks

To commit without runnning `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Generate a performance matrix 
#### 0. create available_datasets.py dynamically (to be implemented)
#### 1. Create the arguments for the HPO and run them on META as follows
```bash
python submission/create_hpo_args.py --command_file_name ARGS_FILE_NAME #--> args file outputted
sbatch submission/meta_kakaobrain_optimized_per_dataset.sh # --> set ARGS_FILE parameter to newly created ARGS_FILE_NAME, set experiment_group to EXPERIMENT_DIR, set budgets
```
#### 2. Generate incumbent configs
```bash
mkdir src/configs/INC_OUTPUT_DIR
python src/hpo/incumbents_to_config.py --output_dir src/configs/INC_OUTPUT_DIR --experiment_group_dir EXPERIMENT_DIR # --> .yaml configs outputted to EXPERIMENT_DIR
```
#### 3. Evaluate configurations
```bash
python submission/create_datasets_x_configs_args.py --config_path INC_OUTPUT_DIR --command_file_name EVAL_ARGS_FILE_NAME # --> EVAL_ARGS_FILE_NAME stored in submission/
sbatch submission/meta_kakaobrain_datasets_x_configs.sh # --> set ARGS_FILE to EVAL_ARGS_FILE_NAME, set --experiment_group to EVALUATION_DIR_PATH, evaluations stored in EVALUATION_DIR_PATH
```

#### 4. Once the evaluation directory has been generated, generate the pandas DataFrames and csv files with the following command
```bash
python src/hpo/performance_matrix_from_evaluation.py --experiment_group_dir EVALUATION_DIR_PATH
```

### Pre-Computing meta features

To pre-compute meta features run

```bash
python -m src.meta_features.precompute_meta_features --dataset_path DATASET_PATH
```

the default output_path is `src/meta_features/meta_features.yaml`.

### Making a submission

To create a submission `.zip` for the codalab platform run

```bash
python submission/codalab.py
```

This uses the settings in `src/configs/default.yaml`.


## Project Structure

```
├── experiments/                           <<  Files generated during runtime
│
├── install/                               <<  Requirements and scripts for installation
│
├── src/                                   <<  Source code
│   └── winner_<TRACK>/                    <<  Winner code for <TRACK>
│   └── competition/                       <<  Competition source code
│       └── run_local_test.py              <<  Execute competition evaluation locally
│
├── submission/                            <<  Submission utilities
│    └── competition.py                    <<  Create codalab submission
│
└── justfile                               <<  Command runner file akin to Makefile
```


## License

[Apache 2.0](LICENSE)
