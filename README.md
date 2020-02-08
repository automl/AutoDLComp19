# AutoDLComp19
AutoDL Competition 2019


## Installation

Activate a conda python3.5 environment then run
```bash
bash install/requirements_gcc.sh
pip install -r install/requirements.txt
bash install/requirements_torch.sh
bash install/install_precommit.sh
bash install/install_just.sh
bash install/install_winner_cv.sh
```


## Usage


### Running locally

To run the competition evaluation locally run
```bash
python -m src.competition.run_local_test \
    --dataset_dir DATASET_DIR \
    --code_dir src \
    --model_config CONFIG \
    --experiment_dir EXPERIMENT_DIR \
    --time_budget 1200
```

CONFIG corresponds to one of `src/configs/thomas_configs`.

If you want to overwrite the output dir (for repeated local testing for example), supply the `--overwrite` flag.

### Do not run pre-commit hooks

To commit without runnning `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Making a submission (DEPRECIATED)

To create a submission `.zip` for the codalab platform run

```bash
python submission/competition.py
```

This uses the settings in `src/config.hjson` to determine the modality, lookup paths, finetuning strategy, pretrained_parameters and model to load, hyperparameters, etc. To change the settings, you can either edit `src/config.hjson` or via arguments:


```bash
python submission/competition.py --lr 1e-4
```

You need to specify which model parameter files you want to include in the submission. You can do this via editing the `active_model_files` attribute, e.g.,

```json
active_model_files: ["resnet18-5c106cde"]  # With respect to model_dir
```

To include a python package that is not included on the competition platform, edit the `extra_packages` attribute, e.g.,

```json
extra_packages: [".miniconda/envs/autodl/lib/python3.5/site-packages/hjson"]
```



## Project Structure

```
├── experiments/                           <<  Logs and other files generated during runtime
│
├── install/                               <<  Requirements and scripts for installation
│
├── src/                                   <<  Source code
│   └── winner_<TRACK>/                    <<  Winner code for <TRACK>
│   └── competition/                       <<  Competition source code
│       └── run_local_test.py              <<  Execute competition evaluation locally
│
└── submission/                            <<  Submission utilities
    └── competition.py                     <<  Automatic generation of competition submissions
```


## License

[Apache 2.0](LICENSE)
