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

### Making a submission

To create a submission `.zip` for the codalab platform run

```bash
python submission/codalab.py
```

This uses the settings in `src/configs/general.yaml`.


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
