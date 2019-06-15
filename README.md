# AutoDLComp19
AutoDL Competition Scripts 2019


## Project Structure

```
├── .miniconda/                            <<  Project-local miniconda installation
│
├── competition/                           <<  Competition source code for local test
│   ├── ingestion_program/                 <<  Main execution library
│   ├── sample_result_submission/          <<  General output dir
│   ├── sample_submission/                 <<  Example submission code
│   │   └── model.py                       <<  Pytorch version of model.py from competition
│   ├── scoring_output/                    <<  Scoring output dir
│   ├── scoring_program/                   <<  Source code to produce AUL score
│   └── run_local_test.py                  <<  Execute competition evaluation locally
│
├── datasets/                              <<  Raw and processed datasets
│
├── experiments/                           <<  Logs and other files generated during runtime
│   └── cluster_oe/                        <<  Output and error files from clusters
│
├── models/                                <<  Parameters obtained offline (e.g., pretrained)
│
├── reports/                               <<  Analysis and results as tex, html, ...
│
├── src/                                   <<  Source code
│   ├── image/
│   │   ├── download_pretrained_models.py  <<  Download pretrained models
│   │   ├── models.py                      <<  Architectures and parameters
│   │   ├── online_concrete.py             <<  Training and inference strategies
│   │   ├── online_meta.py                 <<  Model/parameter selection, finetuining, ..
│   │   └── pretrained_models.hjson        <<  Models for download_pretrained_models.py
│   ├── ...
│   ├── video/
│   ├── config.hjson                       <<  Execution parameters for model.py
│   ├── dataloading.py                     <<  Dataloading utilities
│   ├── model.py                           <<  Main file for competition submission
│   └── utils.py                           <<  Utility code
│
├── submission/                            <<  Submission utilities
│   └── competition.py                     <<  Automatic generation of competition submissions
│
└── utils/                                 <<  General purpose scripts (formating, setup, ..)
```


## Setup

To setup the python environment or to update an environment run
```bash
bash utils/setup_environment.sh
```

1. If there is no miniconda installation, downloads the miniconda installer and installs it into `.miniconda`
1. Updates conda
1. If there is no autodl environment, installs it as specified in `utils/.environment.yml`
1. If there is an autodl environment and there are changes to `utils/.environment.yml` performs updates

To setup all tasks run
```bash
bash utils/setup_tasks.sh
```

This currently only does something on the meta cluster, where it symlinks `datasets/image` to the collected tfrecord datasets.


To setup all models (download pretrained weights), run
```bash
bash utils/setup_models.sh
```
This currently only downloads image models.


Setting up installs pre-commit. Pre-commit allows to specify, configure and share git pre-commit hooks. Formatting and format checking is therefore performed when committing. Please ensure you installed it. When in doubt, from the project's root run:
```bash
pre-commit install
```

To commit without runnning pre-commit supply the `--no-verify` option to `git commit`.

## Usage


### Running locally

To run the competition evaluation locally run
```bash
python competition/run_local_test.py \
    --dataset_dir datasets/DATASET \
    --code_dir MODELY_PY_FOLDER \
    --job_id JOB_ID \
    --task_id TASK_ID

# E.g.,
python competition/run_local_test.py \
    --dataset_dir datasets/public_data/Chucky \
    --code_dir competition/sample_submission \
    --job_id [SOME_INTEGER] \
    --task_id [SOME_INTEGER]
```

If you want to overwrite the output dir (for repeated local testing for example), supply the `--overwrite` flag.

If you want to open an interactive job session on the cluster run from the login node:
```bash
srun -p meta_gpu-x --pty bash
```
Then you can activate your environment and run the above scripts.

The script `submission/meta_cluster_array_job.sh` provides an examplefor running an array job. On the login node run:
```bash
sbatch submission/meta_cluster_array_job.sh
```


### Making a submission

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


### Miscellaneous

Activating an environment without having the conda installation in your `PATH`:
```bash
source .miniconda/bin/activate autodl
```

To run the pre-commit scripts manually run
```bash
bash utils/format.sh
```

## Deinstallation

To remove the conda installation run

```bash
bash utils/clean_conda.sh
```
