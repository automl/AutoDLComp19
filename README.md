# AutoDLComp19
AutoDL Competition Scripts 2019


## Project Structure

```
├── .miniconda          <<  Project-local miniconda installation
│
├── datasets            <<  Raw and processed datasets
│
├── experiments         <<  Logs and other files generated during runtime
│   └── cluster_oe      <<  Output and error files from clusters
│
├── models              <<  Parameters obtained offline (e.g., pretrained)
│
├── reports             <<  Analysis and results as tex, html, ...
│
├── src                 <<  Source code
│   ├── video
│   └── image
│
├── submission          <<  Submission scripts for clusters
│
└── utils               <<  General purpose scripts (formating, setup, ..)
```


## Setup

To setup the python environment or to update an environment run

```bash
bash utils/setup_environment.sh
```

1. If there is no miniconda installation, downloads the miniconda installer and installs it into `.miniconda`
1. Updates conda
1. If there is no autodl environment, installs it as specified in `utils/environment.yml`
1. If there is an autodl environment and there are changes to `utils/environment.yml` performs updates

To setup all tasks run

```bash
bash utils/setup_tasks.sh
```

This currently only does something on the meta cluster, where it symlinks `datasets/image` to the collected tfrecord datasets.

## Usage


Activating an environment without having the conda installation in your `PATH`:

```bash
source .miniconda/bin/activate autodl
```

Setting up installs pre-commit. Pre-commit allows to specify, configure and share git pre-commit hooks. Formatting and format checking is therefore performed when committing. Please ensure you installed it. When in doubt, from the project's root run:
```bash
pre-commit install
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
