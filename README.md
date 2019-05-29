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
└── submission           <<  Submission scripts for clusters
```


## Setup

To setup the python environment or to update an environment run

```bash
bash utils/setup_environment.sh
```

1. If there is no miniconda installation, downloads the miniconda installer from the web and installs miniconda into `.miniconda`
1. Updates conda
1. If there is no autodl environment, installs it as specified in `utils/environment.yml`
1. If there is an autodl environment and there are changes to `utils/environment.yml` performs updates 

## Usage


(De)activating an environment without having the conda installation in your `PATH`:

```bash
source .miniconda/bin/activate autodl
source .miniconda/bin/deactivate
```

To format all python code, sort imports, and display unused imports in `src/` run:

```bash
bash utils/format.sh  # currently only formats in src/image
```

## Deinstallation

To remove the conda installation run

```bash
bash utils/clean_conda.sh
```
