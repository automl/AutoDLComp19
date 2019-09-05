#!/usr/bin/env python
import argparse  # noqa: E402
import os
import re
import site
import subprocess
import zipfile

import numpy as np
from utils import BASEDIR, Config

BRANCHNAME = subprocess.check_output(
    ["git", "status"]
).strip().decode('utf8').split('\n')[0][10:].replace('/', '_')
COMMITHASH = subprocess.check_output(["git", "log"]).strip().decode('utf8')[8:16]
PYTHON_LIB_PATH = site.getsitepackages()[0]
PRETRAINED_WEIGHTS_PATH = os.path.join(BASEDIR, 'torchhome', 'checkpoints')

if __name__ == "__main__":  # noqa: C901
    # Construct base CLI, later add args dynamically from config file too
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission_dir", default=os.path.join(BASEDIR, "../submissions")
    )
    parser.add_argument("--code_dir", default=BASEDIR)
    parser.add_argument("--out_file", default=None)
    pargs = parser.parse_args()
    pargs.submission_dir = os.path.abspath(pargs.submission_dir)
    pargs.code_dir = os.path.abspath(pargs.code_dir)
    pargs.out_file = os.path.abspath(
        pargs.out_file
    ) if pargs.out_file is not None else None

    filename = (
        (BRANCHNAME[-20:] if len(BRANCHNAME) > 20 else BRANCHNAME)  # 20
        + ':'  # 1
        + COMMITHASH  # 7
        + '.zip'  # 4
    )  # Maxsize is 32 including extension
    out_file = (
        os.path.join(pargs.submission_dir, filename)
        if pargs.out_file is None else pargs.out_file
    )

    filelist = []
    whitelist = []
    blacklist = [
        'src/sideload_config.json', 'run.log', '.pth', '.pth.tar', '.tar',
        'pack_submission.py', 'bohb_auc.py', 'run_benchmark.py', 'README.md'
    ]

    config = Config(os.path.join(pargs.code_dir, 'config.hjson'))
    for p, f in config.include['packages'].items():
        if p == 'apex' and not config.use_amp:
            continue
        for fp in f:
            if os.path.isdir(os.path.join(PYTHON_LIB_PATH, fp)):
                for path, subdirs, files in os.walk(os.path.join(PYTHON_LIB_PATH, fp)):
                    for name in files:
                        fileabspath = os.path.join(path, name)
                        if (
                            (
                                '__pycache__' in fileabspath or '.gitkeep' in fileabspath
                                or np.any([e in fileabspath for e in blacklist])
                            ) and not np.any([e in fileabspath for e in whitelist])
                        ):
                            continue
                        if not os.path.isfile(fileabspath):
                            print(
                                '\033[91mZip does not support symlinks!!!: {}\033[0m'.
                                format(fileabspath)
                            )
                            continue
                        print('Adding to zip-list: {}'.format(fileabspath))
                        filelist.append(fileabspath)
            else:
                fileabspath = os.path.join(PYTHON_LIB_PATH, fp)
                print('Adding to zip-list: {}'.format(fileabspath))
                filelist.append(fileabspath)

    for p in config.include['pretrained_weights']:
        whitelist.append(p)

    for path, subdirs, files in os.walk(pargs.code_dir, onerror=print):
        for name in files:
            fileabspath = os.path.join(path, name)
            if (
                (
                    '__pycache__' in fileabspath or '.gitkeep' in fileabspath or
                    np.any([e in fileabspath for e in blacklist])
                ) and not np.any([e in fileabspath for e in whitelist])
            ):
                continue
            if not os.path.isfile(fileabspath):
                print(
                    '\033[91mZip does not support symlinks!!!: {}\033[0m'.
                    format(fileabspath)
                )
                continue
            print('Adding to zip-list: {}'.format(fileabspath))
            filelist.append(fileabspath)

    if not os.path.isdir(pargs.submission_dir):
        os.mkdir(pargs.submission_dir)

    if os.path.isfile(out_file):
        err = 'Maybe you forgot to commit and push? Anyway ' '{}' ' already exists.'.format(
            out_file
        )
        raise FileExistsError(err)

    with zipfile.ZipFile(out_file, 'w') as zfile:
        for f in filelist:
            zf = re.sub(r'^' + BASEDIR, '', re.sub(r'^' + PYTHON_LIB_PATH, '', f))
            zfile.write(f, zf, zipfile.ZIP_DEFLATED)
        fsize = os.path.getsize(out_file) / 1024.**2
        if fsize > 300.:
            print(
                '\033[1m\033[91mSubmission is bigger than 300 MB! Please double check!\033[0m'
            )
        print(
            '\033[92mFinished zipping. File has been created at '
            '{0}'
            '\nand is {1:.2f} MB in size.\033[0m'.format(out_file, fsize)
        )
