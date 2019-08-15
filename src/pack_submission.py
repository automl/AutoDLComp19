#!/usr/bin/env python
import argparse  # noqa: E402
import zipfile
import os
import site
import subprocess
from utils import Config, BASEDIR

BRANCHNAME = subprocess.check_output(["git", "status"]).strip().decode('utf8').split('\n')[0][10:].replace('/', '_')
COMMITHASH = subprocess.check_output(["git", "log"]).strip().decode('utf8')[8:16]
PYTHON_LIB_PATH = site.getsitepackages()[0]
PRETRAINED_WEIGHTS_PATH = os.path.join(BASEDIR, 'torchhome', 'checkpoints')

if __name__ == "__main__":
    # Construct base CLI, later add args dynamically from config file too
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", default=os.path.join(BASEDIR, "../submissions"))
    parser.add_argument("--code_dir", default=BASEDIR)
    parser.add_argument("--out_file", default=None)
    pargs = parser.parse_args()
    pargs.submission_dir = os.path.abspath(pargs.submission_dir)
    pargs.code_dir = os.path.abspath(pargs.code_dir)
    pargs.out_file = os.path.abspath(pargs.out_file) if pargs.out_file is not None else None

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

    config = Config(os.path.join(pargs.code_dir, 'config.hjson'))
    for p, f in config.include['packages'].items():
        if p == 'apex' and not config.use_amp:
            continue
        filelist.append(os.path.join(PYTHON_LIB_PATH, f))

    for p in config.include['pretrained_weights']:
        whitelist.append(os.path.join(PRETRAINED_WEIGHTS_PATH, p))

    for path, subdirs, files in os.walk(pargs.code_dir, onerror=print):
        if PRETRAINED_WEIGHTS_PATH in path:
            print('Skipping Folder {}'.format(path))
            continue
        for name in files:
            fileabspath = os.path.join(path, name)
            if (
                '__pycache__' in fileabspath
                or '.gitkeep' in fileabspath
            ):
                continue
            filelist.append(os.path.join(path, name))

    filelist += whitelist

    if not os.path.isdir(pargs.submission_dir):
        os.mkdir(pargs.submission_dir)

    if os.path.isfile(out_file):
        err = 'Maybe you forgot to commit and push? Anyway ''{}'' already exists.'.format(out_file)
        raise FileExistsError(err)

    with zipfile.ZipFile(out_file, 'w') as zfile:
        for f in filelist:
            print('Adding to archive: {}'.format(f))
            zfile.write(f, f.replace(BASEDIR, ''), zipfile.ZIP_DEFLATED)
        fsize = os.path.getsize(out_file) / 1024.**2
        if fsize > 300.:
            print('\033[1m\033[91mSubmission is bigger than 300 MB! Please double check!\033[0m')
        print('\033[92mFinished zipping. File has been created at ''{0}''\nand is {1:.2f} MB in size.\033[0m'.format(
            out_file, fsize
        ))
