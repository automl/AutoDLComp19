import os
import shutil
import hjson
import sys
sys.path.append('.')
from src.utils import Config
import argparse

if __name__ == '__main__':
    # TODO(Danny): Do not copy stuff, but add stuff to zip file 1 by 1 for space
    # efficiency
    # TODO(Danny): Add packaging of python libs

    # Construct base CLI, later add args dynamically from config file too
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_dir', default="colab_submission")
    parser.add_argument('--code_dir', default="src")
    parser.add_argument('--zip_name', default="colab_submission")

    # Construct CLI dynamically from default config
    config = Config("src/config.hjson")  # Hardcode to circumvent argparse issue
    for key, value in config.__dict__.items():  # TODO(Danny): iter interface or smth
        parser.add_argument('--' + key, default=value, type=type(value))

    # Overwrite config values with parsed values
    args = parser.parse_args()
    args_dict = vars(args)
    for key, _ in config.__dict__.items():
        config.__dict__[key] = args_dict[key]

    # Create submission directory
    if os.path.isdir(args.submission_dir):
        shutil.rmtree(args.submission_dir)  # shutil does not work with pathlib
    ignore = shutil.ignore_patterns("__pycache__", "config.hjson")
    shutil.copytree(args.code_dir, args.submission_dir, ignore=ignore)

    # Copy active models
    for model_file in config.active_model_files:
        model_name = model_file + ".pth"
        shutil.copyfile(
            config.model_dir + "/" + model_name, args.submission_dir + "/" + model_name
        )

    # Set model dir to correct value with respect to the submission
    config.model_dir = "."  # TODO(Danny): Check if this is correct for submission


    config.write(args.submission_dir + "/config.hjson")

    # Zip everything
    shutil.make_archive(args.zip_name, 'zip', args.submission_dir)
