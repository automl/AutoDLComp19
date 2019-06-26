import argparse  # noqa: E402
import os
import shutil
import sys

from src.utils import Config  # noqa: E402

sys.path.append(".")  # isort:skip

if __name__ == "__main__":
    # TODO(Danny): Do not copy stuff, but add stuff to zip file 1 by 1 for space
    # efficiency
    # TODO(Danny): Add packaging of python libs

    # Construct base CLI, later add args dynamically from config file too
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", default=".codalab_submission")
    parser.add_argument("--code_dir", default="src")
    parser.add_argument("--zip_name", default="codalab_submission")
    parser.add_argument(
        "--no_clean_up", action="store_true", help="Do not delete submission dir"
    )

    # Construct CLI dynamically from default config
    config = Config("src/config.hjson")  # Hardcode to circumvent argparse issue
    for key, value in config.__dict__.items():  # TODO(Danny): iter interface or smth
        parser.add_argument("--" + key, default=value, type=type(value))

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

    # Include extra packages
    for extra_package in config.extra_packages:
        shutil.copytree(
            extra_package,
            args.submission_dir + "/" + os.path.basename(extra_package),
            ignore=ignore,
        )

    if config.include_mixed_precision:
        # Include extra packages
        for pkg in config.mixed_precision_package:
            shutil.copytree(
                pkg,
                args.submission_dir + "/" + os.path.basename(pkg),
                ignore=ignore,
            )

    # Write
    config.is_codalab_submission = True
    config.write(args.submission_dir + "/config.hjson")

    # Zip everything and clean up
    shutil.make_archive(args.zip_name, "zip", args.submission_dir)
    if not args.no_clean_up:
        shutil.rmtree(args.submission_dir)  # TODO(Danny): Maybe wrap in try: finally:
