from copy import deepcopy


def construct_model_config(config, default_config):
    mc = deepcopy(default_config)

    # yapf: disable
    mc["autocv"]["dataset"]["cv_valid_ratio"] = config["cv_valid_ratio"]
    mc["autocv"]["dataset"]["max_valid_count"] = config["max_valid_count"]
    mc["autocv"]["dataset"]["log2_max_size"] = 2 ** config["log2_max_size"]
    mc["autocv"]["dataset"]["max_times"] = config["max_times"]
    mc["autocv"]["dataset"]["train_info_sample"] = config["train_info_sample"]
    mc["autocv"]["dataset"]["enough_count"]["image"] = config["enough_count_image"]
    mc["autocv"]["dataset"]["enough_count"]["video"] = config["enough_count_video"]

    mc["autocv"]["dataset"]["steps_per_epoch"] = config["steps_per_epoch"]
    mc["autocv"]["conditions"]["early_epoch"] = config["early_epoch"]
    mc["autocv"]["conditions"]["skip_valid_score_threshold"] = config["skip_valid_score_threshold"]
    mc["autocv"]["conditions"]["test_after_at_least_seconds"] = config[
        "test_after_at_least_seconds"]
    mc["autocv"]["conditions"]["test_after_at_least_seconds_max"] = config[
        "test_after_at_least_seconds_max"]
    mc["autocv"]["conditions"]["test_after_at_least_seconds_step"] = config[
        "test_after_at_least_seconds_step"]
    mc["autocv"]["conditions"]["threshold_valid_score_diff"] = config["threshold_valid_score_diff"]
    mc["autocv"]["conditions"]["max_inner_loop_ratio"] = config["max_inner_loop_ratio"]

    mc["autocv"]["optimizer"]["lr"] = config["lr"]
    mc["autocv"]["optimizer"]["min_lr"] = config["min_lr"]
    mc["autocv"]["dataset"]["batch_size"] = config["batch_size"]
    # yapf: enable

    return mc