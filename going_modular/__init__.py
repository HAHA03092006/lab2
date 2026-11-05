"""
going_modular — CIFAR-10 Transfer Learning helpers.

Import theo module (cách dùng trong notebook):
    from going_modular import data_setup, engine, evaluate, model_builder, utils

Hoặc flat import:
    from going_modular import (
        create_train_val_test_dataloaders,
        get_pretrained_transform,
        build_model,
        count_trainable_params,
        count_total_params,
        train, train_step, evaluate_step, eval_model,
        predict_probs, predict_dataloader,
        compute_confusion_matrix, per_class_accuracy, top_confusions,
        plot_history, plot_confusion_matrix, plot_sample_predictions,
        set_seed, get_device, save_model, load_model,
        save_training_metadata, load_training_artifacts,
    )
"""

from .data_setup import (
    create_train_val_test_dataloaders,
    get_pretrained_transform,
    stratified_split_indices,
)

from .model_builder import (
    build_model,
    count_trainable_params,
    count_total_params,
    count_params,
    describe_model,
    format_model_summary,
    get_param_groups,
    get_transforms,
    set_train_mode,
    ModelConfig,
    ARCH_SPECS,
    CIFAR10ModelV0,
    CIFAR10ModelV1,
    MODEL_NAMES,
)

from .engine import (
    train,
    train_step,
    evaluate_step,
    eval_model,
)

from .evaluate import (
    predict_probs,
    predict_dataloader,
    compute_confusion_matrix,
    per_class_accuracy,
    top_confusions,
    plot_history,
    plot_confusion_matrix,
    plot_sample_predictions,
)

from .utils import (
    set_seed,
    get_device,
    get_env_info,
    create_writer,
    save_model,
    load_model,
    save_config,
    save_history,
    load_experiment,
    save_test_metrics,
    load_test_metrics,
    artifacts_exist,
    save_checkpoint,
    load_checkpoint,
    save_training_metadata,
    load_training_artifacts,
    accuracy_fn,
    print_train_time,
)

__all__ = [
    # data
    "create_train_val_test_dataloaders",
    "get_pretrained_transform",
    "stratified_split_indices",
    # model
    "build_model",
    "count_trainable_params",
    "count_total_params",
    "CIFAR10ModelV0",
    "CIFAR10ModelV1",
    "MODEL_NAMES",
    # engine
    "train",
    "train_step",
    "evaluate_step",
    "eval_model",
    # evaluate
    "predict_probs",
    "predict_dataloader",
    "compute_confusion_matrix",
    "per_class_accuracy",
    "top_confusions",
    "plot_history",
    "plot_confusion_matrix",
    "plot_sample_predictions",
    # utils
    "set_seed",
    "get_device",
    "get_env_info",
    "create_writer",
    "save_model",
    "load_model",
    "save_config",
    "save_history",
    "load_experiment",
    "save_test_metrics",
    "load_test_metrics",
    "artifacts_exist",
    "save_checkpoint",
    "load_checkpoint",
    "save_training_metadata",
    "load_training_artifacts",
    "accuracy_fn",
    "print_train_time",
]