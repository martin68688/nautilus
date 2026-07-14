"""Built-in fixed-holdout task definitions.

The catalog describes file layout and scoring only. Prediction columns are
read from each task's sample submission during preparation, so wide tasks do
not need a duplicated, hand-maintained class list here.
"""

from copy import deepcopy


TASKS = {
    "spooky-author-identification": {
        "metric": "multiclass_log_loss",
        "id_column": "id",
        "prediction_columns": "sample_submission",
        "normalize_probabilities": True,
        "public_subdir": "prepared/public",
        "private_labels": "prepared/private/test.csv",
        "sample_submission": "sample_submission.csv",
    },
    "leaf-classification": {
        "metric": "multiclass_log_loss",
        "id_column": "id",
        "prediction_columns": "sample_submission",
        "normalize_probabilities": True,
        "public_subdir": "prepared/public",
        "private_labels": "prepared/private/test.csv",
        "sample_submission": "sample_submission.csv",
    },
    "aerial-cactus-identification": {
        "metric": "binary_roc_auc",
        "id_column": "id",
        "prediction_columns": ["has_cactus"],
        "normalize_probabilities": False,
        "public_subdir": "prepared/public",
        "private_labels": "prepared/private/test.csv",
        "sample_submission": "sample_submission.csv",
    },
    "new-york-city-taxi-fare-prediction": {
        "metric": "rmse",
        "id_column": "key",
        "prediction_columns": ["fare_amount"],
        "normalize_probabilities": False,
        "public_subdir": "prepared/public",
        "private_labels": "prepared/private/test.csv",
        "sample_submission": "sample_submission.csv",
    },
}


def task_spec(task_id: str) -> dict:
    try:
        return deepcopy(TASKS[task_id])
    except KeyError as exc:
        supported = ", ".join(sorted(TASKS))
        raise ValueError(
            f"Unknown fixed-holdout task {task_id!r}; supported tasks: {supported}"
        ) from exc
