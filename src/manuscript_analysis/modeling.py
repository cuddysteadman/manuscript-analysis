from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import tensorflow as tf

from . import artifacts
from .contract import N_RUNS
from .data import apply_scaler, fit_scaler


def get_models(model_dir: str | Path, *, expected_count: int | None = N_RUNS):
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    seed_dirs = [path for path in model_dir.glob("seed*") if path.is_dir()]
    try:
        seed_dirs.sort(key=lambda path: int(path.name.removeprefix("seed")))
    except ValueError as error:
        raise ValueError(
            f"Model directories under {model_dir} must be named seed<N>"
        ) from error

    if expected_count is not None and len(seed_dirs) != expected_count:
        raise ValueError(
            f"Expected {expected_count} seed models in {model_dir}; "
            f"found {len(seed_dirs)}"
        )

    models = {}
    for seed_dir in seed_dirs:
        required = ("model.keras", "scaler.npz", "metrics.json")
        if missing := [name for name in required if not (seed_dir / name).is_file()]:
            raise FileNotFoundError(
                f"Incomplete model run {seed_dir}; missing {', '.join(missing)}"
            )
        threshold = json.loads((seed_dir / "metrics.json").read_text())["session_tuned_threshold"]
        models[seed_dir.name] = (
            tf.keras.models.load_model(seed_dir / "model.keras"),
            np.load(seed_dir / "scaler.npz", allow_pickle=False),
            float(threshold),
        )
    return models


def build_lstm(input_shape, config):
    regularizer = tf.keras.regularizers.l2(config["l2_reg"])
    model = tf.keras.Sequential(name="unsorted_arch_lstm")
    model.add(tf.keras.layers.Input(shape=input_shape, name="input"))
    # Normalizing a one-wide feature axis turns every timestep into zero.
    if config["add_layernorm"] and input_shape[-1] > 1:
        model.add(tf.keras.layers.LayerNormalization(name="input_ln"))
    nodes = list(config["layer_nodes"])
    for index, width in enumerate(nodes):
        model.add(tf.keras.layers.LSTM(width, return_sequences=index < len(nodes) - 1, recurrent_dropout=config["recurrent_dropout"], kernel_regularizer=regularizer, recurrent_regularizer=regularizer, name=f"LSTM_{index}"))
        model.add(tf.keras.layers.Dropout(config["dropout"], name=f"drop_lstm_{index}"))
    model.add(tf.keras.layers.Flatten(name="flatten"))
    model.add(tf.keras.layers.Dense(config["dense_nodes"], activation="relu", kernel_regularizer=regularizer, name="dense"))
    model.add(tf.keras.layers.Dropout(.2, name="drop_dense"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="out"))
    return model


def fit_lstm_model(
    X_train, y_train, X_val, y_val, train_weights, *, input_shape,
    model_config, train_config, seed, verbose=0, callbacks=(),
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model = build_lstm(input_shape, model_config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(train_config["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=train_config["patience"],
            restore_best_weights=True,
        ),
        *callbacks,
    ]
    if train_config["lr_plateau"]:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=.5, patience=5, min_lr=1e-6,
        ))
    started = time.time()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        sample_weight=train_weights, epochs=train_config["epochs"],
        batch_size=train_config["batch_size"], callbacks=callbacks, verbose=verbose,
    )
    return model, history, time.time() - started


def _threshold(labels, probabilities):
    if np.unique(labels).size < 2:
        raise ValueError("Threshold tuning requires both classes")
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    valid = np.isfinite(thresholds)
    return float(thresholds[valid][np.argmax(tpr[valid] - fpr[valid])]) if valid.any() else .5


def _aggregate(recording_ids, probabilities, labels):
    ids = np.unique(recording_ids)
    output_p, output_y, counts = [], [], []
    for recording in ids:
        mask = recording_ids == recording
        values = probabilities[mask]
        weights = np.exp(4 * np.abs(values - .5))
        output_p.append(np.sum(weights * values) / np.sum(weights))
        output_y.append(int(labels[mask][0]))
        counts.append(int(mask.sum()))
    return ids, np.asarray(output_p), np.asarray(output_y), np.asarray(counts)


def aggregate_recordings(recording_ids, probabilities, labels):
    return _aggregate(np.asarray(recording_ids), np.asarray(probabilities), np.asarray(labels))


def probability_metrics(labels, probabilities, prefix):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    auc = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else np.nan
    return {
        f"{prefix}_auc": auc,
        f"{prefix}_average_precision": float(average_precision_score(labels, probabilities)),
        f"{prefix}_brier": float(brier_score_loss(labels, probabilities)),
        f"{prefix}_log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def _metrics(labels, probabilities, threshold, prefix):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        f"{prefix}_threshold": float(threshold), f"{prefix}_tn": int(tn), f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn), f"{prefix}_tp": int(tp),
        f"{prefix}_accuracy": float(accuracy_score(labels, predicted)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        f"{prefix}_precision": float(precision_score(labels, predicted, zero_division=0)),
        f"{prefix}_recall": float(recall_score(labels, predicted, zero_division=0)),
        f"{prefix}_f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def classification_metrics(labels, probabilities, prefix, threshold=.5):
    return _metrics(np.asarray(labels), np.asarray(probabilities), threshold, prefix)


def _prediction_frame(pipeline, seed, split, labels, probabilities, recording_ids, threshold, label_to_class, feature_set=None, blocks=True):
    frame = pd.DataFrame({"pipeline": pipeline, "seed": seed, "split": split, "session": recording_ids, "class_label": labels, "class_name": [label_to_class[int(value)] for value in labels], "probability_ad": probabilities})
    if feature_set is not None:
        frame.insert(1, "feature_set", feature_set)
    frame["predicted_label"] = (frame.probability_ad >= threshold).astype(int)
    frame["predicted_class"] = frame.predicted_label.map(label_to_class)
    if blocks:
        frame["block_index_within_session"] = frame.groupby("session").cumcount()
    return frame


def train_condition_seed(
    *, pipeline_name, seed, run_dir, bundle, model_config, train_config,
    n_spikes, label_to_class, feature_set=None, callbacks=(),
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if artifacts.run_is_complete(run_dir):
        return json.loads((run_dir / "metrics.json").read_text())
    used = bundle["used_sessions"].copy()
    if feature_set is not None:
        used.insert(1, "feature_set", feature_set)
    artifacts.write_sessions_used(used, run_dir)
    lo, hi, mean, standard_deviation = fit_scaler(bundle["X_train"])
    X_train = apply_scaler(bundle["X_train"], lo, hi, mean, standard_deviation)
    X_val = apply_scaler(bundle["X_val"], lo, hi, mean, standard_deviation)
    X_test = apply_scaler(bundle["X_test"], lo, hi, mean, standard_deviation)
    model, history, train_time = fit_lstm_model(
        X_train, bundle["y_train"], X_val, bundle["y_val"], bundle["train_w"],
        input_shape=(n_spikes, len(bundle["features"])), model_config=model_config,
        train_config=train_config, seed=seed, callbacks=callbacks,
    )
    train_p = model.predict(X_train, verbose=0).ravel()
    val_p = model.predict(X_val, verbose=0).ravel()
    test_p = model.predict(X_test, verbose=0).ravel()
    block_threshold = _threshold(bundle["y_val"], val_p)
    _, train_rec_p, train_rec_y, _ = _aggregate(
        bundle["train_rec_ids"], train_p, bundle["y_train"]
    )
    val_ids, val_rec_p, val_rec_y, val_counts = _aggregate(
        bundle["val_rec_ids"], val_p, bundle["y_val"]
    )
    test_ids, test_rec_p, test_rec_y, test_counts = _aggregate(
        bundle["test_rec_ids"], test_p, bundle["y_test"]
    )
    session_threshold = _threshold(
        np.concatenate([train_rec_y, val_rec_y]),
        np.concatenate([train_rec_p, val_rec_p]),
    )
    kwargs = {"label_to_class": label_to_class, "feature_set": feature_set}
    predictions = {
        "block_predictions_validation.csv": _prediction_frame(
            pipeline_name, seed, "val", bundle["y_val"], val_p,
            bundle["val_rec_ids"], block_threshold, **kwargs
        ),
        "block_predictions_test.csv": _prediction_frame(
            pipeline_name, seed, "test", bundle["y_test"], test_p,
            bundle["test_rec_ids"], block_threshold, **kwargs
        ),
        "session_predictions_validation.csv": _prediction_frame(
            pipeline_name, seed, "val", val_rec_y, val_rec_p, val_ids,
            session_threshold, blocks=False, **kwargs
        ).assign(n_blocks=val_counts, threshold=session_threshold),
        "session_predictions_test.csv": _prediction_frame(
            pipeline_name, seed, "test", test_rec_y, test_rec_p, test_ids,
            session_threshold, blocks=False, **kwargs
        ).assign(n_blocks=test_counts, threshold=session_threshold),
    }
    history_frame = pd.DataFrame(history.history).assign(pipeline=pipeline_name, seed=seed, epoch=lambda frame: np.arange(1, len(frame) + 1))
    if feature_set is not None:
        history_frame.insert(1, "feature_set", feature_set)
    history_frame.to_csv(run_dir / "history.csv", index=False)
    auc = lambda y, p: float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan
    metrics = {
        "pipeline": pipeline_name, "seed": seed, "architecture": "UNSORTED_ARCH",
        "features": list(bundle["features"]), "n_features": len(bundle["features"]),
        "train_time_s": round(train_time, 3), "epochs_run": len(history.history["loss"]),
        "best_val_auc": float(np.max(history.history["val_auc"])), "n_params": int(model.count_params()),
        "n_train_blocks": len(bundle["y_train"]), "n_val_blocks": len(bundle["y_val"]), "n_test_blocks": len(bundle["y_test"]),
        "n_train_sessions": int(used.split.eq("train").sum()), "n_val_sessions": len(val_ids), "n_test_sessions": len(test_ids),
        "n_test_ctrl_sessions": int((test_rec_y == 0).sum()), "n_test_ad_sessions": int((test_rec_y == 1).sum()),
        "block_auc": auc(bundle["y_test"], test_p), "block_average_precision": float(average_precision_score(bundle["y_test"], test_p)),
        "block_brier": float(brier_score_loss(bundle["y_test"], test_p)), "block_log_loss": float(log_loss(bundle["y_test"], test_p, labels=[0, 1])),
        "session_auc": auc(test_rec_y, test_rec_p), "session_average_precision": float(average_precision_score(test_rec_y, test_rec_p)),
        "session_brier": float(brier_score_loss(test_rec_y, test_rec_p)), "session_log_loss": float(log_loss(test_rec_y, test_rec_p, labels=[0, 1])),
        **_metrics(bundle["y_test"], test_p, block_threshold, "block_tuned"), **_metrics(bundle["y_test"], test_p, .5, "block_naive"),
        **_metrics(test_rec_y, test_rec_p, session_threshold, "session_tuned"), **_metrics(test_rec_y, test_rec_p, .5, "session_naive"),
    }
    if feature_set is not None:
        metrics["feature_set"] = feature_set
    artifacts.write_training_run(
        run_dir,
        model=model,
        scaler=(lo, hi, mean, standard_deviation),
        features=bundle["features"],
        predictions=predictions,
        history=history_frame,
        metrics=metrics,
        completion={
            "pipeline": pipeline_name,
            "seed": seed,
            "feature_set": feature_set,
            "completed_at_unix": time.time(),
        },
    )
    del model, X_train, X_val, X_test
    tf.keras.backend.clear_session()
    gc.collect()
    return metrics
