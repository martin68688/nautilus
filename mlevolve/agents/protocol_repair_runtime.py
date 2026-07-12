"""Small dependency-free runtime guard for staged protocol repairs."""

from __future__ import annotations

import hashlib
import json


def _ids(values):
    if values is None:
        return set()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return {str(value) for value in values}


def _digest(values):
    payload = "\n".join(sorted(_ids(values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ProtocolProvenanceGuard:
    """Record and validate data scopes without retaining sensitive row values."""

    schema = "mlevolve_protocol_provenance_v1"

    def __init__(self):
        self.partitions = {}
        self.fits = []
        self.predictions = []
        self.selections = []
        self.final_evaluations = []
        self.violations = []
        self.frozen = False

    def register_partition(self, name, sample_ids):
        ids = _ids(sample_ids)
        self.partitions[str(name)] = ids
        outer_train = self.partitions.get("outer_train", set())
        outer_holdout = self.partitions.get("outer_holdout", set())
        if outer_train and outer_holdout and outer_train & outer_holdout:
            self.violations.append("outer_train overlaps outer_holdout")

    def record_fit(self, component, fit_ids, *, purpose="training"):
        ids = _ids(fit_ids)
        holdout = self.partitions.get("outer_holdout", set())
        if ids & holdout:
            self.violations.append(f"{component} fit scope includes outer_holdout")
        self.fits.append({"component": str(component), "purpose": str(purpose), "ids": _digest(ids)})

    def record_prediction(self, component, train_ids, predict_ids, *, purpose):
        train, predict = _ids(train_ids), _ids(predict_ids)
        if purpose == "oof" and train & predict:
            self.violations.append(f"{component} OOF prediction rows overlap training rows")
        self.predictions.append({
            "component": str(component), "purpose": str(purpose),
            "train_ids": _digest(train), "predict_ids": _digest(predict),
        })

    def record_selection(self, kind, selection_ids):
        ids = _ids(selection_ids)
        holdout = self.partitions.get("outer_holdout", set())
        if ids & holdout:
            self.violations.append(f"{kind} selection scope includes outer_holdout")
        if self.frozen:
            self.violations.append(f"{kind} selection occurred after protocol freeze")
        self.selections.append({"kind": str(kind), "ids": _digest(ids)})

    def freeze(self):
        if not self.selections:
            self.violations.append("protocol frozen before recording model/ensemble selection")
        self.frozen = True

    def record_final_evaluation(self, sample_ids):
        ids = _ids(sample_ids)
        holdout = self.partitions.get("outer_holdout", set())
        if not self.frozen:
            self.violations.append("final evaluation occurred before protocol freeze")
        if not holdout or ids != holdout:
            self.violations.append("final evaluation scope is not exactly outer_holdout")
        if self.final_evaluations:
            self.violations.append("outer_holdout evaluated more than once")
        self.final_evaluations.append(_digest(ids))

    def assert_clean(self):
        required = ("outer_train", "outer_holdout")
        for name in required:
            if not self.partitions.get(name):
                self.violations.append(f"required partition missing: {name}")
        if not self.fits:
            self.violations.append("no fit scopes recorded")
        if not self.predictions:
            self.violations.append("no prediction scopes recorded")
        if not self.final_evaluations:
            self.violations.append("final evaluation was not recorded")
        if self.violations:
            raise RuntimeError("Protocol provenance failed: " + "; ".join(dict.fromkeys(self.violations)))

    def emit(self):
        payload = {
            "schema": self.schema,
            "status": "clean" if not self.violations else "blocked",
            "violations": list(dict.fromkeys(self.violations)),
            "counts": {
                "partitions": len(self.partitions),
                "fits": len(self.fits),
                "predictions": len(self.predictions),
                "selections": len(self.selections),
                "final_evaluations": len(self.final_evaluations),
            },
        }
        print("MLEVOLVE_PROTOCOL_PROVENANCE=" + json.dumps(payload, sort_keys=True))
        return payload
