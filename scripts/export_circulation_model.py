"""Export the parameters of the pre-trained circulation (ROSC) model.

The model was trained with scikit-learn 1.5.1 and stored as pickles
(``circ-classification_model.joblib``: ``sklearn.svm.SVC``,
``circ-classification_scaler.joblib``: ``sklearn.preprocessing.RobustScaler``).
This script writes the fitted parameters as plain arrays to
``src/vitabel/assets/circulation_model.npz``, which
:mod:`vitabel.analysis.circulation.model` uses for prediction without
scikit-learn. It is kept to document how the file was produced; running it
needs the two pickles and scikit-learn / joblib, which are not dependencies
of vitabel::

    uv run --with scikit-learn==1.5.1 --with joblib \\
        python scripts/export_circulation_model.py MODEL.joblib SCALER.joblib
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

TARGET = Path(__file__).parents[1] / "src/vitabel/assets/circulation_model.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="circ-classification_model.joblib")
    parser.add_argument("scaler", type=Path, help="circ-classification_scaler.joblib")
    parser.add_argument("--output", type=Path, default=TARGET)
    args = parser.parse_args()

    svc = joblib.load(args.model)
    scaler = joblib.load(args.scaler)
    # binary RBF SVC with probability estimates; the prediction in vitabel
    # implements exactly this case
    assert type(svc).__name__ == "SVC" and svc.kernel == "rbf"
    assert len(svc.classes_) == 2 and svc.probability
    assert type(scaler).__name__ == "RobustScaler"

    np.savez_compressed(
        args.output,
        support_vectors=svc.support_vectors_,
        dual_coef=svc.dual_coef_[0],
        intercept=svc.intercept_[0],
        gamma=svc._gamma,
        prob_a=svc.probA_[0],
        prob_b=svc.probB_[0],
        classes=svc.classes_,
        scaler_center=scaler.center_,
        scaler_scale=scaler.scale_,
    )
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
