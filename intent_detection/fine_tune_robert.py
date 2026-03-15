"""
Deprecated — use ``intent_detection/train.py`` instead.

    python intent_detection/train.py --model roberta

This file is kept for backwards compatibility only.  Running it will emit a
``DeprecationWarning`` and delegate to the unified training script.
"""
import os
import subprocess
import sys
import warnings

MODEL_KEY = "roberta"

if __name__ == "__main__":
    warnings.warn(
        f"fine_tune_robert.py is deprecated. "
        f"Use: python intent_detection/train.py --model {MODEL_KEY}",
        DeprecationWarning,
        stacklevel=1,
    )
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    sys.exit(
        subprocess.call(
            [sys.executable, script, "--model", MODEL_KEY] + sys.argv[1:]
        )
    )
