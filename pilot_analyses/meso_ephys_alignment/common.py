"""Settings shared by the meso_ephys pilot (fixed in HYPOTHESIS.md)."""
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

BWM_SESSIONS = ["5c0c560e-9e1f-45e9-b66e-e4ee7855be84",   # NR_0028
                "6274dda8-3a59-4aa1-95f8-a8a549c46a26",   # CSH_ZAD_025
                "2bdf206a-820f-402f-920a-9e86cd5388a4"]   # UCLA037
MESO_SESSIONS = ["90029e53-b014-4bb1-98e6-15878e8a25ad",  # SP054
                 "fa10ff03-be1f-4b67-9e1e-87bf6979c967",  # SP075
                 "a4afea1e-ad72-433d-9498-95ddc54252fe"]  # SP076
SUBJECT = {"5c0c560e": "NR_0028", "6274dda8": "CSH_ZAD_025", "2bdf206a": "UCLA037",
           "90029e53": "SP054", "fa10ff03": "SP075", "a4afea1e": "SP076"}

GREY_BIN = 1.
N_NEURONS = 30
N_DRAWS = 20
K = 3
K_MAX = 10
N_SHIFTS = 500
VARIABLES = ("stim", "choice", "feedback", "prior")
# (event, start, stop) per variable; same order as VARIABLES
WINDOWS = {
    "bwm": {"stim": ("stimOn_times", 0.0, 0.1), "choice": ("firstMovement_times", -0.1, 0.0),
            "feedback": ("feedback_times", 0.0, 0.2), "prior": ("stimOn_times", -0.4, -0.1)},
    "meso": {"stim": ("stimOn_times", 0.0, 0.3), "choice": ("firstMovement_times", -0.3, 0.0),
             "feedback": ("feedback_times", 0.0, 0.3), "prior": ("stimOn_times", -0.4, -0.1)},
}

SENSORY_EXACT = {"SSs", "LGd", "LP", "PO", "VPM", "VPL", "VPLpc", "MGv", "MGd", "SCs"}
MOTOR = {"MOp", "MOs", "CP", "ACB", "GPe", "SNr", "SCm", "MRN", "GRN", "IRN", "PRNr", "PRNc",
         "ZI", "VAL", "VM", "PAG"}


def region_class(area):
    a = str(area)
    if (a.startswith(("VIS", "AUD", "SSp-")) and a != "VISC") or a in SENSORY_EXACT:
        return "sensory"
    if a in MOTOR:
        return "motor"
    return "other"


def meso_area(label):
    """Allen area without its layer suffix (VISp2/3 -> VISp)."""
    return re.sub(r"\d.*$", "", str(label))


def beryl(acronyms):
    from iblatlas.regions import BrainRegions
    return BrainRegions().acronym2acronym(np.asarray(acronyms), mapping="Beryl")
