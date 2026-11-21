"""Every decision in the project lives here. Nothing is hard-coded elsewhere."""
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIGURES, TABLES, MODELS = OUT / "figures", OUT / "tables", OUT / "models"

COHORT_CSV = PROCESSED / "pediatric_cohort.csv"
ANALYSIS_CSV = PROCESSED / "analysis.csv"


def ensure_dirs():
    """Make the folders the scripts write into. Safe to call every time."""
    for d in (RAW, PROCESSED, FIGURES, TABLES, MODELS):
        d.mkdir(parents=True, exist_ok=True)


# ---- study design -----------------------------------------------------------
SEED = 42
N_SPLITS, N_REPEATS = 5, 10          # repeated stratified CV
N_BOOTSTRAP = 2000

# Hankinson 1999 only covers children from age 8. Scoring the same children
# with every equation means starting at 8, not 6. 01_audit.py reports the loss.
AGE_MIN, AGE_MAX = 8, 17

OUTCOME = "asthma_current"           # primary; "asthma_ever" is the secondary
PRIMARY_GROUPS = ["Black", "White"]  # the contrast the hypothesis is about

# Every equation type is forced to flag this share of children, so the
# comparison is about WHO gets flagged, not HOW MANY.
FLAG_RATE = 0.15

# ---- the experiment: equation types differ ONLY in these three columns ------
COMMON_FEATURES = ["Age", "Sex", "Height", "BMI"]

EQUATION_TYPES = {
    # no equation at all - litres straight off the spirometer
    "raw":     ["Baseline_FEV1_L", "Baseline_FVC_L", "Baseline_FEV1_FVC_Ratio"],

    # published pair -> external validity
    "legacy":  ["FEV1_z_hank", "FVC_z_hank", "Ratio_z_hank"],
    "neutral": ["FEV1_Zscores_GLOBAL", "FVC_Zscores_GLOBAL",
                "FEV1_FVC_ratio_GLOBAL_Zscores"],

    # our own pair, same specification, race in vs race out -> internal validity
    "ownfit_race":   ["FEV1_z_own_race", "FVC_z_own_race", "Ratio_z_own_race"],
    "ownfit_norace": ["FEV1_z_own_norace", "FVC_z_own_norace",
                      "Ratio_z_own_norace"],
}

# the two comparisons the hypothesis actually rests on
CONTRASTS = [("legacy", "neutral"), ("ownfit_race", "ownfit_norace")]

# ---- how race maps onto the Hankinson 1999 categories -----------------------
# Hankinson published three groups. NHANES has five. "Other Hispanic" and
# "Other / multiracial" children have no category of their own, so a clinic
# scores them with the Caucasian equation by default. That default is part of
# what we are studying, so it is written down here rather than hidden in code.
HANKINSON_ETHNICITY = {
    "White": 1,               # Caucasian
    "Black": 2,               # African-American
    "Mexican American": 3,    # Mexican-American
    "Other Hispanic": 1,      # no category -> default
    "Other / multiracial": 1, # no category -> default
}
HANKINSON_DEFAULTED = ["Other Hispanic", "Other / multiracial"]

# ---- nothing matching these may ever enter the feature matrix ---------------
# race (the moderator), identifiers, survey design, the outcome and anything
# derived from it, and every pre-computed flag in the source file.
BANNED_EXACT = {
    "SEQN", "Race", "race", "RIDRETH1", "RIAGENDR", "RIDAGEYR",
    "Height2_x_Race", "WTMEC2YR", "wtmec6yr", "SDMVSTRA", "SDMVPSU", "SDDSRVYR",
}
BANNED_SUBSTRINGS = (
    "asthma", "MCQ",                                   # outcome and its source
    "LLN", "ULN", "Normal_", "Obstruction", "PRISm",   # pre-computed flags
    "Restrictive", "Mixed_", "OneVariable",
    "flag", "pred_", "_Quality",
)
