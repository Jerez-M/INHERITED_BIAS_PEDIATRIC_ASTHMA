# The Inherited Bias — pediatric asthma

Does scoring the same breath with a race-adjusted rather than a race-neutral
reference equation change **which** children with asthma get flagged — even
when race is never given to the model?

## The design in one paragraph

Every child is scored five different ways. The same children, the same model,
the same folds, the same seed, and every run is forced to flag the same number
of children. The only thing that changes is the three lung-function columns the
model is allowed to see. If one scoring then misses more Black children with
asthma than another, it is not because it flagged fewer children. It is because
it flagged different ones.

## The five equation types

| Equation type   | How the breath is scored                                              | What it is for                           |
| --------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| `raw`           | Litres straight off the spirometer. No equation.                      | The floor. Shows what the equation adds. |
| `legacy`        | Hankinson 1999, race-specific (via `pyspiro`)                         | published pair → **external validity**   |
| `neutral`       | GLI Global 2022, race-neutral (`*_GLOBAL` columns in the source file) | published pair → **external validity**   |
| `ownfit_race`   | our own equation, Hankinson's functional form, fitted per race        | own pair → **internal validity**         |
| `ownfit_norace` | the same equation, race left out                                      | own pair → **internal validity**         |

The published pair is what hospitals actually use, so it is the result that
matters for policy. But it compares two equations built by two different
research teams on two different populations, so a difference could come from
any of those choices. The own-fit pair removes that: same data, same
specification, same code, one difference — race in or race out. If both pairs
point the same way, that is the strongest version of the finding.

## Run order

    python scripts/00_download_and_merge.py --spirometry data/raw/NHANES_2007_2012_Only_Acceptable_Spirometry_Values.csv
    python scripts/01_audit.py
    python scripts/02_equations.py
    python scripts/03_reclassification.py
    python scripts/04_train.py --equation-type raw
    python scripts/04_train.py --equation-type legacy
    python scripts/04_train.py --equation-type neutral
    python scripts/04_train.py --equation-type ownfit_race
    python scripts/04_train.py --equation-type ownfit_norace
    python scripts/05_compare.py
    python scripts/06_simulate.py
    python scripts/07_construct.py

Or all of it at once:

    bash run_all.sh --n-repeats 3   # quick pass while you are still changing things
    bash run_all.sh                 # the full run

(`bash run_all.sh` rather than `./run_all.sh`, so it works the same on
Windows under Git Bash, where the execute bit does not survive.)

Each script reads from `data/processed/` and writes to `outputs/`. Re-running
any step is safe.

`04_train.py` is the slow one. The stacking model refits its three learners
inside its own folds, so a full run takes a few minutes per equation type on
a laptop. Use `--n-repeats 2` or `3` while you iterate and drop the flag for
the final numbers. If you only want the headline quickly:

    python scripts/04_train.py --equation-type legacy  --models stack
    python scripts/04_train.py --equation-type neutral --models stack

## What each script answers

| Script                   | Question                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `01_audit.py`            | Who is in the cohort, and are the z-scores already in the source file race-neutral? |
| `02_equations.py`        | What does each equation think of each child? Writes `analysis.csv`.                 |
| `03_reclassification.py` | Who crosses the abnormal line when you swap the equation? **No model involved.**    |
| `04_train.py`            | Trained on one scoring, who does the model miss?                                    |
| `05_compare.py`          | Is the gap bigger under one equation than the other, on the same children?          |
| `06_simulate.py`         | On invented children with identical lungs, does the equation alone produce a gap?   |
| `07_construct.py`        | Does the z-score mean the same thing in both groups?                                |

`03` is worth presenting on its own. It needs no model at all, so nobody can
say the result is an artefact of the machine learning.

### Which script answers which kind of validity

| Validity                                                     | Where it comes from                                                                                                                                                                                            |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Internal** — is the equation really what moved the result? | `06_simulate.py` (the equation is manipulable, race is not, so the simulation is the only place a true experiment is possible) and the `ownfit_race` / `ownfit_norace` pair, which holds everything else fixed |
| **External** — does it hold outside one modelling choice?    | the `legacy` / `neutral` pair, two equations built by different teams on different populations, plus the same result across four models                                                                        |
| **Construct** — do the numbers mean what we say they mean?   | `07_construct.py`: the z-score should track asthma attacks and ER visits (convergent), not BMI (discriminant), and should do so equally well in both groups                                                    |

## Structure

    inherited-bias/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── run_all.sh
    ├── .venv/                 virtualenv, never committed
    ├── config.py              every decision: paths, seed, folds, equation
    │                          types, flag rate, banned columns
    │
    ├── data/
    │   ├── raw/               spirometry CSV + 6 NHANES XPT files
    │   └── processed/
    │       ├── pediatric_cohort.csv    after merge + outcomes
    │       └── analysis.csv            + z-scores from every equation
    │
    ├── src/
    │   ├── data.py            load the cohort, tidy Sex / Height / race
    │   ├── equations.py       Hankinson 1999 + our own race-in / race-out pair
    │   ├── features.py        build X for one equation type (leakage guard here)
    │   ├── models.py          logistic / forest / boosting / stack
    │   ├── evaluate.py        repeated CV, matched flag rate, who gets missed
    │   └── simulate.py        the controlled experiment on invented children
    │
    ├── scripts/               00 … 06, run in order
    ├── notebooks/
    │   ├── explore.ipynb      scratch EDA; nothing downstream depends on it
    │   └── report.ipynb       figures, tables, and the nursing-facing lab
    │
    └── outputs/{figures,tables,models}/

## Setup

    cd inherited-bias
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Three rules

1. **Equation types differ in exactly three columns.** Same rows, same folds,
   same seed, same settings, same threshold rule. `config.EQUATION_TYPES` is
   the only place the difference is written down.
2. **Race is never a feature.** It is the moderator. `src/features.py` raises
   `LeakageError` if race, an identifier, a survey-design variable, the
   outcome, or any pre-computed flag reaches the feature matrix. The guard
   runs on every call and cannot be skipped.
3. **Every run flags the same share of children** (`config.FLAG_RATE`, 15%).
   This is what turns the comparison into a question about _who_ rather than
   _how many_. The threshold is recomputed inside every bootstrap resample so
   the property holds there too. 15% sits a little above the asthma rate in
   the cohort, which is roughly what a screening tool would do; if you change
   it, change it once in `config.py` and re-run everything.

## Things worth knowing before you read the numbers

**The age band starts at 8, not 6.** Hankinson 1999 only published pediatric
coefficients from age 8. Scoring the same children with every equation means
starting where the narrowest one starts. `01_audit.py` reports how many
children this costs.

**Two race groups have no Hankinson category.** Hankinson published three
(Caucasian, African-American, Mexican-American). NHANES has five. A clinic
scores "Other Hispanic" and "Other / multiracial" children with the Caucasian
equation by default, so that is what `config.HANKINSON_ETHNICITY` does — and
it is written down there rather than buried in code, because that default is
itself part of what we are studying.

**Below the lower limit of normal is not a diagnosis.** It means the child
measured lower than 95% of the healthy children the equation was built from.
`03_reclassification.py` says this in its docstring for the same reason.

**The own-fit equations are fitted on children who have never had asthma**,
which is how reference equations are built. Those children are still in the
model as negatives, so there is a small amount of shared information between
the equation and the training data. The published pair (`legacy`, `neutral`)
has no such overlap, because both equations were fitted somewhere else
entirely. That is one more reason to report both pairs.

## Where things go

`src/` holds functions; importing it does nothing. `scripts/` holds the things
you run, in order; each one reads from `data/processed/`, calls `src/`, and
writes to `outputs/`. If two things need the same code, it belongs in `src/`.

EDA happens twice on purpose: `scripts/01_audit.py` produces the numbers that
end up in the report, and `notebooks/explore.ipynb` is scratch. Any figure or
statistic that reaches a slide comes from the script.

## Later

A web front end for exploring the models and comparing the equation types is
planned but deliberately not started. Everything a web layer would need is
already a file in `outputs/tables/`, so it can be added without touching the
analysis.
