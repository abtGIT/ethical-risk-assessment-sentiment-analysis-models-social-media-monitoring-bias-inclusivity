"""
thesis_utils.py
================================================================
Shared helpers for all experiment notebooks.

Abhishek Tomar | PN1196973 | LJMU | April 2026
An Ethical Risk Assessment of Sentiment Analysis Models
in Social Media Monitoring

Import this at the top of every notebook:

    import sys
    sys.path.append('/content/drive/MyDrive/My_Data/upgrad-ljmu-thesis/LJMU Thesis/Implementation')
    from thesis_utils import *

WHY THIS FILE EXISTS
--------------------
Subgroup tagging, preprocessing and metric computation must be
identical across every experiment. If Notebook 6 tags sarcasm one
way and Notebook 8 tags it another, the fairness comparison is
meaningless. Defining everything once here removes that risk.

V2 FIXES (2026-08-09):
  * fairness_fairlearn EOD no longer compares the correctness vector
    against itself (which forced EOD to 0.0 everywhere). It is now
    computed on the real sentiment prediction, one-vs-rest per class,
    so it genuinely cross-checks AIF360 EOD.
  * fairness_aif360 Theil index computed in closed form (was silently
    erroring to NaN). Consistency is set to NaN by design, with a clear
    note, because it needs feature vectors this audit does not have.
"""

import os
import re
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    classification_report, brier_score_loss
)

warnings.filterwarnings("ignore")

# ==================================================================
# 1. PATHS
# ==================================================================
BASE = Path("/content/drive/MyDrive/My_Data/upgrad-ljmu-thesis/LJMU Thesis")

PATHS = {
    "eda":         BASE / "Dataset_Exploration" / "EDA",
    "data":        BASE / "Implementation" / "data",
    "models":      BASE / "Implementation" / "models",
    "predictions": BASE / "Implementation" / "predictions",
    "results":     BASE / "Implementation" / "results",
    "figures":     BASE / "Implementation" / "figures",
}

def setup_dirs():
    """Create the folder structure. Run once at the start of Notebook 5."""
    for name, path in PATHS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {name:<12} -> {path}")

SEED = 42


# ==================================================================
# 2. PREPROCESSING
# ==================================================================
# Emoticon pattern for Sentiment140. Labels in that dataset were derived
# FROM these emoticons, so leaving them in the text lets the model read
# the label instead of learning from words. Stripping is not optional.
EMOTICON_PATTERN = r"[:=;]-?[\)\(DdPpOo\|/\\]|<3|\^_\^"

URL_PATTERN     = r"https?://\S+|www\.\S+"
MENTION_PATTERN = r"@\w+"
REPEAT_PUNCT    = r"([!?.,])\1{2,}"


def clean_text(text, strip_emoticons=False):
    """
    Six-step preprocessing. Deliberately does NOT stem or lemmatise:
    'amazingly' and 'amazing' carry different sarcasm signals and
    collapsing them destroys a feature the study needs.
    """
    if not isinstance(text, str):
        return ""
    t = text
    if strip_emoticons:                       # Sentiment140 only
        t = re.sub(EMOTICON_PATTERN, " ", t)
    t = t.lower()
    t = re.sub(URL_PATTERN, " ", t)
    t = re.sub(MENTION_PATTERN, " ", t)
    t = re.sub(REPEAT_PUNCT, r"\1", t)        # !!!! -> !
    t = re.sub(r"\s+", " ", t).strip()
    return t


def preprocess_column(df, text_col="text", strip_emoticons=False):
    """Apply clean_text to a whole column. Returns a copy."""
    out = df.copy()
    out["text_clean"] = out[text_col].apply(
        lambda x: clean_text(x, strip_emoticons=strip_emoticons)
    )
    return out


# ==================================================================
# 3. LINGUISTIC SUBGROUP TAGGING
# ==================================================================
# These definitions answer the second marker's question directly:
# "the learner should discuss how sarcasm or linguistic subgroup
#  tagging is implemented"

SARCASM_PATTERN = (
    # explicit hashtags and keywords
    r"#(sarcasm|sarcastic|irony|ironic|notreally|jk|justjoking|suuure|riiight)"
    # colloquial irony phrases
    r"|(?:yeah[,.]?\s*right)"
    r"|(?:as\s*if)"
    r"|(?:oh\s*great)"
    r"|(?:just\s*what\s*i\s*needed)"
    r"|(?:wow[,.]?\s*thanks)"
    r"|(?:totally[,!]+)"
    r"|(?:sure[.]+)"
    r"|(?:not\s*at\s*all)"
    r"|(?:big\s*surprise)"
    r"|(?:great\s*job\s+ruining)"
    # negation before positive vocabulary
    r"|(?:not|never)\s+(?:great|amazing|awesome|fantastic|brilliant|"
    r"wonderful|perfect|lovely)"
)

SLANG_TERMS = [
    "lol", "lmao", "omg", "tbh", "imo", "imho", "wtf", "idk", "ngl", "smh",
    "af", "fr", "lowkey", "highkey", "slay", "lit", "goat", "vibe", "fire",
    "wanna", "gonna", "gotta", "kinda", "ur", "bruh", "bro", "fam", "dope",
    "sick", "sus", "bussin", "salty", "cap", "bet", "flex", "ghost", "clout",
    "woke", "ftw", "ftl", "epic", "fail", "rofl", "ttyl", "brb", "kk", "np",
    "nvm", "thx", "gr8", "b4",
]
SLANG_PATTERN = r"\b(" + "|".join(re.escape(t) for t in SLANG_TERMS) + r")\b"

# Thresholds. Fixed before results were seen and not adjusted afterwards.
EMOJI_DENSITY_THRESHOLD = 0.05   # ~1 emoji per 20 tokens
SLANG_RATIO_THRESHOLD   = 0.10   # ~1 slang token per 10 words
FORMAL_MIN_WORDS        = 8      # floor so very short posts aren't "formal"


def count_emojis(text):
    """Requires: pip install emoji"""
    import emoji as _emoji
    return sum(1 for ch in str(text) if ch in _emoji.EMOJI_DATA)


def tag_subgroups(df, text_col="text"):
    """
    Adds linguistic feature columns and a single `subgroup_primary` label.

    Priority when a post matches more than one rule:
        sarcasm > emoji-heavy > slang-heavy > formal > other

    IMPORTANT: subgroup labels are metadata for evaluation only.
    They are never passed to a model as a training feature.
    """
    out = df.copy()
    txt = out[text_col].astype(str)

    out["word_count"] = txt.str.split().str.len().replace(0, 1)

    # emoji
    out["emoji_count"]   = txt.apply(count_emojis)
    out["emoji_density"] = out["emoji_count"] / out["word_count"]
    out["has_emoji"]     = out["emoji_count"] > 0

    # slang
    out["slang_count"] = txt.str.lower().str.count(SLANG_PATTERN)
    out["slang_ratio"] = out["slang_count"] / out["word_count"]
    out["has_slang"]   = out["slang_count"] > 0

    # sarcasm
    out["sarcasm_indicator"] = txt.str.lower().str.contains(
        SARCASM_PATTERN, regex=True, na=False
    )

    # punctuation intensity (used as an engineered feature later)
    out["punct_count"]       = txt.str.count(r"[!?]")
    out["punct_intensity"]   = out["punct_count"] / out["word_count"]

    # boolean subgroup flags
    out["sg_sarcasm"] = out["sarcasm_indicator"]
    out["sg_emoji"]   = out["emoji_density"] > EMOJI_DENSITY_THRESHOLD
    out["sg_slang"]   = out["slang_ratio"]   > SLANG_RATIO_THRESHOLD
    out["sg_formal"]  = (
        ~out["sg_sarcasm"] & ~out["sg_emoji"] & ~out["sg_slang"]
        & (out["word_count"] >= FORMAL_MIN_WORDS)
    )
    out["sg_mixed"] = (
        out[["sg_sarcasm", "sg_emoji", "sg_slang"]].sum(axis=1) > 1
    )

    def _primary(r):
        if r["sg_sarcasm"]: return "sarcasm"
        if r["sg_emoji"]:   return "emoji-heavy"
        if r["sg_slang"]:   return "slang-heavy"
        if r["sg_formal"]:  return "formal"
        return "other"

    out["subgroup_primary"] = out.apply(_primary, axis=1)
    return out


SUBGROUPS = ["formal", "emoji-heavy", "slang-heavy", "sarcasm", "mixed"]
REFERENCE_SUBGROUP = "formal"   # privileged group in all fairness comparisons


# ==================================================================
# 4. SAVING AND LOADING PREDICTIONS  ***CRITICAL***
# ==================================================================
# Experiments 19, 20, 21 and 23 read these files instead of retraining.
# If predictions are not saved here, those experiments become expensive.

def save_predictions(exp_id, model_name, y_true, y_pred, y_proba,
                     subgroups, dataset="TweetEval", extra=None):
    """
    Save one model's test-set output so later experiments can reuse it.

    exp_id      : "exp01", "exp18a" etc.
    model_name  : "LogisticRegression"
    y_proba     : (n_samples, n_classes) probability array
    subgroups   : subgroup_primary values aligned with y_true
    """
    df = pd.DataFrame({
        "y_true":   np.asarray(y_true),
        "y_pred":   np.asarray(y_pred),
        "subgroup": np.asarray(subgroups),
    })

    proba = np.asarray(y_proba)
    for i in range(proba.shape[1]):
        df[f"proba_{i}"] = proba[:, i]

    df["confidence"] = proba.max(axis=1)
    df["correct"]    = (df["y_true"] == df["y_pred"]).astype(int)

    if extra:
        for k, v in extra.items():
            df[k] = v

    fname = f"{exp_id}_{model_name}_{dataset}.parquet".replace(" ", "")
    path  = PATHS["predictions"] / fname
    df.to_parquet(path, index=False)
    print(f"  saved predictions -> {fname}  ({len(df):,} rows)")
    return path


def load_predictions(exp_id, model_name, dataset="TweetEval"):
    """Read back a saved prediction file."""
    fname = f"{exp_id}_{model_name}_{dataset}.parquet".replace(" ", "")
    return pd.read_parquet(PATHS["predictions"] / fname)


def list_predictions():
    """Show everything saved so far."""
    files = sorted(PATHS["predictions"].glob("*.parquet"))
    for f in files:
        print(f"  {f.name}")
    return files


def save_model(model, exp_id, model_name):
    fname = f"{exp_id}_{model_name}.pkl".replace(" ", "")
    with open(PATHS["models"] / fname, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  saved model -> {fname}")


def load_model(exp_id, model_name):
    fname = f"{exp_id}_{model_name}.pkl".replace(" ", "")
    with open(PATHS["models"] / fname, "rb") as fh:
        return pickle.load(fh)


def save_result_table(df, table_name):
    """Save a result table as CSV, ready to paste into the V1 sheet."""
    path = PATHS["results"] / f"{table_name}.csv"
    df.to_csv(path, index=False)
    print(f"  saved table -> {table_name}.csv")
    return path


# ==================================================================
# 5. EVALUATION
# ==================================================================
def evaluate_model(y_true, y_pred, y_proba=None, label=""):
    """
    Standard metric block used in every experiment.

    Macro F1 is the headline metric, not accuracy. A model that
    predicts 'neutral' for everything scores 45.9% accuracy on
    TweetEval while learning nothing about positive or negative.
    """
    res = {
        "Model":       label,
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Precision":   precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall":      recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1":    f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC":         matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_proba is not None:
        res["Brier Score"] = multiclass_brier(y_true, y_proba)
    return res


def multiclass_brier(y_true, y_proba, classes=None):
    """
    Brier Score via one-vs-rest, averaged over classes.

    Needed because AIF360's probability-based metrics (DIR, AOD)
    assume calibrated inputs. Brier > 0.25 means fairness values
    should be qualified in the write-up.
    """
    y_true = np.asarray(y_true)
    proba  = np.asarray(y_proba)
    if classes is None:
        classes = np.unique(y_true)
    scores = []
    for i, c in enumerate(classes):
        if i >= proba.shape[1]:
            break
        binary = (y_true == c).astype(int)
        scores.append(np.mean((proba[:, i] - binary) ** 2))
    return float(np.mean(scores))


def subgroup_report(pred_df, subgroup_col="subgroup"):
    """
    Per-subgroup performance. Fills Table 4 of the V1 sheet.
    Takes a saved prediction dataframe.
    """
    rows = []
    for sg in pred_df[subgroup_col].unique():
        part = pred_df[pred_df[subgroup_col] == sg]
        if len(part) == 0:
            continue
        yt, yp = part["y_true"], part["y_pred"]
        cm = confusion_matrix(yt, yp)
        with np.errstate(divide="ignore", invalid="ignore"):
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tp = np.diag(cm)
            tn = cm.sum() - (fp + fn + tp)
            fpr = np.nanmean(fp / (fp + tn))
            fnr = np.nanmean(fn / (fn + tp))
        rows.append({
            "Subgroup":               sg,
            "Number of Samples":      len(part),
            "Accuracy":               accuracy_score(yt, yp),
            "Macro F1":               f1_score(yt, yp, average="macro", zero_division=0),
            "FPR":                    float(fpr),
            "FNR":                    float(fnr),
            "Misclassification Rate": 1 - accuracy_score(yt, yp),
        })
    return pd.DataFrame(rows).sort_values("Number of Samples", ascending=False)


def worst_group_accuracy(pred_df, subgroup_col="subgroup",
                         exclude=("other",), min_n=30):
    """
    Lowest accuracy across linguistic subgroups.
    Stops a high overall score from hiding total failure on one group.
    """
    accs = {}
    for sg in pred_df[subgroup_col].unique():
        if sg in exclude:
            continue
        part = pred_df[pred_df[subgroup_col] == sg]
        if len(part) < min_n:
            continue
        accs[sg] = accuracy_score(part["y_true"], part["y_pred"])
    if not accs:
        return np.nan, None
    worst = min(accs, key=accs.get)
    return accs[worst], worst


def subgroup_f1_variance(pred_df, subgroup_col="subgroup",
                         exclude=("other",), min_n=30):
    """Variance of Macro F1 across subgroups. High = inconsistent model."""
    f1s = []
    for sg in pred_df[subgroup_col].unique():
        if sg in exclude:
            continue
        part = pred_df[pred_df[subgroup_col] == sg]
        if len(part) < min_n:
            continue
        f1s.append(f1_score(part["y_true"], part["y_pred"],
                            average="macro", zero_division=0))
    return float(np.var(f1s)) if f1s else np.nan


# ==================================================================
# 6. CONFIDENCE-AWARE RISK  (Experiment 21 - Novelty 4)
# ==================================================================
HCER_CONFIDENCE_THRESHOLD = 0.70

def confidence_risk(pred_df, subgroup_col="subgroup"):
    """
    HCER - proportion of a subgroup's errors made at confidence > 0.70
    MCE  - mean confidence across that subgroup's wrong predictions

    Reads only saved prediction files. No retraining.
    """
    rows = []
    for sg in pred_df[subgroup_col].unique():
        part   = pred_df[pred_df[subgroup_col] == sg]
        errors = part[part["correct"] == 0]
        if len(errors) == 0:
            rows.append({
                "Subgroup": sg, "N Misclassified": 0,
                "Misclassification Rate": 0.0,
                "HCER": 0.0, "MCE": 0.0, "HCER > 0.30?": "No",
            })
            continue
        hcer = (errors["confidence"] > HCER_CONFIDENCE_THRESHOLD).mean()
        mce  = errors["confidence"].mean()
        rows.append({
            "Subgroup":               sg,
            "N Misclassified":        len(errors),
            "Misclassification Rate": len(errors) / len(part),
            "HCER":                   float(hcer),
            "MCE":                    float(mce),
            "HCER > 0.30?":           "Yes" if hcer > 0.30 else "No",
        })
    return pd.DataFrame(rows).sort_values("HCER", ascending=False)


# ==================================================================
# 7. FAIRNESS METRICS
# ==================================================================
def _binary_correctness_frame(pred_df, subgroup, reference=REFERENCE_SUBGROUP):
    """
    AIF360 and Fairlearn both expect a binary outcome and a binary
    protected attribute. The framing used throughout this study is:
        favourable outcome = model predicted correctly
        privileged group   = formal posts
        unprivileged group = the subgroup being compared
    """
    part = pred_df[pred_df["subgroup"].isin([reference, subgroup])].copy()
    part["protected"] = (part["subgroup"] == subgroup).astype(int)  # 1 = unprivileged
    part["outcome"]   = part["correct"]
    return part


def fairness_manual(pred_df, subgroup, reference=REFERENCE_SUBGROUP):
    """
    Fallback computation. Runs even if AIF360 is not installed,
    and doubles as a sanity check against the toolkit outputs.
    """
    part = _binary_correctness_frame(pred_df, subgroup, reference)
    if part.empty:
        return {}
    priv   = part[part["protected"] == 0]["outcome"]
    unpriv = part[part["protected"] == 1]["outcome"]
    if len(priv) == 0 or len(unpriv) == 0:
        return {}
    p_priv, p_unpriv = priv.mean(), unpriv.mean()
    return {
        "Subgroup":  subgroup,
        "N":         len(unpriv),
        "SPD":       float(p_unpriv - p_priv),
        "DIR":       float(p_unpriv / p_priv) if p_priv > 0 else np.nan,
        "Priv Acc":  float(p_priv),
        "Unpriv Acc": float(p_unpriv),
    }


def fairness_aif360(pred_df, subgroup, reference=REFERENCE_SUBGROUP):
    """
    AIF360 metrics: SPD, DIR, Theil Index, Consistency.
    Requires: pip install aif360
    """
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import BinaryLabelDatasetMetric
    except ImportError:
        print("  aif360 not installed - falling back to manual computation")
        return fairness_manual(pred_df, subgroup, reference)

    part = _binary_correctness_frame(pred_df, subgroup, reference)
    if part.empty:
        return {}

    df = part[["outcome", "protected"]].copy()
    bld = BinaryLabelDataset(
        df=df,
        label_names=["outcome"],
        protected_attribute_names=["protected"],
        favorable_label=1,
        unfavorable_label=0,
    )
    m = BinaryLabelDatasetMetric(
        bld,
        privileged_groups=[{"protected": 0}],
        unprivileged_groups=[{"protected": 1}],
    )
    out = {
        "Subgroup":    subgroup,
        "N":           int((part["protected"] == 1).sum()),
        "AIF360 SPD":  float(m.statistical_parity_difference()),
        "AIF360 DIR":  float(m.disparate_impact()),
    }

    # Theil index (generalized entropy at alpha=1) computed directly from the
    # benefit vector b_i = 2*outcome (correct) so mean(b)=... . AIF360's own
    # generalized_entropy_index errors on this binary correctness frame in the
    # installed version, so it is computed here in closed form instead. This
    # always returns a real number and matches AIF360's definition:
    #   GE(1) = (1/n) * sum( (b_i/mu) * ln(b_i/mu) )
    # with the convention 0*ln(0) = 0.
    try:
        b = part["outcome"].to_numpy(dtype=float)
        # standard AIF360 benefit mapping: b_i = 1 - y_pred_i + y_true_i,
        # but for a correctness outcome the informative quantity is the
        # correctness rate itself, so b_i = outcome_i works directly.
        mu = b.mean()
        if mu > 0:
            ratio = b / mu
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(ratio > 0, ratio * np.log(ratio), 0.0)
            out["Theil Index"] = float(terms.mean())
        else:
            out["Theil Index"] = np.nan
    except Exception:
        out["Theil Index"] = np.nan

    # Consistency is a k-NN individual-fairness metric that needs feature
    # vectors. This audit runs on saved predictions with no feature matrix,
    # so consistency is not computable here and is reported as NaN by design
    # rather than silently swallowing an error. Individual-level inequality
    # is instead captured by the Theil index above.
    out["Consistency Score"] = np.nan
    return out


def fairness_classification_aif360(pred_df, subgroup,
                                   reference=REFERENCE_SUBGROUP,
                                   positive_class=None):
    """
    EOD and AOD need a ClassificationMetric, which compares true
    labels against predictions. Computed one-vs-rest per class,
    because EOD is only defined for binary outcomes.
    """
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import ClassificationMetric
    except ImportError:
        return {}

    part = pred_df[pred_df["subgroup"].isin([reference, subgroup])].copy()
    if part.empty:
        return {}
    part["protected"] = (part["subgroup"] == subgroup).astype(int)

    classes = sorted(part["y_true"].unique())
    targets = [positive_class] if positive_class is not None else classes

    eods, aods = [], []
    for cls in targets:
        t = part.copy()
        t["label_true"] = (t["y_true"] == cls).astype(int)
        t["label_pred"] = (t["y_pred"] == cls).astype(int)

        d_true = BinaryLabelDataset(
            df=t[["label_true", "protected"]].rename(columns={"label_true": "label"}),
            label_names=["label"], protected_attribute_names=["protected"],
            favorable_label=1, unfavorable_label=0)
        d_pred = BinaryLabelDataset(
            df=t[["label_pred", "protected"]].rename(columns={"label_pred": "label"}),
            label_names=["label"], protected_attribute_names=["protected"],
            favorable_label=1, unfavorable_label=0)

        cm = ClassificationMetric(
            d_true, d_pred,
            privileged_groups=[{"protected": 0}],
            unprivileged_groups=[{"protected": 1}])
        try:
            eods.append(cm.equal_opportunity_difference())
            aods.append(cm.average_odds_difference())
        except Exception:
            pass

    return {
        "Subgroup":   subgroup,
        "AIF360 EOD": float(np.nanmean(eods)) if eods else np.nan,
        "AIF360 AOD": float(np.nanmean(aods)) if aods else np.nan,
    }


def fairness_fairlearn(pred_df, subgroup, reference=REFERENCE_SUBGROUP,
                       positive_class=None):
    """
    Fairlearn equivalents. Cross-checks the AIF360 numbers.
    Requires: pip install fairlearn

    Two metrics are reported:

      DPD - demographic parity difference on the CORRECTNESS outcome
            (favourable = model got it right). This matches AIF360 SPD,
            which is also computed on correctness, so the two are directly
            comparable and 'Frameworks Agree?' is meaningful.

      EOD - equalized odds difference on the ACTUAL sentiment prediction,
            computed one-vs-rest per class and averaged. This mirrors
            fairness_classification_aif360 exactly: real y_true and y_pred,
            subgroup as the sensitive feature.

    NOTE: the previous version computed EOD by passing the correctness
    vector as BOTH y_true and y_pred, which forced EOD to 0.0 for every
    subgroup. That was a bug, not a finding. EOD is now computed against
    the true sentiment labels so it genuinely cross-checks AIF360 EOD.
    """
    try:
        from fairlearn.metrics import (
            demographic_parity_difference, equalized_odds_difference
        )
    except ImportError:
        print("  fairlearn not installed - skipping cross-check")
        return {}

    part = _binary_correctness_frame(pred_df, subgroup, reference)
    if part.empty:
        return {}

    # --- DPD on correctness (comparable to AIF360 SPD) ---
    try:
        dpd = demographic_parity_difference(
            part["outcome"],                    # y_true (correctness)
            part["outcome"],                    # y_pred (correctness) - parity of the
                                                # favourable-outcome rate by group
            sensitive_features=part["protected"])
    except Exception:
        dpd = np.nan

    # --- EOD on the real sentiment prediction, one-vs-rest per class ---
    # equalized_odds_difference needs true labels vs predicted labels, so
    # it must see y_true and y_pred, NOT the correctness vector. Averaging
    # over classes matches the AIF360 ClassificationMetric computation.
    classes = sorted(part["y_true"].unique())
    targets = [positive_class] if positive_class is not None else classes
    eods = []
    for cls in targets:
        yt = (part["y_true"] == cls).astype(int)
        yp = (part["y_pred"] == cls).astype(int)
        # a class present in only one group, or absent entirely, yields an
        # undefined rate; skip it rather than poisoning the mean with a nan
        try:
            e = equalized_odds_difference(
                yt, yp, sensitive_features=part["protected"])
            if not np.isnan(e):
                eods.append(e)
        except Exception:
            continue
    eod = float(np.mean(eods)) if eods else np.nan

    return {
        "Subgroup":      subgroup,
        "Fairlearn DPD": float(dpd),
        "Fairlearn EOD": float(eod),
    }


def full_fairness_audit(pred_df, reference=REFERENCE_SUBGROUP,
                        subgroups=("emoji-heavy", "slang-heavy", "sarcasm", "mixed")):
    """
    Runs both toolkits over every subgroup comparison and merges the
    results. Fills Table 9. This is Experiment 20 / Novelty 6.

    The 'Frameworks Agree?' column is the point of the exercise:
    where two independently written implementations agree, the
    finding is robust. Where they differ, that is reportable.
    """
    rows = []
    present = set(pred_df["subgroup"].unique())
    for sg in subgroups:
        if sg not in present:
            continue
        rec = {}
        rec.update(fairness_aif360(pred_df, sg, reference))
        rec.update(fairness_classification_aif360(pred_df, sg, reference))
        rec.update(fairness_fairlearn(pred_df, sg, reference))

        spd = rec.get("AIF360 SPD", np.nan)
        dpd = rec.get("Fairlearn DPD", np.nan)
        if not (np.isnan(spd) or np.isnan(dpd)):
            rec["Frameworks Agree?"] = "Yes" if abs(abs(spd) - abs(dpd)) < 0.02 else "No"
        else:
            rec["Frameworks Agree?"] = "N/A"

        aod = rec.get("AIF360 AOD", np.nan)
        dir_ = rec.get("AIF360 DIR", np.nan)
        flags = []
        if not np.isnan(aod)  and abs(aod) > 0.10:          flags.append("AOD")
        if not np.isnan(dir_) and not (0.80 <= dir_ <= 1.25): flags.append("DIR")
        rec["Passes Threshold?"] = "No - " + ", ".join(flags) if flags else "Yes"

        rows.append(rec)
    return pd.DataFrame(rows)


# =================================================================
# 8. FAIRNESS DRIFT  (Experiment 22 - Novelty 5)
# ==================================================================
DRIFT_THRESHOLD = 0.10

def fairness_drift(source_audit, target_audit,
                   source_name="TweetEval", target_name="SemEval-2014"):
    """
    Absolute change in each fairness metric between two datasets,
    for the same model and the same subgroup comparison.

    Low drift  -> fairness is intrinsic to the model
    High drift -> fairness is domain-dependent, so the ethical risk
                  assessment only holds for the dataset it was measured on
    """
    rows = []
    for _, s in source_audit.iterrows():
        t = target_audit[target_audit["Subgroup"] == s["Subgroup"]]
        if t.empty:
            continue
        t = t.iloc[0]
        aod_drift = abs(s.get("AIF360 AOD", np.nan) - t.get("AIF360 AOD", np.nan))
        dir_drift = abs(s.get("AIF360 DIR", np.nan) - t.get("AIF360 DIR", np.nan))
        rows.append({
            "Train Dataset":  source_name,
            "Test Dataset":   target_name,
            "Subgroup":       s["Subgroup"],
            "Source AOD":     s.get("AIF360 AOD", np.nan),
            "Target AOD":     t.get("AIF360 AOD", np.nan),
            "Fairness Drift": aod_drift,
            "Source DIR":     s.get("AIF360 DIR", np.nan),
            "Target DIR":     t.get("AIF360 DIR", np.nan),
            "DIR Drift":      dir_drift,
            "Drift > 0.10?":  "Yes" if aod_drift > DRIFT_THRESHOLD else "No",
            "Fairness-Stable or Domain-Sensitive?":
                "Domain-sensitive" if aod_drift > DRIFT_THRESHOLD else "Fairness-stable",
        })
    return pd.DataFrame(rows)


# =================================================================
# 9. BOOTSTRAP CONFIDENCE INTERVALS
# ==================================================================
def bootstrap_ci(y_true, y_pred, metric_fn=None, n_boot=1000,
                 alpha=0.05, seed=SEED):
    """
    Needed because the sarcasm subgroup has only 14 posts in the TweetEval
    test partition. A point estimate on that sample is not defensible on
    its own; the bootstrap CI makes the uncertainty visible.
    """
    if metric_fn is None:
        metric_fn = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    n = len(yt)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            scores.append(metric_fn(yt[idx], yp[idx]))
        except Exception:
            continue
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return float(np.mean(scores)), lo, hi


def mcnemar_test(pred_df_a, pred_df_b):
    """
    Paired comparison of two models on the same test instances.
    Used when comparing LR against Linear SVM in Experiment 19.
    """
    from scipy.stats import chi2
    a = pred_df_a["correct"].values
    b = pred_df_b["correct"].values
    n01 = int(((a == 0) & (b == 1)).sum())   # A wrong, B right
    n10 = int(((a == 1) & (b == 0)).sum())   # A right, B wrong
    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)   # continuity corrected
    p    = 1 - chi2.cdf(stat, df=1)
    return {"statistic": float(stat), "p_value": float(p),
            "n01": n01, "n10": n10}


BONFERRONI_ALPHA = 0.05 / 4   # four subgroup comparisons per model


# =================================================================
# 10. ETHICAL RISK INDEX
# ==================================================================
def score_performance_risk(wga, f1_variance):
    if wga > 0.65 and f1_variance < 0.02: return 1
    if wga < 0.50 or f1_variance > 0.05:  return 3
    return 2

def score_fairness_risk(mean_aod):
    a = abs(mean_aod)
    if a < 0.05: return 1
    if a > 0.10: return 3
    return 2

def score_misclassification_risk(max_hcer, type3_dominant=False):
    if max_hcer > 0.30 or type3_dominant: return 3
    if max_hcer < 0.20:                   return 1
    return 2

def score_explainability_risk(n_models_agreeing):
    if n_models_agreeing >= 3: return 1
    if n_models_agreeing <= 1: return 3
    return 2

def score_generalisation_risk(mean_drift):
    if mean_drift < 0.05: return 1
    if mean_drift > 0.10: return 3
    return 2

def ethical_risk_index(perf, fair, misc, expl, gen):
    total = perf + fair + misc + expl + gen
    level = "Low" if total <= 7 else ("Medium" if total <= 11 else "High")
    return {
        "Performance Risk":       perf,
        "Fairness Risk":          fair,
        "Misclassification Risk": misc,
        "Explainability Risk":    expl,
        "Robustness Risk":        gen,
        "Total Risk Score":       total,
        "Final Ethical Risk Level": level,
    }


# ==================================================================
print("thesis_utils loaded. [V2 FIXED: Fairlearn EOD, Theil index]")
print(f"  seed = {SEED}")
print(f"  subgroup thresholds: emoji > {EMOJI_DENSITY_THRESHOLD}, "
      f"slang > {SLANG_RATIO_THRESHOLD}, formal min words = {FORMAL_MIN_WORDS}")
print(f"  reference subgroup  = {REFERENCE_SUBGROUP}")
