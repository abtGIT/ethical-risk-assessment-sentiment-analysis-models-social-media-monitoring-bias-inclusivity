
"""
final_thesis_utils.py
Shared utilities for the authoritative LJMU final experiment run.

Key corrections relative to the legacy utilities:
- reference group label is `reference-unmarked`, not `formal`
- sarcasm label is `sarcasm-indicated`, not gold sarcasm
- emoji count uses full emoji sequences via emoji.emoji_list
- correctness disparity is reported as CPD / CRR, not SPD / DIR
- model/feature selection is validation/CV-only before a single final test evaluation
- external datasets are evaluation-only and never used for source model selection
"""
from pathlib import Path
import json, pickle, re, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    precision_score, recall_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    average_precision_score
)
warnings.filterwarnings("ignore")

SEED = 42
LABELS = ["negative","neutral","positive"]
LABEL_TO_ID = {v:i for i,v in enumerate(LABELS)}
ID_TO_LABEL = {i:v for v,i in LABEL_TO_ID.items()}

# Existing implementation is read-only source material for the rebuild.
LEGACY_ROOT = Path("/content/drive/MyDrive/My_Data/upgrad-ljmu-thesis/LJMU Thesis/Implementation")
FINAL_ROOT = Path("/content/drive/MyDrive/LJMU_FINAL_EXPERIMENT_RUN")
FINAL = {
    "root": FINAL_ROOT,
    "data": FINAL_ROOT/"final_outputs"/"data",
    "models": FINAL_ROOT/"final_outputs"/"models",
    "predictions": FINAL_ROOT/"final_outputs"/"predictions",
    "results": FINAL_ROOT/"final_outputs"/"results",
    "figures": FINAL_ROOT/"final_outputs"/"figures",
    "manifests": FINAL_ROOT/"final_outputs"/"manifests",
}
LEGACY = {
    "data": LEGACY_ROOT/"data",
    "models": LEGACY_ROOT/"models",
    "predictions": LEGACY_ROOT/"predictions",
    "results": LEGACY_ROOT/"results",
}

def setup_final_dirs():
    for p in FINAL.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)

SARCASM_PATTERN = (
    r"#(?:sarcasm|sarcastic|irony|ironic|notreally|jk|justjoking|suuure|riiight)"
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
    r"|(?:not|never)\s+(?:great|amazing|awesome|fantastic|brilliant|wonderful|perfect|lovely)"
)
SLANG_TERMS = [
    "lol","lmao","omg","tbh","imo","imho","wtf","idk","ngl","smh","af","fr",
    "lowkey","highkey","slay","lit","goat","vibe","fire","wanna","gonna","gotta",
    "kinda","ur","bruh","bro","fam","dope","sick","sus","bussin","salty","cap",
    "bet","flex","ghost","clout","woke","ftw","ftl","epic","fail","rofl","ttyl",
    "brb","kk","np","nvm","thx","gr8","b4"
]
SLANG_RE = re.compile(r"(?<!\w)("+"|".join(sorted(map(re.escape,SLANG_TERMS),key=len,reverse=True))+r")(?!\w)", re.I)
SARCASM_RE = re.compile(SARCASM_PATTERN, re.I)
EMOJI_THRESHOLD = 0.05
SLANG_THRESHOLD = 0.10
REFERENCE_MIN_WORDS = 8
MIN_STABLE_N = 30
FOCAL_GROUPS = ["emoji-heavy","slang-heavy","sarcasm-indicated","reference-unmarked"]
REFERENCE_GROUP = "reference-unmarked"

FREEZE_CONFIG = {
    "emoji_heavy":{"formula":"emoji_count / word_count","operator":">","threshold":EMOJI_THRESHOLD},
    "slang_heavy":{"formula":"slang_count / word_count","operator":">","threshold":SLANG_THRESHOLD,"lexicon":SLANG_TERMS},
    "sarcasm_indicated":{"definition":"predefined surface-cue regex; not gold sarcasm","regex":SARCASM_PATTERN},
    "reference_unmarked":{"definition":"no focal flag AND word_count >= 8","minimum_word_count":REFERENCE_MIN_WORDS},
    "priority":["sarcasm-indicated","emoji-heavy","slang-heavy","reference-unmarked","other"],
    "minimum_n_for_stable_quantitative_interpretation":MIN_STABLE_N,
    "threshold_tuning_policy":"no threshold is changed after observing subgroup sizes",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
EMOTICON_RE = re.compile(r"[:=;]-?[\)\(DdPpOo\|/\\]|<3|\^_\^")

def clean_text(text, strip_emoticons=False):
    t = "" if text is None else str(text)
    if strip_emoticons:
        t = EMOTICON_RE.sub(" ",t)
    t = t.lower()
    t = URL_RE.sub(" ",t)
    t = MENTION_RE.sub(" ",t)
    t = re.sub(r"\s+"," ",t).strip()
    return t

def tag_subgroups(df, text_col="text"):
    import emoji
    out=df.copy()
    raw=out[text_col].fillna("").astype(str)
    out["word_count"]=raw.str.split().str.len().clip(lower=1)
    out["emoji_count"]=raw.map(lambda t: len(emoji.emoji_list(str(t))))
    out["emoji_density"]=out["emoji_count"]/out["word_count"]
    out["slang_count"]=raw.map(lambda t: len(SLANG_RE.findall(str(t).lower())))
    out["slang_ratio"]=out["slang_count"]/out["word_count"]
    out["sarcasm_indicator"]=raw.map(lambda t: bool(SARCASM_RE.search(str(t).lower())))
    out["punct_intensity"]=raw.str.count(r"[!?]")/out["word_count"]
    out["hashtag_count"]=raw.str.count(r"#\w+")
    out["negation_count"]=raw.str.lower().str.count(r"\b(?:not|no|never|cannot|can't|won't|doesn't|don't|didn't|isn't|aren't)\b")

    out["is_emoji_heavy"]=out["emoji_density"]>EMOJI_THRESHOLD
    out["is_slang_heavy"]=out["slang_ratio"]>SLANG_THRESHOLD
    out["is_sarcasm_indicated"]=out["sarcasm_indicator"].astype(bool)
    out["is_reference_candidate"]=(~out["is_emoji_heavy"] & ~out["is_slang_heavy"] &
                                   ~out["is_sarcasm_indicated"] & (out["word_count"]>=REFERENCE_MIN_WORDS))
    def assign(r):
        if r["is_sarcasm_indicated"]: return "sarcasm-indicated"
        if r["is_emoji_heavy"]: return "emoji-heavy"
        if r["is_slang_heavy"]: return "slang-heavy"
        if r["is_reference_candidate"]: return "reference-unmarked"
        return "other"
    out["subgroup"]=out.apply(assign,axis=1)
    return out

class SocialFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.sid_ = SentimentIntensityAnalyzer()
        return self
    def transform(self, X):
        import emoji
        rows=[]
        for txt in pd.Series(X).fillna("").astype(str):
            wc=max(1,len(txt.split()))
            emojis=len(emoji.emoji_list(txt))
            slang=len(SLANG_RE.findall(txt.lower()))
            punct=len(re.findall(r"[!?]",txt))
            hashtags=len(re.findall(r"#\w+",txt))
            neg=len(re.findall(r"\b(?:not|no|never|cannot|can't|won't|doesn't|don't|didn't|isn't|aren't)\b",txt.lower()))
            polarity=self.sid_.polarity_scores(txt)["compound"]
            rows.append([emojis/wc, slang/wc, punct/wc, hashtags/wc, neg/wc, polarity])
        return sp.csr_matrix(np.asarray(rows,dtype=float))
    def get_feature_names_out(self, input_features=None):
        return np.array(["emoji_density","slang_ratio","punct_intensity","hashtag_density","negation_density","vader_compound"])

def evaluate_predictions(y_true,y_pred,y_proba=None,label=""):
    out={
        "Model":label,
        "Accuracy":accuracy_score(y_true,y_pred),
        "Macro F1":f1_score(y_true,y_pred,average="macro",zero_division=0),
        "Weighted F1":f1_score(y_true,y_pred,average="weighted",zero_division=0),
        "Macro Precision":precision_score(y_true,y_pred,average="macro",zero_division=0),
        "Macro Recall":recall_score(y_true,y_pred,average="macro",zero_division=0),
        "MCC":matthews_corrcoef(y_true,y_pred),
        "Cohen Kappa":cohen_kappa_score(y_true,y_pred),
    }
    if y_proba is not None:
        y=np.asarray(y_true); p=np.asarray(y_proba)
        y_one=np.eye(p.shape[1])[y]
        out["Brier"]=float(np.mean((p-y_one)**2))
    return out

def prediction_frame(df,y_true,y_pred,y_proba,model_name,dataset):
    p=np.asarray(y_proba)
    out=pd.DataFrame({
        "row_id":df.get("row_id",pd.Series(np.arange(len(df)))).to_numpy(),
        "text":df["text"].astype(str).to_numpy(),
        "y_true":np.asarray(y_true,dtype=int),
        "y_pred":np.asarray(y_pred,dtype=int),
        "true_label":[ID_TO_LABEL[int(v)] for v in y_true],
        "pred_label":[ID_TO_LABEL[int(v)] for v in y_pred],
        "subgroup":df["subgroup"].astype(str).to_numpy(),
        "model":model_name,
        "dataset":dataset,
    })
    for i,name in ID_TO_LABEL.items():
        out[f"proba_{name}"]=p[:,i]
    out["confidence"]=p.max(axis=1)
    out["correct"]=(out["y_true"]==out["y_pred"]).astype(int)
    return out

def bootstrap_macro_f1(y_true,y_pred,n_boot=1000,seed=SEED):
    rng=np.random.default_rng(seed)
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); n=len(y_true)
    if n<2: return (np.nan,np.nan)
    vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n)
        vals.append(f1_score(y_true[idx],y_pred[idx],average="macro",zero_division=0))
    return tuple(np.quantile(vals,[0.025,0.975]))

def subgroup_performance(pred):
    rows=[]
    for sg,part in pred.groupby("subgroup",observed=True):
        lo,hi=bootstrap_macro_f1(part.y_true,part.y_pred,n_boot=500)
        rows.append({
            "Subgroup":sg,"N":len(part),
            "Accuracy":accuracy_score(part.y_true,part.y_pred),
            "Macro F1":f1_score(part.y_true,part.y_pred,average="macro",zero_division=0),
            "Macro F1 CI Low":lo,"Macro F1 CI High":hi,
            "Misclassification Rate":1-accuracy_score(part.y_true,part.y_pred),
            "Interpretation Status":"stable_quantitative" if len(part)>=MIN_STABLE_N else "exploratory_descriptive_only"
        })
    return pd.DataFrame(rows)

def correctness_parity(pred,subgroup,reference=REFERENCE_GROUP):
    part=pred[pred.subgroup.isin([subgroup,reference])]
    a=part[part.subgroup==subgroup].correct
    b=part[part.subgroup==reference].correct
    if len(a)==0 or len(b)==0: return {}
    ps=float(a.mean()); pr=float(b.mean())
    return {"Subgroup":subgroup,"N":len(a),"Reference N":len(b),
            "Subgroup Correctness Rate":ps,"Reference Correctness Rate":pr,
            "CPD":ps-pr,"CRR":ps/pr if pr>0 else np.nan}

def class_conditional_disparity(pred,subgroup,reference=REFERENCE_GROUP):
    rows=[]
    part=pred[pred.subgroup.isin([subgroup,reference])]
    for cls,name in ID_TO_LABEL.items():
        for metric in ("TPR","FPR"):
            vals={}
            for sg in (subgroup,reference):
                d=part[part.subgroup==sg]
                yt=(d.y_true.to_numpy()==cls).astype(int)
                yp=(d.y_pred.to_numpy()==cls).astype(int)
                cm=confusion_matrix(yt,yp,labels=[0,1])
                tn,fp,fn,tp=cm.ravel()
                vals[sg]=tp/(tp+fn) if metric=="TPR" and (tp+fn)>0 else (fp/(fp+tn) if metric=="FPR" and (fp+tn)>0 else np.nan)
            rows.append({"Subgroup":subgroup,"Class":name,"Metric":metric,
                         "Subgroup Rate":vals[subgroup],"Reference Rate":vals[reference],
                         "Gap":vals[subgroup]-vals[reference]})
    return pd.DataFrame(rows)

def expected_calibration_error(y_true,y_proba,n_bins=10):
    y_true=np.asarray(y_true); p=np.asarray(y_proba)
    conf=p.max(axis=1); pred=p.argmax(axis=1); corr=(pred==y_true).astype(float)
    bins=np.linspace(0,1,n_bins+1); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(conf>lo)&(conf<=hi) if lo>0 else (conf>=lo)&(conf<=hi)
        if mask.any():
            ece += mask.mean()*abs(corr[mask].mean()-conf[mask].mean())
    return float(ece)

def confidence_calibration_by_subgroup(pred,n_bins=10,threshold=0.70):
    rows=[]
    for sg,part in pred.groupby("subgroup",observed=True):
        errors=part[part.correct==0]
        rows.append({
            "Subgroup":sg,"N":len(part),"N Errors":len(errors),
            "Misclassification Rate":1-part.correct.mean(),
            "HCER":float((errors.confidence>threshold).mean()) if len(errors) else 0.0,
            "MCE":float(errors.confidence.mean()) if len(errors) else np.nan,
            "ECE":expected_calibration_error(part.y_true,part[[f"proba_{x}" for x in LABELS]].to_numpy(),n_bins=n_bins),
            "Mean Confidence":float(part.confidence.mean()),
            "Accuracy":float(part.correct.mean()),
            "Interpretation Status":"stable_quantitative" if len(part)>=MIN_STABLE_N else "exploratory_descriptive_only"
        })
    return pd.DataFrame(rows)

def save_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2,default=str)

def save_pickle(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"wb") as f: pickle.dump(obj,f)

def load_pickle(path):
    with open(path,"rb") as f: return pickle.load(f)