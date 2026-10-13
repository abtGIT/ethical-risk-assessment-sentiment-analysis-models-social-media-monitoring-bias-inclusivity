import json, hashlib, re, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

SEED=42
LABELS=["negative","neutral","positive"]
SLANG_TERMS=[
"lol","lmao","omg","tbh","imo","imho","wtf","idk","ngl","smh","af","fr",
"lowkey","highkey","slay","lit","goat","vibe","fire","wanna","gonna","gotta",
"kinda","ur","bruh","bro","fam","dope","sick","sus","bussin","salty","cap",
"bet","flex","ghost","clout","woke","ftw","ftl","epic","fail","rofl","ttyl",
"brb","kk","np","nvm","thx","gr8","b4"]
SLANG_RE=re.compile(r"(?<!\w)("+"|".join(sorted(map(re.escape,SLANG_TERMS),key=len,reverse=True))+r")(?!\w)",re.I)
SARCASM_PATTERN=(
r"#(?:sarcasm|sarcastic|irony|ironic|notreally|jk|justjoking|suuure|riiight)"
r"|(?:yeah[,.]?\s*right)|(?:as\s*if)|(?:oh\s*great)|(?:just\s*what\s*i\s*needed)"
r"|(?:wow[,.]?\s*thanks)|(?:totally[,!]+)|(?:sure[.]+)|(?:not\s*at\s*all)"
r"|(?:big\s*surprise)|(?:great\s*job\s+ruining)"
r"|(?:not|never)\s+(?:great|amazing|awesome|fantastic|brilliant|wonderful|perfect|lovely)")
SARCASM_RE=re.compile(SARCASM_PATTERN,re.I)
URL_RE=re.compile(r"https?://\S+|www\.\S+",re.I)
MENTION_RE=re.compile(r"@\w+")

def clean_text(text):
    t="" if text is None else str(text)
    t=t.lower(); t=URL_RE.sub(" ",t); t=MENTION_RE.sub(" ",t)
    return re.sub(r"\s+"," ",t).strip()

class SocialFeatureTransformer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.sid_=SentimentIntensityAnalyzer(); return self
    def transform(self,X):
        rows=[]
        for txt in pd.Series(X).fillna("").astype(str):
            wc=max(1,len(txt.split()))
            rows.append([
                len(emoji.emoji_list(txt))/wc,
                len(SLANG_RE.findall(txt.lower()))/wc,
                len(re.findall(r"[!?]",txt))/wc,
                len(re.findall(r"#\w+",txt))/wc,
                len(re.findall(r"\b(?:not|no|never|cannot|can't|won't|doesn't|don't|didn't|isn't|aren't)\b",txt.lower()))/wc,
                self.sid_.polarity_scores(txt)["compound"]
            ])
        return sp.csr_matrix(np.asarray(rows,dtype=float))

def load_split(name):
    base="https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/"
    txt=pd.read_csv(base+name+"_text.txt",header=None,sep="\0",engine="python",quoting=3)[0].astype(str)
    lab=pd.read_csv(base+name+"_labels.txt",header=None)[0].astype(int)
    assert len(txt)==len(lab)
    return pd.DataFrame({"text":txt,"label":lab})

def tag_features(df, emoji_thr=.05, slang_thr=.10, min_words=8, priority=("sarcasm","emoji","slang")):
    x=df.copy()
    raw=x.text.fillna("").astype(str)
    x["word_count"]=raw.str.split().str.len().clip(lower=1)
    x["emoji_count"]=raw.map(lambda t:len(emoji.emoji_list(str(t))))
    x["emoji_density"]=x.emoji_count/x.word_count
    x["slang_count"]=raw.map(lambda t:len(SLANG_RE.findall(str(t).lower())))
    x["slang_ratio"]=x.slang_count/x.word_count
    x["sarcasm"]=raw.map(lambda t:bool(SARCASM_RE.search(str(t).lower())))
    flags={
      "sarcasm":x.sarcasm.to_numpy(bool),
      "emoji":(x.emoji_density>emoji_thr).to_numpy(bool),
      "slang":(x.slang_ratio>slang_thr).to_numpy(bool)}
    ref=(~flags["sarcasm"] & ~flags["emoji"] & ~flags["slang"] & (x.word_count.to_numpy()>=min_words))
    out=np.array(["other"]*len(x),dtype=object)
    out[ref]="reference-unmarked"
    label={"sarcasm":"sarcasm-indicated","emoji":"emoji-heavy","slang":"slang-heavy"}
    for key in reversed(priority):
        out[flags[key]]=label[key]
    x["subgroup"]=out
    for k,v in flags.items(): x["flag_"+k]=v
    return x

def macro(y,p,labels=None):
    return f1_score(y,p,average="macro",zero_division=0,labels=labels)

def metric_rows(df, scenario, labels=None):
    ref=df[df.subgroup=="reference-unmarked"]
    ref_f=macro(ref.label,ref.pred,labels)
    rows=[]
    for sg,g in df.groupby("subgroup"):
        rows.append({
          "scenario":scenario,"subgroup":sg,"N":len(g),
          "accuracy":accuracy_score(g.label,g.pred),
          "macro_f1":macro(g.label,g.pred,labels),
          "gap_vs_reference":macro(g.label,g.pred,labels)-ref_f,
          "negative_N":int((g.label==0).sum()),"neutral_N":int((g.label==1).sum()),"positive_N":int((g.label==2).sum()),
          "mean_words":float(g.word_count.mean()),"median_words":float(g.word_count.median())})
    return rows

def bootstrap_gap(df, subgroup, n_boot, labels_eval=None, seed=42):
    a=df[df.subgroup==subgroup][["label","pred"]].to_numpy()
    b=df[df.subgroup=="reference-unmarked"][["label","pred"]].to_numpy()
    rng=np.random.default_rng(seed)
    vals=np.empty(n_boot)
    for i in range(n_boot):
        ia=rng.integers(0,len(a),len(a)); ib=rng.integers(0,len(b),len(b))
        vals[i]=macro(a[ia,0],a[ia,1],labels_eval)-macro(b[ib,0],b[ib,1],labels_eval)
    return np.quantile(vals,[.025,.5,.975])

out=Path("pass_c_outputs"); out.mkdir(exist_ok=True)
tr=load_split("train"); va=load_split("val"); te=load_split("test")
assert (len(tr),len(va),len(te))==(45615,2000,12284)

word=lambda:TfidfVectorizer(preprocessor=clean_text,ngram_range=(1,2),min_df=2,max_df=.98,sublinear_tf=True,max_features=120000)
char=lambda:TfidfVectorizer(preprocessor=clean_text,analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=120000)
features=FeatureUnion([("word",word()),("char",char()),("social",SocialFeatureTransformer())])
pipe=Pipeline([("features",features),("clf",CalibratedClassifierCV(LinearSVC(C=.5,random_state=42),cv=3))])
trainval=pd.concat([tr,va],ignore_index=True)
pipe.fit(trainval.text,trainval.label)
te["pred"]=pipe.predict(te.text)
te_locked=tag_features(te,.05,.10,8,("sarcasm","emoji","slang"))
acc=accuracy_score(te.label,te.pred); mf=macro(te.label,te.pred)
counts=te_locked.subgroup.value_counts().to_dict()
guard={"accuracy":acc,"macro_f1":mf,"counts":counts,
       "python":platform.python_version(),"sklearn":__import__("sklearn").__version__,
       "emoji":emoji.__version__}
assert abs(acc-0.6213774014978835)<1e-12,(acc,mf)
assert abs(mf-0.6053146348313904)<1e-12,(acc,mf)
assert counts=={"reference-unmarked":10388,"other":1112,"emoji-heavy":710,"slang-heavy":60,"sarcasm-indicated":14},counts
(out/"reproduction_check.json").write_text(json.dumps(guard,indent=2))

# #18 emoji threshold sensitivity
rows=[]
for th in [.03,.05,.10]:
    z=tag_features(te,th,.10,8,("sarcasm","emoji","slang")); z["pred"]=te.pred
    rows+=metric_rows(z,f"emoji_threshold={th:.2f}")
pd.DataFrame(rows).to_csv(out/"issue18_emoji_threshold_sensitivity.csv",index=False)

# #19 slang threshold sensitivity and reproducible lexicon
rows=[]
for th in [.05,.10,.15]:
    z=tag_features(te,.05,th,8,("sarcasm","emoji","slang")); z["pred"]=te.pred
    rows+=metric_rows(z,f"slang_threshold={th:.2f}")
pd.DataFrame(rows).to_csv(out/"issue19_slang_threshold_sensitivity.csv",index=False)
(out/"slang_lexicon.txt").write_text("\n".join(SLANG_TERMS)+"\n")
locked_slang=te_locked[te_locked.subgroup=="slang-heavy"].copy()
locked_slang["matched_terms"]=locked_slang.text.map(lambda t:"|".join(SLANG_RE.findall(str(t).lower())))
locked_slang.sample(n=min(20,len(locked_slang)),random_state=42)[["text","matched_terms","label","pred","word_count","slang_ratio"]].to_csv(out/"issue19_manual_review_sample.csv",index=False)

# #21 overlap and priority order
flagcols=["flag_sarcasm","flag_emoji","flag_slang"]
overlap=[]
for mask in range(1,8):
    names=[flagcols[i] for i in range(3) if mask&(1<<i)]
    exact=np.ones(len(te_locked),dtype=bool)
    for i,c in enumerate(flagcols):
        exact &= te_locked[c].to_numpy()==bool(mask&(1<<i))
    overlap.append({"pattern":"+".join(n.replace("flag_","") for n in names),"N_exact":int(exact.sum())})
pd.DataFrame(overlap).to_csv(out/"issue21_prepriority_overlap.csv",index=False)
priority_rows=[]
orders=[
 ("locked_sarcasm_emoji_slang",("sarcasm","emoji","slang")),
 ("emoji_sarcasm_slang",("emoji","sarcasm","slang")),
 ("slang_emoji_sarcasm",("slang","emoji","sarcasm")),
 ("emoji_slang_sarcasm",("emoji","slang","sarcasm"))]
for name,order in orders:
    z=tag_features(te,.05,.10,8,order); z["pred"]=te.pred
    priority_rows+=metric_rows(z,name)
pd.DataFrame(priority_rows).to_csv(out/"issue21_priority_sensitivity.csv",index=False)

# #22 reference selection effects
ref_rows=[]
overall_w=np.array([(te.label==k).mean() for k in [0,1,2]])
for mw in [1,5,8,10,12]:
    z=tag_features(te,.05,.10,mw,("sarcasm","emoji","slang")); z["pred"]=te.pred
    for r in metric_rows(z,f"reference_min_words={mw}"): ref_rows.append(r)
    # common class-weight standardised correctness using overall test class shares
    for sg,g in z[z.subgroup.isin(["reference-unmarked","emoji-heavy","slang-heavy","sarcasm-indicated"])].groupby("subgroup"):
        rec=[]
        for k in [0,1,2]:
            q=g[g.label==k]
            rec.append(float((q.pred==q.label).mean()) if len(q) else np.nan)
        std=float(np.nansum(overall_w*np.array(rec)))
        ref_rows.append({"scenario":f"reference_min_words={mw}:class_standardised",
                         "subgroup":sg,"N":len(g),"accuracy":np.nan,"macro_f1":np.nan,
                         "gap_vs_reference":std,"negative_N":int((g.label==0).sum()),
                         "neutral_N":int((g.label==1).sum()),"positive_N":int((g.label==2).sum()),
                         "mean_words":float(g.word_count.mean()),"median_words":float(g.word_count.median())})
pd.DataFrame(ref_rows).to_csv(out/"issue22_reference_selection_sensitivity.csv",index=False)

# #30/#31 direct gap intervals and bootstrap stability
bg=[]
for nb in [500,2000,5000]:
    for sg in ["emoji-heavy","slang-heavy","sarcasm-indicated","other"]:
        q=bootstrap_gap(te_locked,sg,nb,None,42)
        bg.append({"subgroup":sg,"n_boot":nb,"gap_point":macro(te_locked.loc[te_locked.subgroup==sg,"label"],te_locked.loc[te_locked.subgroup==sg,"pred"])-macro(te_locked.loc[te_locked.subgroup=="reference-unmarked","label"],te_locked.loc[te_locked.subgroup=="reference-unmarked","pred"]),
                   "ci_low":q[0],"bootstrap_median":q[1],"ci_high":q[2]})
pd.DataFrame(bg).to_csv(out/"issue30_31_direct_gap_bootstrap.csv",index=False)

# #51 frozen Sentiment140 exact sample
sfile=Path("pass_c_inputs/df_sentiment140_eval_sample_final.parquet")
sha=hashlib.sha256(sfile.read_bytes()).hexdigest()
assert sha=="d25327ccb390945e945f687c38f84e3973e81199c7b7977f02d1f5a08593b8ed",sha
s=pd.read_parquet(sfile)
text_for_pred="text_eval" if "text_eval" in s.columns else "text"
# discover truth label field as in frozen sample
if "sentiment" in s.columns:
    truth=s.sentiment.astype(str).str.lower().map({"negative":0,"positive":2})
elif "label" in s.columns:
    truth=s.label.map({0:0,4:2}).fillna(s.label)
else:
    raise RuntimeError("No Sentiment140 truth field")
mask=truth.isin([0,2])
s=s.loc[mask].copy(); s["label"]=truth.loc[mask].astype(int).to_numpy()
s["pred"]=pipe.predict(s[text_for_pred].astype(str))
s=tag_features(s,.05,.10,8,("sarcasm","emoji","slang"))
# tag_features copy retained pred/label
sref=s[s.subgroup=="reference-unmarked"]; se=s[s.subgroup=="emoji-heavy"]
src={
 "N":len(s),"accuracy":accuracy_score(s.label,s.pred),
 "macro_f1":macro(s.label,s.pred,[0,2]),"neutral_prediction_rate":float((s.pred==1).mean()),
 "emoji_N":len(se),"emoji_negative_N":int((se.label==0).sum()),"emoji_positive_N":int((se.label==2).sum()),
 "emoji_macro_f1":macro(se.label,se.pred,[0,2]),"reference_N":len(sref),"reference_macro_f1":macro(sref.label,sref.pred,[0,2])}
src["emoji_gap"]=src["emoji_macro_f1"]-src["reference_macro_f1"]
q=bootstrap_gap(s,"emoji-heavy",5000,[0,2],42)
src.update({"emoji_gap_ci_low":float(q[0]),"emoji_gap_bootstrap_median":float(q[1]),"emoji_gap_ci_high":float(q[2])})
assert abs(src["macro_f1"]-0.580373)<5e-6,src
assert len(se)==72 and len(sref)==73512,src
(out/"issue51_sentiment140_emoji_gap.json").write_text(json.dumps(src,indent=2))

# concise combined summary
summary={"reproduction":guard,"sentiment140":src}
(out/"pass_c_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
