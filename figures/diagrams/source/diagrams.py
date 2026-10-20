import subprocess,re
P={'NAVY':('#D6E4F2','#2F5C8A'),'SLATE':('#E4E8EE','#5A6A7E'),'TEAL':('#D3ECE6','#2E7D6E'),'OCHRE':('#FBE7C2','#A8720F'),
   'RED':('#F6D5DA','#A33A4B'),'LIGHT':('#FFFFFF','#7F8C9A'),'GREY':('#EDEDED','#8A8A8A'),'NOTE':('#FFF8E1','#9C8F6A')}
def esc(s): return s.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n')
def build(name,nodes,edges,rankdir='TB',fs=13,extra='',ranks=()):
    L=[f'digraph G {{ rankdir={rankdir}; bgcolor="white"; pad=0.15; nodesep=0.35; ranksep=0.45; newrank=true; {extra}',
       f' node [shape=box, style="rounded,filled", fontname="Liberation Serif", fontsize={fs}, fontcolor="#1A1A1A", penwidth=1.3, margin="0.18,0.08"];',
       ' edge [color="#4A5563", penwidth=1.4, arrowsize=0.8, fontname="Liberation Serif", fontsize=%d, fontcolor="#333333"];'%(fs-2)]
    for nid,lab,sty in nodes:
        f,c=P[sty]; shape=', shape=note, style="filled"' if sty=='NOTE' else ''
        L.append(f' {nid} [label="{esc(lab)}", fillcolor="{f}", color="{c}"{shape}];')
    for e in edges:
        a,b=e[0],e[1]; at=e[2] if len(e)>2 else ''
        L.append(f' {a} -> {b} [{at}];')
    for r in ranks: L.append(' {rank=same; '+'; '.join(r)+';}')
    L.append('}')
    src='\n'.join(L); open(f'./{name}.dot','w').write(src)
    for fmt,extra_args in (('svg',[]),('png',['-Gdpi=300'])):
        subprocess.run(['dot',f'-T{fmt}'+(':cairo' if fmt=='png' else ''),*extra_args,f'./{name}.dot','-o',f'./{name}.{fmt}'],check=True)
NOTE_E='style=dashed, dir=none, color="#A33A4B"'
DASH='style=dashed'
# ---------- Figure 1.1
build('fig1_1',[('a','Social media users\n5.66 billion user identities','NAVY'),('b','User-generated content\nslang, emoji, sarcasm','LIGHT'),
 ('c','Automated sentiment\nmonitoring','SLATE'),('d','Sentiment classifier\nnegative / neutral / positive','TEAL'),('e','Aggregate summary\nsingle accuracy figure reported','OCHRE'),
 ('f','Organisational\ndecisions','NAVY'),('n1','Performance may differ\nacross writing styles','NOTE'),('n2','Aggregate scores may conceal\nsubgroup differences','NOTE')],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('n1','d',NOTE_E),('n2','e',NOTE_E)],ranks=[('n1','d'),('n2','e')])
# ---------- Figure 1.2
build('fig1_2',[('a','Social media sentiment monitoring\ndeployed at scale','NAVY'),('b','Aggregate accuracy reported\nas evidence of adequacy','LIGHT'),
 ('c','Linguistic variation concealed\nwithin the average','OCHRE'),('d','This study: subgroup-level evidence across performance,\ndisparity, reliability, explanation and robustness','TEAL'),
 ('e','Disparity made visible\nand bounded','LIGHT'),('f','Reproducible frozen rules\nfor independent re-audit','LIGHT'),('g','Informed deployment decisions\nrather than assumed adequacy','NAVY')],
 [('a','b'),('b','c'),('c','d'),('d','e'),('d','f'),('e','g'),('f','g')],fs=16)
# ---------- Figure 1.3
build('fig1_3',[('g','Gap in the reviewed literature\nno integrated sentiment-specific audit linking\nlinguistic subgroups, disparity, error/confidence,\nexplanation and external robustness','RED'),
 ('c1','Contribution 1\nReproducible\nlinguistic-subgroup\naudit with frozen\nrules','NAVY'),('c2','Contribution 2\nMulti-dimensional\nethical-risk evidence\nprofile, dimensions\nkept separate','TEAL'),
 ('c3','Contribution 3\nRobustness-qualified\ninterpretation\nlocked pipeline,\nN ≥ 30 coverage rule','OCHRE'),
 ('o','Outcome: evaluation framework and\nevidence synthesis, not a new classifier,\nfairness metric or single risk score','SLATE'),
 ('s','Supporting analyses (not separate novelty)\nAIF360 / Fairlearn equivalence check\ncontrolled model and feature comparison\nSHAP / LIME as diagnostic evidence','LIGHT')],
 [('g','c1'),('g','c2'),('g','c3'),('c1','o'),('c2','o'),('c3','o'),('s','o',DASH+', dir=back')],extra='',ranks=[('c1','c2','c3')])
# fix supporting box below outcome: invert edge direction visually
src=open('./fig1_3.dot').read().replace('s -> o [style=dashed, dir=back];','o -> s [style=dashed, dir=back];')
open('./fig1_3.dot','w').write(src)
for fmt,ex in (('svg',[]),('png',['-Gdpi=300'])): subprocess.run(['dot',f'-T{fmt}'+(':cairo' if fmt=='png' else ''),*ex,'./fig1_3.dot','-o',f'./fig1_3.{fmt}'],check=True)
# ---------- Figure 2.1
build('fig2_1',[('r','Raw social media text\ne.g. TweetEval: “Going to see\n#FantasticBeasts and I am LIT”','LIGHT'),('p','Preprocessing\ncleaning, normalisation','SLATE'),
 ('i','Informal language\nemoji, slang, sarcasm, hashtags','RED'),('f','Feature extraction\nTF-IDF, n-grams,\nsocial-media features','SLATE'),
 ('m','Sentiment classification\nmodel (e.g. LR, SVM, NB)','NAVY'),('o','Sentiment output\nnegative / neutral / positive','TEAL'),('u','Monitoring use\npublic opinion, customer\nfeedback, crisis response','OCHRE')],
 [('r','p'),('p','f'),('f','m'),('m','o'),('o','u'),('i','p','style=dashed, color="#A33A4B", label="  affects"'),('i','f','style=dashed, color="#A33A4B", label="  affects"'),('i','o','style=dashed, color="#A33A4B", label="  can distort"')],ranks=[('i','f')])
# ---------- Figure 2.2
build('fig2_2',[('t','TweetEval post (raw text)','NAVY'),('r','Frozen precedence rule\nsarcasm → emoji → slang\n→ reference → other','SLATE'),
 ('fg','Focal linguistic groups','LIGHT'),('cr','Comparison and residual','LIGHT'),('s','Sarcasm-\nindicated\nsurface-cue\nrule match','RED'),('e','Emoji-heavy\nemoji/word\nratio > 0.05','TEAL'),
 ('sl','Slang-heavy\nslang/word\nratio > 0.10\n51-term lexicon','OCHRE'),('ru','Reference-\nunmarked\nno focal marker\n≥ 8 words','NAVY'),('ot','Other\nno focal\nassignment','GREY')],
 [('t','r'),('r','fg'),('r','cr'),('fg','s'),('fg','e'),('fg','sl'),('cr','ru'),('cr','ot')],fs=16,extra='nodesep=0.2;')
# ---------- Figure 2.3
build('fig2_3',[('ref','Reference-unmarked\n(comparison baseline)','NAVY'),('foc','Focal linguistic group\nemoji / slang / sarcasm-indicated','TEAL'),
 ('cpd','CPD\nCorrectness Parity\nDifference','LIGHT'),('crr','CRR\nCorrectness Rate\nRatio','LIGHT'),('cc','Class-conditional\none-vs-rest TPR and FPR gaps','LIGHT'),
 ('aif','AIF360 / Fairlearn\nequivalence cross-check','OCHRE'),('out','Bounded linguistic disparity evidence\n(not demographic fairness)','SLATE')],
 [('ref','cpd'),('ref','crr'),('ref','cc'),('foc','cpd'),('foc','crr'),('foc','cc'),('cpd','aif'),('crr','aif'),('cc','aif'),('aif','out')],ranks=[('ref','foc')])
# ---------- Figure 3.1 (section numbers aligned with the current Chapter 3 headings)
build('fig3_1',[('a','3.2–3.3  Philosophy, approach\nand research design','NAVY'),('b','3.4  Research questions and\nmethodological alignment','SLATE'),
 ('c','3.5–3.6  Datasets, preparation\nand subgroup design','TEAL'),('d','3.7  Comparative modelling and\nfeature evaluation strategy','OCHRE'),
 ('e','3.8  Multi-dimensional\nevaluation framework','TEAL'),('f','3.9–3.10  Experimental design,\nuncertainty and generalisation','SLATE'),
 ('g','3.11  Reliability, validity\nand reproducibility','NAVY'),('h','3.12  Ethical considerations\nand limitations','RED'),('i','3.13  Evaluation metrics','LIGHT')],
 [('a','b'),('b','c'),('c','d'),('d','e','constraint=false'),('e','f'),('f','g'),('g','h'),('h','i')],ranks=[('a','e'),('b','f'),('c','g'),('d','h')])
# ---------- Figure 3.2
build('fig3_2',[('a','Existing literature\non sentiment and fairness','NAVY'),('b','Identified research gap\nlinguistic disparity under-examined','OCHRE'),
 ('c','Four research questions\nRQ1–RQ4','TEAL'),('d','Analytical rules fixed\nbefore test inspection','RED'),('e','Experimental testing\nunder pre-specified conditions','SLATE'),
 ('f','Evidence evaluated\nagainst stated questions','NAVY'),('n','Frozen before results:\nsubgroup definitions · thresholds\nslang lexicon · precedence rule','NOTE')],
 [('a','b'),('b','c'),('c','d'),('d','e'),('e','f'),('d','n',NOTE_E)],fs=16,ranks=[('e','n')])
# ---------- Figure 3.3
build('fig3_3',[('m','Methodological\nstrategy','NAVY'),('p','Philosophy\nPositivist','LIGHT'),('a','Approach\nDeductive','LIGHT'),('me','Method\nExperimental, quantitative','LIGHT'),
 ('l','Logic\nComparative','LIGHT'),('u','Unit of analysis\nPost and linguistic subgroup','LIGHT'),('e','Evaluation\nMulti-dimensional','LIGHT')],
 [('m','p'),('m','a'),('m','me'),('m','l'),('m','u'),('m','e')],rankdir='LR',fs=16)
# ---------- Figure 3.5
build('fig3_5',[('tw','TweetEval\nsocial media · three-class','NAVY'),('n','External data never influences\nmodel or feature selection','NOTE'),
 ('dev','Development\nvalidation-based model and feature selection','SLATE'),('se','SemEval-2014\nreview text · three-class','TEAL'),('lt','One locked test evaluation\nRQ1 · RQ2 · RQ3','NAVY'),
 ('s140','Sentiment140\nsocial media · binary','OCHRE'),('ds','Domain / task-shift\nstress evaluation (RQ4)','LIGHT'),('br','Binary external\nrobustness evaluation (RQ4)','LIGHT')],
 [('tw','dev'),('n','dev',NOTE_E),('dev','lt'),('lt','ds'),('lt','br'),('se','ds',DASH),('s140','br',DASH)],ranks=[('tw','n'),('se','lt','s140')])
# ---------- Figure 3.9
build('fig3_9',[('o','Official TweetEval\nsentiment partitions\ntrain 45,615\nvalidation 2,000 · test 12,284','NAVY'),
 ('cl','Classes (neg / neu / pos)\ntrain 7,093 / 20,673 / 17,849\nval. 312 / 869 / 819\ntest 3,972 / 5,937 / 2,375','LIGHT'),
 ('rk','Raw text kept\nfor tagging; markers\npreserved','TEAL'),('fr','Frozen subgroup rules\non raw text (test split)\nreference-unmarked 10,388\nemoji-heavy 710 · slang-heavy 60\nsarcasm-indicated 14 · other 1,112','TEAL'),
 ('cc','Conservative cleaning\nlower-case · remove URLs\nand @user mentions\nno stemming, lemmatisation\nor stop-word removal','SLATE'),
 ('n','Emoji-bearing posts\ntrain 0 · validation 0 · test 809\nmean words per post\ntrain 19.2 · test 14.9','NOTE'),
 ('au','Exact-overlap audit\nzero overlap across all\nthree split pairs','OCHRE'),
 ('ex','External targets normalised\nwith the same functions\nSentiment140: frozen 100,000-row\ndeduplicated binary sample\nSemEval-2014: 1,758 aspect rows\n(1,011 unique sentences)\nno refitting, retraining or threshold tuning','LIGHT')],
 [('o','cl','dir=none, style=dashed'),('o','rk'),('rk','fr'),('o','cc'),('fr','n',NOTE_E),('fr','au'),('cc','au'),('au','ex')],ranks=[('o','cl'),('fr','cc')])
# ---------- Figure 3.10
build('fig3_10',[('t','TweetEval test post (raw text, N = 12,284)','NAVY'),('r','Frozen precedence rule, once per post\nsarcasm → emoji → slang\n→ reference-unmarked → other','SLATE'),
 ('s','Sarcasm-indicated\nN = 14\nirony hashtag or\nstock ironic phrase\n“SORRY, HARRY...\nI’M LEAVING YOU FOR\nYOUR TWIN: MARCEL\n#JK” (neutral)','RED'),
 ('e','Emoji-heavy\nN = 710\nemoji / word > 0.05\n“Thanksgiving food is\nbetter the second day\n[turkey][heart-eyes]”\n(positive)','TEAL'),
 ('sl','Slang-heavy\nN = 60\n51-term lexicon,\nslang / word > 0.10\n“Going to see\n#FantasticBeasts\nand I am LIT” (positive)','OCHRE'),
 ('ru','Reference-unmarked\nN = 10,388\nno focal marker, ≥ 8 words\n“Like Grayson Allen, Adam\nis now operating at\n50 percent.” (neutral)','NAVY'),
 ('ot','Other\nN = 1,112\nno focal marker,\n< 8 words\n“This is great!”\n(positive)','GREY'),('n','Interpretation rule\nN ≥ 30: quantitative reporting\nN < 30: exploratory only','NOTE')],
 [('t','r'),('r','s'),('r','e'),('r','sl'),('r','ru'),('r','ot'),('ot','n',NOTE_E)],fs=18,extra='nodesep=0.15;',ranks=[('s','e','sl'),('ru','ot','n')])
# fix 3.10 layout: push reference/other row below the focal row
s=open('./fig3_10.dot').read().replace(' ot -> n [', ' e -> ru [style=invis]; sl -> ot [style=invis];\n ot -> n [')
open('./fig3_10.dot','w').write(s)
for fmt,ex in (('svg',[]),('png',['-Gdpi=300'])): subprocess.run(['dot',f'-T{fmt}'+(':cairo' if fmt=='png' else ''),*ex,'./fig3_10.dot','-o',f'./fig3_10.{fmt}'],check=True)
# ---------- Figure 3.11
build('fig3_11',[('b','Baseline model comparison\nfive model families · word TF-IDF · validation Macro F1','NAVY'),('f','Feature representation comparison\nword · character · social-only · hybrid','TEAL'),
 ('p5','Five common-representation\npipelines retained for RQ2','TEAL'),('o','Controlled optimisation\n5-fold train-only CV (C ∈ {0.5, 1, 2})','OCHRE'),
 ('s','Final selection\ncalibrated Linear SVM + hybrid features, C = 0.5','SLATE'),('l','Locked pipeline\nrefit train + validation; one test evaluation\nRQ1 · RQ3 · RQ4','NAVY')],
 [('b','f'),('b','p5',DASH),('f','o'),('o','s'),('s','l')],fs=15,ranks=[('f','p5')])
# ---------- Figure 3.12 (RQ1 label aligned with the RQ1 wording)
build('fig3_12',[('l','Locked\npipeline and\nfrozen outputs','NAVY'),('r1','RQ1 Subgroup performance\nMacro F1 · class-level · gap\n(overall: accuracy · MCC · kappa)','LIGHT'),
 ('r2','RQ2 Linguistic disparity\nCPD · CRR\nclass-conditional TPR/FPR','LIGHT'),('r3a','RQ3 Confidence, calibration\nHCER · MCE · ECE','LIGHT'),
 ('r3b','RQ3 Errors, explainability\nerror census\nSHAP/LIME sample','LIGHT'),('r4','RQ4 Robustness\nsource-to-target\nMacro F1 change','LIGHT'),
 ('e','Evidence kept\nseparate: no\ntotal score\nor ranking','RED')],
 [('l','r1'),('l','r2'),('l','r3a'),('l','r3b'),('l','r4'),('r1','e'),('r2','e'),('r3a','e'),('r3b','e'),('r4','e')],rankdir='LR')
# ---------- Figure 3.13
build('fig3_13',[('s1','Stage 1  Protocol, data\nand subgroup freeze','NAVY'),('s2','Stage 2  Controlled baseline\nmodel comparison','NAVY'),
 ('s3','Stage 3  Feature comparison,\noptimisation and pipeline lock','NAVY'),('n','Pipeline locked:\ntest split opened once','NOTE'),('s4','Stage 4  Locked TweetEval\ntest evaluation','TEAL'),
 ('s5','Stage 5  RQ1 subgroup\nperformance audit','TEAL'),('s6','Stage 6  RQ2 five-model\ndisparity audit','TEAL'),('s7','Stage 7  RQ3 errors,\nconfidence and XAI','TEAL'),
 ('s8','Stage 8  RQ4 external\nrobustness','OCHRE'),('s9','Stage 9  Multi-dimensional\nethical-risk evidence synthesis','SLATE')],
 [('s1','s2'),('s2','s3'),('s3','s4'),('s3','n',NOTE_E),('s4','s5'),('s4','s6'),('s4','s7'),('s4','s8'),('s5','s9'),('s6','s9'),('s7','s9'),('s8','s9')],ranks=[('s3','n'),('s5','s6'),('s7','s8')],
 extra='')
s=open('./fig3_13.dot').read().replace(' s5 -> s9 [', ' s5 -> s7 [style=invis]; s6 -> s8 [style=invis];\n s5 -> s9 [')
open('./fig3_13.dot','w').write(s)
for fmt,ex in (('svg',[]),('png',['-Gdpi=300'])): subprocess.run(['dot',f'-T{fmt}'+(':cairo' if fmt=='png' else ''),*ex,'./fig3_13.dot','-o',f'./fig3_13.{fmt}'],check=True)
# ---------- Figure 3.14
build('fig3_14',[('c','Locked in-domain comparator\nTweetEval held-out test','NAVY'),('mb','Matched binary-truth\nsource subset','LIGHT'),('tc','Three-class\nsource reference','LIGHT'),
 ('s140','Sentiment140 binary target\nfrozen 100,000-row sample','OCHRE'),('se','SemEval-2014\nthree-class target','TEAL'),
 ('d1','Δ Macro F1 · accuracy\nneutral-prediction rate','LIGHT'),('d2','Δ Macro F1 · accuracy\nunder domain/task shift','LIGHT'),
 ('cq','Coverage-qualified subgroup transfer\nN ≥ 30 rule; sparse groups not interpreted','RED')],
 [('c','mb'),('c','tc'),('mb','s140'),('tc','se'),('s140','d1'),('se','d2'),('d1','cq'),('d2','cq')],fs=16)
# ---------- Figure 4.1
build('fig4_1',[('u','final_thesis_utils.py\nfrozen rules · metrics · thresholds','NAVY'),('f0','FINAL_00\nconfiguration freeze and run manifest','OCHRE'),
 ('aux','AUX_01\nSarcOji proxy validation (auxiliary)','GREY'),('seq','FINAL_01 – FINAL_07\nexperiment sequence','TEAL'),
 ('fa','Frozen artefacts\npredictions · probabilities · metric CSV files','SLATE'),('th','Thesis Chapters 5 and 6','NAVY')],
 [('u','f0'),('u','aux',DASH),('f0','seq'),('seq','fa'),('fa','th'),('aux','th',DASH)],fs=15,ranks=[('f0','aux')])
# ---------- Figure 4.2
build('fig4_2',[('o','Official TweetEval partitions\n45,615 / 2,000 / 12,284','NAVY'),('cl','Conservative cleaning\nURLs and mentions removed','SLATE'),
 ('st','Subgroup tagging on raw text\nfrozen rules and precedence','TEAL'),('v','Vectorisers and classifier fitted on\ndevelopment data only; refit on\ntrain + validation after lock','OCHRE'),
 ('li','Locked model inference\non held-out test','NAVY'),('pa','Prediction artefacts\ny_true · y_pred · probabilities\nconfidence · subgroup','TEAL'),('c','Consumed by RQ1–RQ4\nwithout retraining','LIGHT')],
 [('o','cl'),('o','st'),('cl','v'),('v','li'),('li','pa'),('st','pa'),('pa','c')],fs=14)
# ---------- Figure 4.3
build('fig4_3',[('f0','FINAL_00\nconfiguration and freeze','NAVY'),('f1','FINAL_01\ndata and subgroups','TEAL'),('f2','FINAL_02\nmodel development and lock','OCHRE'),
 ('aux','AUX_01\nSarcOji proxy\nvalidation','GREY'),('f3','FINAL_03\nRQ1 subgroups','SLATE'),('f4','FINAL_04\nRQ2 disparity audit','SLATE'),
 ('f5','FINAL_05\nRQ3 errors, XAI','SLATE'),('f6','FINAL_06\nRQ4 external robustness','SLATE'),('f7','FINAL_07\nethical-risk synthesis','NAVY')],
 [('f0','f1'),('f1','f2'),('f1','aux',DASH),('f2','f3'),('f2','f4'),('f2','f5'),('f2','f6'),('f3','f7'),('f4','f7'),('f5','f7'),('f6','f7'),('aux','f7',DASH)],fs=17,ranks=[('f3','f4'),('f5','f6')])
s=open('./fig4_3.dot').read().replace(' f3 -> f7 [', ' f3 -> f5 [style=invis]; f4 -> f6 [style=invis];\n f3 -> f7 [')
open('./fig4_3.dot','w').write(s)
for fmt,ex in (('svg',[]),('png',['-Gdpi=300'])): subprocess.run(['dot',f'-T{fmt}'+(':cairo' if fmt=='png' else ''),*ex,'./fig4_3.dot','-o',f'./fig4_3.{fmt}'],check=True)
print('diagrams built')
