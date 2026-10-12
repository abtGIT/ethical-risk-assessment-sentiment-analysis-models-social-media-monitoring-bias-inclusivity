"""Recompute the Chapter 3 figure statistics from the OFFICIAL TweetEval files.
Source: https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/{train,val,test}_{text,labels}.txt"""
import emoji, re, numpy as np, statistics as st, sys
D=sys.argv[1] if len(sys.argv)>1 else 'tweeteval'
L={0:'negative',1:'neutral',2:'positive'}; S={}
for s in ('train','val','test'):
    S[s]=([l.rstrip('\n') for l in open(f'{D}/{s}_text.txt',encoding='utf8')],[int(l) for l in open(f'{D}/{s}_labels.txt')])
print('FIGURE 3.6 class counts and shares')
for s,(tx,lb) in S.items(): print(f'  {s:5s} N={len(lb):6d}', [(L[k], lb.count(k), round(100*lb.count(k)/len(lb),2)) for k in (0,1,2)])
print('FIGURE 3.7 mean whitespace words per post')
for s,(tx,lb) in S.items(): print(f'  {s:5s}', round(st.mean(len(t.split()) for t in tx),2))
print('FIGURE 3.7 right panel, quartiles/median/whiskers by class, ALL SPLITS POOLED')
for k in (0,1,2):
    w=np.array([len(t.split()) for s in S for t,l in zip(*S[s]) if l==k]); q1,m,q3=np.percentile(w,[25,50,75]); i=q3-q1
    print(f'  {L[k]:8s} Q1 {q1:.0f} median {m:.0f} Q3 {q3:.0f} whiskers {w[w>=q1-1.5*i].min()}-{w[w<=q3+1.5*i].max()}')
print('FIGURE 3.8 emoji-bearing posts (emoji library)')
for s,(tx,lb) in S.items():
    e=sum(1 for t in tx if emoji.emoji_count(t)>0); print(f'  {s:5s} {e} ({100*e/len(tx):.2f}%)')
SL="lol lmao omg tbh imo imho wtf idk ngl smh af fr lowkey highkey slay lit goat vibe fire wanna gonna gotta kinda ur bruh bro fam dope sick sus bussin salty cap bet flex ghost clout woke ftw ftl epic fail rofl ttyl brb kk np nvm thx gr8 b4".split()
print('FIGURE 3.8 any slang-lexicon term (51-term frozen lexicon, simple tokeniser - approximation of frozen tagger)')
for s,(tx,lb) in S.items():
    k=sum(1 for t in tx if any(w in SL for w in re.findall(r"[a-z0-9']+",t.lower()))); print(f'  {s:5s} {k} ({100*k/len(tx):.2f}%)')
print('FIGURE 3.4 sample records found in official test split')
samp=[("Westworld.Damn.There is life after Game of Thrones.",2),("The Rocket Richard race or the Cy Young race?  #NHL",1),("8 is the reason I'm a Linux user.",0),("Tonight feels like a coffee cake kinda night. Let's beat #NationalFastFoodDay!!",2),("Selfie Stick, Extendable Monopod with Bluetooth Remote and Tripod Stand for iPhone 7 6S",1),("This...is upsetting and a manipulative way of diverting the conversation about gun control.",0),("Going to see #FantasticBeasts and I am LIT",2),("Like Grayson Allen, Adam is now operating at 50 percent.",1)]
tx,lb=S['test']; idx={}
for i,t in enumerate(tx): idx.setdefault(' '.join(t.split()),[]).append((i+1,lb[i]))
for s,l in samp:
    h=idx.get(' '.join(s.split())); print(f'  {"FOUND" if h else "MISSING"} line {h[0][0] if h else "-"} label {h[0][1] if h else "-"} (figure label {l}) :: {s[:50]}')
