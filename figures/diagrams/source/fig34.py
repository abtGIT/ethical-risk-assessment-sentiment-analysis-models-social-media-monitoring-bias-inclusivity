import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, textwrap
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif'],'svg.fonttype':'none'})
lines=[l.rstrip('\n') for l in open('./tweeteval/test_text.txt',encoding='utf8')]; labs=[int(l) for l in open('./tweeteval/test_labels.txt')]
keys=['Westworld.Damn.There is life after Game of Thrones','The Rocket Richard race or the Cy Young race?','8 is the reason I\'m a Linux user','Tonight feels like a coffee cake kinda night','Selfie Stick, Extendable Monopod','This...is upsetting and a manipulative way','Going to see #FantasticBeasts and I am LIT','Like Grayson Allen, Adam is now operating at 50 percent']
rows=[]
for k in keys:
    i=[n for n,l in enumerate(lines) if l.startswith(k)]; assert len(i)>=1,k; i=i[0]; rows.append((lines[i].strip(),labs[i]))
name={0:'negative',1:'neutral',2:'positive'}
W,H=1830/300,1242/300; fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')
x0,x1,x2,x3=0.08,4.45,5.05,5.92; y=0.28
ax.text(x0,y,'tweet_eval · configuration: sentiment · split: test (12,284 rows)',fontsize=10.5,fontweight='bold',va='bottom'); y+=0.1
hh=0.3; ax.add_patch(plt.Rectangle((x0,y),x3-x0,hh,facecolor='#E8EEF4',edgecolor='#9AA6B2',linewidth=0.8))
for xx,t in ((x0+0.1,'text'),(x1+0.08,'label'),(x2+0.08,'label name')): ax.text(xx,y+hh/2,t,fontsize=10,fontweight='bold',va='center')
y+=hh
for txt,l in rows:
    wrapped=textwrap.wrap(txt,62,drop_whitespace=False) if len(txt)>62 else [txt]
    rh=0.12+0.17*len(wrapped)
    ax.add_patch(plt.Rectangle((x0,y),x3-x0,rh,facecolor='white',edgecolor='#C3CCD5',linewidth=0.7))
    for k,wl in enumerate(wrapped): ax.text(x0+0.1,y+0.07+0.17*k+0.085,wl.strip(),fontsize=9.5,va='center')
    ax.text(x1+0.08,y+rh/2,str(l),fontsize=9.5,va='center'); ax.text(x2+0.08,y+rh/2,name[l],fontsize=9.5,va='center'); y+=rh
for xx in (x1,x2): ax.plot([xx,xx],[0.38,y],color='#C3CCD5',linewidth=0.7)
ax.text(x0,y+0.22,'Label mapping: 0 = negative, 1 = neutral, 2 = positive. Source: official cardiffnlp TweetEval sentiment test split.',fontsize=9,style='italic',va='center')
fig.savefig('./fig3_4.png',dpi=300); fig.savefig('./fig3_4.svg')
print(rows)
