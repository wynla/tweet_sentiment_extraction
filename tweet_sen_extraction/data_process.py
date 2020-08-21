import pandas as pd

tw=pd.read_csv("input/twitter_augmented.csv")
# ds=tw['selected_text'].str.split(',', expand=True).stack().reset_index(level = 1,drop = True)#将1级索引去
# data_new = tw.drop(['selected_text'], axis=1).join(ds.to_frame( name = 'selected_text'))
id = list(tw['textID'])
sentiment = list(tw['sentiment'])
texts = list(tw['text'])
selected_text = list(tw['selected_text'])
all = []
for i,ts,sts,sti in zip(id,texts,selected_text,sentiment):
    for t,st in zip(eval(ts),eval(sts)):
        all.append([i,t,st,sti])
data_new=pd.DataFrame(data=all,columns=['textID','text','selected_text','sentiment'])
data_new.to_csv("twitter_augmented.csv")