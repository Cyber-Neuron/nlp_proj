'''
Created on Dec 12, 2018

@author: dan
'''
from nltk.corpus import sentiwordnet as swn
import string
import fire
maketrans = str.maketrans

senti_dic={}
translate_dict = dict((c, " ") for c in string.punctuation)
translate_map = maketrans(translate_dict)
def _processing(line):
        try:
            line=line.lower()
            text = line.translate(translate_map)
            for t in text.split():  # sentence segmentation
                if t not in senti_dic:
                    senti_dic[t]=None
        except:
            print("exp:",line)
        
def _get_senti(s_s):
    slst=[s_s.pos_score(),s_s.neg_score(),s_s.obj_score()]
    return (str(slst.index(max(slst))) ,",".join([str(s) for s in slst]))           

def senti(file_path):
    with open(file_path) as f:
        for line in f:
            _processing(line)
    for t in senti_dic:
        try:
            senti_synset=list(swn.senti_synsets(t))
            if len(senti_synset)==0:
                continue
            s=senti_synset[0]
            senti_dic[t]=_get_senti(s)
        except:
            print("EXP:",t)
            continue
    with open(file_path+".senti.json","w") as w:
        for s in senti_dic:
            if senti_dic[s] is None:
                continue
            w.write(s+"\t"+senti_dic[s][0]+"\t"+senti_dic[s][1])
            w.write("\n")
fire.Fire()