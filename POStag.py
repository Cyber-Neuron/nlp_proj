'''
Created on Dec 12, 2018

@author: dan
'''
import fire

from nltk.tag.perceptron import PerceptronTagger
import string
import nltk
maketrans = str.maketrans
translate_dict = dict((c, " ") for c in string.punctuation)
translate_map = maketrans(translate_dict)
tagger = PerceptronTagger()
tags_dic = {}
import redis
r = redis.Redis(
    host='127.0.0.1',
     port=7777, 
     #password=''
    )
def _store_tag(name,rs,k_prefix):
    if rs:
        try:
            r.incr(k_prefix+"_"+name)
        except:
            print ("Redis Error:",name)
    else:
        if tag not in tags_dic:
            tags_dic[tag] = 1
        else:
            tags_dic[tag] += 1
def _processing(line):
        try:
            line = line.lower()
            text = line.translate(translate_map)
            return text
        except:
            print("exp:", line)
            return None
def tag(file_path,k_prefix,use_redis=True):
    with open(file_path) as f:
        for line in f:
            line = _processing(line)
            if line is None:
                continue
            try:
                tokens = nltk.word_tokenize(line)
                tags = nltk.tag._pos_tag(tokens, None, tagger,lang="eng")
                for t in tags:
                    tag = t[1]
                    _store_tag(tag,use_redis,k_prefix)
            except:
                print("EXP:", line)
                continue
    
    if not use_redis:
#         print (tags_dic)
        with open(file_path+".POS","w") as w:
            for tag in tags_dic:
                w.write(tag+"\t"+str(tags_dic[tag]))
                w.write("\n")
def getrst(key):
    with open(key+".POSrst","w") as w:
        for k in r.keys(key+"*"):
            w.write(k.decode("utf-8").split("_")[1]+"\t"+str(r.get(k).decode("utf-8") )+"\n")
                        
fire.Fire()