from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from scipy.stats import ks_2samp
import json
base_dir = "POStags/"
flst = ["cd.POSrst", "imdb.POSrst", "sst.POSrst",
"elec.POSrst", "movie.POSrst", "video.POSrst"
      ]
CD={"pos":3574,"neg":4732,"obj":68352}
Video={"pos":1700,"neg":2303,"obj":29946}
Movie={"pos":3831,"neg":5312,"obj":77807}
Elec={"pos":2544,"neg":3309,"obj":51224}
IMDB={"pos":3588,"neg":4797,"obj":62233}
SST={"pos":452,"neg":456,"obj":5311}
sw_dic={"CD":CD,"Video":Video,"Movie":Movie,"Elec":Elec,"IMDB":IMDB,"SST":SST}
fd_dic = {}
tags_dic = {}
def loadFD():
    for fname in flst:
        with open(base_dir + fname) as f:
            for line in f:
                line = line.strip()
                tag = line.split("\t")[0]
                count = int(line.split("\t")[1])
                if fname not in fd_dic:
                    fd = {}
                    fd[tag] = count
                    fd_dic[fname] = fd
                else:
                    fd_dic[fname][tag] = count
                if tag not in tags_dic:
                    tags_dic[tag] = 1
                else:
                    tags_dic[tag] += 1
def sort_dic(odic):
    return {k:odic[k]for k in sorted(odic.iterkeys())}

loadFD()
POS = [k for k in tags_dic.keys() if tags_dic[k] == 6 and len(k) > 0]
for fl in flst:
    fd_dic[fl] = {k:v for k, v in fd_dic[fl].iteritems() if k in POS}
pos_array = np.zeros((len(flst), len(flst)))
# fd_dic=sorted(fd_dic.iteritems(), key=lambda (k,v): (v,k))
print ",".join(fd_dic.keys())
for dk, dv in fd_dic.iteritems():
    dv = sort_dic(dv)
    for ik, iv in fd_dic.iteritems():
        # if "imdb" in dk  or "sst" in dk:
            # print dk,ik,entropy(dv.values(),iv.values())
            iv = sort_dic(iv)
            kl = entropy(dv.values(), iv.values())
            pos_array[fd_dic.keys().index(dk)][fd_dic.keys().index(ik)] = kl  # [0]
    print dk, ",", json.dumps(pos_array[fd_dic.keys().index(dk)].tolist()).replace("],", "\n").replace("[", "").replace("]", "").replace(" ", "")
pos_sarray = np.zeros((6, 6))
print ",".join(sw_dic.keys())
for dk, dv in sw_dic.iteritems():
    dv = sort_dic(dv)
    for ik, iv in sw_dic.iteritems():
        # if "imdb" in dk  or "sst" in dk:
            # print dk,ik,entropy(dv.values(),iv.values())
            iv = sort_dic(iv)
            kl = entropy(dv.values(), iv.values())
            pos_sarray[sw_dic.keys().index(dk)][sw_dic.keys().index(ik)] = kl  # [0]
    print dk, ",", json.dumps(pos_sarray[sw_dic.keys().index(dk)].tolist()).replace("],", "\n").replace("[", "").replace("]", "").replace(" ", "")
