import numpy as np
import pandas as pd
import csv, os, random, sys
import scipy.stats

def _eval_all(emb_simset):
    inp_emb = {}
    for wordvec in emb_simset.iterrows():
        word, vec = wordvec[1][0], wordvec[1][1:].tolist()
        vec = np.fromiter(map(float, vec[1:]), dtype = np.float32)
        norm = np.linalg.norm(vec)
        inp_emb[word] = vec/norm if (norm != 0) else [vec]
    score_dict = {}
    score_dict['score'] = 0
    for root,dirs,files in os.walk('/home/u15410/yizhen/emsemble/AutoencodedMetaEmbedding/VecShare/vecshare/Test_Input'):
        files = [testfile for testfile in files if testfile[0]!='.']
        for testfile in files:
            f_path = root+'/'+testfile
            print (f_path)
#             score_dict[testfile[:-4].strip().lower().replace(" ", "_").replace("-", "_")] = _eval_sim(f_path, inp_emb)
            score_dict[testfile[:-4].strip().lower()] = _eval_sim(f_path, inp_emb)
            if  testfile != 'mc-30.csv':
                score_dict['score'] += _eval_sim(f_path, inp_emb)/(len(files)-1)
    return score_dict

def _eval_sim(testfile,inp_emb):
    test, emb = np.empty(0), np.empty(0)
    testdrop = np.empty(0)
    spearman_corr = 0
    mean_vec = np.mean(np.asarray(list(inp_emb.values()) ), axis = 0)

    with open(testfile, 'r') as comp_test:
        tests_csv = csv.reader(comp_test)
        total = 0
        cnt=0
        for line in tests_csv:
            total+=1
            try:
                word1, word2 = line[0], line[1]
            except:
                print (line)
            wordvec_1 = inp_emb[word1] if word1 in inp_emb else mean_vec
            wordvec_2 = inp_emb[word2] if word2 in inp_emb else mean_vec
            if word1 not in inp_emb and word2 not in inp_emb:
                cnt+=1
            test = np.append(test, float(line[2]))
            if np.any(wordvec_1) and np.any(wordvec_2):
                emb = np.append(emb, np.dot(wordvec_1, wordvec_2))
            else:
                emb = np.append(emb, 0)
        print ("Total word pairs: {0}, missing: {1}".format(total, cnt))
    spearman_corr += scipy.stats.spearmanr(test, emb).correlation
        
            
#             if (word1 in inp_emb) and (word2 in inp_emb):
#                 wordvec_1, wordvec_2 = inp_emb[word1], inp_emb[word2]
#                 test = np.append(test, float(line[2]))
#                 if np.any(wordvec_1) and np.any(wordvec_2):
#                     emb = np.append(emb, np.dot(wordvec_1, wordvec_2))
#                 else:
#                     emb = np.append(emb, 0)
#             else:
#                 testdrop = np.append(testdrop, float(line[2]))

#     mean_vec = np.mean(np.asarray(list(inp_emb.values()) ), axis = 0)
        
#     for i in range (0,5):
#         embdrop = np.empty(0)
#         for j in range (0, len(testdrop)):
#             temp_test, temp_emb = np.empty(0), np.empty(0)
#             randvec1 = random.choice(inp)
#             randvec2 = random.choice(inp)
#             if np.any(randvec1) and np.any(randvec2):
#                 embdrop = np.append(embdrop, np.dot(randvec1, randvec2))
#             else:
#                 embdrop = np.append(embdrop, 0)

#         temp_test = np.append(test, testdrop)
#         temp_emb  = np.append(emb, embdrop)

#         spearman_corr += (scipy.stats.spearmanr(temp_test, temp_emb).correlation)/5
    return spearman_corr

if __name__ == '__main__':
	filename = sys.argv[1]
	tes = pd.read_csv(filename,quoting=3,  error_bad_lines=False)
	print (_eval_all(tes))
