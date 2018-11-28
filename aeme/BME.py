import numpy as np
import argparse 
import os
from utils import Logger, Utils
from tqdm import tqdm
import sklearn.preprocessing as skpre
from sklearn.decomposition import PCA

LOG = './log/'
CHECKPOINT = './checkpoint/'

METHODS = ['CONC', 'PCA']

class BME(object):
    def __init__(self, **kwargs):
        self.input_list = kwargs['input']
        self.dim = kwargs['dim']
        self.method = kwargs['method']
        self.output_path = kwargs['output']
        self.logger = Logger(self.method)
        self.utils = Utils(self.logger.log)
        
    def load_data(self):
        # load data and preprocessing
        src_dict_list = [self.utils.load_emb(path) for path in self.input_list]
        # find intersection of words
        self.inter_words = list(set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list]))
        self.logger.log('Intersection Words: %s' % len(self.inter_words)) 
        self.sources = np.asarray(list(zip(*[skpre.normalize([src_dict[word] for word in self.inter_words]) for src_dict in src_dict_list])))
        del src_dict_list
        print (len(self.inter_words)==len(self.sources))
        print ("Data loaded")

    def build_method(self):
        if self.method == 'CONC':
            vectors = np.reshape(self.sources, (len(self.sources), -1))
            self.logger.log('concatenate completed') 
        else:
            conc = np.reshape(self.sources, (len(self.sources), -1))
            self.logger.log('PCA into %s dimensions' % self.dim)
            print(conc.shape)
            pca = PCA(n_components=self.dim)
            vectors = pca.fit_transform(conc)
        print(vectors.shape)
        output_emb = {self.inter_words[i]: vectors[i] for i in range(len(self.inter_words))}
        self.utils.save_emb(output_emb, self.output_path)      

def main():
    """
    -m: method, svd or concatenation
    -d: for svd, how many dimension 
    """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', dest = 'inputs', type=str, required = True, nargs='+', help = 'source embs directories')
    add_arg('-m', dest = 'method', type = str, required = True, help = 'which method used to generate meta_emb')
    add_arg('-d', dest = 'dim', type = int, default = 300, help = 'for pca, output dimensions')
    add_arg('-o', dest='output', type=str,    required=True, help='directory of yielded meta-embedding')

    args = parser.parse_args()
    assert args.method in METHODS

    params = {'input': tuple(args.inputs),
                'dim': args.dim,
                'method': args.method,
                'output': args.output}
    bme = BME(**params)
    log = bme.logger.log
    log('Source paths: %s' % (params['input'],))
    log('Output path: %s' % params['output'])
    bme.load_data()
    bme.build_method()

if __name__ == '__main__':
    main()

    
    

