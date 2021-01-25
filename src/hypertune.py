import make_embeddings as e
import itertools
from collections import defaultdict
from gensim.models import KeyedVectors

""" Brute-force hyperparameter finder for Cuneiform Embeddings  

                                                        asahala 2020

This script naively tests multiple parameters to find optimal settings
for a given data set against a set of gold standards. By default
the results are written in TSV format into ´output_hypertune.log´

  spearman   pearson   params...
  ...                                                               """


""" Define hyperparameters to be tested. PMI and SVD parameters
are separated to avoid re-reading the corpus when not necessary """

# SET YOUR TEST PARAMETERS HERE ======================================
# Key names must be ones used in Embeddings object constructor!
parameter_grid = {'window_size': [1,2,3,4],
                  'subsampling_rate': [None],
                  'k_factor': [0,1,2,3],
                  'dirty_stopwords': [False],
                  'dynamic_window': [False, True]}

pmi_parameter_grid = {'shift_type': [0],
                      'alpha': [None, 0.75],
                      'lambda_': [None], 
                      'threshold': [0,3,5,7]}

dimensions = [60, 300]

""" Define evaluation datasets """
filename = 'akk.wpl'
datasets = ['gold.tsv', 'gold000.tsv', 'gold100.tsv', 'gold200.tsv']
output = 'output_hypertune.log'

# ===================================================================

def make_params(grid):
    """ Create test settings """ 
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

parameters = make_params(parameter_grid)
pmi_parameters = make_params(pmi_parameter_grid)
combs = len(parameters) * len(pmi_parameters) * len(dimensions)
TESTRUN = 1
print('Parameter combinations to be tested: %i' % combs)


def evaluate_model(datasets, p, pmi_p, d):
    """ :param datasets           list of gold standard file names 
        :param p                  current test parameters
        :param pmi_p              current PMI test parameters
        :param d                  current dimensionality """

    global TESTRUN

    vecs = KeyedVectors.load_word2vec_format('DELETEME.VEC', binary=False)
    spearman_scores = []
    pearson_scores = []

    """ Evaluate against all given data sets and get avg. scores """
    for dataset in datasets:
        results = vecs.evaluate_word_pairs(dataset,
                                           restrict_vocab=10000000,
                                           dummy4unknown=False)
        pearson, spearman, oov_ratio = results
        spearman_scores.append(spearman[0])
        pearson_scores.append(pearson[0])

    avg_spearman = round(sum(spearman_scores) / len(spearman_scores), 3)
    avg_pearson = round(sum(pearson_scores) / len(pearson_scores), 3)

    #by_spear[avg_spearman].append(p)
    #by_pear[avg_pearson].append(p)
    all_ =  {**p, **pmi_p}
    logline = '\t'.join(['%s=%s' % (k, str(v)) for k, v in sorted(all_.items())])
    line = str(avg_spearman) + '\t' + str(avg_pearson) + '\t' + logline + "\tdim=" + str(d)
    print(str(TESTRUN).zfill(2) + ': ' + line[0:60] + '...')
    TESTRUN += 1

    """ Write logfile """
    with open(output, 'a') as f:
        f.write(line + '\n')


def test_model(corpus_file, chunksize, datasets, pmi_params, params, **kwargs):    
    """ :param corpus_file              test corpus file name
        :param chunksize                number of words per chunk
        :param datasets                 list of gold standard file names
        :param params                   current parameters to be tested
        :param **kwargs                 params passed to embeddings object
    """    

    """ Initialize model and count co-occurrence matrix """
    embs = e.Cooc(filename, chunksize, verbose=False, **kwargs)
    embs.count_cooc()

    """ Backup co-occurrence matrix because lambda smoothing modifies it """
    cooc_matrix = embs.cooc_matrix
    
    for pmi_p in pmi_params:
        embs.cooc_matrix = cooc_matrix
        embs.calculate_pmi(**pmi_p)
        for dim in dimensions:
            embs.factorize(dim)
            embs.save_vectors('DELETEME.VEC') 
            evaluate_model(datasets, params, pmi_p, dim)

for i, p in enumerate(parameters):
    test_model(filename, 400000, datasets, pmi_parameters, p, **p)
    #evaluate_model(datasets, p)

#for k, v in sorted(by_spear.items(), reverse=True):
#    print(k, v)
