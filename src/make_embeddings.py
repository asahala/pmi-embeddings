#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import itertools
import math
import numpy as np
import os
import random
import sys
import time
from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

__version__ = "2021-18-01 16:15"

# Globals
PADDING = "#"
LACUNA = "_"
STOP = "<STOP>"

# Pretty print dividers 
DIV = "> " + "--" * 16

""" *´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*

 Word embeddings for cuneiform languages                           asahala 2020

 This script combines findings in several PMI+SVD and related research papers 
 to test their applicability in fragmentary and repetitive low-resource 
 languages such as Sumerian and Akkadian.

  + Dirichlet Smoothing                  (Jungmaier et al. 2020)
  + Context similarity weighting (CSW)   (Sahala & Linden 2020)
  + Shifted PMI                          (Levy et al. 2015)
  + Dynamic Context Window (DCW)         (Sahlgren 2006)
  + Subsampling                          (Mikolov et al. 2013)
  + Dirty and clean stopwords            (Mikolov et al. 2013)
  + Context Distribution Smoothing (CDS) (Levy et al. 2015)
  + Eigenvalue weighting                 (Caron 2001)
  + Window scaling                       (Church & Hanks 1990)
  
 Input format:
   Lemmatized text one word per line. Use symbol ´#´ to set window
   span constraints (i.e. text or paragraph boundaries) and ´_´
   to indicate stop words and lacunae in Cuneiform text.

 Output format:
   Word2vec compatible raw text word vector files.

 TODO:
  - Handle stopwords and lacunae differently from each other
  - Asymmetric co-occurrences for stopwords

 Credits:
  Code cannibalized from Jacob Jungmaier and Omer Levy is credited where
  it is due.

*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`*.*´`* """ 

def get_size(obj, seen=None):

    """ Recursively finds size of objects
        copied from: https://stackoverflow.com/questions/449560/ """

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class CSW:

    """ Context Similarity Weighting as in Sahala & Linden (2020). 
    ´Improving Word Association Measures with Context Similarity Weighting´
    KDIR 2020.

    Improves results in repetitive corpora such as Oracc.

         2020-12-09        Improved memory usage. Uses now lookup instead 
                           of storing all the windows. Still a better 
                           solution is needed for reasonable space 
                           complexity.

                           Lookup indices should be stored as generators 
                           but I haven't yet found an efficient way to do 
                           it because the bigrams would have to be extracted 
                           in sorted order. Using itertools for chaining 
                           generators is not helpful.
                            
                           contexts = [[c1], [c2], ... [cn]]
                           lookup = {(bigram): [i, ...], ...}

                           where i refer to index of the contexts where bigram
                           was seen. 

                           TODO: - removed duplicate contexts
                                 - store bigrams only from a -> b """


    def __init__(self):
        self.contexts = []
        self.context_lookup = defaultdict(list)
        self.s = 0


    @property
    def size(self):
        #a = sum(len(v) for k, v in self.context_lookup.items())\
        #    + sum(len(c) for c in self.contexts)\
        #    + len(self.context_lookup.keys())
        x = max(self.context_lookup.values(), key=len)
        print(len(self.context_lookup.keys()), 'keys')
        print(len(self.contexts), len(set(self.contexts)), 'conts')
        print(len(x))
        print(len(set(x)))
        #print(self.context_lookup.keys())
        print(self.context_lookup[(43, 653)], '\n', self.context_lookup[(653, 43)])
        return 1#get_size(self.contexts) + get_size(self.context_lookup)

        
    def add_lookup(self, bigram, id_, discard):
        """ Add bigram to lookup if the requirements are met

        :param bigram              bigram in (int, int) format
        :param id_                 window index id
        :param discard             set of words to be discarded

        :type bigram               (int, int)
        :type id_                  int
        :type discard              {int, ...} """

        keyword, collocate = bigram
        if collocate < 0:
            return
        if keyword in discard:
            return
        if collocate in discard:
            return
        self.context_lookup[bigram].append(id_)
        return True


    def add_context(self, context):
        """ Add context window to lookup
        :param context            context window
        :type context             [int, ...] """

        self.contexts.append(context)


    def _get_phi(self, contexts):
        """ Return context similarity weight φ over the
        given co-occurrence contexts 

        :param contexts           list of context windows
        :type contexts            [[int, ...], ...]"""
        
        if len(contexts) == 1:
            return 1.0
        else:
            prop_vector = []
            for index in zip(*contexts[::-1]):
                Vi = len(set(index) - set([-1, -2, -3]))
                Wi = len([i for i in index if i >= 0])
                proportion = Vi / max(Wi, 1)
                if proportion > 0:
                    prop_vector.append(proportion)
            if len(prop_vector) == 0:
                return 1
            return sum(prop_vector) / len(prop_vector)


    def compute(self, dim, k_factor=0, verbose=False):
        """ Returns Context similarity Weights as a sparse
        matrix 

        :param dim            vocabulary size
        :param k_factor       CSW k-factor
        
        :type dim             int
        :type k_factor        int / float """
        
        i = 0
        weight_matrix = lil_matrix((dim, dim), dtype=float)
        items = len(self.context_lookup)

        for bigram, indices in self.context_lookup.items():
            w1, w2 = bigram
            contexts = [self.contexts[i] for i in indices]
            d = self._get_phi(contexts) ** k_factor
            weight_matrix[w1, w2] = d
            
            if i % 100000 == 0 and verbose:
                print("   %i / %i" % (i, items), end="\r")
            i += 1

        if verbose:
            print("   %i / %i" % (i, items), end="\r")

        return weight_matrix.tocsr()


class Cooc:

    """ Co-occurrence calculations. Takes in a file and returns a
    Context Similarity Weighted PMI matrix """

    def __init__(self, filename, chunksize=400000, dynamic_window=False,
                 min_count=1, subsampling_rate=0.0, window_size=5,
                 k_factor=3, window_scaling=False, verbose=True):

        """ 
        :param filename            input file
        :param chunksize           lines per chunk
        :param min_count           discard word below this frequency
        :param subsampling_rate    subsampling rate as in word2vec
        :param verbose             print processing info
        :param distance_scaling    take distance in to account in cooc counts
        :param window_size         window size (keyword included)
        :param k_factor            CSW k-factor
        :param window_scaling      scale co-occurrences with window size
        
        :type filename             str
        :type chunksize            int
        :type min_count            int
        :type subsampling_rate     float
        :type verbose              bool
        :type distance_scaling     bool
        :type window_size          int
        :type k_factor             int / float
        :type window_scaling       bool """
    
        self.filename = filename
        self.chunksize = chunksize
        self.min_count = min_count
        self.subsampling_rate = subsampling_rate
        self.verbose = verbose
        self.dynamic_window = dynamic_window
        self.k_factor = k_factor
        self.window_size = window_size
        self.window_scaling = window_scaling
        self.csw = self.k_factor > 0

        self.rand = random.Random(0)
        self.word_count = defaultdict(int)
        self.pad = [-3] * self.window_size 

        self.context_weights = CSW()
        self.window_id = -1

        self.chunks = 0
        self.time = 0

        """ Scale co-occurrence freqs by window size to ensure
        Σ f(a,b) = Σ f(a) = Σ f(b) = N over PMI(*,*). Church & Hanks 1990
        only do this for asymmetric windows. This attempts to generalize
        it for symmetric windows as well. """

        self.scale = 1
        if self.window_scaling:
            if self.window_size > 1:
                self.scale = (self.window_size - 1) * 2
            else:
                self.scale = 2


    @property
    def pmi_matrix(self):
        return self.pmi


    def _read_file(self, only_vocab=False):
        """ Read file line-by-line to save memory. The file is read into chunks
        and spans. Chunks are simply N line long parts of the file and spans 
        are lines, paragraphs, sentences or texts within the chunks. When
        co-occurrences are calculated, the window is not allowed to extend
        from span to another!

        To customize the span symbol, change global variable ´PADDING´

        :param only_vocab            Ignore words that do not satisfy
                                     given constratins such as min_count

        :type only_vocab             bool """
        
        span = []
        chunks = []
        i = 0
        with open(self.filename, encoding='utf-8') as corpus:
            word = corpus.readline().rstrip()
            while word:
                i += 1
                if word != PADDING:
                    if only_vocab:
                        """ Ignore rare words completely if mincount is set;
                        Levy et al. (2015) call this the dirty method """
                        if word in self.vocabulary_set:
                            span.append(self.word_to_id[word])
                    else:
                        span.append(word)
                else:
                    """ append spans into chunks and yield """
                    if i >= self.chunksize:
                        chunks.append(span)
                        yield chunks
                        span = []
                        chunks = []
                        i = 0
                    else:
                        chunks.append(span)
                        span = []
                word = corpus.readline().rstrip()
            chunks.append(span)
            yield chunks        


    def _count_freqs(self):
        """ Count word freqs, make vocabulary and subsampling dictionary """
        spans = 0
        corpus_size = 0

        if self.verbose:
            print(DIV)
            print("> Reading %s..." % self.filename)

        for chunks in self._read_file(only_vocab=False):
            self.chunks += 1
            for span in chunks:
                spans += 1
                for word in span:
                    self.word_count[word] += 1
                    corpus_size += 1

        """ Make vocabulary of words satisfying freq constraints """
        self.vocabulary = [w for w, c in sorted(self.word_count.items(),
                                                key=lambda x:x[1],
                                                reverse=True)
                           if c >= self.min_count]

        """ Represent vocabulary as set for faster filtering """
        self.vocabulary_set = frozenset(self.vocabulary)

        """ Make word->id mapping ordered by word frequency. This allows
        representing the vocabulary as sparse matrix row and column indices
        and saves some memory in CSW """
        self.word_to_id = {w: i for i, w in
                           enumerate(self.vocabulary) if w != LACUNA}

        """ Give negative index for all meta symbols """
        self.word_to_id[STOP] = -2
        self.word_to_id[LACUNA] = -1

        """ Make a set of words that are discarded in CSW, by default
        all words that cannot possibly have more than one co-occurrence with
        any other word (i.e. we want to only save windows that matter) """
        self.discard = frozenset([self.word_to_id[w] for w in self.vocabulary_set
                        if self.word_count[w] < 2])

        """ Prepare subsampling: calculate discard probabilities """
        if self.subsampling_rate:
            ssr = self.subsampling_rate * corpus_size
            self.subsampling_dict = {self.word_to_id[w]: 1-math.sqrt(ssr/c)
                                     for w, c in self.word_count.items()
                                     if c > ssr}

        if self.verbose:
            lacunae = self.word_count.get(LACUNA, 0)
            print("   Corpus statistics:")
            print("      spans       {}".format(spans))
            print("      tokens      {}".format(corpus_size))
            print("      types       {}".format(len(self.vocabulary_set)))
            print("      lacunae     {}".format(lacunae))
            print("      frag. rate: %.2f" % (lacunae/corpus_size))

            if self.subsampling_rate:
                print("> Subsampled words: {}".format(ssr))
                print("> Words in subsampling dictionary: {}"\
                      .format(len(self.subsampling_dict)))


    def _extract_windows(self, chunk):
        """ Extract symmetric oc-occurrence windows from text chunks"""
        wz = self.window_size
        distance = 1

        """ Split input text into windows; note that at this point
        the word are already represented as integers! 0 is the most
        common word 1 the second etc. -2 is a padding and -1 a lacuna. """
        for w in zip(*[chunk[i:] for i in range(1+wz*2)]):
            center = w[wz]
            
            """ Skip paddings, stopwords and lacunae """
            if center < 0:
                continue
                    
            """ If CSW is on, save co-occurrence context """
            if self.csw:
                if center not in self.discard:
                    context = w[0:wz] + w[wz+1:]
                    self.context_weights.add_context(context)
                    self.window_id += 1

            for i, bigram in enumerate(itertools.product([center], w)):
                """ Disallow center word occurring with itself """
                if i == wz:
                    continue

                """ Add window id to lookup if CSW is on. The lookup
                contains all contexts where (a, b) is attested """
                if self.csw:
                    self.context_weights.add_lookup(bigram,
                                                    self.window_id,
                                                    self.discard)
                """ Add co-occurrence to sparse matrix if not stopword,
                lacuna or padding. If dynamic window is set, reduce
                co-occurrence count based on words' mutual distance """
                if bigram[-1] >= 0:
                    self.row.append(center)
                    self.col.append(bigram[-1])
                    if self.dynamic_window:
                        distance = 1 / abs(i - wz)
                    self.data.append(1 * distance)

  
    def count_cooc(self):
        """ This method creates a co-occurrence matrix of
        the given corpus """

        """ Initialize word counts to set dimensions, subsampling,
        word indices etc. """
        st = time.time()
        self._count_freqs()

        """ Initialize vocab x vocab sized sparse co-occurrence matrix """
        dim = len(self.vocabulary)
        self.cooc_matrix = csr_matrix((dim, dim), dtype=int)

        et = time.time() - st
        self.time += et

        if self.verbose:
            print(">   (%.2f seconds)" % (et))
            print(DIV)
            print("> Extracting bigrams...")
        i = 1

        """ Extract windows and count co-occurrences. Read the input text
        second time but discard subsampled and too rare words if min_count
        is set.  """
        ctime = 0
        ttime = 0
        for chunks in self._read_file(only_vocab=True):

            """ Initialize matrix row, col and data """
            self.row = []
            self.col = []
            self.data = []

            st = time.time()
            if self.verbose:
                print('   %i / %i (%.2f)' % (i, self.chunks, ctime), 
                      sep='\t', end='\r')

            """ Skip empty spans (full of lacunae), do subsampling if set and
            pad the windows to equal length for CSW """
            for span in chunks:
                if not span or len(span) == 1:
                    continue

                # Would be faster if done in _read_file()
                if self.subsampling_rate:
                    span = [w for w in span if (w not in self.subsampling_dict
                                or self.rand.random() > self.subsampling_dict[w])]

                padded = self.pad + span + self.pad
                self._extract_windows(padded)

            """ Update co-occurrence matrix """
            self.cooc_matrix += csr_matrix((self.data, (self.row, self.col)),
                                           shape=(dim, dim), dtype=float)
            
            ctime = round(time.time() - st, 3)
            ttime += ctime
            i += 1

        if self.verbose:
            print('> Matrix size: {} ({} kB)'\
                  .format(np.prod(self.cooc_matrix.shape),
                          round(self.cooc_matrix.data.nbytes/1000)))
            print('> Non-zero elements: %i' % (self.cooc_matrix.nnz)) 
            print('    (%.2f seconds)' % (ttime))
            print(DIV)

        self.time += ttime

        """ Calculate CSW and take Hadamard product of the co-occurrence matrix
        and the context similiarity weight matrix. """

        if self.csw:
            st = time.time()
            if self.verbose:
                print('> Calculating context similarities...')
            self.cooc_matrix = self.cooc_matrix.multiply(
                self.context_weights.compute(dim, self.k_factor, self.verbose)
            )

            et = time.time() - st
            if self.verbose:
                print('    (%.2f seconds)' % (et))
                print(DIV)
                # getting the size of the lookup is very slow
                #print(self.context_weights.size)
            del self.context_weights

        self.time += et

    
    def calculate_pmi(self, shift_type=0, alpha=None, lambda_=None, threshold=0):

        """ Calculate Shifted PMI matrix with various modifications

        :param lambda_                  Dirichlet smoothing
        :param alpha                    Context distribution smoothing
        :param threshold                PMI shift value k (see formulae below)
        :param shift_type               set PMI shift type
                                          0: Jungmaier et al. 2020
                                          1: Levy & Goldberg 2014
                                          2: Experimental: linear addition

        :type lambda_                   float (recommended: 0.0001)
        :type alpha                     float (recommended: 0.75)
        :type threshold                 integer (useful values: 0-10)
        :type shift_type                integer (0, 1, 2)
                                          0: if PMI(a,b) < -k, PMI(a,b) = 0
                                          1: max(PMI(a,b) - log2(k), 0)
                                          2: max(PMI(a,b) + k, 0)

        For PMI(*,*) it is more efficient to calculate the scores using
        elementwise matrix multiplication thanks to optimization in 
        numpy/scipy as in Levy et al. 2015. For each element (a,b) in the 
        matrix:

                         p(a,b)                               1
        PMI(a,b) = log2 --------   =  log2 ( N * f(a,b) * -------- )
                        p(a)p(b)                          f(a)f(b)    


        """

        st = time.time()
        if self.verbose:
            print("> Calculating PMI...")

        sum_a = np.array(self.cooc_matrix.sum(axis=1))[:, 0]
        sum_b = np.array(self.cooc_matrix.sum(axis=0))[0, :]

        """ Dirichlet and context distribution smoothing as in

           Jacob Jungmaier (Accessed: 2020-12-01):
              https://github.com/jungmaier/dirichlet-smoothed-word-embeddings/
           Omer Levy (Accessed: 2019-05-30)
              https://bitbucket.org/omerlevy/hyperwords/ """

        if lambda_ is not None:
            if self.verbose:
                print("> Dirichlet smoothing λ={}".format(lambda_))
            sum_a += (lambda_ * self.cooc_matrix.shape[0])
            sum_b += (lambda_ * self.cooc_matrix.shape[0])
            self.cooc_matrix.data = self.cooc_matrix.data + lambda_
        if alpha is not None:
            if self.verbose:
                print("> Context distribution smoothing α={}".format(alpha))
            sum_b = sum_b ** alpha

        sum_total = sum_b.sum()
        self.pmi = csr_matrix(self.cooc_matrix)
        
        """ Scale co-oc frequencies by window size; this has to
        be done after row and column summation """
        if self.scale != 1:
            self.pmi *= self.scale
        
        """ Calculate PMI """
        with np.errstate(divide='ignore'):
            sum_a = np.reciprocal(sum_a)
            sum_b = np.reciprocal(sum_b)

        self.pmi = self.pmi.multiply(sum_a)\
                           .multiply(sum_b[:, None]) * sum_total
        self.pmi.data = np.log2(self.pmi.data)

        """ Apply threshold for Shifted PMI. """
        if threshold is not None:
            if shift_type == 0:
                self.pmi.data[self.pmi.data < -threshold] = 0
            elif shift_type == 1:
                self.pmi.data *= np.log2(threshold)
                self.pmi.data[self.pmi.data < 0] = 0
            elif shift_type == 2:
                self.pmi.data += threshold
                self.pmi.data[self.pmi.data < 0] = 0
    
        et = time.time() - st
        self.time += et

        if self.verbose:
            print('    (%.2f seconds)' % (et))
            print(DIV)


    def factorize(self, dimensions, eigenvalue_weighting=0.0):

        """ Perform truncated SVD for the PMI matrix. 
        
        Adapted from Jungmaier 
        htps://github.com/jungmaier/dirichlet-smoothed-word-embeddings """

        # Do not modify pmi_matrix directly
        m = self.pmi_matrix

        if self.verbose:
            print("> SVD...", end="\r")
        if eigenvalue_weighting == 1:
            svd = TruncatedSVD(n_components=dimensions, random_state=0)
            m = svd.fit_transform(m)
        elif eigenvalue_weighting == 0:
            m, _, _ = randomized_svd(m, n_components=dimensions,
                                 random_state=0)
        else:
            m, s, _ = randomized_svd(m, n_components=dimensions,
                                 random_state=0)
            sigma = np.zeros((m.shape[0], m.shape[1]))
            sigma = np.diag(s**eigenvalue_weighting)
            m = m.dot(sigma)

        if self.verbose:
            print("> Normalizing vectors...", end="\r")
        self.svd_matrix = normalize(m, norm='l2', axis=1, copy=False)
        del m


def save_word_vectors(file_name, embeddings):
    
    """ Adapted from Jungmaier
       https://github.com/jungmaier/dirichlet-smoothed-word-embeddings
                                                 (Accessed: 2020-12-01)
    
    Saves word vectors from a word vector matrix to a text file 
    in word2vec rawtext format.

    :param file_name              vector file name
    :param embeddings             embeddings object

    :type file_name               str
    :type embeddings              Cooc() """

    word_to_id = embeddings.word_to_id
    vocab = word_to_id.keys()
    verbose = embeddings.verbose
    word_vector_matrix = embeddings.svd_matrix

    st = time.time()
    if verbose:
        print("> Saving word vectors for {} most frequent words:"
              .format(len(vocab)))

    with open(file_name, "w") as vector_file:
        vector_file.write(str(word_vector_matrix.shape[0]) + " "
                          + str(word_vector_matrix.shape[1]) + "\n")

        for i, word in enumerate(vocab, start=1):
            vector_file.write(word
                              + " "
                              + " ".join([str(value)
                                          for value
                                          in word_vector_matrix
                                          [word_to_id[word], :]])
                              + "\n")
            if verbose:
                if i % 1000 == 0:
                    print("> {} of {} word vectors saved.".format(i, len(vocab)),
                          end="\r")
                elif i == len(vocab):
                    et = time.time() - st
                    print("> {} of {} word vectors saved."\
                          .format(i, len(vocab)))
                    print('    (%.2f seconds)' % (et))
                    print(DIV)


if __name__ == "__main__":

    """ Modified and extended from Jacob Jungmaier:
           https://github.com/jungmaier/dirichlet-smoothed-word-embeddings
                                                 (Accessed: 2020-12-01) """

    desc = "Calculates word embeddings by using Shifted PPMI \
    (Levy, Dagan & Goldberg 2015), SVD, Dirichlet Smoothing \
    (Jungmaier, Kassner & Roth 2020) and Context Similarity Weighting \
    (Sahala & Linden 2020)."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('corpus_file',
                        help='Text file with word tokens separated \
                              by newlines.')
    parser.add_argument('word_vector_filename',
                        help='Desired name of the word vector file.')
    parser.add_argument('--window_size', '-w', type=int, default=3,
                        help='Size of the window around a word for \
                              co-occurrence counts, integer. Default: 5.')
    parser.add_argument('--cds', type=float, default=None,
                        help='Context Distribution Smoothing\
                        (if used, recommended α = 0.75)')
    parser.add_argument('--window_scaling', action='store_true', default=False,
                        help='Scale co-oc by window size')
    parser.add_argument('--dynamic_window', action='store_true', default=False,
                        help='Dynamic Window as in GloVE and Word2vec')
    parser.add_argument('--k_value', '-k', type=float, default=3.0,
                        help='CSW k-value')
    parser.add_argument('--min_count', '-m', type=int, default=1,
                        help='Minimal word count for words to process, \
                              integer, useful range 1-10. Default: 1.')
    parser.add_argument('--subsampling', '-s', type=float, default=0.0,
                        help='Subsampling rate (similar to word2vec). \
                              Default: 0.0 (no subsampling).')
    parser.add_argument('--chunk_size', '-c', type=int, default=400000,
                        help='Chunk size in bytes for chunkwise processing \
                              of the corpus. Default: 3000000. If memory \
                              overflow encountered, choose lower value.')
    parser.add_argument('--dimensions', '-d', type=int, default=300,
                        help='Size of word embeddings, integer. Default: 300.')
    parser.add_argument('--smoothing_factor', '-l', type=float, default=None,
                        help='Smoothing factor lambda, float, \
                              useful range: 0-1. Default: 0.0001.')
    parser.add_argument('--threshold', '-t', type=int, default=None,
                        help='Shifted PPMI threshold:\
                              Default: 0 -> PPMI.')
    parser.add_argument('--shift_type', type=int, default=0,
                        help='PMI matrix shift type. \
                        0 = replace PMI(a,b) < -k with 0, \
                        1 = max(PMI(a,b) - log(k), 0), \
                        2 = max(PMI(a,b) + k, 0).')
    parser.add_argument('--eigenvalue_weighting', '-e', type=float,
                        default=0.0, help='Weighting of singular values \
                                           (Sigma), float, useful range: 0-1. \
                                           Default: 0.0 (singular values are \
                                           not considered).')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output.')
    args = parser.parse_args()


    print(args)

    """ Init embeddings object """
    embeddings = Cooc(args.corpus_file,
                      chunksize=args.chunk_size,
                      window_size=args.window_size,
                      min_count=args.min_count,
                      subsampling_rate=args.subsampling,
                      k_factor=args.k_value,
                      dynamic_window=args.dynamic_window,
                      window_scaling=args.window_scaling,
                      verbose=args.verbose)

    """ Make CSW and Cooc matrix """
    embeddings.count_cooc()

    """ Make PMI matrix """
    embeddings.calculate_pmi(shift_type=args.shift_type,
                             alpha=args.cds,
                             lambda_=args.smoothing_factor,
                             threshold=args.threshold)

    """ Factorize sparse matrix  """
    embeddings.factorize(dimensions=args.dimensions,
                         eigenvalue_weighting=args.eigenvalue_weighting)

    """ Save word vectors from embeddings object """
    save_word_vectors(file_name=args.word_vector_filename,
                      embeddings=embeddings)
