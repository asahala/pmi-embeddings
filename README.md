# pmi-embeddings
State-of-the-art count-based word embeddings for low-resource languages with a special focus on historical languages, especially Akkadian and Sumerian.

1. ```src/make_vectors.py``` creates PMI-based word embeddings from the source corpus.
2. ```src/hypertune.py``` tests hyperparameters (by using brute force) to find the best settings for the given data set (requires a gold standard).
3. ```corpora/akkadian.zip``` a zipped test corpora of Akkadian language.
3. ```eval/gold.tsv``` an initial version of the [Akkadian gold standard](https://www.helsinki.fi/en/news/language-culture/creating-a-gold-standard-for-akkadian-word-embeddings).

## Requirements
```Python 3.6``` or newer, [numpy](https://numpy.org/), [scipy](https://www.scipy.org/) and [sklearn](https://scikit-learn.org/stable/). For evaluation scripts you will also need [gensim](https://pypi.org/project/gensim/). Tested on Linux and Windows 7/10.

## Features
```make_vectors.py``` is a fast and efficient way to build PMI-based word embeddings from small (a few million words) text corpora. It combines findings from several research papers:

+ Dirichlet Smoothing (Jungmaier et al. 2020)
+ Context Similarity Weighting (Sahala & Linden 2020)
+ Shifted PMI (Levy et al. 2015) with experimental variants
+ Dynamic Context Windows (Sahlgren 2006)
+ Subsampling (Mikolov et al. 2013)
+ Context Distribution Smoothing (Levy et al. 2015)
+ Eigenvalue Weighting (Caron 2001)
+ Window Scaling (Church & Hanks 1990)

### Input format
Lemmatized UTF-8 endoded text one word per line. Use symbol ´#´ to set window span constraints (i.e. text or paragraph boundaries) and ´_´ to indicate stop words and lacunae in cuneiform texts.

### Output format
Word2vec compatible raw text word vectors.

### Parameters and usage
Run the script from the commmand line ```python3 make_vectors.py corpusfile vectorfile [parameters]```. See [this document](https://docs.google.com/document/d/1TjVWqrhalCDjkOQf-JLk1jmC6N83MWGUIEVjbJpm9Es) for detailed information about the parameters and references. 

### Performance
On Intel i5-2400 3.10GHz using a corpus of 1M words and a window size of 3, takes ca. 35 seconds to make basic embeddings and 50 seconds to make CSW-embeddings. Although this is quite fast, it is good to keep in mind when using ```hypertune.py```.

## Todo
+ Clean and dirty stopword handling
+ Add corpus pre-processing scripts
+ Jypyter API


