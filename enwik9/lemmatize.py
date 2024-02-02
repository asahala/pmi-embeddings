import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

""" Generator-based lemmatizer for large Enwik9 chunks.

Might take a long time to run, but should not crash due to
memory error.

asahala 2023 """

def lemmatize_data(text):
    """ Lemmatize untokenized text using NLTK
    :param text      input text
    :type text       str """

    d = []
    itot = len(word_tokenize(text))
    i = 0
    for word, tag in pos_tag(word_tokenize(text)):

        if i % 1000 == 0:
            print(i, itot)
    
        wntag = tag[0].lower()
        wntag = wntag if wntag in ('a', 'r', 'n', 'v') else None
        if not wntag:
            lemma = word
        else:
            lemma = wnl.lemmatize(word, wntag)

        d.append(lemma)

        i += 1

    return d


def lemmatize_large_data(text):
    """ Lemmatize untokenized text using NLTK
    :param text      input text
    :type text       stt """

    #d = []
    #itot = len(word_tokenize(text))
    #i = 0

    chunk = []
    part = 0
    for e, w in enumerate(text, start=1):

        chunk.append(w)
        if e % 50000 == 0:
            with open(f'chunks/enwik9-2M-{str(part).zfill(3)}.txt', 'a', encoding='utf-8') as output:
                print(f'Processing chunk {e}')
                for word, tag in pos_tag(chunk):
                    wntag = tag[0].lower()
                    wntag = wntag if wntag in ('a', 'r', 'n', 'v') else None
                    if not wntag:
                        lemma = word
                    else:
                        lemma = wnl.lemmatize(word, wntag)
                    output.write(lemma + '\n')
                #print(f'Written to output {len(chunk)}')
                chunk = []
                #output.write('#' + '\n')

        if e % 2000000 == 0:
            #output.write('<EOF>\n')
            part += 1
            
                  
def parse_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().splitlines():
            yield line

def parse_large_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield line[:-1]

def write_data(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')
        #f.write('#\n')

text = parse_large_data('data/enwik9-40M.txt')
lemmatize_large_data(text)



'''
print('parsing')
text = ' '.join(list(parse_data('data/enwik9-40M.txt')))
print('lemmatizing')
data = lemmatize_data(text)
print('writing')
write_data('data/enwik9-40M-part1-lem.txt', data)
    
'''


