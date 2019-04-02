from pre_processing_wiki import file2sent
from collections import Counter, defaultdict
from functools import partial, reduce
import multiprocessing
from tqdm import tqdm

corpus_list = []  # type: List[str]


def load_corpus_list(path: str):
    global corpus_list
    with open(path, 'r') as f:
        corpus_list = f.read().split('\n')


def get_ngram(sents: list, n: int) -> Counter:
    ngram = []
    for sent in sents:
        ngram += [tuple(sent[i - n:i]) for i in range(len(sent) + 1) if i >= n]
    return Counter(ngram)


def multi_processing(batch_file: list, mapper: callable, reducer: callable) -> list:
    pool = multiprocessing.Pool()
    ngram_counts = pool.map(mapper, batch_file)
    pool.close()
    pool.join()
    return reduce(reducer, ngram_counts)


def list_split(data: list, batch_size: int) -> List[list]:
    return [data[x:x+batch_size] for x in range(0, len(data), batch_size)]


def get_ngram_batch(input_list: list, batch_size: int = 252) -> Counter:
    ngram_freq = Counter()
    corpus_batch = list_split(input_list, batch_size)

    for batch in tqdm(corpus_batch):
        sents = multi_processing(batch, file2sent, lambda x,y: x+y)
        for n in [1, 2, 3]:
            ngram_freq += Counter({k: v for k, v in get_ngram(sents, n).items() if v > 1})

    return ngram_freq


# ------------------------------------------------------------------------------
# save the results on disk
import pickle


def save_obj(obj, file_name):
    pickle.dump(obj, open(file_name, 'wb'))


def load_obj(file_name):
    obj = pickle.load(open(file_name, 'rb'))


if __name__ == '__main__':
    corpus_name = 'input_corpus.txt'
    load_corpus_list(corpus_name)
    ngram_freq = get_ngram_batch(corpus_list)
    save_obj(ngram_freq, 'output/wiki_gram_123.model')


