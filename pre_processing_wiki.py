import re
import jieba

jieba.setLogLevel(20)
zh_pattern = re.compile('[\u4e00-\u9fa5]')  # chinese word pattern


def chinese_in_string(input_string):
    return zh_pattern.search(input_string)

def file2sent(file_path:str):            # return a list(file) of list(sent)
    text = str(open(file_path, 'r').read())
    text = re.sub("<.+?>", '', text)        # delete html tags
    text = re.sub("\(.+?\)", '', text)  # delete html tags
    # text = re.sub("\d{4}\d+", '9999', text)         # convert all numbers > 9999 to 9999
    sents = re.findall("[\w,\d]+", text)    # break text to sentences
    sents = [s for s in sents if zh_pattern.search(s)]  # only consider sentence who have chinese word
    sents = [jieba.lcut(s) for s in sents]  # cut sent via jieba

    # print(sents)
    return sents


if __name__ == '__main__':
    sents = file2sent('/nlp_project/big_files/wiki_corpus/wiki_chs/AG/wiki_42')
    # for s in sents:
    #     print(''.join(s))
    print('avg len of', len(sents), 'sents:', sum(len(s) for s in sents) / len(sents))