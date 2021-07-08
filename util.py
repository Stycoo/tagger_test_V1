'''
Main Task of This File: str2token + token2id
'''
import numpy as np
import re
import jieba

def normalization(text):
    '''
    Main Purpose: data cleaning (illegal chars & sensitive words)
    :param text: string
    :return: cleaned string
    '''
    pattern = re.compile("")
    # unfinished

def str2token(text, task_type)->list:
    # currently, we don't consider CWS
    if task_type == 'POS':
        tokens = list(jieba.cut(text))
    elif task_type == 'NER':
        tokens = list(text)
    else:
        raise Exception("Wrong task type input! (NER OR POS)")
    return tokens

def token2id(words, word2id, max_seq_len):
    def func(word):
        if word in word2id:
            return word2id[word]
        else:
            return word2id['<unk>']

    ids = [func(word) for word in words]

    if len(ids) >= max_seq_len:
        ids = ids[:max_seq_len]
        ids = np.asarray(ids).reshape([-1, max_seq_len])
        return ids
    else:
        ids.extend([0] * (max_seq_len - len(ids)))
        ids = np.asarray(ids).reshape([-1, max_seq_len])
        return ids