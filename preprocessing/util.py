import git
import json
import re
import pickle
import time
import enum
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim import models, corpora
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',200)



def to_token(line, useful_token = None):
    final_token = []   # 最后的token序列
    lmtzr = WordNetLemmatizer()
    stopwords_en = stopwords.words('english')
    
    tokens = re.split('[^0-9a-zA-Z]+', line)
    ret = []
    for token in tokens:
        ret.extend(line_to_tokens(token))
    # tokens = line_to_tokens(line)
    if useful_token:
        for token in ret:
            token = token.lower()                
            if token not in useful_token:          
                continue
            token = lmtzr.lemmatize(token, 'v')   
            final_token.append(token) 
    else:
        for token in ret:
            token = token.lower()                
            if token in stopwords_en:          
                continue
            token = lmtzr.lemmatize(token, 'v')   
            final_token.append(token) 

    return final_token

def eval_counter(df_series):
    new_series = []
    for i in df_series:
        new_series.append(eval(i))
    return new_series

class StateType(enum.IntEnum):
    INITIAL_STATE = 0
    UPPERCASE_STATE = 1
    LOWERCASE_STATE = 2
    NUMBER_STATE = 3
    SPECIAL_STATE = 4
    
    
def line_to_tokens(code):
    """
    采用简单的字符类型的方式将代码进行切分
    upper | upper lower
    upper | number
    upper | special
    lower | upper
    lower | number
    lower | special
    number | upper
    number | lower
    number | special
    special | upper
    special | lower
    special | number
    结果示例："foo  ,1" -> ["foo", "  ", ",", "1"]
    """
    # normal state transitions that will result in splitting
    normal_transitions = [
      (StateType.UPPERCASE_STATE, StateType.NUMBER_STATE),
      (StateType.UPPERCASE_STATE, StateType.SPECIAL_STATE),
      (StateType.LOWERCASE_STATE, StateType.UPPERCASE_STATE),
      (StateType.LOWERCASE_STATE, StateType.NUMBER_STATE),
      (StateType.LOWERCASE_STATE, StateType.SPECIAL_STATE),
      (StateType.NUMBER_STATE, StateType.UPPERCASE_STATE),
      (StateType.NUMBER_STATE, StateType.LOWERCASE_STATE),
      (StateType.NUMBER_STATE, StateType.SPECIAL_STATE),
      (StateType.SPECIAL_STATE, StateType.UPPERCASE_STATE),
      (StateType.SPECIAL_STATE, StateType.LOWERCASE_STATE),
      (StateType.SPECIAL_STATE, StateType.NUMBER_STATE)]
    # output, state
    tokens = []
    state = StateType.INITIAL_STATE
    next_state = None
    memory = []
    for i, inputchar in enumerate(code):
        if inputchar.isupper():
            next_state = StateType.UPPERCASE_STATE
        elif inputchar.islower():
            next_state = StateType.LOWERCASE_STATE
        elif inputchar.isdigit():
            next_state = StateType.NUMBER_STATE
        else:
            next_state = StateType.SPECIAL_STATE

    # splitting cases
        if (state, next_state) in normal_transitions:
            tokens.append(''.join(memory))   # TheShape  -  The存储起来 Shape
            memory = []
        elif (state, next_state) == (StateType.UPPERCASE_STATE,
                                 StateType.LOWERCASE_STATE) and len(memory) > 1: # VSShape - VS Shape
            tokens.append(''.join(memory[:-1]))  
            memory = [memory[-1]]
        elif (state, next_state) == (StateType.SPECIAL_STATE,
                                 StateType.SPECIAL_STATE):
            if inputchar in [' ', '\t'] and inputchar == code[i-1]:   # 如果是空格或者\t 并且前一个字符也是，那么
                if len(memory) >= 20:   # 如果长度大于20，直接生成一个token
                    tokens.append(''.join(memory))
                    memory = []
            elif inputchar.isspace() or code[i-1].isspace(): # 如果是空格直接停止
                tokens.append(''.join(memory))
                memory = []

    # put inputchar into memory, always
        memory.append(inputchar)
        state = next_state
    if memory:
        tokens.append(''.join(memory))
    return tokens


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def funcs_preprocess(item):
    keyword = ['auto', 'double', 'int', 'struct', 'break', 'else', 'long', 'switch', 
          'case', 'enum', 'register', 'typedef', 'char', 'extern', 'return', 'union', 
          'const', 'float', 'short', 'unsigned', 'continue', 'for', 'signed', 'void',
          'default', 'goto', 'sizeof', 'volatile', 'do', 'if', 'while', 'static']
    ret = re.split(r'[^0-9a-zA-Z]', item)
    ret = list(set(ret))
    ret = [item for item in ret if item and item not in keyword]
    return ' '.join(ret)



def string_preprocess(ret):
    ret = ret.replace(r"\r\n", ' ').replace(r"\n", ' ').replace(r"\r", ' ')
    ret = re.sub(r' +', ' ', ret) 
    return ret


def savefile(data, path):
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()
    
    
def readfile(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


    
def As_in_B(As: list, B:str):
    """
    获取A列表中元素item有多少出现在字符串B中
    """
    cnt = 0
    for A in As:
        if A in B:
            cnt += 1
    return cnt


def re_search(query: str, item: str):
    """
    正则匹配，有返回List，没有返回None
    """
    return re.findall(query, item)


def inter_token(*array_list):
    arr = array_list[0]
    for array in array_list:
        arr = arr & array
    return arr

def union_list(*items):
    ret = []
    for item in items:
        ret.extend(item)
    return ret




