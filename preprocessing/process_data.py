import os
import time
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from util import *

import re
import gc
import os
import time
import re
import logging
import math
import string
import warnings
import git
import string
import json
from glob import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from nltk.corpus import stopwords
from collections import Counter
from util import *
warnings.filterwarnings("ignore")

def get_commit_tokens(reponame, token_path, gitlogpath, pool_num = 5):
    t1 = time.time()
    tokens = set()
    print(reponame)
    with open(gitlogpath+'Log_{}.txt'.format(reponame), 'r',errors='ignore') as fp:
        stop = False
        #  avoid its too large
        while not stop:
            lines = []
            for i in range(10000000):
                line = fp.readline()
                if line == '':
                    stop = True
                    break
                lines.append(line)
                break
            token = multi_process_line(lines, pool_num)
            tokens.update(token)
            stop = True
            
    t2 = time.time()
    # save tokens to file
    with open(token_path+'tokens_{}.txt'.format(reponame), 'w+') as fp:
        fp.write(str(tokens))
        
def multi_process_line(lines, pool_num = 5):
    with Pool(pool_num) as p:
        res = list(
            tqdm(p.imap(to_token, lines), total=len(lines),
                 desc='process'))
        p.close()
        p.join()
    ret = set()
    for item in res:
        ret.update(item)
    return ret


def get_tokens_from_path(path):
    with open(path, 'r',  errors='ignore') as fp:
        lines = fp.readlines()
    return set(multi_process_line(lines))


# ================= get code data =================
def tokenize(item, stopword_list):
    with open("../data/tokens/"+'useful_tokens.txt', 'r') as fp:
        useful_tokens = eval(fp.read())
    return [token for token in to_token(item, useful_tokens) if token not in stopword_list and len(token) > 1]



def get_code_info(repo, commit):
    outputs = read_code_diff(repo, commit).split('\n')
    files, filepaths, funcs = [], [], []
    token_list = []
    for line in outputs:
        if line.startswith('diff --git'):
            line = line.lower()
            files.append(line.split(' ')[-1].strip().split('/')[-1])
            filepaths.append(line.split(" ")[-1].strip())
        elif line.startswith('@@ '):
            line = line.lower()
            funcs.append(line.split('@@')[-1].strip())
        elif (line.startswith('+') and not line.startswith('++')) or (
                line.startswith('-') and not line.startswith('--')):
            line = line.lower()
            token_list.extend(tokenize(line[2:], stopword_list))
    token_counter = Counter(token_list)
    return [commit, files, filepaths, funcs, token_counter]


def mid_func(item):
    return get_code_info(*item)


def multi_process_code(repo, commits, poolnum=5):
    length = len(commits)
    with Pool(poolnum) as p:
        ret = list(
            tqdm(p.imap(mid_func, zip([repo] * length, commits)),
                 total=length,
                 desc='get commits info'))
        p.close()
        p.join()
    return ret
# =================preprocess_data.py=================

def re_func(item):
    res = []
    find = re.findall("(([a-zA-Z0-9]+_)+[a-zA-Z0-9]+.{2})", item)
    for item in find:
        item = item[0]
        if item[-1] == ' ' or item[-2] == ' ':
            res.append(item[:-2])

    find = re.findall("(([a-zA-Z0-9]+_)*[a-zA-Z0-9]+\(\))", item)
    for item in find:
        item = item[0]
        res.append(item[:-2])

    find = re.findall("(([a-zA-Z0-9]+_)*[a-zA-Z0-9]+ function)", item)
    for item in find:
        item = item[0]
        res.append(item[:-9])
    return res

def re_filepath(item):
    res = []
    find = re.findall(
        '(([a-zA-Z0-9]|-|_|/)+\.(cpp|cc|cxx|cp|CC|hpp|hh|C|c|h|py|php|java))',
        item)
    for item in find:
        res.append(item[0])
    return res


def re_file(item):
    res = []
    find = re.findall(
        '(([a-zA-Z0-9]|-|_)+\.(cpp|cc|cxx|cp|CC|hpp|hh|C|c|h|py|php|java))',
        item)
    for item in find:
        res.append(item[0])
    return res

def get_tokens(text, List):
    return set([item for item in List if item in text])

# =================preprocess_data.py   get_commit_data (226) =================
def get_info(repo, commit):
    reponame = repo.remotes.origin.url.split('.git')[0].split('/')[-1]

    outputs = read_code_diff(reponame, commit).split('\n')

    temp_commit = repo.commit(commit)
    # data to be collected
    weblinks, bug, issue, cve = [], [], [], []
    filepaths, funcs = [], []
    addcnt, delcnt = 0, 0
    # get commit message
    mess = temp_commit.message
    # get weblink bugID issueID cveID
    link_re = r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    weblinks.extend(re.findall(link_re, mess))
    bug.extend(re.findall('[bB]ug[^0-9]{0,5}([0-9]{1,7})[^0-9]', mess))
    issue.extend(re.findall('[iI]ssue[^0-9]{0,5}([0-9]{1,7})[^0-9]', mess))
    cve.extend(re.findall('[CVEcve]{3}-[0-9]+-[0-9]+', mess))
    # get commit time
    datetime = pd.Timestamp(temp_commit.committed_date, unit='s')
    datetime = '{:04}{:02}{:02}'.format(datetime.year, datetime.month,
                                        datetime.day)

    for line in outputs:
        # get weblink bugID issueID cveID in code diff
        weblinks.extend(re.findall(link_re, line))
        bug.extend(re.findall('[bB]ug[^0-9]{0,5}([0-9]{1,7})[^0-9]', line))
        issue.extend(re.findall('[iI]ssue[^0-9]{0,5}([0-9]{1,7})[^0-9]', line))
        cve.extend(re.findall('[CVEcve]{3}-[0-9]+-[0-9]+', line))
        # get filepaths and funcnames in code diff
        # get added and deleted lines of code
        if line.startswith('diff --git'):
            filepath = line.split(' ')[-1].strip()[2:]
            filepaths.append(filepath)
        elif line.startswith('@@ '):
            funcname = line.split('@@')[-1].strip()
            funcname = funcs_preprocess(funcname)
            funcs.append(funcname)
        else:
            if line.startswith('+') and not line.startswith('++'):
                addcnt = addcnt + 1
            elif line.startswith(
                    '-') and not line.startswith('--'):
                delcnt = delcnt + 1

    return set(weblinks), set(bug), set(issue), set(cve), datetime, set(
        filepaths), set(funcs), addcnt, delcnt


def get_commit_info(data):
    out = get_info(data[0], data[1])
    return (data[1], out)
# get_commit_token

def multi_process_get_commit_info(repo, commits, number_of_works = 5):
    length = len(commits)
    with Pool(number_of_works) as p:
        ret = list(
            tqdm(p.imap(get_commit_info, zip(*([repo] * length, commits))),
                 total=length,
                 desc='get commits info'))
        p.close()
        p.join()
    return ret

#######################  tokens.py get commit tokens #######################
def get_commit_token(input):
    reponame, commit = input
    repo = git.Repo(gitrepo_path + reponame)
    if not os.path.exists(gitcommit_path+'{}/{}'.format(reponame, commit)):
        try:
            temp_commit = repo.commit(commit)
            mess = temp_commit.message.replace('\r\n', ' ').replace('\n', ' ')
            mess = to_token(mess, useful_tokens)
        except:
            commit_meta_data = json.load(open('../data/commit_meta/{}/{}.json'.format(reponame, commit)))
            mess = commit_meta_data['message'].replace('\r\n', ' ').replace('\n', ' ')
            mess = to_token(mess, useful_tokens)

        filepaths, funcs,  codes = [], [], []
        outputs = read_code_diff(reponame, commit).split('\n')
        for line in outputs:
            if line.startswith('diff  --git'):
                filepath = line.split(' ')[-1].strip()[2:] 
                filepaths.extend(to_token(filepath, useful_tokens))
            elif line.startswith('@@ '):
                funcname = line.split('@@')[-1].strip()
                funcname = funcs_preprocess(funcname)
                funcs.extend(to_token(funcname, useful_tokens))
            else:
                
                line_tokens = to_token(line[1:], useful_tokens)
                codes.extend(line_tokens)

        total_token = union_list(mess, filepaths, funcs, codes)
        with open(gitcommit_path+'{}/{}'.format(reponame, commit), 'w') as fp:
            fp.write(str(total_token))
    return None


def multi_process_get_commit_token(repo, commits, number_of_workers):
    length=len(commits)
    with Pool(number_of_workers) as p: 
        list(tqdm(p.imap(get_commit_token, zip([repo for i in range(length)], commits)) ,total=length,desc='多进程处理commits'))
        p.close() 
        p.join()

### ==================  tokens.py IDF(line 200)  ================== ###

def get_vuln_token(vuln_token):
    token_dict = {}
    for token in useful_tokens:
        token_dict[token] = 0
    for token in tqdm(vuln_token, ncols=80, desc='执行任务' + ' pid:' + str(os.getpid())):
        for item in token:
            if item in useful_tokens:
                token_dict[item] += 1
    return token_dict


def multi_process_get_vuln_token(vuln_tokens, number_of_workers):
    length = len(vuln_tokens)
    with Pool(number_of_workers) as p:
        result = list(p.imap(get_vuln_token, vuln_tokens))
        p.close()
        p.join()
    return result


def load_commit_token(filepaths):
    token_dict = dict()
    for token in useful_tokens:
        token_dict[token] = 0

    for file in tqdm(filepaths, ncols=80, desc='执行任务' + ' pid:' + str(os.getpid())):
        with open(file, 'r') as fp:
            commit_token = set(eval(fp.read()))
        for item in commit_token:
            try:
                token_dict[item] += 1
            except:
                pass
        del commit_token
        gc.collect()
    return token_dict

def multi_process_load_commit_token(file_list, number_of_workers):
    length = len(file_list)
    with Pool(number_of_workers) as p:
        result = list(p.imap(load_commit_token, file_list))
        p.close()
        p.join()
    return result

def read_code_diff(reponame, commit):

    if os.path.exists(code_diff_path2):
        code_diff = pickle.load(open(code_diff_path2, 'rb'))
        return code_diff
    elif os.path.exists(code_diff_path1):
        with open(code_diff_path1, 'r') as f:
            code_diff = f.read()
        return code_diff
    else:
        return ''


def re_bug(item):
    find = re.findall('bug.{0,3}([0-9]{2, 5})', item)
    return set(find)


def re_cve(item):
    return set(re.findall('(cve-[0-9]{4}-[0-9]{1,7})', item))


def token(item, stopword_list):
    return [token for token in to_token(item, useful_tokens) if token not in stopword_list and len(token) > 1]

if __name__ == '__main__':


    ''' 
    generate the useful_token 

    '''
    basepath = '../preprocessing'
    gitrepo_path = "../gitrpepo"
    tokenpath = "../data/tokens/"
    gitlogpath = "../data/gitlog/"
    gitcommit_path = '../data/gitcommit/'
    code_data_path = '../data/code_data'
    commit_info_path = '../data/commit_info'

    file_path_lst = [tokenpath, gitlogpath, gitcommit_path, code_data_path]

    for i in file_path_lst:
        if not os.path.exists(i):
            os.makedirs(i)

    number_of_workers = mp.cpu_count()
    
    data_df = pd.read_csv('../data/data.pkl')
    
    repos = data_df.repo.unique()
    print("-------- Start Get Token From {} Repos --------".format(len(repos)))
        
    print('The number of positive samples is {}'.format(len(data_df)))
    
    for reponame in repos:
        t1 = time.time()
        logpath = gitlogpath+"Log_{}.txt".format(reponame)
        savefile_name = tokenpath+"tokens_{}.txt".format(reponame)
        if not os.path.exists(logpath) and not os.path.exists(savefile_name):
            print("-------- Start get git log : {} {} --------".format(reponame, repos.tolist().index(reponame)))
            os.chdir(gitrepo_path + reponame)
            os.system('git log -p --color=never > '+logpath)
            get_commit_tokens(reponame, tokenpath, gitlogpath, pool_num=number_of_workers)
            os.remove(logpath)
        t2 = time.time()
    os.chdir(basepath)
    
    print("-------- Start get commit info --------")
    
    print("-------- Start generate useful tokens --------")
    
    '''
    获取repo中的token, 计算useful and unuseful token.
    '''
    paths = [tokenpath+'tokens_{}.txt'.format(reponame) for reponame in repos]
    
    print(f'Get {len(paths)} tokens files!')
    
    commit_tokens = set()
    
    for path in tqdm(paths):
        with open(path, 'r') as fp:
            commit_token = eval(fp.read())
            commit_tokens.update(commit_token)
        # remove unuse file
        os.remove(path)
        
    with open(tokenpath+'tokens_commit.txt', 'w')  as fp:
        fp.write(str(commit_tokens))

    vuln_token = set()
    vuln_token.update(get_tokens_from_path("../data/vuln_data.csv"))
    with open(tokenpath+'/tokens_vuln.txt', 'w+') as fp:
        fp.write(str(vuln_token))
        
    print(f'vuln tokens: {len(vuln_token)}')
       
    ### get useful token
    total_tokens = vuln_token | commit_tokens
    useful_tokens = vuln_token & commit_tokens
    unuseful_tokens = total_tokens - useful_tokens
    print('total words: ',len(total_tokens))
    print('total useful words: ',len(useful_tokens))
    print('total useful words',len(unuseful_tokens))
    
    with open(tokenpath+'useful_tokens.txt', 'w+') as fp:
        fp.write(str(useful_tokens))
    with open(tokenpath+'unuseful_tokens.txt', 'w+') as fp:
        fp.write(str(unuseful_tokens))

    '''========='''    


    print("load tokens")
    with open(tokenpath+'useful_tokens.txt', 'r') as fp:
        useful_tokens = eval(fp.read())

    with open(tokenpath+'unuseful_tokens.txt', 'r') as fp:
        unused_tokens = eval(fp.read())
    
    print(f'useful tokens: {len(useful_tokens)}')
    with open("../data/vuln_type_impact.json", 'r') as f:
        vuln_type_impact = json.load(f)

    vuln_type = set(vuln_type_impact.keys())
    vuln_impact = set()
    for value in vuln_type_impact.values():
        vuln_impact.update(value)

    stopword_list = stopwords.words('english') + list(string.punctuation)

    print('========================== Start Processing CVE Data Description to Get Elements ==========================')
    
    vuln_data = pd.read_csv('../data/vuln_data.csv', encoding = 'latin-1')
    vuln_data['functions'] = vuln_data['desc'].apply(re_func)
    vuln_data['files'] = vuln_data['desc'].apply(re_file)
    vuln_data['filepaths'] = vuln_data['desc'].apply(re_filepath)
    vuln_data['vuln_type'] = vuln_data['desc'].apply(lambda item: get_tokens(item, vuln_type))
    vuln_data['vuln_impact'] = vuln_data['desc'].apply(lambda item: get_tokens(item, vuln_impact))
    vuln_data['desc_token'] = vuln_data['desc'].apply(lambda item: tokenize(item, stopword_list))
    vuln_data['desc_token_counter'] = vuln_data['desc_token'].apply(lambda item: Counter(item))
    vuln_data.to_csv('../{}/data/vuln_data.csv'.format(data_folder), index=False)


    # ====================get code data====================

    print("# ====================get code data====================\n")

    dataset = pickle.load(open('../data/Dataset.pkl', 'rb'))
    dataset = reduce_mem_usage(dataset)
  
    print("the shape of dataset is {}".format(dataset.shape))
    
    for reponame in repos:
        savepath = code_data_path+'/code_data_' + reponame + '.csv'
        if os.path.exists(savepath):
            continue
        print('\n' + reponame+" Processing...")
        commits = dataset[dataset.repo == reponame].commit.unique()
        code_data = multi_process_code(reponame, commits, number_of_workers)
        code_df = pd.DataFrame(
            code_data,
            columns=['commit', 'code_files', 'code_filepaths', 'code_funcs', 'code_token_counter'])
        
        code_df.to_csv(savepath, index=False)
    
    ''' ======================================'''


    '''mess_df'''

    print('======== Processing commit message data ========\n')

    commit_mess_data = []
    for reponame in repos:
        repo = git.Repo(gitrepo_path + '/' + reponame)
        df_tmp = dataset[dataset.repo == reponame]
        for commit in tqdm(df_tmp.commit.unique()):
            mess = repo.commit(commit).message.lower()
            type_set = set()
            for value in vuln_type:
                if value in mess:
                    type_set.add(value)
            impact_set = set()
            for value in vuln_impact:
                if value in mess:
                    impact_set.add(value)
            
            bugs = re_bug(mess)
            cves = re_cve(mess)
            mess_token = token(mess, stopword_list)

            commit_mess_data.append([
                commit, bugs, cves, type_set, impact_set,
                Counter(mess_token)
            ])

    commit_mess_data = pd.DataFrame(commit_mess_data,
                                    columns=[
                                        'commit', 'mess_bugs', 'mess_cves',
                                        'mess_type', 'mess_impact',
                                        'mess_token_counter'
                                    ])
    commit_mess_data.to_csv('../data/mess_data.csv', index=False)
    

    print("======== Processing commit data ========\n")

    if not os.path.exists(commit_info_path):
        os.mkdir(commit_info_path)

    for reponame in repos:
        save_file = commit_info_path + '/' + reponame + '_commit_info'
        if not os.path.exists(save_file):
            repo = git.Repo(gitrepo_path + reponame)
            commits = dataset[dataset.repo == reponame].commit.unique()
            result = multi_process_get_commit_info(repo, commits, number_of_workers)
            savefile(dict(result), commit_info_path + '/' + reponame + '_commit_info')
        else:
            continue

    
    '''======================================================='''
    vuln_data['cwedesc'] = vuln_data['cwedesc'].apply(lambda item:to_token(item, useful_tokens))
    vuln_data['desc'] = vuln_data['desc'].apply(lambda item:to_token(item, useful_tokens))
    vuln_data['total'] = vuln_data['cwedesc'] + vuln_data['desc']

    repo_commits = pickle.load(open('../data/repo_commit.pkl', 'rb'))

    for reponame in repos:
        logging.info(reponame+ '正在处理....')
        print(reponame+ '正在处理....')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        repo_gitcommit_path = gitcommit_path+reponame
        if not os.path.exists(repo_gitcommit_path):
            os.makedirs(repo_gitcommit_path)

        repo = git.Repo(gitrepo_path + reponame)
        multi_process_get_commit_token(reponame, repo_commits[reponame], number_of_workers)
    
    print('==================== IDF ====================') 
    files = glob('../data/gitcommit/*/*')
    length = len(files)
    print('total nunber of commit files: ', length)
    file_lst = []
    for i in range(number_of_workers):
        tmp = files[math.floor(i / number_of_workers * length)
                                :math.floor((i + 1) / number_of_workers * length)]
        file_lst.append(tmp)
    
    result = multi_process_load_commit_token(file_lst, number_of_workers)
    
    token_dict = {}
    for token in useful_tokens:
        token_dict[token] = 0
        
    for dic in tqdm(result):
        for token in dic.keys():
            token_dict[token] += dic[token]


            
    vuln_tokens = list(vuln_data['total'])
    
    length_vuln = len(vuln_tokens)
    
    vuln_commit = []
    for i in range(number_of_workers):
        tmp = vuln_tokens[math.floor(
            i / number_of_workers * length_vuln):math.floor((i + 1) / number_of_workers * length_vuln)]
        vuln_commit.append(tmp)

    result = multi_process_get_vuln_token(vuln_commit, number_of_workers)
    
    for dic in tqdm(result):
        for token in dic.keys():
            token_dict[token] += dic[token]
    for item in token_dict.keys():
        token_dict[item] = np.log((len(files)+len(vuln_tokens)) / (token_dict[item]+1))

    with open('../data/token_IDF.txt', 'w') as fp:
        fp.write(str(token_dict))
