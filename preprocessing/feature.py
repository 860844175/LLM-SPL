import datetime
import glob
import pandas as pd
from collections import Counter
import gc
import os
import logging
from util import *
import shutil
import warnings
warnings.filterwarnings("ignore")


def weblinks_bug_issue_cve(weblinks, bug, issue, cve, row):
    issue_cnt = len(issue)
    web_cnt = len(weblinks)
    bug_cnt = len(bug)
    cve_cnt = len(cve)

    def search_in_nvd_links(pre_token, tokens, links):
        cnt = 0
        for link in links:
            if pre_token and pre_token not in link:
                continue
            for token in tokens:
                if token.lower() in link:
                    cnt += 1
                    break
        return cnt

    web_match_nvd_links = search_in_nvd_links([], weblinks, row['links'])
    issue_match_nvd_links = search_in_nvd_links('issue', issue, row['links'])
    bug_match_nvd_links = search_in_nvd_links('bug', bug, row['links'])
    cve_match = As_in_B(cve, row['cve'])
    return issue_cnt, web_cnt, bug_cnt, cve_cnt

def vuln_commit_token(reponame, commit, cwedesc_tokens, desc_tokens):
    vuln_tokens = union_list(desc_tokens, cwedesc_tokens)

    with open(
            '../data/gitcommit/{}/{}'.format(
                reponame, commit), 'r') as fp:
        commit_tokens = eval(fp.read())

    commit_tokens_set = set(commit_tokens)

    inter_token_total = inter_token(set(vuln_tokens), commit_tokens_set)
    inter_token_total_cnt = len(inter_token_total)
    inter_token_total_ratio = inter_token_total_cnt / len(vuln_tokens)

    inter_token_cwe = inter_token(set(cwedesc_tokens), commit_tokens_set)
    inter_token_cwe_cnt = len(inter_token_cwe)
    inter_token_cwe_ratio = inter_token_cwe_cnt / (1 + len(cwedesc_tokens))

    return vuln_tokens, commit_tokens, inter_token_total_cnt, inter_token_total_ratio, inter_token_cwe_cnt, inter_token_cwe_ratio


def feature_time(committime, cvetime):
    committime = datetime.datetime.strptime(committime, '%Y%m%d')
    cvetime = datetime.datetime.strptime(cvetime, '%Y%m%d')
    time_dis = abs((cvetime - committime).days)
    return time_dis  

def get_feature(row, commit_info):
    commit = row['commit']
    reponame = row['repo']
    weblinks, bug, issue, cve, datetime, filepaths, funcs, addcnt, delcnt = commit_info[commit]

    issue_cnt, web_cnt, bug_cnt, cve_cnt = weblinks_bug_issue_cve(weblinks, bug, issue, cve, row)

    time_dis = feature_time(
        str(datetime), str(row['cvetime']))

    vuln_tokens, commit_tokens, inter_token_total_cnt, inter_token_total_ratio, inter_token_cwe_cnt, inter_token_cwe_ratio \
        = vuln_commit_token(reponame, commit, row['cwedesc'], row['desc'])

    c_vuln = Counter(vuln_tokens)
    c_commit = Counter(commit_tokens)
    len_vuln_tokens = len(vuln_tokens) + 1
    len_commit_tokens = len(commit_tokens) + 1
    vuln_tfidf = []
    commit_tfidf = []
    for token in token_IDF:
        vuln_tfidf.append(c_vuln[token]/len_vuln_tokens * token_IDF[token])
        commit_tfidf.append(
            c_commit[token]/len_commit_tokens * token_IDF[token])
    tfidf = cosine_similarity(vuln_tfidf, commit_tfidf)

    return addcnt, delcnt, issue_cnt, web_cnt, bug_cnt, cve_cnt, \
        time_dis, inter_token_cwe_cnt, inter_token_cwe_ratio, tfidf
        
def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]]
                    for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos 

def get_vuln_idf(bug, links, cve, cves):
    cve_match = 0
    for item in cves:
        if item in cve.lower():
            cve_match = 1
            break

    bug_match = 0
    for link in links:
        if 'bug' in link or 'Bug' in link:
            for item in bug:
                if item.lower() in link:
                    bug_match = 1
                    break

    return bug_match, cve_match

def get_vuln_loc(nvd_items, commit_items):
    same_cnt = 0
    for commit_item in commit_items:
        for nvd_item in nvd_items:
            if nvd_item in commit_item:
                same_cnt += 1
                break
    same_ratio = same_cnt / (len(commit_items)+1)
    unrelated_cnt = len(nvd_items) - same_cnt
    return same_cnt, same_ratio, unrelated_cnt


def get_vuln_type_relete(nvd_type, nvd_impact, commit_type, commit_impact, vuln_type_impact):
    l1, l2, l3 = 0, 0, 0
    
    for nvd_item in nvd_type:
        for commit_item in commit_type:
            if nvd_item == commit_item:
                l1 += 1
            else:
                l3 += 1
    for nvd_item in nvd_type:
        for commit_item in commit_impact:
            if commit_item in vuln_type_impact.get(nvd_item):
                l2 += 1
            else:
                l3 += 1
    for commit_item in commit_type:
        for nvd_item in nvd_impact:
            if nvd_item in vuln_type_impact.get(commit_item):
                l2 += 1
            else:
                l3 += 1
    cnt = l1 + l2 + l3+1
    return l1/cnt, l2/cnt, (l3+1)/cnt


def get_vuln_desc_text(c1, c2):
    c3 = c1 and c2
    same_token = c3.keys()
    shared_num = len(same_token)
    shared_ratio = shared_num / (len(c1.keys())+1)
    c3_value = list(c3.values())
    if len(c3_value) == 0:
        c3_value = [0]
    return shared_num, shared_ratio, max(c3_value), sum(c3_value), np.mean(c3_value), np.var(c3_value)


if __name__ == '__main__':
    
    gitpath = '../data/gitrepo/'
    commit_info_path = '../data/commit_info/'
    dataset_foldler_path = '../data/dataset/'
    
    if not os.path.exists(dataset_foldler_path):
        os.makedirs(dataset_foldler_path)
    
    with open('../data/token_IDF.txt', 'r') as f:
        token_IDF = eval(f.read())
        
    with open("../data/vuln_type_impact.json", 'r') as f:
        vuln_type_impact = json.load(f)
    
    dataset_df = pickle.load(open('../data/Dataset.pkl', 'rb'))
    dataset_df = reduce_mem_usage(dataset_df)
    
    original_shape = dataset_df.shape[0]
    

    vuln_df = pd.read_csv('../data/vuln_data.csv')
    vuln_df['desc'] = vuln_df['desc'].apply(eval)
    vuln_df['desc_token'] = vuln_df['desc_token'].apply(eval)
    vuln_df['desc_token_counter'] = eval_counter(vuln_df['desc_token_counter'])
    vuln_df['links'] = vuln_df['links'].apply(eval)
    vuln_df['cwedesc'] = vuln_df['cwedesc'].apply(eval)
    vuln_df['cvetime'] = vuln_df['cvetime'].astype(str)
    vuln_df['functions'] = vuln_df['functions'].apply(eval)
    vuln_df['files'] = vuln_df['files'].apply(eval)
    vuln_df['filepaths'] = vuln_df['filepaths'].apply(eval)
    vuln_df['vuln_type'] = vuln_df['vuln_type'].apply(eval)
    vuln_df['vuln_impact'] = vuln_df['vuln_impact'].apply(eval)
    vuln_df = reduce_mem_usage(vuln_df)
 
    mess_df = pd.read_csv('../data/mess_data.csv')
    mess_df['mess_bugs'] = mess_df['mess_bugs'].apply(eval)
    mess_df['mess_cves'] = mess_df['mess_cves'].apply(eval)
    mess_df['mess_type'] = mess_df['mess_type'].apply(eval)
    mess_df['mess_impact'] = mess_df['mess_impact'].apply(eval)
    mess_df['mess_token_counter'] = eval_counter(mess_df['mess_token_counter'])
    mess_df = reduce_mem_usage(mess_df)
    
    
    dataset_df = dataset_df.merge(vuln_df, how = 'left', on = 'cve').merge(mess_df, how = 'left', on = 'commit')
    
    assert original_shape == dataset_df.shape[0]
    
    repos = dataset_df.repo.unique()
    
    for reponame in repos:

        dirpath = 'tmp/' + reponame
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        tmp_df = dataset_df[dataset_df.repo == reponame]
        repo = git.Repo(gitpath + reponame)
        
        commits = tmp_df.commit
        
        commit_info = readfile(commit_info_path + f'{reponame}_commit_info')
        
        total_cnt = tmp_df.shape[0]
        
        each_cnt = 1000
        epoch = int((total_cnt + each_cnt)/each_cnt)
        logging.info('epoch: {}'.format(epoch))
        
        t1 = time.time()
        
        for i in tqdm(range(epoch)):
            if os.path.exists(dirpath + '/{:04}.csv'.format(i)):
                continue
            df = tmp_df.iloc[i * each_cnt: min((i + 1) * each_cnt, total_cnt)]
            df["addcnt"], df["delcnt"], df["issue_cnt"], df["web_cnt"], df["bug_cnt"], df["cve_cnt"],\
                df["time_dis"], df["inter_token_cwe_cnt"], df["inter_token_cwe_ratio"], \
                df["vuln_commit_tfidf"] = zip(
                    *df.apply(lambda row: get_feature(row, commit_info), axis=1))
            df['total_cnt'] = df['addcnt'] + df['delcnt']
            df.drop(['desc', 'links', 'cwedesc','cvetime'], axis = 1, inplace = True)
            
            df.to_csv(dirpath + '/{:04}.csv'.format(i), index = False)
        t2 = time.time()
        
        logging.info('{}共耗时：{} min'.format(reponame, (t2 - t1) / 60))
        gc.collect()
        
        files = glob.glob(dirpath + '/*.csv')
        
        m = {}
        for file in files:
            idx = int(re.search('([0-9]+).csv', file).group(1))
            m[idx] = file
            
        l = []
        for i in range(epoch):
            tmp = pd.read_csv(m[i])
            l.append(tmp)
            
        data_df = pd.concat(l)
        data_df.to_csv('../dataset/Dataset_{}.csv'.format(reponame), index=False)    

        dirpath1 = 'tmp1/'+reponame

        if not os.path.exists(dirpath1):
            os.makedirs(dirpath1)
            
        code_df = pd.read_csv('../data/code_data/code_data_{}.csv'.format(reponame))
        code_df = code_df.drop_duplicates()
        code_df['code_files'] = code_df['code_files'] .apply(eval)
        code_df['code_funcs'] = code_df['code_funcs'].apply(eval)
        code_df['code_filepaths'] = code_df['code_filepaths'] .apply(eval)
        code_df['code_token_counter'] =eval_counter(code_df['code_token_counter'])
        
        
        tmp_df = dataset_df[dataset_df.repo == reponame]
        tmp_df = (tmp_df.merge(code_df, how='left', on='commit'))
        commits = tmp_df.commit.unique()
        total_cnt = tmp_df.shape[0]
        each_cnt = 1000
        epoch = int((total_cnt+each_cnt)/each_cnt) 
        
        t1 = time.time()
        for i in tqdm(range(epoch)):
            if os.path.exists(dirpath1+'/{:04}.csv'.format(i)):
                continue
            df = tmp_df.iloc[i * each_cnt: min((i + 1) * each_cnt, total_cnt)]
            df['cve_match'], df['bug_match'] = zip(*df.apply(
                lambda row: get_vuln_idf(row['mess_bugs'], row['links'], row['cve'], row['mess_cves']), axis=1))
            
            df['filepath_same_cnt'], df['filepath_same_ratio'], df['filepath_unrelated_cnt'] = zip(*df.apply(
                lambda row: get_vuln_loc(row['filepaths'], row['code_filepaths']), axis=1))
            
            df['func_same_cnt'], df['func_same_ratio'], df['func_unrelated_cnt'] = zip(*df.apply(
                lambda row: get_vuln_loc(row['functions'], row['code_funcs']), axis=1))
            
            df['file_same_cnt'], df['file_same_ratio'], df['file_unrelated_cnt'] = zip(*df.apply(
                lambda row: get_vuln_loc(row['files'], row['code_files']), axis=1))
            
            df['mess_shared_num'], df['mess_shared_ratio'], df['mess_max'], df['mess_sum'], df['mess_mean'], df['mess_var'] = zip(*df.apply(
                lambda row: get_vuln_desc_text(row['desc_token_counter'], row['mess_token_counter']), axis=1))
            
            df['code_shared_num'], df['code_shared_ratio'], df['code_max'], df['code_sum'], df['code_mean'], df['code_var'] = zip(*df.apply(
                lambda row: get_vuln_desc_text(row['desc_token_counter'], row['code_token_counter']), axis=1))
            
            df['vuln_type_1'], df['vuln_type_2'], df['vuln_type_3'] = zip(*df.apply(
                lambda row: get_vuln_type_relete(row['vuln_type'], row['vuln_impact'], row['mess_type'], row['mess_impact'], vuln_type_impact), axis=1))
        
            df.drop(['mess_bugs', 'links', 'mess_cves', 'functions', 'code_funcs', 'filepaths', 'code_filepaths',
                    'files', 'code_files', 'vuln_type', 'vuln_impact', 'mess_type', 'mess_impact',
                    'desc_token_counter', 'mess_token_counter', 'code_token_counter'],
                    axis=1, inplace=True)
            df.to_csv(dirpath1+'/{:04}.csv'.format(i), index=False)
        t2 = time.time()
        logging.info('{}共耗时：{} min'.format(reponame, (t2 - t1) / 60))
        print('{}共耗时：{} min'.format(reponame, (t2 - t1) / 60))
        gc.collect()
    
        files = glob.glob(dirpath1+'/*.csv')
        m = {}
        l = []
        for file in files:
            idx = int(re.search('([0-9]+).csv', file).group(1))
            m[idx] = file
        l = [pd.read_csv(m[i]) for i in range(epoch)]
        data_df2 = pd.concat(l)

        tmp_columns = ['cve_match', 'bug_match', 'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt', 'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',
                    'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',
                    'mess_shared_num', 'mess_shared_ratio', 'mess_max', 'mess_sum', 'mess_mean', 'mess_var',
                    'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var']
        
        data_df[tmp_columns] = data_df2[tmp_columns]
        data_df.to_csv('../dataset/Dataset_{}.csv'.format(reponame), index=False)
    
    pd.concat([pd.read_csv(i) for i in glob.glob('../dataset/*')]).to_csv('../dataset/total_dataset.csv', index=False)
    
        

