import time
import logging
import re
import requests
import lxml
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString, Tag

import argparse


headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
    # "Cookie": "" 
}


import subprocess

def git_clone(repo_url, directory=None):
    # 构建git clone命令
    cmd = ["git", "clone", repo_url]
    if directory:
        cmd.append(directory)  # 如果指定了目录，则添加到命令中

    try:
        # 执行命令
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("仓库已克隆成功")
        print(result.stdout)  # 打印标准输出
    except subprocess.CalledProcessError as e:
        # 如果出现错误，打印错误信息
        print("错误信息：", e.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data input')

    parser.add_argument('data_path', type = str)

    parser.add_argument('cve_desc_path', type = str, default = '../data/cve_desc.csv')

    parser.add_argument('output_path', type = str, default = '../data/vuln_data.csv')

    parser.add_arguemnt('git_repos', type = str, default = '../data/repo_list.pkl')

    args = parser.parse_args()

    data_path = args.data_path

    cve_desc_path = args.cve_desc_path

    repo_list_path = args.git_repos

    output_path = args.output_path

    data = pd.read_csv(data_path)

    cve_list = data.cve.unique().tolist()

    repos_list = pickle.load(open(repo_lsit_path), 'rb')

    for repo_url in repos_list:
        git_clone(repo_url, '../gitrepo/')
    
    ### get cve time
    result_list = []
    for cve in tqdm(cve_list):
        page = 'https://cve.mitre.org/cgi-bin/cvename.cgi?name='+cve
        res = requests.get(url=page,  headers=headers)
        time.sleep(5) # Prevent frequent visits
        cvetime = re.search('<td><b>([0-9]{8})</b></td>', res.text).group(1)
        result_list.append((cve, cvetime))

    df = pd.DataFrame(result_list, columns = ['cve', 'cvetime'])

    ### get nvd info
    result_list = []
    for cve in tqdm(cve_list):
        page = 'https://nvd.nist.gov/vuln/detail/'+cve
        try:
            links = []
            cwe = ()
            res = requests.get(url=page,  headers=headers)
            soup = BeautifulSoup(res.text, 'lxml')
            tbody = soup.find(attrs={'data-testid': "vuln-hyperlinks-table"}).tbody
            for tr in tbody.children:
                if isinstance(tr, NavigableString): continue
                tds = tr.findAll('td')
                if 'Patch' in tds[1].text:
                    links.append(tds[0].a['href'])
            tbody = soup.find(attrs={'data-testid': "vuln-CWEs-table"}).tbody
            for tr in tbody.children:
                if isinstance(tr, NavigableString): continue
                tds = tr.findAll('td')
                cwe = (tds[0].text, tds[1].text)
        except Exception as e:
            logging.info(url + " ")
        time.sleep(5) # Prevent frequent visits
        result_list.append([cve, links, cwe])

    df2 = pd.DataFrame(result_list, columns=['cve', 'links', 'cwe'])
    df2 = df2.drop_duplicates(['cve']).reset_index(drop=True)

    df2['cwedesc'] = df2['cwe'].apply(lambda items:  items[1] if len(items) else '')
    df2['cwedesc'] = df2['cwedesc'].fillna('')
    df2['cwedesc'] = df2['cwedesc'].apply(lambda x: to_token(x))
        
    df3 = pd.read_csv(cve_desc_path)
    df3 = df.merge(df2[['cve', 'links', 'cwedesc']], how='left', on='cve').merge(df3, how='left', on='cve')
    df3.to_csv(output_path, index = False)