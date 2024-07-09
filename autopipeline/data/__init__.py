import pkg_resources
import pandas as pd
import importlib.resources as resources
import numpy as np
import os
from huggingface_hub import hf_hub_download, list_repo_files

def get_persuasion_effect_data():
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/17/persuasive-17.csv')
    df = pd.read_csv(data_file_path)
    df = df[['sentence']]
    return df

def get_toxic_data():
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/24/toxic.csv')
    df = pd.read_csv(data_file_path)
    df = df[['original_sentence']]
    return df

def get_dog_whistle_data():
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/3/dogwhistle.csv')
    df = pd.read_csv(data_file_path)
    df = df[['Linguistic Context']]
    return df

def get_legal_text():
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-text.csv')
    df = pd.read_csv(data_file_path)
    return df

def get_legal_doc():
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-doc.csv')
    df = pd.read_csv(data_file_path)
    return df

class QUIET_ML:
    def __init__(self, repo_id='chuxuan/quiet-ml', cache_dir=None):
        self.repo_id = repo_id
        self.repo_type = 'dataset'
        self.cache_dir = cache_dir if cache_dir else os.path.join(os.getcwd(), 'huggingface_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        data_file_path = pkg_resources.resource_filename('autopipeline', f'data/queries.csv')
        df = pd.read_csv(data_file_path)
        self.queries = df
        data_file_path = pkg_resources.resource_filename('autopipeline', f'data/desc.csv')
        df = pd.read_csv(data_file_path)
        self.desc = df


    def download_file(self, filename):
        file_path = hf_hub_download(repo_id=self.repo_id, filename=filename, repo_type=self.repo_type, cache_dir=self.cache_dir)
        return file_path

    def query_text(self, qid):
        return self.queries.loc[self.queries["QID"] == qid, "Query"].iloc[0]
    
    def query_desc(self, qid):
        return self.desc.loc[self.desc["QID"] == qid, "Description"].iloc[0]
    
    def query_data(self, qid):
        if qid in [83, 84, 85, 86, 87, 88, 89, 92, 93, 98, 99, 100, 101]:
            df = get_legal_text()
        elif qid in [90, 91, 94, 95, 96, 97, 102, 103, 104, 105, 106, 107, 108, 109, 110, 116, 117, 118, 119, 120]:
            df = get_legal_doc()
        else:
            data_file_path = self.download_file(f'quiet-ml/{qid}/data.csv')
            df = pd.read_csv(data_file_path)
        return df
    
    def get_filtered_files(self, directory, prefix='answer', suffix='.csv'):
        files = list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
        filtered_files = [f for f in files if f.startswith(f'{directory}/{prefix}') and f.endswith(suffix)]
        return filtered_files

    def download_filtered_files(self, directory, prefix='answer', suffix='.csv'):
        filtered_files = self.get_filtered_files(directory, prefix, suffix)
        downloaded_files = [self.download_file(f) for f in filtered_files]
        return downloaded_files
    
    def query_answer(self, qid):
        # if qid > 82:
        #     return []
        if qid == 8:
            return [0.9401709401709402]
        if qid == 84:
            return ['ORDINAL']
        elif qid == 29:
            dataframes = [38.51447912749153]
        elif qid == 30:
            dataframes = [39.32038834951456]
        elif qid == 41:
            dataframes = [0.4225370762711864]
        elif qid == 42:
            dataframes = ['joy']
        elif qid == 44:
            dataframes = [1340]
        elif qid == 46:
            dataframes = [8190]
        elif qid == 59:
            dataframes = [np.array(['agreement', 'answer', 'appreciation', 'disagreement', 'elaboration', 'humor', 'question'], dtype = object)]
        elif qid == 60:
            dataframes = [0.38878080415045396]
        elif qid == 69:
            dataframes = [33.324558]
        elif qid == 70:
            dataframes = [46.809986]
        elif qid == 75:
            dataframes = [24]
        elif qid == 78:
            dataframes = [-29.513333333333332]
        elif qid == 31:
            dataframes = [['Jessica Rabbit', 'Tina Carlyle', 'Susie Diamond', 'Sugar Kane Kowalczyk', 'Dorothy Vallens', 'Ellen Aim']]
        elif qid == 71:
            dataframes = [['William Munny','Ned Logan','Butch Cassidy','The Sundance Kid',"Patrick Floyd 'Pat' Garrett", "Roy O'Bannon"]]
        elif qid == 82:
            dataframes = ['Evidence']
        elif qid == 85:
            dataframes = [np.array(['ADP', 'PROPN', 'PUNCT', 'SPACE', 'PRON', 'NOUN', 'AUX', 'VERB', 'DET', 'SCONJ', 'ADV', 'ADJ', 'PART', 'NUM', 'CCONJ', 'SYM', 'X'], dtype = object), 'NOUN']
        elif qid == 86:
            return [np.array(['ADP', 'PROPN', 'PUNCT', 'SPACE', 'PRON', 'NOUN', 'AUX', 'VERB', 'DET', 'SCONJ', 'ADV', 'ADJ', 'PART', 'NUM', 'CCONJ', 'SYM', 'X'], dtype = object)]
        elif qid == 87:
            return ['NOUN']
        elif qid == 99:
            return ['PERSON']
        elif qid == 101:
            return ['X']
        elif qid == 106:
            return ['NORP']
        elif qid == 100:
            return [np.array(['SPACE', 'DET', 'NOUN', 'VERB', 'SCONJ', 'AUX', 'ADV', 'ADP', 'PRON', 'PUNCT', 'ADJ', 'PART', 'PROPN', 'CCONJ', 'NUM', 'X'], dtype = object)]
        elif qid == 119:
            dataframes = [np.array(['SPACE', 'PUNCT', 'PROPN', 'SCONJ', 'DET', 'NOUN', 'VERB', 'CCONJ', 'AUX', 'ADJ', 'ADP', 'PRON', 'NUM', 'PART'], dtype = object)]
        elif qid == 108:
            return ['INTJ']
        elif qid == 105:
            return [np.array(['PROPN', 'NOUN', 'ADP', 'SPACE', 'PUNCT', 'SCONJ', 'DET', 'CCONJ', 'ADJ', 'VERB', 'PART', 'NUM', 'PRON', 'AUX', 'ADV', 'X', 'INTJ', 'SYM'], dtype = object), ['PROPN', 'NOUN', 'ADP', 'SPACE', 'PUNCT', 'SCONJ', 'DET', 'CCONJ', 'ADJ', 'VERB', 'PART', 'NUM', 'PRON', 'AUX', 'ADV', 'X', 'INTJ', 'SYM']]
        elif qid == 114:
            return [-0.4301570813700501]
        elif qid == 98:
            dataframes = [np.array(['Keith Melvin Dubray', 'Billy Howard', 'Michael Daniel Herrington', 'Bruce D. Switzer', 'Ralph Lowe', 'the Circuit Court'], dtype = object)]
        elif qid == 116:
            dataframes = [np.array(['STIPULATION', 'Mohamed Ba'], dtype = object)]
        elif qid == 117:
            return ['LOC']
        elif qid == 120:
            return ['PROPN']
        elif qid == 89:
            dataframes = [[' An attorney admits that the record was tendered late due to a mistake on his part . We find that such error, admittedly made by a criminal defendant, is good cause to grant motion .',
  ' Keith Melvin Dubray, by his attorney, has filed a motion for a rule on the clerk . His attorney admitted it was his fault that the record was not timely tendered .',
  ' Per Curiam. reviewed the record tendered late due to a mistake on his part .',
  ' The transcript of the case was not timely filed and it was no fault of the appellant . We find that such an error, admittedly made by the attorneys for a criminal defendant, is good cause to grant the motion .',
  ' Curtis and Billy Howard, brothers, were charged with and convicted of aggravated robbery and theft of property . Curtis, as an habitual offender, was sentenced to consecutive terms of life and 30 years . Billy Howard was convicted of concurrent terms of 10',
  ' Appellant, Michael Daniel Herrington, by his attorney, has filed for a rule on the clerk . His attorney, Bruce D. Switzer, admits that the record was tendered late due to a mistake on his part .',
  ' The transcript of the case was not timely filed and it was no fault of the appellant . The appellant’s former attorney, Ralph Lowe, admitted by affidavit attached to the motion that the transcript was filed late due to a mistake on',
  ' The transcript of the case was not timely filed and it was no fault of the appellant . His attorney admits that the transcript was filed late due to a mistake on his part .',
  ' An attorney admits that the trial court’s order granting an extension of time was not timely filed and it was no fault of the appellant . We find that such an error, admittedly made by the attorney for a criminal defendant, is',
  ' Petitioner pleaded guilty to aggravated robbery, in the Circuit Court of Randolph County . He was sentenced as an habitual offender to 20 years imprisonment . Petitioner subsequently filed numerous pro se petitions in circuit court, all of which raised grounds for post']]
        else:
            dataframes = []

        directory_name = f'quiet-ml/{qid}'
        downloaded_files = self.download_filtered_files(directory_name)
        dataframes.extend([pd.read_csv(file) for file in downloaded_files])

        return dataframes
    
    # Function to load query, data, desc, and answer altogether
    def query(self, qid):
        query = self.query_text(qid)
        data = self.query_data(qid)
        # answer = self.query_answer(qid)[0]
        answer = self.query_answer(qid)
        description = self.query_desc(qid)
        return {
            "query": query,
            "data": data,
            "desc": description,
            "answer": answer
        }


