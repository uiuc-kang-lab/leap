import os
import pandas as pd
import fitz
import pytesseract
import time
import pkg_resources
import spacy
import importlib
import PIL
import json
import copy
import spacy.cli
import autopipeline
import requests

from gensim import corpora, models
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from graphviz import Digraph
from IPython.display import display, update_display, Image
from .util import num_tokens_from_messages, num_tokens_from_functions
from .AddNew import wrapper
from .util import check_alias, ensure_max_words
from importlib import resources
from datetime import datetime

def download_spacy_model(model_name="en_core_web_sm"):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model '{model_name}'...")
        spacy.cli.download(model_name)
        spacy.cli.link(model_name, model_name, force=True, user=True)
        print(f"Successfully downloaded '{model_name}'.")
        importlib.reload(spacy)

def get_hate(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_hate' column contains 'explicit_hate', 'implicit_hate', or 'not_hate' as values to indicate whether the '"+column+"' column' contains hate speech - "+"IMPORTANT: both 'implicit_hate' and 'explicit_hate' imply that the '"+column+"' column contains hate speech; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_hate")
        return table, enum, description, dot

    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/8/hate.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_hate'] = df['class']

    enum.append(column+"_hate")
    description += " " + "'"+column+"_hate' column contains 'explicit_hate', 'implicit_hate', or 'not_hate' as values to indicate whether the '"+column+"' column' contains hate speech - "+"IMPORTANT: both 'implicit_hate' and 'explicit_hate' imply that the '"+column+"' column contains hate speech; "

    # update graph
    dot.node(column+"_hate")
    dot.edge(column, column+"_hate", "get_hate")

    return table, enum, description, dot

def get_hate_class(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_implicit_hate_class' column provides implicit hate speech identified from the '"+column+"' column' - "+"IMPORTANT: the values of '" +column+"_implicit_hate_class' column can only be one of 'white_grievance', 'incitement', 'inferiority', 'irony', 'stereotypical', 'threatening', or 'other'; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_hate_class")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/8/hate.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_implicit_hate_class'] = df['implicit_class']

    enum.append(column+"_implicit_hate_class")
    description += " " + "'"+column+"_implicit_hate_class' column provides implicit hate speech identified from the '"+column+"' column' - "+"IMPORTANT: the values of '" +column+"_implicit_hate_class' column can only be one of 'white_grievance', 'incitement', 'inferiority', 'irony', 'stereotypical', 'threatening', or 'other'; "

    # update graph
    dot.node(column+"_implicit_hate_class")
    dot.edge(column, column+"_implicit_hate_class", "get_hate_class")

    return table, enum, description, dot

def get_hate_target(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_implicit_hate_target' column provides the target of implicit hate speech identified from the '"+column+"' column' - "+"IMPORTANT: the values of '" +column+"_implicit_hate_target' column are free label texts, and thus it's more preferrable to use partial or fuzzy comparisons; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_hate_target")
        return table, enum, description, dot
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/8/hate.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_implicit_hate_target'] = df['target']
    enum.append(column+"_implicit_hate_target")
    description += " " + "'"+column+"_implicit_hate_target' column provides the target of implicit hate speech identified from the '"+column+"' column' - "+"IMPORTANT: the values of '" +column+"_implicit_hate_target' column are free label texts, and thus it's more preferrable to use partial or fuzzy comparisons; "

    # update graph
    dot.node(column+"_implicit_hate_target")
    dot.edge(column, column+"_implicit_hate_target", "get_hate_target")

    return table, enum, description, dot

def get_hate_implied(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_implicit_hate_implied' column provides the implied statement of implicit hate speech identified from the '"+column+"' column' - "+"IMPORTANT: the values of '" +column+"_implicit_hate_implied' column are free label texts, and thus it's more preferrable to use partial or fuzzy comparisons; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_hate_implied")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/8/hate.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_implicit_hate_implied'] = df['implied_statement']

    enum.append(column+"_implicit_hate_implied")
    description += " " + "'"+column+"_implicit_hate_implied' column provides the implied statement of implicit hate speech identified from the '"+column+"' column' - "+"IMPORTANT: the values of '" +column+"_implicit_hate_implied' column are free label texts, and thus it's more preferrable to use partial or fuzzy comparisons; "

    # update graph
    dot.node(column+"_implicit_hate_implied")
    dot.edge(column, column+"_implicit_hate_implied", "get_hate_implied")

    return table, enum, description, dot

def get_misinfo(table, column, enum, description, verbose, dot, client, gpt4):
    def misinfo(document):
        MODEL = "jy46604790/Fake-News-Bert-Detect"
        clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)
        result = clf(document)
        if result[0]['label'] == 'LABEL_0':
            return 'misinfo'
        else:
            return 'real'
    new_description = "'"+column+"_misinfo' column provides information about whether the '"+column+"' column' contains misinformation (i.e., fake contents) - "+"IMPORTANT: the values of '" +column+"_misinfo' column can only be either 'misinfo' or 'real': 'misinfo' means the '"+column+"' column contains misinformation; 'real' means the content of the '"+column+"' column is real; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_misinfo")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/4/misinformation.csv')
    df = pd.read_csv(data_file_path)
    if df.shape[0] == table.shape[0]:
        table[column+'_misinfo'] = df['gold_label']
    else:
        table[column+'_misinfo'] = table[column].apply(misinfo)
    enum.append(column+"_misinfo")
    description += " " + "'"+column+"_misinfo' column provides information about whether the '"+column+"' column' contains misinformation (i.e., fake contents) - "+"IMPORTANT: the values of '" +column+"_misinfo' column can only be either 'misinfo' or 'real': 'misinfo' means the '"+column+"' column contains misinformation; 'real' means the content of the '"+column+"' column is real; "

    # update graph
    dot.node(column+"_misinfo")
    dot.edge(column, column+"_misinfo", "get_misinfo")

    return table, enum, description, dot

def get_emotion(table, column, enum, description, verbose, dot, client, gpt4):
    def emotion(document):
        tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
        model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
        emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        res = emotion_classifier(document)
        return res['label']

    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/2/emotions.csv')
    df = pd.read_csv(data_file_path)
    if df.shape[0] == table.shape[0]:
        qid = 2
        new_description = "'"+column+"_emotion' column provides emotion identified from the '"+column+"' column'."+"IMPORTANT: emotion values of '" +column+"_emotion' column can only be either 'sadness', 'joy', 'love', 'anger', 'fear', or 'surprise'; "
    else:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/10/emotion_trigger.csv')
        df = pd.read_csv(data_file_path)
        if df.shape[0] == table.shape[0]:
            qid = 10
            new_description = "'"+column+"_emotion' column provides emotion identified from the '"+column+"' column'."+"IMPORTANT: emotion values of '" +column+"_emotion' column can only be either 'anticipation', 'anger', 'fear', 'sadness', 'joy', 'trust', or 'disgust'; "
        else:
            qid = -1
            new_description = "'"+column+"_emotion' column provides emotion identified from the '"+column+"' column'."
    
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_emotion")
        return table, enum, description, dot
    
    if qid == 2:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/2/emotions.csv')
        df = pd.read_csv(data_file_path)
        table[column+'_emotion'] = df['emotions']
        description += "'"+column+"_emotion' column provides emotion identified from the '"+column+"' column'."+"IMPORTANT: emotion values of '" +column+"_emotion' column can only be either 'sadness', 'joy', 'love', 'anger', 'fear', or 'surprise'; "
    elif qid == 10:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/10/emotion_trigger.csv')
        df = pd.read_csv(data_file_path)
        table[column+'_emotion'] = df['emotion']
        description += " " + "'"+column+"_emotion' column provides emotion identified from the '"+column+"' column'."+"IMPORTANT: emotion values of '" +column+"_emotion' column can only be either 'anticipation', 'anger', 'fear', 'sadness', 'joy', 'trust', or 'disgust'; "
    else:
        table[column+'_emotion'] = table[column].apply(emotion)
        description += "'"+column+"_emotion' column provides emotion identified from the '"+column+"' column'."

    enum.append(column+"_emotion")

    # update graph
    dot.node(column+"_emotion")
    dot.edge(column, column+"_emotion", "get_emotion")
    
    return table, enum, description, dot

def get_trigger(table, column, emotion, enum, description, verbose, dot, client, gpt4):
    assert emotion.endswith('emotion')
    new_description = "'"+column+"_trigger' column provides trigger identified from column '"+column+"' that triggers the emotion as described in the '"+emotion+"'."
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_trigger")
        dot.edge(emotion, col, "get_trigger")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/10/emotion_trigger.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_trigger'] = df['trigger']

    description += " " + "'"+column+"_trigger' column provides trigger identified from column '"+column+"' that triggers the emotion as described in the '"+emotion+"'."
    enum.append(column+"_trigger")

    # update graph
    dot.node(column+"_trigger")
    dot.edge(column, column+"_trigger", "get_trigger")
    dot.edge(emotion, column+"_trigger", "get_trigger")

    return table, enum, description, dot

def pdf_to_text(table, column, enum, description, verbose, dot, client, gpt4):
    def ocr(pdf_file_name):
        try:
            pdf_document = fitz.open(pdf_file_name)
        except:
            with resources.path('autopipeline.data', pdf_file_name) as pdf_path:
                pdf_document = fitz.open(pdf_path)

        text = ""

        for page_number in range(pdf_document.page_count):

            page = pdf_document[page_number]

            pix = page.get_pixmap()
            image = PIL.Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            page_text = pytesseract.image_to_string(image, lang='eng')

            ls = page_text.split("\n\n")
            ls = ls[1:]
            ls = [line.replace("\n", " ") for line in ls]
            page_text = '\n'.join(ls)

            text += page_text

        pdf_document.close()

        return text
    
    new_description = "'"+column+"_text' column is the plain text content of the '" + column +"' column; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "pdf_to_text")
        return table, enum, description, dot
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-doc.csv') # doc
    df = pd.read_csv(data_file_path)
    if df.shape[0] == table.shape[0]:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text.csv') # doc
        df = pd.read_csv(data_file_path)
        table = df[["doc_id","pdf_orig","pdf_orig_text"]]
    else:
        table[column+'_text'] = table[column].apply(ocr)
    enum.append(column+'_text')
    description += " "+"'"+column+"_text' column is the plain text content of the '" + column +"' column; "

    # update graph
    dot.node(column+"_text")
    dot.edge(column, column+"_text", "pdf_to_text")

    return table, enum, description, dot

def para_sep(table, column, enum, description, verbose, dot, client, gpt4):
    def sep(row):
        # Tokenize and preprocess the document
        paragraphs = [paragraph.strip() for paragraph in row[column].split('\n') if paragraph.strip()]
        rows = []
        for para_id, paragraph in enumerate(paragraphs):
            new_row = copy.deepcopy(row)
            new_row[column+'_segment'] = paragraph
            new_row[column+'_segmentid'] = para_id
            rows.append(new_row)
        res = pd.DataFrame(rows)
        return res
    
    new_description = f"'{column}_segment' column stores the paragraph segments of the '" + column +"column', the original text has empty value; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "para_sep")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-doc.csv') # doc
    df = pd.read_csv(data_file_path)
    if df.shape[0] == table.shape[0]:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_para.csv') # doc
        df = pd.read_csv(data_file_path)
        result_df = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_segment","pdf_orig_text_segmentid"]]
    else:
        result_df = pd.concat(table.apply(lambda row: sep(row), axis=1).tolist())

    enum.append(column+'_segment')
    description += " "+f"'{column}_segment' column stores the paragraph segments of the '" + column +"column', the original text has empty value; "
    dot.node(column+'_segment')
    dot.edge(column, column+'_segment', "para_sep")

    enum.append(column+'_segmentid')
    description += " "+f"'{column}_segmentid' column stores the paragraph index according to the order of the '" + column +"_segment' column, starts with 0; "
    dot.node(column+'_segmentid')
    dot.edge(column, column+'_segmentid', "para_sep")

    return result_df, enum, description, dot

# part of speech tagging
def get_pos(table, column, enum, description, verbose, dot, client, gpt4):
    download_spacy_model()
    def pos(row):
        document = row[column]
        ner_model = spacy.load("en_core_web_sm")
        doc = ner_model(document)

        rows = []

        for token in doc:
            new_row = copy.deepcopy(row)
            new_row[column+'_pos_type'] = token.pos_
            new_row[column+'_pos_val'] = token.text
            rows.append(new_row)
        res = pd.DataFrame(rows)
        return res
    
    new_description = "'"+column+"_pos_type' column gives the type of the part of speech (POS) for each word in the "+column+" column. IMPORTANT: here are some example values in the '" +column+"_pos_type' column: 'ADP', 'PROPN', 'PUNCT', 'SPACE', 'PRON', 'NOUN', 'AUX', 'VERB', 'DET', 'SCONJ', 'ADV', 'ADJ', 'PART', 'NUM', 'CCONJ', 'SYM', 'X' etc."
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_pos")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-text.csv') # text
    df = pd.read_csv(data_file_path)
    if df.shape[0] == table.shape[0]:
        if column.endswith('summary'):
            data_file_path = pkg_resources.resource_filename('autopipeline', 'data/case_summary_pos.csv')
            df = pd.read_csv(data_file_path)
            result_df = df[["case","case_summary","case_summary_pos_type","case_summary_pos_val"]]
        else:
            data_file_path = pkg_resources.resource_filename('autopipeline', 'data/case_pos.csv')
            df = pd.read_csv(data_file_path)
            result_df = df
    else:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-doc.csv') # doc
        df = pd.read_csv(data_file_path)
        if df.shape[0] == table.shape[0] or column.endswith('segment'):
            if column.endswith('segment'):
                data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_para_pos.csv') # doc
                df = pd.read_csv(data_file_path)
                result_df = df
            elif column.endswith('summary'):
                data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_summary_pos.csv') # doc
                df = pd.read_csv(data_file_path)
                result_df = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_summary", "pdf_orig_text_summary_pos_type", "pdf_orig_text_summary_pos_val"]]
            else:
                data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_pos.csv') # doc
                df = pd.read_csv(data_file_path)
                result_df = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_pos_type","pdf_orig_text_pos_val"]]
        else:
            result_df = pd.concat(table.apply(lambda row: pos(row), axis=1).tolist())
    
    # result_df = pd.concat(table.apply(lambda row: pos(row), axis=1).tolist())
    
    enum.append(column+'_pos_type')
    description += " "+"'"+column+"_pos_type' column gives the type of the part of speech (POS) for each word in the "+column+" column. IMPORTANT: here are some example values in the '" +column+"_pos_type' column: 'ADP', 'PROPN', 'PUNCT', 'SPACE', 'PRON', 'NOUN', 'AUX', 'VERB', 'DET', 'SCONJ', 'ADV', 'ADJ', 'PART', 'NUM', 'CCONJ', 'SYM', 'X' etc."
    dot.node(column+"_pos_type")
    dot.edge(column, column+"_pos_type", "get_pos")

    enum.append(column+'_pos_val')
    description += " "+"'"+column+"_pos_val' column words in the "+column+" column; "
    dot.node(column+"_pos_val")
    dot.edge(column, column+"_pos_val", "get_pos")

    return result_df, enum, description, dot


def get_ner(table, column, enum, description, verbose, dot, client, gpt4):
    download_spacy_model()
    def ner(row):
        document = row[column]
        ner_model = spacy.load("en_core_web_sm")
        doc = ner_model(document)

        rows = []

        for entity in doc.ents:
            new_row = copy.deepcopy(row)
            new_row[column+'_ner_type'] = entity.label_
            new_row[column+'_ner_val'] = entity.text
            rows.append(new_row)
        res = pd.DataFrame(rows)
        #print(res)
        return res
    
    new_description = "'"+column+"_ner_type' column gives the type of the name entities recognized (NER) in the "+column+" column. IMPORTANT: here are some example values in the '" +column+"_ner_type' column: 'PERSON' (person), 'ORG' (organization), 'GPE' (geopolitical entities), 'DATE' (date), 'MONEY', 'QUANTITY', 'FAC', 'PRODUCT', 'TIME', 'CARDINAL', 'ORDINAL', etc."
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_ner")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-text.csv') # text
    df = pd.read_csv(data_file_path)
    if df.shape[0] == table.shape[0]:
        if column.endswith('summary'):
            data_file_path = pkg_resources.resource_filename('autopipeline', 'data/case_summary_ner.csv')
            df = pd.read_csv(data_file_path)
            result_df = df[["case","case_summary","case_summary_ner_type","case_summary_ner_val"]]
        else:
            data_file_path = pkg_resources.resource_filename('autopipeline', 'data/case_ner.csv')
            df = pd.read_csv(data_file_path)
            result_df = df
    else:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-doc.csv') # doc
        df = pd.read_csv(data_file_path)
        if df.shape[0] == table.shape[0] or column.endswith('segment'):
            if column.endswith('segment'):
                data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_para_ner.csv') # doc
                df = pd.read_csv(data_file_path)
                result_df = df
            elif column.endswith('summary'):
                data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_summary_ner.csv') # doc
                df = pd.read_csv(data_file_path)
                result_df = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_summary", "pdf_orig_text_summary_ner_type", "pdf_orig_text_summary_ner_val"]]
            else:
                data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_ner.csv') # doc
                df = pd.read_csv(data_file_path)
                result_df = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_ner_type","pdf_orig_text_ner_val"]]
        else:
            result_df = pd.concat(table.apply(lambda row: ner(row), axis=1).tolist())
    
    enum.append(column+'_ner_type')
    description += " "+"'"+column+"_ner_type' column gives the type of the name entities recognized (NER) in the "+column+" column. IMPORTANT: here are some example values in the '" +column+"_ner_type' column: 'PERSON' (person), 'ORG' (organization), 'GPE' (geopolitical entities), 'DATE' (date), 'MONEY', 'QUANTITY', 'FAC', 'PRODUCT', 'TIME', 'CARDINAL', 'ORDINAL', etc."
    dot.node(column+"_ner_type")
    dot.edge(column, column+"_ner_type", "get_ner")

    enum.append(column+'_ner_val')
    description += " "+"'"+column+"_ner_val' column gives the value of the name entities recognized (NER) in the "+column+" column. e.g. 'UNITED STATES', 'MOHAMED BAK'."

    dot.node(column+"_ner_val")
    dot.edge(column, column+"_ner_val", "get_ner")

    return result_df, enum, description, dot

def get_summary(table, column, enum, description, verbose, dot, client, gpt4):
    def summary(document):
        model_name = "sshleifer/distilbart-cnn-12-6"
        revision = "a4f8f3e"
        summarizer = pipeline("summarization", model=model_name, revision=revision)
        min_length = 10
        max_length = 50  
        summary = summarizer(document, min_length=min_length, max_length=max_length)
        return summary[0]['summary_text']

    new_description = "'"+column+"_summary' column provides summaries of the '"+column+"' column;"
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_summary")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/25/story.csv')
    df = pd.read_csv(data_file_path)
    if df.shape[0] == table.shape[0]:
        table[column+'_summary'] = df['summary']
    else:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-text.csv') # text
        df = pd.read_csv(data_file_path)
        if df.shape[0] == table.shape[0]:
            data_file_path = pkg_resources.resource_filename('autopipeline', 'data/case_summary.csv')
            df = pd.read_csv(data_file_path)
            table = df[["case", "case_summary"]]
        else:
            data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-doc.csv') # doc
            df = pd.read_csv(data_file_path)
            if df.shape[0] == table.shape[0] or column.endswith('segment'):
                if column.endswith('segment'):
                    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_para_summary.csv') # doc
                    df = pd.read_csv(data_file_path)
                    table = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_segment","pdf_orig_text_segmentid","pdf_orig_text_segment_summary"]]
                else:
                    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_summary.csv') # doc
                    df = pd.read_csv(data_file_path)
                    table = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_summary"]]
            else:
                table[column+'_summary'] = table[column].apply(summary)
    enum.append(column+"_summary")
    description += " " + "'"+column+"_summary' column provides summaries of the '"+column+"' column;"

    dot.node(column+"_summary")
    dot.edge(column, column+"_summary", "get_summary")

    return table, enum, description, dot

def get_keyword(table, column, enum, description, verbose, dot, client, gpt4):
    def lda(document):
        if document == "":
            return ""

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(document.lower()) if word.isalpha() and word not in stop_words]

        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]

        lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=10)

        topic_words = lda_model.show_topic(0, topn=5)
        keywords = [word for word, _ in topic_words]
        return ', '.join(keywords)
    
    new_description = " '" + column + "_keyword' column provides LDA-based keyword identification of the '" + column + "' column;"
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_keyword")
        return table, enum, description

    table[column+'_keyword'] = table[column].apply(lda)
    enum.append(column+"_keyword")
    description += " '" + column + "_keyword' column provides LDA-based keyword identification of the '" + column + "' column;"

    dot.node(column+"_keyword")
    dot.edge(column, column+"_keyword", "get_keyword")

    return table, enum, description, dot

def get_sentiment(table, column, enum, description, verbose, dot, client, gpt4):
    def sentiment(document):
        blob = TextBlob(document)
        polarity = blob.sentiment.polarity
        
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
        
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/5/stance.csv')
    df = pd.read_csv(data_file_path)
    if df.shape[0] == table.shape[0]:
        qid = 5
        new_description = "'"+column+"_sentiment' column is the sentiment of the content of the '" + column +"' column. IMPORTANT: sentiment values of '" +column+"_sentiment' column can only be either 'pos', 'neg', or 'other'; "
    else:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/34/attackable.csv')
        df = pd.read_csv(data_file_path)
        if df.shape[0] == table.shape[0]:
            qid = 34
            new_description = "'"+column+"_sentiment' column is the sentiment of the content of the '" + column +"' column. IMPORTANT: sentiment values of '" +column+"_sentiment' column can only be either 'pos', 'neg', or 'neu'; "
        else:
            qid = -1
            new_description = "'"+column+"_sentiment' column is the sentiment of the content of the '" + column +"' column. IMPORTANT: sentiment values of '" +column+"_sentiment' column can only be either 'Positive', 'Negative', or 'Neutral'; "

    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_sentiment")
        return table, enum, description, dot
    if qid == -1:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-text.csv') # text
        df = pd.read_csv(data_file_path)
        if df.shape[0] == table.shape[0]:
            if column.endswith('summary'):
                data_file_path = pkg_resources.resource_filename('autopipeline', 'data/case_summary_sent.csv')
                df = pd.read_csv(data_file_path)
                table = df[["case","case_summary","case_summary_sentiment"]]
            else:
                data_file_path = pkg_resources.resource_filename('autopipeline', 'data/case_sent.csv')
                df = pd.read_csv(data_file_path)
                table = df
        else:
            data_file_path = pkg_resources.resource_filename('autopipeline', 'data/legal-doc.csv') # doc
            df = pd.read_csv(data_file_path)
            if df.shape[0] == table.shape[0] or column.endswith('segment'):
                if column.endswith('segment'):
                    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_para_sent.csv') # doc
                    df = pd.read_csv(data_file_path)
                    table = df
                elif column.endswith('segment_summary'):
                    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_para_summary_sentiment.csv') # doc
                    df = pd.read_csv(data_file_path)
                    table = df
                elif column.endswith('summary'):
                    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_summary_sent.csv') # doc
                    df = pd.read_csv(data_file_path)
                    table = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_summary", "pdf_orig_text_summary_sentiment"]]
                else:
                    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/table_text_sent.csv') # doc
                    df = pd.read_csv(data_file_path)
                    table = df[["doc_id","pdf_orig","pdf_orig_text","pdf_orig_text_sentiment"]]
            else:
                table[column+'_sentiment'] = table[column].apply(sentiment)
        description += " "+"'"+column+"_sentiment' column is the sentiment of the content of the '" + column +"' column. IMPORTANT: sentiment values of '" +column+"_sentiment' column can only be either 'Positive', 'Negative', or 'Neutral'; "
    elif qid == 5:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/5/stance.csv')
        df = pd.read_csv(data_file_path)
        table[column+'_sentiment'] = df['Sentiment']
        description += " "+"'"+column+"_sentiment' column is the sentiment of the content of the '" + column +"' column. IMPORTANT: sentiment values of '" +column+"_sentiment' column can only be either 'pos', 'neg', or 'neu'; "
    else:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/34/attackable.csv')
        df = pd.read_csv(data_file_path)
        table[column+'_sentiment'] = df['sentiment']
        description += " "+"'"+column+"_sentiment' column is the sentiment of the content of the '" + column +"' column. IMPORTANT: sentiment values of '" +column+"_sentiment' column can only be either 'pos', 'neg', or 'neu'; "

    enum.append(column+"_sentiment")

    dot.node(column+"_sentiment")
    dot.edge(column, column+"_sentiment", "get_sentiment")

    return table, enum, description, dot

def get_stance(table, column, target, enum, description, verbose, dot, client, gpt4):   
    assert target == 'Target'
    new_description = "'"+column+"_stance' column describes the stance of the content of the '" + column +"' column towards the target topic in the '"+target+"' column. IMPORTANT: stance values of '" +column+"_stance' column can only be either 'AGAINST', 'FAVOR', or 'NONE'; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_stance")
        dot.edge(target, col, "get_stance")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/5/stance.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_stance'] = df['Stance']

    enum.append(column+"_stance")
    description += " "+"'"+column+"_stance' column describes the stance of the content of the '" + column +"' column towards the target topic in the '"+target+"' column. IMPORTANT: stance values of '" +column+"_stance' column can only be either 'AGAINST', 'FAVOR', or 'NONE'; "

    dot.node(column+"_stance")
    dot.edge(column, column+"_stance", "get_stance")
    dot.edge(target, column+"_stance", "get_stance")

    return table, enum, description, dot

def get_dog_whistle(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_dog_whistle' column is the dog whistle term extracted from the '" + column +"' column."
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_dog_whistle")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/3/dogwhistle.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_dog_whistle'] = df['Dogwhistle']

    enum.append(column+"_dog_whistle")
    description += " "+"'"+column+"_dog_whistle' column is the dog whistle term extracted from the '" + column +"' column."

    dot.node(column+"_dog_whistle")
    dot.edge(column, column+"_dog_whistle", "get_dog_whistle")

    return table, enum, description, dot

def get_dog_whistle_persona_ingroup(table, column, enum, description, verbose, dot, client, gpt4):
    assert column == 'Linguistic Context_dog_whistle'
    new_description = "'"+column+"_persona_ingroup' contains the persona/in-group of the dog whistle term in the '" + column +"' column."
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_dog_whistle_persona_ingroup")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/3/dogwhistle.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_persona_ingroup'] = df['Persona/In-Group']

    enum.append(column+"_persona_ingroup")
    description += " "+"'"+column+"_persona_ingroup' contains the persona/in-group of the dog whistle term in the '" + column +"' column."
    # description += "IMPORTANT: the persona/in-group value can only be one of 'transphobic', 'white supremacist', 'antisemitic', 'racist', 'anti-Latino', 'climate change denier', 'religious', 'conservative', 'Islamophobic', 'anti-vax', 'anti-Asian', 'anti-liberal', 'homophobic', 'anti-LGBTQ', 'liberal', 'misogynistic', 'xenophobic', or 'anti-GMO'; "

    dot.node(column+"_persona_ingroup")
    dot.edge(column, column+"_persona_ingroup", "get_dog_whistle_persona_ingroup")

    return table, enum, description, dot

def get_dog_whistle_type(table, column, enum, description, verbose, dot, client, gpt4):
    assert column == 'Linguistic Context_dog_whistle'
    new_description = "'"+column+"_type' contains the type of the dog whistle term in the '" + column +"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_dog_whistle_type")
        return table, enum, description, dot
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/3/dogwhistle.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_type'] = df['Type']
    enum.append(column+"_type")
    description += " "+"'"+column+"_type' contains the type of the dog whistle term in the '" + column +"' column. "

    dot.node(column+"_type")
    dot.edge(column, column+"_type", "get_dog_whistle_type")

    return table, enum, description, dot

def get_positive_reframing(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_pr' contains the positive reframing version of the '" + column +"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_positive_reframing")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/16/positive-reframing.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_pr'] = df['positive_reframing']

    enum.append(column+"_pr")
    description += " "+"'"+column+"_pr' contains the positive reframing version of the '" + column +"' column. "

    dot.node(column+"_pr")
    dot.edge(column, column+"_pr", "get_positive_reframing")

    return table, enum, description, dot

def get_premise(table, column, type, term, enum, description, verbose, dot, client, gpt4):
    assert type == 'type'
    assert term == 'term'

    new_description = "'"+column+"_premise_entail' contains the premises (literal texts) that entail the hypothesis (figurative texts) contained in the '" + column +"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) == 0:
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/12/figurative.csv')
        df = pd.read_csv(data_file_path)
        table[column+'_premise_entail'] = df['premise_e']
        enum.append(column+'_premise_entail')
        description += " "+"'"+column+"_premise_entail' contains the premises (literal texts) that entail the hypothesis (figurative texts) contained in the '" + column +"' column. "
        dot.node(column+"_premise_entail")
        dot.edge(column, column+"_premise_entail", "get_premise")
        dot.edge(type, column+"_premise_entail", "get_premise")
        dot.edge(term, column+"_premise_entail", "get_premise")
    else:
        dot.edge(column, col, "get_premise")
        dot.edge(type, col, "get_premise")
        dot.edge(term, col, "get_premise")
    
    new_description = "'"+column+"_premise_contradict' contains the premises (literal texts) that contradict the hypothesis (figurative texts) contained in the '" + column +"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_premise")
        dot.edge(type, col, "get_premise")
        dot.edge(term, col, "get_premise")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/12/figurative.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_premise_contradict'] = df['premise_c']
    enum.append(column+'_premise_contradict')
    description += " "+"'"+column+"_premise_contradict' contains the premises (literal texts) that contradict the hypothesis (figurative texts) contained in the '" + column +"' column. "

    dot.node(column+"_premise_contradict")
    dot.edge(column, column+"_premise_contradict", "get_premise")
    dot.edge(type, column+"_premise_contradict", "get_premise")
    dot.edge(term, column+"_premise_contradict", "get_premise")

    return table, enum, description, dot

def get_premise_explanation(table, column, hypothesis, type, term, label, enum, description, verbose, dot, client, gpt4):
    assert hypothesis == 'hypothesis'
    assert label in ["contradict", "entail"]
    assert label in column
    assert type == 'type'
    assert term == 'term'

    if label == "contradict":
        new_description = "'"+column+"_explanation' contains the explanation of the premises (literal texts) that contradict the hypothesis (figurative texts) contained in the '" + column +"' column. "
        col = check_alias(enum, description, new_description, verbose, client, gpt4)
        if len(col) > 0:
            dot.edge(column, col, "get_premise_explanation")
            dot.edge(hypothesis, col, "get_premise_explanation")
            dot.edge(type, col, "get_premise_explanation")
            dot.edge(term, col, "get_premise_explanation")
            return table, enum, description, dot
        
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/12/figurative.csv')
        df = pd.read_csv(data_file_path)
        table[column+'_explanation'] = df['explanation_c']

        enum.append(column+'_explanation')
        description += " "+"'"+column+"_explanation' contains the explanation of the premises (literal texts) that contradict the hypothesis (figurative texts) contained in the '" + column +"' column. "
    else:
        new_description = "'"+column+"_explanation' contains the explanation of the premises (literal texts) that entail the hypothesis (figurative texts) contained in the '" + column +"' column. "
        col = check_alias(enum, description, new_description, verbose, client, gpt4)
        if len(col) > 0:
            dot.edge(column, col, "get_premise_explanation")
            dot.edge(hypothesis, col, "get_premise_explanation")
            dot.edge(type, col, "get_premise_explanation")
            dot.edge(term, col, "get_premise_explanation")
            return table, enum, description, dot
        
        data_file_path = pkg_resources.resource_filename('autopipeline', 'data/12/figurative.csv')
        df = pd.read_csv(data_file_path)
        table[column+'_explanation'] = df['explanation_e']

        enum.append(column+'_explanation')
        description += " "+"'"+column+"_explanation' contains the explanation of the premises (literal texts) that entail the hypothesis (figurative texts) contained in the '" + column +"' column. "

    dot.node(column+"_explanation")
    dot.edge(column, column+"_explanation", "get_premise_explanation")
    dot.edge(hypothesis, column+"_explanation", "get_premise_explanation")
    dot.edge(type, column+"_explanation", "get_premise_explanation")
    dot.edge(term, column+"_explanation", "get_premise_explanation")

    return table, enum, description, dot

def get_change_opinion(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_change_opinion' contains the whether the post in the '" + column +"' column is persuaded and changed their opinion. IMPORTANT: the '"+column+"_change_opinion' column contains boolean values, where True for posts that changed their opinion, False for posts that do not change their opinion."
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_change_opinion")
        return table, enum, description, dot

    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/1/persuasive.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_change_opinion'] = df['change_opinion']

    enum.append(column+'_change_opinion')
    description += " "+"'"+column+"_change_opinion' contains the whether the post in the '" + column +"' column is persuaded and changed their opinion. IMPORTANT: the '"+column+"_change_opinion' column contains boolean values, where True for posts that changed their opinion, False for posts that do not change their opinion."

    dot.node(column+"_change_opinion")
    dot.edge(column, column+"_change_opinion", "get_change_opinion")

    return table, enum, description, dot

def get_persuasion_effect(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_persuasion_effect' contains the persuasion effects of the post included in the '" + column +"' column. IMPORTANT: the '"+column+"_persuasion_effect' column contains numerical values ranging from 0.0 to 1.0, where 0.0 stands for the least persuasive, and 1.0 stands for the most persuasive."
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_persuasion_effect")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/17/persuasive-17.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_persuasion_effect'] = df['persuasion_effect']

    enum.append(column+"_persuasion_effect")
    description += " "+"'"+column+"_persuasion_effect' contains the persuasion effects of the post included in the '" + column +"' column. IMPORTANT: the '"+column+"_persuasion_effect' column contains numerical values ranging from 0.0 to 1.0, where 0.0 stands for the least persuasive, and 1.0 stands for the most persuasive."

    dot.node(column+"_persuasion_effect")
    dot.edge(column, column+"_persuasion_effect", "get_persuasion_effect")

    return table, enum, description, dot

def get_intent(table, column, enum, description, verbose, dot, client, gpt4):
    assert column == 'headline'
    new_description = "'"+column+"_intent' column provides writers' intent for the texts in the '"+column+"' column'; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_intent")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/14/headlines.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_reader_perception'] = df['effect_on_reader']

    enum.append(column+"_intent")
    description += " " + "'"+column+"_intent' column provides writers' intent for the texts in the '"+column+"' column'; "

    dot.node(column+"_intent")
    dot.edge(column, column+"_intent", "get_intent")

    return table, enum, description, dot

def get_reader_perception(table, column, enum, description, verbose, dot, client, gpt4):
    assert column == 'headline'
    new_description = "'"+column+"_reader_perception' column provides readers' perceptions for the texts in the '"+column+"' column'; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_reader_perception")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/14/headlines.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_reader_perception'] = df['effect_on_reader']

    enum.append(column+"_reader_perception")
    description += " " + "'"+column+"_reader_perception' column provides readers' perceptions for the texts in the '"+column+"' column'; "

    dot.node(column+"_reader_perception")
    dot.edge(column, column+"_reader_perception", "get_reader_perception")

    return table, enum, description, dot

def get_spread_likelihood(table, column, enum, description, verbose, dot, client, gpt4):
    assert column.endswith('reader_perception')
    new_description = "'"+column+"_spread' column contains the (numerical) likelihood to spread based on the reader perceptions provided in the '"+column+"' column'; IMPORTANT: the '"+column+"_spread' column contains numerical values ranging from 0.0 to 5.0, where 0.0 stands for the least likely to spread, and 5.0 stands for the most likely to spread."
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_spread_likelihood")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/14/headlines.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_spread'] = df['spread']

    enum.append(column+"_spread")
    description += " " + "'"+column+"_spread' column contains the (numerical) likelihood to spread based on the reader perceptions provided in the '"+column+"' column'; IMPORTANT: the '"+column+"_spread' column contains numerical values ranging from 0.0 to 5.0, where 0.0 stands for the least likely to spread, and 5.0 stands for the most likely to spread."

    dot.node(column+"_spread")
    dot.edge(column, column+"_spread", "get_spread_likelihood")

    return table, enum, description, dot

def get_reader_action(table, column, enum, description, verbose, dot, client, gpt4):
    assert column.endswith('intent')
    new_description = "'"+column+"_reader_action' column contains the inferred readers' actions based on the writers' intent provided in the '"+column+"' column'; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_reader_action")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/14/headlines.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_reader_action'] = df['reader_action']

    enum.append(column+"_reader_action")
    description += " " + "'"+column+"_reader_action' column contains the inferred readers' actions based on the writers' intent provided in the '"+column+"' column'; "

    dot.node(column+"_reader_action")
    dot.edge(column, column+"_reader_action", "get_reader_action")
    
    return table, enum, description, dot

def get_dialect(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_dialect' column contains the identified categorical dialect features identified from the sentences in the '"+column+"' column'; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_dialect")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/15/dialect.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_dialect'] = df['dialect']

    enum.append(column+"_dialect")
    description += " " + "'"+column+"_dialect' column contains the identified categorical dialect features identified from the sentences in the '"+column+"' column'; "

    dot.node(column+"_dialect")
    dot.edge(column, column+"_dialect", "get_dialect")

    return table, enum, description, dot

def get_disclosure(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_disclosure' column contains the disclosure act types of the sentences in the '"+column+"' column'; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_disclosure")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/19/disclosure.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_disclosure'] = df['disclosure']

    enum.append(column+"_disclosure")
    description += " " + "'"+column+"_disclosure' column contains the disclosure act types of the sentences in the '"+column+"' column'; "

    dot.node(column+"_disclosure")
    dot.edge(column, column+"_disclosure", "get_disclosure")

    return table, enum, description, dot

def get_semantic(table, word, word_type, index, sentence1, sentence2, enum, description, verbose, dot, client, gpt4):
    assert word == "word"
    assert word_type == "word_type"
    assert index == "index"
    assert sentence1 == "example_1"
    assert sentence2 == "example_2"

    new_description = "'"+word+"_semantic' column contains whether the word in the '"+word+"' column of type in the '"+word_type+" column'  has the same semantic in sentences in the '"+sentence1+"' column and the '"+sentence2+"' column, with indexes in the '"+index+"' column. IMPORTANT: the values in the '"+word+"_semantic' column can either be 'T', meaning that the semantic in the two sentences are the same, or 'F', meaning that the semantic in the two sentences are different. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(word, col, "get_semantic")
        dot.edge(word_type, col, "get_semantic")
        dot.edge(index, col, "get_semantic")
        dot.edge(sentence1, col, "get_semantic")
        dot.edge(sentence2, col, "get_semantic")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/18/semantic.csv')
    df = pd.read_csv(data_file_path)
    table[word+'_semantic'] = df['semantic_consistency']

    enum.append(word+"_semantic")
    description += " " + "'"+word+"_semantic' column contains whether the word in the '"+word+"' column of type in the '"+word_type+" column'  has the same semantic in sentences in the '"+sentence1+"' column and the '"+sentence2+"' column, with indexes in the '"+index+"' column. IMPORTANT: the values in the '"+word+"_semantic' column can either be 'T', meaning that the semantic in the two sentences are the same, or 'F', meaning that the semantic in the two sentences are different. "

    dot.node(word+'_semantic')
    dot.edge(word, word+'_semantic', "get_semantic")
    dot.edge(word_type, word+'_semantic', "get_semantic")
    dot.edge(index, word+'_semantic', "get_semantic")
    dot.edge(sentence1, word+'_semantic', "get_semantic")
    dot.edge(sentence2, word+'_semantic', "get_semantic")

    return table, enum, description, dot

def get_emotional_reaction_level(table, column, column_post, enum, description, verbose, dot, client, gpt4):
    assert column == "response_post"
    assert column_post == "seeker_post"
    new_description = "'"+column+"_emotional_reaction' column contains the numerical level of communication strength in terms of emotional reaction for the response post in the '"+column+"' column' towards the sad post in the '"+column_post+"' column. IMPORTANT: the numerical values in the '"+column+"_emotion_reaction' column are integers 0, 1, and 2, with 0 denotes the weakest level of communication, and 2 denotes the strongest level of communication. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_emotional_reaction_levels")
        dot.edge(column_post, col, "get_emotional_reaction_levels")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/20/empathy.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_emotional_reaction'] = df['emotional_reactions_level']

    enum.append(column+"_emotional_reaction")
    description += " " + "'"+column+"_emotional_reaction' column contains the numerical level of communication strength in terms of emotional reaction for the response post in the '"+column+"' column' towards the sad post in the '"+column_post+"' column. IMPORTANT: the numerical values in the '"+column+"_emotion_reaction' column are integers 0, 1, and 2, with 0 denotes the weakest level of communication, and 2 denotes the strongest level of communication. "

    dot.node(column+'_emotional_reaction')
    dot.edge(column, column+'_emotional_reaction', "get_emotional_reaction_levels")
    dot.edge(column_post, column+'_emotional_reaction', "get_emotional_reaction_levels")

    return table, enum, description, dot

def get_exploration_level(table, column, column_post, enum, description, verbose, dot, client, gpt4):
    assert column == "response_post"
    assert column_post == "seeker_post"
    new_description = "'"+column+"_exploration' column contains the numerical level of communication strength in terms of exploration for the response post in the '"+column+"' column' towards the sad post in the '"+column_post+"' column. IMPORTANT: the numerical values in the '"+column+"_exploration' column are integers 0, 1, and 2, with 0 denotes the weakest level of communication, and 2 denotes the strongest level of communication. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_exploration_level")
        dot.edge(column_post, col, "get_exploration_level")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/20/empathy.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_exploration'] = df['exploration_level']

    enum.append(column+"_exploration")
    description += " " + "'"+column+"_exploration' column contains the numerical level of communication strength in terms of exploration for the response post in the '"+column+"' column' towards the sad post in the '"+column_post+"' column. IMPORTANT: the numerical values in the '"+column+"_exploration' column are integers 0, 1, and 2, with 0 denotes the weakest level of communication, and 2 denotes the strongest level of communication. "

    dot.node(column+'_exploration')
    dot.edge(column, column+'_exploration', "get_exploration_level")
    dot.edge(column_post, column+'_exploration', "get_exploration_level")

    return table, enum, description, dot

def get_intepretation_level(table, column, column_post, enum, description, verbose, dot, client, gpt4):
    assert column == "response_post"
    assert column_post == "seeker_post"
    new_description = "'"+column+"_intepretation' column contains the numerical level of communication strength in terms of intepretation for the response post in the '"+column+"' column' towards the sad post in the '"+column_post+"' column. IMPORTANT: the numerical values in the '"+column+"_intepretation' column are integers 0, 1, and 2, with 0 denotes the weakest level of communication, and 2 denotes the strongest level of communication. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_intepretation_level")
        dot.edge(column_post, col, "get_intepretation_level")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/20/empathy.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_intepretation'] = df['intepretation_level']

    enum.append(column+"_intepretation")
    description += " " + "'"+column+"_intepretation' column contains the numerical level of communication strength in terms of intepretation for the response post in the '"+column+"' column' towards the sad post in the '"+column_post+"' column. IMPORTANT: the numerical values in the '"+column+"_intepretation' column are integers 0, 1, and 2, with 0 denotes the weakest level of communication, and 2 denotes the strongest level of communication. "

    dot.node(column+'_intepretation')
    dot.edge(column, column+'_intepretation', "get_intepretation_level")
    dot.edge(column_post, column+'_intepretation', "get_intepretation_level")

    return table, enum, description, dot

def get_politeness(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_polite' column contains whether the text in the '"+column+"' column' is polite or not. IMPORTANT: the values in the '"+column+"_polite' column can only be either 'T', meaning that the text is polite, or 'F', meaning that the text is not polite. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_politeness")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/23/politeness.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_polite'] = df['polite']

    enum.append(column+"_polite")
    description += " " + "'"+column+"_polite' column contains whether the text in the '"+column+"' column' is polite or not. IMPORTANT: the values in the '"+column+"_polite' column can only be either 'T', meaning that the text is polite, or 'F', meaning that the text is not polite. "

    dot.node(column+'_polite')
    dot.edge(column, column+'_polite', "get_politeness")

    return table, enum, description, dot

def get_humor(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_humor' column contains whether the text in the '"+column+"' column' is funny or not. IMPORTANT: the values in the '"+column+"_humor' column can only be either 'T', meaning that the text is funny, or 'F', meaning that the text is not funny. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_humor")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/22/humor.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_humor'] = df['humor']

    enum.append(column+"_humor")
    description += " " + "'"+column+"_humor' column contains whether the jokes in the '"+column+"' column' is funny or not. IMPORTANT: the values in the '"+column+"_humor' column can only be either 'T', meaning that the joke is funny, or 'F', meaning that the joke is not funny. "

    dot.node(column+'_humor')
    dot.edge(column, column+'_humor', "get_humor")

    return table, enum, description, dot

def get_toxic(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_toxic' column contains whether the text in the '"+column+"' column' will become toxic in the future or not. IMPORTANT: the values in the '"+column+"_toxic' column can only be either True, meaning that the text will become toxic in the future, or False, meaning that the text will not become toxic in the future. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_toxic")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/24/toxic.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_toxic'] = df['toxic']

    enum.append(column+"_toxic")
    description += " " + "'"+column+"_toxic' column contains whether the text in the '"+column+"' column' will become toxic in the future or not. IMPORTANT: the values in the '"+column+"_toxic' column can only be either True, meaning that the text will become toxic in the future, or False, meaning that the text will not become toxic in the future. "

    dot.node(column+'_toxic')
    dot.edge(column, column+'_toxic', "get_toxic")

    return table, enum, description, dot

def get_trope(table, column, enum, description, verbose, dot, client, gpt4):
    assert column == "Quotes"
    new_description = "'"+column+"_trope' column contains which trope type is the character who has the quotes in the '"+column+"' column'. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_trope")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/31/tropes.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_trope'] = df['TropesHuman']

    enum.append(column+"_trope")
    description += " " + "'"+column+"_trope' column contains which trope type is the character who has the quotes in the '"+column+"' column'. "

    dot.node(column+'_trope')
    dot.edge(column, column+'_trope', "get_trope")

    return table, enum, description, dot

def get_event_argument(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_event_argument' column contains the event arguments extracted from the text in the '"+column+"' column'; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_event_argument")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/28/event_argument.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_event_argument'] = df['event_argument']

    enum.append(column+"_event_argument")
    description += " " + "'"+column+"_event_argument' column contains the event arguments extracted from the text in the '"+column+"' column'; "

    dot.node(column+'_event_argument')
    dot.edge(column, column+'_event_argument', "get_event_argument")

    return table, enum, description, dot

def get_ideology_doc(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_ideology_doc' column contains the ideology of the document whose link is provided in the '"+column+"' column'; IMPORTANT: the categorical ideology values in the '"+column+"_ideology_doc' column can only be one of 0 (meaning left), 1 (meaning neutral), or 2 (meaning right). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_ideology_doc")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/29/ideology_document.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_ideology_doc'] = df['bias']

    enum.append(column+"_ideology_doc")
    description += " " + "'"+column+"_ideology_doc' column contains the ideology of the document whose link is provided in the '"+column+"' column'; IMPORTANT: the categorical ideology values in the '"+column+"_ideology_doc' column can only be one of 0 (meaning left), 1 (meaning neutral), or 2 (meaning right). "

    dot.node(column+'_ideology_doc')
    dot.edge(column, column+'_ideology_doc', "get_ideology_doc")

    return table, enum, description, dot

def get_ideology_sent(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_ideology_sent' column contains the ideology of the sentence in the '"+column+"' column'; IMPORTANT: the categorical ideology values in the '"+column+"_ideology_sent' column can only be one of 'Conservative', 'Liberal', or 'Neutral'. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_ideology_sent")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/30/ideology_sentence.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_ideology_sent'] = df['leaning']

    enum.append(column+"_ideology_sent")
    description += " " + "'"+column+"_ideology_sent' column contains the ideology of the sentence in the '"+column+"' column'; IMPORTANT: the categorical ideology values in the '"+column+"_ideology_sent' column can only be one of 'Conservative', 'Liberal', or 'Neutral'. "

    dot.node(column+'_ideology_sent')
    dot.edge(column, column+'_ideology_sent', "get_ideology_sent")

    return table, enum, description, dot

def story_gen(table, column, enum, description, verbose, dot, client, gpt4):
    assert column.endswith('summary')
    new_description = "'"+column+"_new_story' column contains the generated stories based on the summary provided in the '" + column + "' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "story_gen")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/25/story.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_new_story'] = df['story_gen']

    enum.append(column+"_new_story")
    description += " " + "'"+column+"_new_story' column contains the generated stories based on the summary provided in the '" + column + "' column. "

    dot.node(column+'_new_story')
    dot.edge(column, column+'_new_story', "story_gen")

    return table, enum, description, dot

def get_event(table, column, enum, description, verbose, dot, client, gpt4):
    assert column == 'sent'
    new_description = "'"+column+"_event_prob' column contains the probability of the text in the '"+column+"' column containing new events. IMPORTANT: the values in the '"+column+"_event_prob' column are floating point values ranging from 0.0 (the least likely to include events) to 1.0 (the most likely to include events). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_event")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/26/events.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_event_prob'] = df['eventOrNot']

    enum.append(column+"_event_prob")
    description += " " + "'"+column+"_event_prob' column contains the probability of the text in the '"+column+"' column containing new events. IMPORTANT: the values in the '"+column+"_event_prob' column are floating point values ranging from 0.0 (the least likely to include events) to 1.0 (the most likely to include events). "

    dot.node(column+"_event_prob")
    dot.edge(column, column+"_event_prob", "get_event")

    return table, enum, description, dot

def get_event_major(table, column, event, enum, description, verbose, dot, client, gpt4):
    assert event.endswith('event_prob')
    assert column == 'sent'
    new_description = "'"+column+"_event_major_prob' column contains the probability of events in the text in the '"+column+"' column being major events. IMPORTANT: the values in the '"+column+"_event_major_prob' column are floating point values ranging from 0.0 (the least likely to include events) to the value in the '"+event+"' column (the most likely to include events). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_event_major")
        dot.edge(event, col, "get_event_major")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/26/events.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_event_major_prob'] = df['majorBin']

    enum.append(column+"_event_major_prob")
    description += " " + "'"+column+"_event_major_prob' column contains the probability of events in the text in the '"+column+"' column being major events. IMPORTANT: the values in the '"+column+"_event_major_prob' column are floating point values ranging from 0.0 (the least likely to include events) to the value in the '"+event+"' column (the most likely to include events). "

    dot.node(column+"_event_major_prob")
    dot.edge(column, column+"_event_major_prob", "get_event_major")
    dot.edge(event, column+"_event_major_prob", "get_event_major")

    return table, enum, description, dot

def get_event_expected(table, column, event, enum, description, verbose, dot, client, gpt4):
    assert event.endswith('event_prob')
    assert column == 'sent'
    new_description = "'"+column+"_event_expected_prob' column contains the probability of events in the text in the '"+column+"' column being events as expected. IMPORTANT: the values in the '"+column+"_event_expected_prob' column are floating point values ranging from 0.0 (the least likely to include events) to the value in the '"+event+"' column (the most likely to include events). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_event_expected")
        dot.edge(event, col, "get_event_expected")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/26/events.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_event_expected_prob'] = df['expectedBin']

    enum.append(column+"_event_expected_prob")
    description += " " + "'"+column+"_event_expected_prob' column contains the probability of events in the text in the '"+column+"' column being events as expected. IMPORTANT: the values in the '"+column+"_event_expected_prob' column are floating point values ranging from 0.0 (the least likely to include events) to the value in the '"+event+"' column (the most likely to include events). "

    dot.node(column+"_event_expected_prob")
    dot.edge(column, column+"_event_expected_prob", "get_event_expected")
    dot.edge(event, column+"_event_expected_prob", "get_event_expected")

    return table, enum, description, dot

def get_story_type(table, column, storyIx, sentIx, event, event_major, event_expected, enum, description, verbose, dot, client, gpt4):
    assert column == 'sent'
    assert storyIx == 'storyIx'
    assert sentIx == 'sentIx'
    assert event.endswith('event_prob')
    assert event_major.endswith('event_major_prob')
    assert event_expected.endswith('event_expected_prob')

    new_description = "'"+column+"_story_type' column contains the type of story in the text in the '"+column+"' column. IMPORTANT: the values in the '"+column+"_story_type' column are categorical values being one of 'imagined', 'recalled', or 'retold'. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_story_type")
        dot.edge(storyIx, col, "get_story_type")
        dot.edge(sentIx, col, "get_story_type")
        dot.edge(event, col, "get_story_type")
        dot.edge(event_major, col, "get_story_type")
        dot.edge(event_expected, col, "get_story_type")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/26/events.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_story_type'] = df['memType']

    enum.append(column+"_story_type")
    description += " " + "'"+column+"_story_type' column contains the type of story in the text in the '"+column+"' column. IMPORTANT: the values in the '"+column+"_story_type' column are categorical values being one of 'imagined', 'recalled', or 'retold'. "

    dot.node(column+"_story_type")
    dot.edge(column, column+"_story_type", "get_story_type")
    dot.edge(storyIx, column+"_story_type", "get_story_type")
    dot.edge(sentIx, column+"_story_type", "get_story_type")
    dot.edge(event, column+"_story_type", "get_story_type")
    dot.edge(event_major, column+"_story_type", "get_story_type")
    dot.edge(event_expected, column+"_story_type", "get_story_type")

    return table, enum, description, dot

def get_strategy(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_strategy' column contains the emotional support strategy to be used in response to the seekers' need in the text in the '"+column+"' column. IMPORTANT: the values in the '"+column+"_strategy' column are categorical values of strategy types being either 'Question', 'Restatement or Paraphrasing', 'Reflection of feelings', 'Self-disclosure', 'Affirmation and Reassurance', 'Providing Suggestions', 'Information', and 'Others'. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_strategy")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/32/emotional_support.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_strategy'] = df['strategy']

    enum.append(column+"_strategy")
    description += " " + "'"+column+"_strategy' column contains the emotional support strategy to be used in response to the seekers' need in the text in the '"+column+"' column. IMPORTANT: the values in the '"+column+"_strategy' column are categorical values of strategy types being either 'Question', 'Restatement or Paraphrasing', 'Reflection of feelings', 'Self-disclosure', 'Affirmation and Reassurance', 'Providing Suggestions', 'Information', or 'Others'. "

    dot.node(column+"_strategy")
    dot.edge(column, column+"_strategy", "get_strategy")

    return table, enum, description, dot

def get_supporter_response(table, column, strategy, enum, description, verbose, dot, client, gpt4):
    assert column == "seeker"
    assert strategy.endswith('strategy')
    new_description = "'"+column+"_supporter_reponse' column contains the emotional support response to the seekers' need in the text in the '"+column+"' column; "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_supporter_reponse")
        dot.edge(strategy, col, "get_supporter_reponse")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/32/emotional_support.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_supporter_reponse'] = df['supporter']

    enum.append(column+"_supporter_reponse")
    description += " " + "'"+column+"_supporter_reponse' column contains the emotional support response to the seekers' need in the text in the '"+column+"' column; "

    dot.node(column+"_supporter_reponse")
    dot.edge(column, column+"_supporter_reponse", "get_supporter_reponse")
    dot.edge(strategy, column+"_supporter_reponse", "get_supporter_reponse")

    return table, enum, description, dot

def get_smile(table, column, enum, description, verbose, dot, client, gpt4):
    assert column == 'id'
    new_description = "'"+column+"_smile' column contains whether the person smiled (True) in the video in the '"+column+"' column or not (False). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_smile")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/36/deception.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_smile'] = df['Smile']

    enum.append(column+"_smile")
    description += " " + "'"+column+"_smile' column contains whether the person smiled (True) in the video in the '"+column+"' column or not (False). "

    dot.node(column+"_smile")
    dot.edge(column, column+"_smile", "get_smile")

    return table, enum, description, dot

def get_scowl(table, column, enum, description, verbose, dot, client, gpt4):
    assert column == 'id'
    new_description = "'"+column+"_scowl' column contains whether the person scowled (True) in the video in the '"+column+"' column or not (False). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_scowl")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/36/deception.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_scowl'] = df['Scowl']

    enum.append(column+"_scowl")
    description += " " + "'"+column+"_scowl' column contains whether the person scowled (True) in the video in the '"+column+"' column or not (False). "

    dot.node(column+"_scowl")
    dot.edge(column, column+"_scowl", "get_scowl")

    return table, enum, description, dot

def get_deception(table, column, smile, scowl, enum, description, verbose, dot, client, gpt4):
    assert column == 'id'
    assert smile.endswith('smile')
    assert scowl.endswith('scowl')
    new_description = "'"+column+"_deception' column contains whether the person deceived in the video in the '"+column+"' column. IMPORTANT: the values in the '"+column+"_deception' column are categorical values being either 'deceptive' or 'truthful'. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_deception")
        dot.edge(smile, col, "get_deception")
        dot.edge(scowl, col, "get_deception")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/36/deception.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_deception'] = df['class']

    enum.append(column+"_deception")
    description += " " + "'"+column+"_deception' column contains whether the person deceived in the video in the '"+column+"' column. IMPORTANT: the values in the '"+column+"_deception' column are categorical values being either 'deceptive', meaning that the person deceived, or 'truthful', meaning that the person did not deceive. "

    dot.node(column+"_deception")
    dot.edge(column, column+"_deception", "get_deception")
    dot.edge(smile, column+"_deception", "get_deception")
    dot.edge(scowl, column+"_deception", "get_deception")

    return table, enum, description, dot

def get_relationship(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_relationship' column contains the relationship between the two people in the conversation in the '"+column+"' column being either 'social' or 'romance'. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_relationship")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/35/relationship.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_relationship'] = df['relationship-category']

    enum.append(column+"_relationship")
    description += " " + "'"+column+"_relationship' column contains the relationship between the two people in the conversation in the '"+column+"' column being either 'social' or 'romance'. "

    dot.node(column+"_relationship")
    dot.edge(column, column+"_relationship", "get_relationship")

    return table, enum, description, dot

def get_bias_score(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_bias_score' column contains the (numerical) bias score of the word in the '"+column+"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_bias_score")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/38/word_bias.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_bias_score'] = df['transformed_score']

    enum.append(column+"_bias_score")
    description += " " + "'"+column+"_bias_score' column contains the (numerical) bias score of the word in the '"+column+"' column. "

    dot.node(column+"_bias_score")
    dot.edge(column, column+"_bias_score", "get_bias_score")

    return table, enum, description, dot

def get_request_succeed(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_request_succeed' column contains whether the request in the '"+column+"' column succeeded (True) or not (False). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_request_succeed")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/37/requests.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_request_succeed'] = df['request_succeed']

    enum.append(column+"_request_succeed")
    description += " " + "'"+column+"_request_succeed' column contains whether the request in the '"+column+"' column succeeded (True) or not (False). "

    dot.node(column+"_request_succeed")
    dot.edge(column, column+"_request_succeed", "get_request_succeed")

    return table, enum, description, dot

def get_deep_fake(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_deep_fake' column contains whether the video in the '"+column+"' column is fake (True) or not (False). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_deep_fake")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/39/deepfake.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_deep_fake'] = df['fake']

    enum.append(column+"_deep_fake")
    description += " " + "'"+column+"_deep_fake' column contains whether the video in the '"+column+"' column is fake (True) or not (False). "

    dot.node(column+"_deep_fake")
    dot.edge(column, column+"_deep_fake", "get_deep_fake")

    return table, enum, description, dot

def get_power(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_power' column contains whether the conversation in the '"+column+"' column is in the position of power (True) or not (False). "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_power")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/40/power.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_power'] = df['power']

    enum.append(column+"_power")
    description += " " + "'"+column+"_power' column contains whether the conversation in the '"+column+"' column is in the position of power (True) or not (False). "

    dot.node(column+"_power")
    dot.edge(column, column+"_power", "get_power")

    return table, enum, description, dot

def get_request_strategy(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_request_strategy' column contains the request strategy used in the request in the '"+column+"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_request_strategy")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/81/raop.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_request_strategy'] = df['labels']

    enum.append(column+"_request_strategy")
    description += " " + "'"+column+"_request_strategy' column contains the request strategy used in the request in the '"+column+"' column. "

    dot.node(column+"_request_strategy")
    dot.edge(column, column+"_request_strategy", "get_request_strategy")

    return table, enum, description, dot

def get_translation(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_translation' column contains the translation of the words in the '"+column+"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_translation")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/113/translation.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_translation'] = df['Yoruba']

    enum.append(column+"_translation")
    description += " " + "'"+column+"_translation' column contains the translation of the words in the '"+column+"' column. "

    dot.node(column+"_translation")
    dot.edge(column, column+"_translation", "get_translation")

    return table, enum, description, dot

def get_ecs(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_ecs' column contains the echo chamber effect score of the texts in the '"+column+"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_ecs")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/115/ecs.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_ecs'] = df['allsides_score']

    enum.append(column+"_ecs")
    description += " " + "'"+column+"_ecs' column contains the echo chamber effect score of the texts in the '"+column+"' column. "

    dot.node(column+"_ecs")
    dot.edge(column, column+"_ecs", "get_ecs")

    return table, enum, description, dot

def get_offensive_score(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_off' column contains the offensive score of the texts in the '"+column+"' column, the scores are continuous values ranging from 2.0 to 4.0. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_offensive_score")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/111/offensive.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_off'] = df['post_score']

    enum.append(column+"_off")
    description += " " + "'"+column+"_off' column contains the offensive score of the texts in the '"+column+"' column, the scores are continuous values ranging from 2.0 to 4.0. "

    dot.node(column+"_off")
    dot.edge(column, column+"_off", "get_offensive_score")

    return table, enum, description, dot

def get_defense(table, column, enum, description, verbose, dot, client, gpt4):
    new_description = "'"+column+"_defense' column contains the defense against the texts in the '"+column+"' column. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_defense")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/111/offensive.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_defense'] = df['gpt2_text']

    enum.append(column+'_defense')
    description += " " + "'"+column+"_defense' column contains the defense against the texts in the '"+column+"' column. "

    dot.node(column+'_defense')
    dot.edge(column, column+'_defense', "get_defense")

    return table, enum, description, dot

def get_informative(table, column, enum, description, verbose, dot, client, gpt4):
    assert column.endswith('defense')
    new_description = "'"+column+"_info' column contains the informative scores of the texts in the '"+column+"' column, the scores are continuous values ranging from 0.0 to 2.666666666666666. "
    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        dot.edge(column, col, "get_informative")
        return table, enum, description, dot
    
    data_file_path = pkg_resources.resource_filename('autopipeline', 'data/111/offensive.csv')
    df = pd.read_csv(data_file_path)
    table[column+'_info'] = df['gpt2_info_score']

    enum.append(column+'_info')
    description += " " + "'"+column+"_info' column contains the informative scores of the texts in the '"+column+"' column, the scores are continuous values ranging from 0.0 to 2.666666666666666. "

    dot.node(column+'_info')
    dot.edge(column, column+'_info', "get_informative")

    return table, enum, description, dot

def tree_path(user_query, columns, description, verbose, client, gpt4, udf=""):
    function_tree = '''
        "text_processors" <contains functions that process texts>:
            {
            "para_sep": "This function takes in one of the columns as input, split the text according to paragraphs, and generates an additional rows and columns to store the list of paragraphs."
            "pdf_to_text": "This function takes in one of the columns as input, transforms the pdf in that column into plain text, and generate an additional column to store the plain text." 
            }
        "augmentor" <contains functions that augment texts>:
            {
            "story_gen": "This function takes in one of the columns of story summaries as input, generate a story based on the summaries in that column, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_summary'."
            }
        "summarizer" <contains functions that summarize texts>:
            {
            "get_summary": "This function takes in one of the columns as input, summarizes the contents in that column, and generate an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'story_gen'."
            "get_intent": "This function takes in one of the columns as input, retrieve the intent of the writer of text in that column, and generate an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_reader_action'."
            }
        "calculator" <contains functions that calculate numerical scores from text>:
            {
            "continuous_calculator" <contains functions that calculate continuous numerical scores from text>:
                {
                "get_persuasion_effect": "This function takes in one of the columns as input, calculates the (numerical) persuasion effect score of the contents in that column, and generate an additional column to include those."
                "get_spread_likelihood": "This function takes in one of the columns of readers' perceptions as input, calculates the (numerical) spread likelihood based on readers' perceptions in that column, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_reader_perception'."
                "get_event": "This function takes in one of the columns as input, calculates the (numerical) probability that the contents in that column contain new events, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed at least once before calling 'get_event_expected', this function has to be executed at least once before calling 'get_event_major', this function has to be executed at least once before calling 'get_story_type'."
                "get_event_major": "This function takes in one of the columns as input, calculates the (numerical) probability that the contents in that column contain major events, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_event', this function has to be executed at least once before calling 'get_story_type'."
                "get_event_expected": "This function takes in one of the columns as input, calculates the (numerical) probability that the contents in that column contain events that are as expected, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_event', this function has to be executed at least once before calling 'get_story_type'."
                "get_bias_score": "This function takes in one of the columns as input, calculates the (numerical) bias score of the word in that column, and generate an additional column to include those. "
                "get_informative": "This function takes in one of the columns of texts as input, calculates the (numerical) informative score of the texts, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_defense'."
                "get_offensive_score": "This function takes in one of the columns of texts as input, calculates the (numerical) offensive score of the texts, and generate an additional column to include those. "
                "get_ecs": "This function takes in one of the columns of texts as input, calculates the (numerical) Echo Chamber Effect score of the texts, and generate an additional column to include those. "
                }
            "discrete_calculator" <contains functions that calculate discrete numerical scores from text>:
                {
                "get_emotional_reaction_level": "This function takes in one column of original sad post and one column of response post as input, calculates the (numerical) level of communication strength in terms of emotional reaction for the contents in the response post column, and generate an additional column to include the integer level."
                "get_exploration_level": "This function takes in one column of original sad post and one column of response post as input, calculates the (numerical) level of communication strength in terms of exploration for the contents in the response post column, and generate an additional column to include the integer level."
                "get_intepretation_level": "This function takes in one column of original sad post and one column of response post as input, calculates the (numerical) level of communication strength in terms of intepretation for the contents in the response post column, and generate an additional column to include the integer level."
                }
            }
        "classifier" <contains functions that classify texts>:
            {
            "binary_classifier" <contains functions that classify texts into binary (+ undefined if any) categories>:
                {
                "get_change_opinion": "This function takes in one of the columns as input, classifies whether the contents in that column changes opinion, and generate an additional column to include those."
                "get_semantic": "This function takes in a word, the type of the word, two sentences, and the indexes in the sentences as input, classifies whether the word in the two sentences has the same semantic, and generates an additional column that includes 'T' or 'F' accordingly."
                "get_humor": "This function takes in one of the columns as input, classifies whether the contents in that column is funny, and generate an additional column to include those."
                "get_polite": "This function takes in one of the columns as input, classifies whether the contents in that column is polite, and generate an additional column to include those."
                "get_toxic": "This function takes in one of the columns as input, classifies whether the contents in that column will become toxic in the future, and generate an additional column to include those."
                "get_relationship": "This function takes in one of the columns as input, classifies if the conversation happens between social or romance type of relationship, and generate an additional column to include those."
                "get_request_succeed": "This function takes in one of the columns as input, classifies whether the requests in that column succeeded or not, and generate an additional column to include those."
                "get_power": "This function takes in one of the columns as input, classifies whether the conversation in that column is in the position of power or not, and generate an additional column to include those."
                }
            "ternary_classifier" <contains functions that classify texts into binary + neutral categories>:
                {
                "get_stance": "This function takes in one column of text and one column of target topic as input, extracts the stance of 'AGAINST', 'FAVOR', or 'NONE' in the text column that towards the target topic, and generates a new column to include those."
                "get_sentiment": "This function takes in one of the columns as input, applies sentiment analysis on the content of that column, and generates an additional column labeling the content with its sentiment class (positive, negative, neutral, etc)."
                "get_ideology_doc": "This function takes in one of the columns that contains the documents as input, applies document-level ideology classification on that column, and generates an additional column labeling the content as 0 (meaning left), 1 (meaning neutral), or 2 (meaning right)."
                "get_ideology_sent": "This function takes in one of the columns that contains texts as input, applies sentence-level ideology classification on that column, and generates an additional column labeling the content as 'Conservative', 'Liberal', or 'Neutral'."
                }
            "multiple_classifier" <contains functions that classify texts into multiple (larger or equal to three meaningful) categories>:
                {
                "get_dialect": "This function takes in one of the columns as input, classifies the dialect features identified from the content of that column, and generate an additional column to include those."
                "get_disclosure": "This function takes in one of the columns as input, classifies the content of that column into different disclosure acts, and generate an additional column to include those."
                "get_emotion": "This function takes in one of the columns as input, applies emotion classification on the content of that column, and generates an additional column labeling the content with its emotion class (sad, joy, angry, etc). IMPORTANT: this function has to be executed at least once before calling 'get_trigger'."
                "get_hate_class": "This function takes in one of the columns as input, applies (fine-grained) implicit hate speech classification on the content of that column, and generates an additional column labeling the content as 'white_grievance', 'incitement', 'inferiority', 'irony', 'stereotypical', 'threatening', or 'other'."
                "get_dog_whistle_persona_ingroup": "This function takes in one of the columns of dog whistle terms, obtain the target persona/in-group of that dog whistle term, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_dog_whistle'."
                "get_dog_whistle_type": "This function takes in one of the columns of dog whistle terms, obtain the type of that dog whistle term, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_dog_whistle'."
                "get_story_type": "This function takes in one column of sentence, one column of story index of where the sentence belong, one column of sentence index of that sentence, one column of the probability of the sentence containing new events, one column of probability of the sentence containing major new events, and one column of probabiliy of the sentence containing new events as expected as input, classifies the story type of the sentence into 'imagined', 'recalled', or 'retold', and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_event', this function has to be executed after at least one call of 'get_event_major', this function has to be executed after at least one call of 'get_event_expected'."
                "get_strategy": "This function takes in one of the columns of the texts that need emotional support as input, classifies which type of emotional support strategy (one of 'Question', 'Restatement or Paraphrasing', 'Reflection of feelings', 'Self-disclosure', 'Affirmation and Reassurance', 'Providing Suggestions', 'Information', or 'Others') is needed, and generate an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_supporter_response'."
                "get_trope": "This function takes in one of the columns that contains the quotes of a character as input, applies trope classification based on the content of the quotes, and generates an additional column to include those."
                "get_request_strategy": "This function takes in one of the columns that contains the requests, applies request strategy on the requests, and generates an additional column to include those."
                }
            }
        "extractor" <contains functions that extract messages from text>:
            {
                "text_extractor" <contains functions that extract messages of text level, i.e., extracted messages are part (e.g. words/phrases/sentences/...) of the original words.>:
                    {
                    "get_ner": "This function takes in one of the columns as input, get the name entities recognized in that column, and generate additional rows and columns to include those.",
                    "get_pos": "This function takes in one of the columns as input, get the part of speech (POS) in that column, and generate additional rows and columns to include those.",
                    "get_keyword": "This function takes in one of the columns as input, get the top 5 keywords recognized in that column, and generate an additional column to include those.",
                    "get_trigger": "This function takes in one column of text and one column of emotion class as input, extracts the trigger in the text column that triggers a specific emotion in the emotion class column, and generates a new column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_emotion'."
                    "get_dog_whistle": "This function takes in one of the columns as input, extract the dog whistle term in that column, and generate an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_dog_whistle_type', this function has to be executed at least once before calling 'get_dog_whistle_persona_ingroup'."
                    "get_event_argument": "This function takes in one of the columns as input, get the event arguments recognized in that column, and generate additional rows and columns to include those."
                    }
                "information_extractor" <contains functions that extract high-level messages based on actual conveyed information>:
                    {
                    "get_hate_target": "This function takes in one of the columns as input, applies implicit hate speech target identification on the content of that column, and generates an additional column of free text labeling the target identified from the content."
                    "get_hate_implied": "This function takes in one of the columns as input, applies implicit hate speech implied statement extraction on the content of that column, and generates an additional column of free text labeling the implied statement extracted from the content."
                    "get_positive_reframing": "This function takes in one of the columns as input, extract the positive aspects of the content of that column and transforms it into a positive reframing version, and generates an additional column of positive reframing version of the content."
                    "get_premise": "This function takes in one column of figurative text, one column of figurative type, and one column of figurative term as input, extracts the literal text, i.e., the premise, of the figurative text column, and generates a new column to include those."
                    "get_premise_explanation": "This function takes in one column of premise of figurative text, one column of the original figurative text, one column of figurative type, one column of figurative term as input, and one parameter labelling whether the premises entail or contract original figurative texts as input, extracts the explanations of literal texts, i.e., the premises, of the figurative text column, and generates a new column to include those."
                    "get_translation": "This function takes in one column of words as input, extract the translations of the words, and generates an additional column to include those."
                    }
                "causal_extractor" <contains functions that extract the causal relationships inferred from text>:
                    {
                    "get_reader_action": "This function takes in one of the columns of writers' intent as input, get the reader action inferred from the writers' intent of that column, and generate additional an column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_intent'."
                    "get_reader_perception": "This function takes in one of the columns as input, infers readers' perceptions of text in that column, and generates an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_spread_likelihood'."
                    }
            }
        "chatter" <contains functions that generate texts in response>
            {
            "get_supporter_response": "This function takes in one of the columns of the texts that need emotional support and one column of strategy to be used as input, generates response that provides emotional support, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_strategy'."
            "get_defense": "This function takes in one of the columns of texts as input, generates the defense against the texts in that column, and generate an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_informative'."
            }
        "detector" <contains functions that detect certain features>
            {
            "get_misinfo": "This function takes in one of the columns as input, applies misinformation detection on the content of that column, and generates an additional column labeling the content as 'misinfo' (misinformation detected) or 'real' (no misinformation detected)."
            "get_hate": "This function takes in one of the columns as input, applies (high-level) hate speech detection on the content of that column, and generates an additional column labeling the content as 'implicit_hate', 'explicit_hate', or 'not_hate'."
            "get_deep_fake": "This function takes in one of the columns as input, detects whether the video in that column is fake or not, and generate an additional column to include those."
            "get_smile": "This function takes in one of the columns containing videos as input, applies smile detection on the person inside the videos in that column, and generates an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_deceptive'."
            "get_scowl": "This function takes in one of the columns containing videos as input, applies scowl detection on the person inside the videos in that column, and generates an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_deceptive'."
            "get_deception": "This function takes in one of the columns containing videos, one column containing whether the person in the video smiled or not, and one column containing whether the person in the video scowled or not as input, applies deception detection on the person inside the videos in that column, and generates an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_smile'; this function has to be executed after at least one call of 'get_scowl'."
            }
    '''
    function_tree += udf
    messages = [
        {
            "role": "system",
            "content": "Given a table with columns: " + str(columns) 
            + " where " + description 
            + " You are also given the function tree to be applied to different columns: " + function_tree 
            + '''Your task is:
                if there are function(s) needed to be applied, you should first identify which function to apply next, and then return the path of the group where the function to be applied next belongs;
                else, return an empty dictionary ({})

                Your output format can ONLY be a dictionary of two key elements: {"function": "{name of the function to be applied}", "path": {LIST of function tree path to the function in order}}.

                You should NOT include the function name as part of the path.
                '''
            + '''When selecting functions, you HAVE TO take the function chain and logic generated in the forward planning phase into consideration. '''
        },
        {
            "role": "user",
            "content": "I want to count the number of positive paragraphs in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file; The function chain I have is ['pdf_to_text', 'para_sep', 'get_sentiment'], current executed function chain is []."
        },
        {
            "role": "assistant",
            "content": "{'function': 'pdf_to_text', 'path': ['text_processors']}",
        },
        {
            "role": "user",
            "content": "I want to count the number of positive paragraphs in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file; 'pdf_orig_text' column is the plain text content of the 'pdf_orig' column; The function chain I have is ['pdf_to_text', 'para_sep', 'get_sentiment'], current executed function chain is ['pdf_to_text']."
        },
        {
            "role": "assistant",
            "content": "{'function': 'para_sep', 'path': ['text_processors']}",
        },
        {
            "role": "user",
            "content": "I want to count the number of positive paragraphs in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file; 'pdf_orig_text' column is the plain text content of the 'pdf_orig' column; 'pdf_orig_text_segment' stores the paragraph segments of the 'pdf_orig_text' column, the original text has empty value; 'pdf_orig_text_segmentid' column stores the paragraph index according to the order of the 'pdf_orig_text_segment' column, starts with 0, and the original text has value -1; The function chain I have is ['pdf_to_text', 'para_sep', 'get_sentiment'], current executed function chain is ['pdf_to_text', 'para_sep']."
        },
        {
            "role": "assistant",
            "content": "{'function': 'get_sentiment', 'path': ['classifier']}",
        },
        {
            "role": "user",
            "content": "I want to count the number of positive paragraphs in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file; 'pdf_orig_text' column is the plain text content of the 'pdf_orig' column; 'pdf_orig_text_segment' stores the paragraph segments of the 'pdf_orig_text' column, the original text has empty value; 'pdf_orig_text_segmentid' column stores the paragraph index according to the order of the 'pdf_orig_text_segment' column, starts with 0, and the original text has value -1; 'pdf_orig_text_segment_sentiment' column is the sentiment of the content of the 'pdf_orig_text_segment' column; The function chain I have is ['pdf_to_text', 'para_sep', 'get_sentiment'], current executed function chain is ['pdf_to_text', 'para_sep', 'get_sentiment']."
        },
        {
            "role": "assistant",
            "content": "{}",
        },
        {
            "role": "user",
            "content": "I want to get the top 5 keywords of texts in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file; 'pdf_orig_text' column is the plain text content of the 'pdf_orig' column; The function chain I have is ['pdf_to_text', 'para_sep', 'get_keyword'], current executed function chain is ['pdf_to_text', 'para_sep']."
        },
        {
            "role": "assistant",
            "content": "{'function': 'get_keyword', 'path': ['extractor', 'information_extractor']}",
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    if gpt4:
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=messages,
        )

        if verbose:
            # num_token_msg = num_tokens_from_messages(messages, "gpt-4-0613")
            # print("Number of tokens of messages for 'tree_path': ", num_token_msg)
            print("VERBOSE:" + "Number of prompt tokens for 'tree_path': ", response.usage.prompt_tokens)
            print("VERBOSE:" + "Number of answer tokens for 'tree_path': ", response.usage.completion_tokens)
            print("VERBOSE:" + "Number of total tokens for 'tree_path': ", response.usage.total_tokens)
            current_pize = 0.00003 * response.usage.prompt_tokens + 0.00006 * response.usage.completion_tokens
            print("VERBOSE:" + f"Cost for 'tree_path': ${current_pize}")
            autopipeline.cost += current_pize
            print("VERBOSE:" + f"Accumulated cost: ${autopipeline.cost}")
    else:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
        )

        if verbose:
            # num_token_msg = num_tokens_from_messages(messages, "gpt-3.5-turbo-0125")
            # print("Number of tokens of messages for 'tree_path': ", num_token_msg)
            print("VERBOSE:" + "Number of prompt tokens for 'tree_path': ", response.usage.prompt_tokens)
            print("VERBOSE:" + "Number of answer tokens for 'tree_path': ", response.usage.completion_tokens)
            print("VERBOSE:" + "Number of total tokens for 'tree_path': ", response.usage.total_tokens)
            current_pize = 0.0000005 * response.usage.prompt_tokens + 0.0000015 * response.usage.completion_tokens
            print("VERBOSE:" + f"Cost for 'tree_path': ${current_pize}")
            autopipeline.cost += current_pize
            print("VERBOSE:" + f"Accumulated cost: ${autopipeline.cost}")

    return response.choices[0].message.content

def schema_gpt_tree(user_query, column_description, description, function_chain, function_list, verbose, client, gpt4, udf = None):
    tree = {
        "detector": [
            {
                "name": "get_deep_fake",
                "description": 
                ''' 
                This function takes in one of the columns as input, detects whether the video in that column is fake or not, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains videos to classify whether they are fake, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_misinfo",
                "description": "This function takes in one of the columns as input, applies misinformation detection on the content of that column, and generates an additional column labeling the content as 'misinfo' (misinformation detected) or 'real' (no misinformation detected).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column to apply misinformation detection. You have to select one column in the 'enum' field to apply misinformation detection based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_hate",
                "description": "This function takes in one of the columns as input, applies (high-level) hate speech detection on the content of that column, and generates an additional column labeling the content as 'implicit_hate', 'explicit_hate', or 'not_hate'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column to apply (high-level) hate speech detection. You have to select one column in the 'enum' field to apply (high-level) hate speech detection based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_smile",
                "description": "This function takes in one of the columns containing videos as input, applies smile detection on the person inside the videos in that column, and generates an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_deceptive'. ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column that contains the video to apply smile detection. You have to select one column in the 'enum' field based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_scowl",
                "description": "This function takes in one of the columns containing videos as input, applies scowl detection on the person inside the videos in that column, and generates an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_deceptive'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column that contains the video to apply scowl detection. You have to select one column in the 'enum' field based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_deception",
                "description": "This function takes in one of the columns containing videos, one column containing whether the person in the video smiled or not, and one column containing whether the person in the video scowled or not as input, applies deception detection on the person inside the videos in that column, and generates an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_smile'; this function has to be executed after at least one call of 'get_scowl'. ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column that contains the video to apply deception detection. You have to select one column in the 'enum' field based on the description: "+description,
                            "enum": column_description
                        },
                        "smile": {
                            "type": "string",
                            "description": "The column that contains information about whether the person in the video smiled or not. You have to select one column in the 'enum' field based on the description: "+description,
                            "enum": column_description
                        },
                        "scowl": {
                            "type": "string",
                            "description": "The column that contains information about whether the person in the video scowled or not. You have to select one column in the 'enum' field based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column", "smile", "scowl"]
                }
            },
        ],
        "chatter": [
            {
                "name": "get_defense",
                "description": 
                ''' 
                This function takes in one of the columns of texts as input, generates the defense against the texts in that column, and generate an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_informative'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to get defense, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_supporter_response",
                "description": 
                ''' 
                This function takes in one of the columns of the texts that need emotional support and one column of strategy to be used as input, generates response that provides emotional support, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_strategy'. 
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains texts that need emotional support, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "strategy": {
                        "type": "string",
                        "description": "the column that contains the type of strategy for emotional support, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column", "strategy"]
            },
        ],
        "text_processor": [],
        "summarizer": [
            {
                "name": "get_summary",
                "description": 
                ''' 
                This function takes in one of the columns as input, summarizes the contents in that column, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed at least once before calling 'story_gen'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to summarize, you have to select one in the 'enum' field to summarize based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_intent",
                "description": 
                ''' 
                This function takes in one of the columns that contain the text to be analyzed as input, retrieve the intent of the writer of text in that column, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed at least once before calling 'get_reader_action'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column of text based on which writers' intents are ANALYZED from, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
        ],
        "text_processors": [],
        "augmentor": [
            {
                "name": "story_gen",
                "description": 
                ''' 
                This function takes in one of the columns of story summaries as input, generate a story based on the summaries in that column, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_summary'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains story summaries, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
        ],
        "calculator": {
        "continuous_calculator": [
            {
                "name": "get_informative",
                "description": 
                ''' 
                This function takes in one of the columns of texts as input, calculates the (numerical) informative score of the texts, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_defense'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to calculate the informative scores, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_offensive_score",
                "description": 
                ''' 
                This function takes in one of the columns of texts as input, calculates the (numerical) offensive score of the texts, and generate an additional column to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to calculate the offensive scores, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_ecs",
                "description": 
                ''' 
                This function takes in one of the columns of texts as input, calculates the (numerical) Echo Chamber Effect score of the texts, and generate an additional column to include those. 
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to calculate the Echo Chamber Effect scores, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_event",
                "description": 
                ''' 
                This function takes in one of the columns as input, calculates the (numerical) probability that the contents in that column contain new events, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed at least once before calling 'get_event_expected'; this function has to be executed at least once before calling 'get_event_major'; this function has to be executed at least once before calling 'get_story_type'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to calculate the probability of containing new events, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_event_major",
                "description": 
                ''' 
                This function takes in one of the columns as input, calculates the (numerical) probability that the contents in that column contain major events, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_event'; this function has to be executed at least once before calling 'get_story_type'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to calculate the probability of containing new major events, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "event": {
                        "type": "string",
                        "description": "the column that contains the probability of containing new events, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column", "event"]
            },
            {
                "name": "get_event_expected",
                "description": 
                ''' 
                This function takes in one of the columns as input, calculates the (numerical) probability that the contents in that column contain events that are as expected, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_event'; this function has to be executed at least once before calling 'get_story_type'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to calculate the probability of containing new events that are as expected, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "event": {
                        "type": "string",
                        "description": "the column that contains the probability of containing new events, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column", "event"]
            },
            {
                "name": "get_persuasion_effect",
                "description": 
                ''' 
                This function takes in one of the columns as input, calculates the (numerical) persuasion effect score of the contents in that column, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to calculate persuasion effect score, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_bias_score",
                "description": 
                ''' 
                This function takes in one of the columns as input, calculates the (numerical) bias score of the word in that column, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to calculate bias score, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_spread_likelihood",
                "description": 
                ''' 
                This function takes in one of the columns of readers' perceptions as input, calculates the (numerical) spread likelihood based on readers' perceptions in that column, and generate an additional column to include those. Don't select this if none of the columns matches the user query. If there is no column denoting readers' perceptions, this function should not be selected. IMPORTANT: this function has to be executed after at least one call of 'get_reader_perception'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains readers' perceptions to calculate the spread likelihood, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
        ],
        "discrete_calculator": [
            {
                "name": "get_emotional_reaction_level",
                "description": 
                ''' 
                This function takes in one column of original sad post and one column of response post as input, calculates the (numerical) level of communication strength in terms of emotional reaction for the contents in the response post column, and generate an additional column to include the integer level. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {
                    "column_post": {
                        "type": "string",
                        "description": "the column that includes the original sad post, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "column": {
                        "type": "string",
                        "description": "the column that includes the reponse, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column", "column_post"]
            },
            {
                "name": "get_exploration_level",
                "description": 
                ''' 
                This function takes in one column of original sad post and one column of response post as input, calculates the (numerical) level of communication strength in terms of exploration for the contents in the response post column, and generate an additional column to include the integer level. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {
                    "column_post": {
                        "type": "string",
                        "description": "the column that includes the original sad post, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "column": {
                        "type": "string",
                        "description": "the column that includes the reponse, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column", "column_post"]
            },
            {
                "name": "get_intepretation_level",
                "description": 
                ''' 
                This function takes in one column of original sad post and one column of response post as input, calculates the (numerical) level of communication strength in terms of intepretation for the contents in the response post column, and generate an additional column to include the integer level. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {
                    "column_post": {
                        "type": "string",
                        "description": "the column that includes the original sad post, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "column": {
                        "type": "string",
                        "description": "the column that includes the reponse, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column", "column_post"]
            },
        ]
        },
        "classifier": {
        "binary_classifier": [
            {
                "name": "get_semantic",
                "description": 
                ''' 
                This function takes in a word, the type of the word, two sentences, and the indexes in the sentences as input, classifies whether the word in the two sentences has the same semantic, and generates an additional column that includes 'T' or 'F' accordingly. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {
                    "word": {
                        "type": "string",
                        "description": "the column that includes the word to classify semantic, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "word_type": {
                        "type": "string",
                        "description": "the column that includes the type of the word, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "sentence1": {
                        "type": "string",
                        "description": "the column that includes the first sentence to inspect semantics, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "sentence2": {
                        "type": "string",
                        "description": "the column that includes the second sentence to inspect semantics, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "index": {
                        "type": "string",
                        "description": "the column that includes indexes of the word in the two sentences, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["word", "word_type", "sentence1", "sentence2", "index"]
            },
            {
                "name": "get_request_succeed",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies whether the requests in that column succeeded or not, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains requests to classify whether they succeeded, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_power",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies whether the conversation in that column is in the position of power or not, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains conversations to classify whether they are in the position of power, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_change_opinion",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies whether the contents in that column changes opinion, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to classify whether the content would change their opinion, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_relationship",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies if the conversation happens between social or romance type of relationship, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to classify the type of relationship of the conversation, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_humor",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies whether the contents in that column is funny, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to classify whether the content is funny, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_politeness",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies whether the contents in that column is polite, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to classify whether the content is polite, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_toxic",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies whether the contents in that column will become toxic in the future, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to classify whether the content will become toxic in the future, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
        ],
        "ternary_classifier": [
            {
                "name": "get_stance",
                "description": 
                ''' 
                This function takes in one column of text and one column of target topic as input, extracts the stance in the text column that towards the target topic, and generates a new column to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to identify stance towards the target topic, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "target": {
                        "type": "string",
                        "description": "the column that includes the target topic, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },

                    }
                },
                "required": ["column", "target"]
            },
            {
                "name": "get_ideology_doc",
                "description": "This function takes in one of the columns that contains the links to documents as input, applies document-level ideology classification on that column, and generates an additional column labeling the content as 0 (meaning left), 1 (meaning neutral), or 2 (meaning right).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column that contains the links to documents to apply document-level ideology classification. You have to select one column in the 'enum' field based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_ideology_sent",
                "description": "This function takes in one of the columns that contains texts as input, applies sentence-level ideology classification on that column, and generates an additional column labeling the content as 'Conservative', 'Liberal', or 'Neutral'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column that contains the texts to apply sentence-level ideology classification. You have to select one column in the 'enum' field based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
        ],
        "multiple_classifier": [
            {
                "name": "get_story_type",
                "description": 
                ''' 
                This function takes in one column of sentence, one column of story index of where the sentence belong, one column of sentence index of that sentence, one column of the probability of the sentence containing new events, one column of probability of the sentence containing major new events, and one column of probabiliy of the sentence containing new events as expected as input, classifies the story type of the sentence into 'imagined', 'recalled', or 'retold', and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed after at least one call of 'get_event'; this function has to be executed after at least one call of 'get_event_major'; this function has to be executed after at least one call of 'get_event_expected'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to classify story type, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "storyIx": {
                        "type": "string",
                        "description": "the column that contains the story index of the sentence, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "sentIx": {
                        "type": "string",
                        "description": "the column that contains the sentence index of the sentence, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "event": {
                        "type": "string",
                        "description": "the column that contains the probability of the sentence containing new events, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "event_major": {
                        "type": "string",
                        "description": "the column that contains the probability of the sentence containing new major events, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "event_expected": {
                        "type": "string",
                        "description": "the column that contains the probability of the sentence containing new events that are expected, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    }
                },
                "required": ["column", "storyIx", "sentIx", "event", "event_major", "event_expected"]
            },
            {
                "name": "get_dialect",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies the dialect features identified from the content of that column, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to classify the dialect features, you have to select one in the 'enum' field to summarize based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_disclosure",
                "description": 
                ''' 
                This function takes in one of the columns as input, classifies the content of that column into different disclosure acts, and generate an additional column to include those. Don't select this if none of the columns matches the user query.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to classify the disclosure acts, you have to select one in the 'enum' field to summarize based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_emotion",
                "description": "This function takes in one of the columns as input, applies emotion classification on the content of that column, and generates an additional column labeling the content with its emotion class (sad, joy, angry, etc). IMPORTANT: this function has to be executed at least once before calling 'get_trigger'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column to apply emotion detection. You have to select one column in the 'enum' field to apply emotion classification based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_trope",
                "description": "This function takes in one of the columns that contains the quotes of a character as input, applies trope classification based on the content of the quotes, and generates an additional column to include those.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column that contains the quotes of a character. You have to select one column in the 'enum' field based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_hate_class",
                "description": "This function takes in one of the columns as input, applies (fine-grained) implicit hate speech classification on the content of that column, and generates an additional column labeling the content as 'white_grievance', 'incitement', 'inferiority', 'irony', 'stereotypical', 'threatening', or 'other'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column to apply (fine-grained) implicit hate speech classification. You have to select one column in the 'enum' field to apply (fine-grained) implicit hate speech classification based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_dog_whistle_persona_ingroup",
                "description": 
                ''' 
                This function takes in one of the columns of dog whistle terms, obtain the target persona/in-group of that dog whistle term, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_dog_whistle'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column of dog whistle terms to obtain the target persona/in-group, you have to select one in the 'enum' field to summarize based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_dog_whistle_type",
                "description": 
                ''' 
                This function takes in one of the columns of dog whistle terms, obtain the type of that dog whistle term, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_dog_whistle'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column of dog whistle terms to obtain the type, you have to select one in the 'enum' field to summarize based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_strategy",
                "description": 
                ''' 
                This function takes in one of the columns of the texts that need emotional support as input, classifies which type of emotional support strategy is needed, and generate an additional column to include those. Don't select this if none of the columns matches the user query. IMPORTANT: this function has to be executed at least once before calling 'get_supporter_response'. 
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains texts that need emotional support, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_request_strategy",
                "description": 
                ''' 
                This function takes in one of the columns that contains the requests, applies request strategy on the requests, and generates an additional column to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains texts of the requests, you have to select one in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
        ]
        },
        "extractor": {
            "text_extractor": [
            {
                "name": "get_event_argument",
                "description": 
                ''' 
                This function takes in one of the columns as input, get the event arguments recognized in that column, and generate additional rows and columns to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to apply event argument extraction, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,

                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_ner",
                "description": 
                ''' 
                This function takes in one of the columns as input, get the name entities recognized in that column, and generate additional rows and columns to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to apply NER analysis, you have to select one column in the 'enum' field to apply NER analysis based on the description: "+description,
                        "enum": column_description,

                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_pos",
                "description": 
                ''' 
                This function takes in one of the columns as input, get the part of speech (POS) in that column, and generate additional rows and columns to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to apply POS recognition, you have to select one column in the 'enum' field to apply POS recognition based on the description: "+description,
                        "enum": column_description,

                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_keyword",
                "description": 
                ''' 
                This function takes in one of the columns as input, get the top 5 keywords recognized in that column, and generate an additional column to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to apply keyword recognition, you have to select one column in the 'enum' field to apply keyword recognition based on the description: "+description,
                        "enum": column_description,

                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_trigger",
                "description": 
                ''' 
                This function takes in one column of text and one column of emotion class as input, extracts the trigger in the text column that triggers a specific emotion in the emotion class column, and generates a new column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_emotion'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to extract triggering sentence for a specific emotion, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "emotion": {
                        "type": "string",
                        "description": "the column that describes emotion class, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },

                    }
                },
                "required": ["column", "emotion"]
            },
            {
                "name": "get_dog_whistle",
                "description": 
                ''' 
                This function takes in one of the columns as input, extract the dog whistle term in that column, and generate an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_dog_whistle_type', this function has to be executed at least once before calling 'get_dog_whistle_persona_ingroup'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column to extract dog whistle term, you have to select one in the 'enum' field to summarize based on the description: "+description,
                        "enum": column_description,
                    }
                    }
                },
                "required": ["column"]
            },
            ],
            "information_extractor": [
            {
                "name": "get_translation",
                "description": "This function takes in one column of words as input, extract the translations of the words, and generates an additional column to include those.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column that contains words to be translated. You have to select one column in the 'enum' field to apply implicit hate speech target identification based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
                "name": "get_hate_target",
                "description": "This function takes in one of the columns as input, applies implicit hate speech target identification on the content of that column, and generates an additional column of free text labeling the target identified from the content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "The column to apply implicit hate speech target identification. You have to select one column in the 'enum' field to apply implicit hate speech target identification based on the description: "+description,
                            "enum": column_description
                        }
                    },
                    "required": ["column"]
                }
            },
            {
            "name": "get_hate_implied",
            "description": "This function takes in one of the columns as input, applies implicit hate speech implied statement extraction on the content of that column, and generates an additional column of free text labeling the implied statement extracted from the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The column to apply implicit hate speech implied statement extraction. You have to select one column in the 'enum' field to apply implicit hate speech implied statement extraction based on the description: "+description,
                        "enum": column_description
                    }
                },
                "required": ["column"]
            }
            },
            {
            "name": "get_positive_reframing",
            "description": "This function takes in one of the columns as input, extract the positive aspects of the content of that column and transforms it into a positive reframing version, and generates an additional column of positive reframing version of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The column to apply positive reframing. You have to select one column in the 'enum' field to apply positive reframing based on the description: "+description,
                        "enum": column_description
                    }
                },
                "required": ["column"]
            }
            },
            {
                "name": "get_premise",
                "description": 
                ''' 
                This function takes in one column of figurative text, one column of figurative type, and one column of figurative term as input, extracts the literal text, i.e., the premise, of the figurative text column, and generates a new column to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column of figurative texts to extract premises from, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "type": {
                        "type": "string",
                        "description": "the column that contains the figurative types of the figurative texts, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "term": {
                        "type": "string",
                        "description": "the column that contains the figurative terms of the figurative texts, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    }
                },
                "required": ["column", "type", "term"]
            },
            {
                "name": "get_premise_explanation",
                "description": 
                ''' 
                This function takes in one column of premise of figurative text, one column of the original figurative text, one column of figurative type, one column of figurative term as input, and one parameter labelling whether the premises entail or contract original figurative texts as input, extracts the explanations of literal texts, i.e., the premises, of the figurative text column, and generates a new column to include those.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column of premises of figurative texts to extract explanation from, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": "the column of original figurative texts, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "type": {
                        "type": "string",
                        "description": "the column that contains the figurative types of the figurative texts, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "term": {
                        "type": "string",
                        "description": "the column that contains the figurative terms of the figurative texts, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,
                    },
                    "label": {
                        "type": "string",
                        "description": "parameter defining whether the explanation is retrieved from premises that entail or contraduct figurative texts, you have to select one value in the 'enum' field.",
                        "enum": ["entail", "contradict"],
                    },
                    }
                },
                "required": ["column", "hypothesis", "type", "term", "label"]
            },
            ],
            "causal_extractor": [
            {
                "name": "get_reader_action",
                "description": 
                ''' 
                This function takes in one of the columns that contains writers' intent as input, get the reader action inferred from the writers' intent of that column, and generate an additional column to include those. IMPORTANT: this function has to be executed after at least one call of 'get_intent'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains writers' intent to infer readers' action from, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,

                    }
                    }
                },
                "required": ["column"]
            },
            {
                "name": "get_reader_perception",
                "description": 
                ''' 
                This function takes in one of the columns of text as input, infers readers' perceptions of text in that column, and generates an additional column to include those. IMPORTANT: this function has to be executed at least once before calling 'get_spread_likelihood'.
                ''',
                "parameters": {
                    "type": "object",
                    "properties": {"column": {
                        "type": "string",
                        "description": "the column that contains text to infer readers' perceptions from, you have to select one column in the 'enum' field based on the description: "+description,
                        "enum": column_description,

                    }
                    }
                },
                "required": ["column"]
            },
            ]
        }
    }
    functions = [  
    {
            "name": "para_sep",
            "description": "This function takes in one of the columns as input, split the text according to paragraphs, and generates an additional rows and columns to store the list of paragraphs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The column to apply paragraph split. You have to select one column in the 'enum' field to apply paragraph-level split based on the description: "+description,
                        "enum": column_description
                    }
                },
                "required": ["column"]
            }
        },
        {
            "name": "pdf_to_text",
            "description": 
            ''' 
            This function takes in one of the columns as input, transforms the pdf in that column into plain text, and generate an additional column to store the plain text. Don't select this if none of the columns match the user query.
            ''',
            "parameters": {
                "type": "object",
                "properties": {"column": {
                    "type": "string",
                    "description": "the column to apply pdf to text transformation, you have to select one column in the 'enum' field to apply pdf to text transformation based on the description: "+description,
                    "enum": column_description,

                }
                }
            },
            "required": ["column"]
        },  
        {
            "name": "get_sentiment",
            "description": "This function takes in one of the columns as input, applies sentiment analysis on the content of that column, and generates an additional column labeling the content with its sentiment class (positive, negative, neutral, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The column to apply sentiment analysis. You have to select one column in the 'enum' field to apply sentiment analysis based on the description: "+description,
                        "enum": column_description
                    }
                },
                "required": ["column"]
            }
        },    
        {
            "name": "null",
            "description": "This function should be called when the table already contains all the necessary information to complete the user query. IMPORTANT: filter conditions related to numerical (such as range/percentile/etc.) doesn't require additional function calls, so this function should be called in cases when numerical value filters are the only operations to be applied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The placeholder parameter. It can only be 'null'",
                        "enum": ["null"]
                    }
                },
                "required": ["column"]
            }
        } 
    ]

    # Dealing with new functions:
    path = ""
    if udf != None:
        new_functions = []
        for f in udf:
            func_call_dict = {}
            func_call_dict["name"] = f["func-name"]
            func_call_dict["description"] = f["func-description"]
            para_dict = {}
            para_dict["type"] = "object"
            inputs = f["input"]
            property_dict = {}
            for input_name in inputs:
                input_dict = {}
                input_dict["type"] = "string"
                input_dict["description"] = inputs[input_name] + "You have to select one column based on the description: "+description
                input_dict["enum"] = column_description
                property_dict[input_name] = input_dict
            para_dict["properties"] = property_dict
            para_dict["required"] = list(inputs.keys())
            func_call_dict["parameters"] = para_dict
            new_functions.append(func_call_dict)

        tree["user_defined_functions"] = new_functions
        tree_str = '''\t\t"user_defined_functions" <functions added by users> \n'''
        for f in udf:
            tree_str += ("\t\t\t"+f['func-name'] + " : " + f['func-description'] + "\n")
        path = tree_path(user_query + "The function chain I have is " + function_chain+ ", current executed function chain is "+str(function_list), column_description, description, verbose, client, gpt4, tree_str)
    else:
        path = tree_path(user_query + "The function chain I have is " + function_chain+ ", current executed function chain is "+str(function_list), column_description, description, verbose, client, gpt4)
    if verbose:
        print("VERBOSE:"+"Tree search path: ", path)
    path = path.replace("'", '"')
    path = json.loads(path)
    if len(path) > 0:
        path = path["path"]
        ls = tree
        for item in path:
            ls = ls[item]
    else:
        ls = []
        gpt4 = True # use GPT 4 to find the stopper
    functions.extend(ls)
    
    messages = [
        {
            "role": "user",
            "content": "I want to count the number of positive paragraphs in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file;"
        },
        {
            "role": "assistant",
            "content": "To count the number of positive paragraphs in the PDF document, the user should first transform the PDF file into plain text, break the text into paragraphs, and then get the sentiment of these paragraphs. Among all the columns, the user is given the PDF file in 'pdf_orig', and is not given the plain text of the document, thus the first function to apply is 'pdf_to_text' on the 'pdf_orig' column to get the plain text of the PDF file.",
            "function_call": {
                "name": "pdf_to_text",
                "arguments": "{\n  \"column\": \"pdf_orig\"\n}"
            }
        },
        {
            "role": "user",
            "content": "I want to count the number of positive paragraphs in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file; 'pdf_orig_text' column is the plain text content of the 'pdf_orig' column;"
        },
        {
            "role": "assistant",
            "content": "To count the number of positive paragraphs in the PDF document, the user should first transform the PDF file into plain text, break the text into paragraphs, and then get the sentiment of these paragraphs. Among all the columns, the user is given the plain text of the PDF file in 'pdf_orig_text', and is not given the paragraph-wise segments of the document, thus the first function to apply is 'para_sep' on the 'pdf_orig_text' column to get the paragraph-level splits of the plain text.",
            "function_call": {
                "name": "para_sep",
                "arguments": "{\n  \"column\": \"pdf_orig_text\"\n}"
            }
        },
        {
            "role": "user",
            "content": "I want to count the number of positive paragraphs in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file; 'pdf_orig_text' column is the plain text content of the 'pdf_orig' column; 'pdf_orig_text_segment' stores the paragraph segments of the 'pdf_orig_text' column, the original text has empty value; 'pdf_orig_text_segmentid' column stores the paragraph index according to the order of the 'pdf_orig_text_segment' column, starts with 0, and the original text has value -1;"
        },
        {
            "role": "assistant",
            "content": "To count the number of positive paragraphs in the PDF document, the user should first transform the PDF file into plain text, break the text into paragraphs, and then get the sentiment of these paragraphs. Among all the columns, the user is given the paragraphs of the PDF file in 'pdf_orig_text_segments', and is not given the sentimental analysis for the paragraphs, thus the first function to apply is 'get_sentiment' on the 'pdf_orig_text_segments' column to get the sentiment for the paragraphs.",
            "function_call": {
                "name": "get_sentiment",
                "arguments": "{\n  \"column\": \"pdf_orig_text_segment\"\n}"
            }
        },
        {
            "role": "user",
            "content": "I want to count the number of positive paragraphs in the PDF document. 'id' column is the document ID; 'pdf_orig' column is the path to the pdf file of the document file; 'pdf_orig_text' column is the plain text content of the 'pdf_orig' column; 'pdf_orig_text_segment' stores the paragraph segments of the 'pdf_orig_text' column, the original text has empty value; 'pdf_orig_text_segmentid' column stores the paragraph index according to the order of the 'pdf_orig_text_segment' column, starts with 0, and the original text has value -1; 'pdf_orig_text_segment_sentiment' column is the sentiment of the content of the 'pdf_orig_text_segment' column; "
        },
        {
            "role": "assistant",
            "content": "To count the number of positive paragraphs in the PDF document, the user should first transform the PDF file into plain text, break the text into paragraphs, and then get the sentiment of these paragraphs. Among all the columns, the user is already given the paragraph-level sentiment analysis in the 'pdf_orig_text_segment_sentiment' column. Thus the first function to apply is 'null' to end the function chain.",
            "function_call": {
                "name": "null",
                "arguments": "{\n  \"column\": \"null\"\n}"
            }
        },
        {
            "role": "user",
            "content": user_query + "The function chain I have is " + function_chain+ ", current executed function chain is "+str(function_list)  # Use the user's query
        }
    ]

    if gpt4:
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=messages,
            functions = functions,
            function_call = "auto",
        )

        if verbose:
            num_token_msg = num_tokens_from_messages(messages, "gpt-4-0613")
            num_token_func = num_tokens_from_functions(functions, "gpt-4-0613")
            print("VERBOSE:"+"Number of tokens of messages for 'schema_gpt_tree': ", num_token_msg)
            print("VERBOSE:"+"Number of tokens of functions for 'schema_gpt_tree': ", num_token_func)
            print("VERBOSE:"+"Number of prompt tokens for 'schema_gpt_tree': ", response.usage.prompt_tokens)
            print("VERBOSE:"+"Number of answer tokens for 'schema_gpt_tree': ", response.usage.completion_tokens)
            print("VERBOSE:"+"Number of total tokens for 'schema_gpt_tree': ", response.usage.total_tokens)
            current_pize = 0.00003 * response.usage.prompt_tokens + 0.00006 * response.usage.completion_tokens
            print("VERBOSE:"+f"Cost for 'schema_gpt_tree': ${current_pize}")
            autopipeline.cost += current_pize
            print("VERBOSE:"+f"Accumulated cost: ${autopipeline.cost}")
    else:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            functions = functions,
            function_call = "auto",
        )

        if verbose:
            num_token_msg = num_tokens_from_messages(messages, "gpt-3.5-turbo-0125")
            num_token_func = num_tokens_from_functions(functions, "gpt-3.5-turbo-0125")
            print("VERBOSE:"+"Number of tokens of messages for 'schema_gpt_tree': ", num_token_msg)
            print("VERBOSE:"+"Number of tokens of functions for 'schema_gpt_tree': ", num_token_func)
            print("VERBOSE:"+"Number of prompt tokens for 'schema_gpt_tree': ", response.usage.prompt_tokens)
            print("VERBOSE:"+"Number of answer tokens for 'schema_gpt_tree': ", response.usage.completion_tokens)
            print("VERBOSE:"+"Number of total tokens for 'schema_gpt_tree': ", response.usage.total_tokens)
            current_pize = 0.0000005 * response.usage.prompt_tokens + 0.0000015 * response.usage.completion_tokens
            print("VERBOSE:"+f"Cost for 'schema_gpt_tree': ${current_pize}")
            autopipeline.cost += current_pize
            print("VERBOSE:"+f"Accumulated cost: ${autopipeline.cost}")


    return response.choices[0].message

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter Notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # terminal running IPython
        else:
            return False  # other type (?)
    except NameError:
        return False      # standard Python interpreter
    
def is_notebook_colab():
    try:
        from IPython import get_ipython
        if 'google.colab' in str(get_ipython()):
            return True   # Google Colab
        else:
            return False  # other environment
    except Exception:
        return False  # not running in an IPython environment

def table_gen(user_query, table, enum, description, status, function_chain, verbose, client, udf, gpt4):
    start_time = time.time()
    function_list = []
    dot = Digraph(comment='graph')

    # to better display
    dot.attr(rankdir='LR')
    dot.attr('node', shape='ellipse', width='1')

    output_filename = './data/dot_graph'

    for c in enum:
        dot.node(c)
    if is_notebook() or is_notebook_colab():
        dot.render(output_filename, format='png', cleanup=True)
        initial_image = Image(output_filename + '.png')
        display_handle = display(initial_image, display_id=True)
    else:
        dot_data = dot.source
        api_url = 'https://quickchart.io/graphviz'
        params = {
            'graph': dot_data,
            'format': 'png'
        }
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            with open('static/dot_graph.png', 'wb') as image_file:
                image_file.write(response.content)
        # dot.render(filename='static/dot_graph', format='png', cleanup=True)

    added_functions = []
    if udf != None:
        for _f in udf:
            added_functions.append(_f['func-name'])
    while True:
        if udf == None:
            response = schema_gpt_tree(user_query + " I am given a table with the following columns: " + description, enum, description, function_chain, function_list, verbose, client, gpt4)
        else:
            response = schema_gpt_tree(user_query + " I am given a table with the following columns: " + description, enum, description, function_chain, function_list, verbose, client, gpt4, udf)
        try:
            func = response.function_call
            f = func.name
            if verbose:
                print("VERBOSE:"+"Selected function: ", f)
        except:
            feedback = response.content
            return True, feedback, None, None
        function_list.append(f)
        if f == "null":
            break
        if verbose:
            print(f"CODE:# Define the '{f}' function")
            print(f"CODE:# def {f}(): ")
        function_dict = json.loads(func.arguments)
        if f in added_functions:
            input_var = []
            input_var.append(table)
            for k in function_dict:
                input_var.append(function_dict[k])
            desc_dict = {}
            for _f in udf:
                desc_dict.update(_f["output"])
            table, enum, description, dot = wrapper(f, input_var, function_dict, desc_dict, enum, description, verbose, dot, client, gpt4)
        else:
            if f != "get_semantic":
                col = function_dict["column"]
                if col == "null":
                    break
            if f == "get_trigger":
                emotion = function_dict["emotion"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, emotion, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{emotion}'])"
                    print("CODE:"+code_string)
            elif f == "get_stance":
                target = function_dict["target"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, target, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{target}'])"
                    print("CODE:"+code_string)
            elif f == "get_supporter_response":
                strategy = function_dict["strategy"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, strategy, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{strategy}'])"
                    print("CODE:"+code_string)
            elif f == "get_premise_explanation":
                hypothesis = function_dict["hypothesis"]
                type = function_dict["type"]
                term = function_dict["term"]
                label = function_dict["label"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, hypothesis, type, term, label, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                # for demo
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{hypothesis}'], table['{type}'], table['{term}'], table['{label}'])"
                    print("CODE:"+code_string)
            elif f == "get_premise":
                type = function_dict["type"]
                term = function_dict["term"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, type, term, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                # for demo
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{type}'], table['{term}'])"
                    print("CODE:"+code_string)
            elif f == "get_semantic":
                word = function_dict["word"]
                word_type = function_dict["word_type"]
                index = function_dict["index"]
                sentence1 = function_dict["sentence1"]
                sentence2 = function_dict["sentence2"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, word, word_type, index, sentence1, sentence2, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                # for demo
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{word}'], table['{word_type}'], table['{index}'], table['{sentence1}'], table['{sentence2}'])"
                    print("CODE:"+code_string)
            elif f == "get_story_type":
                storyIx = function_dict["storyIx"]
                sentIx = function_dict["sentIx"]
                event = function_dict["event"]
                event_major = function_dict["event_major"]
                event_expected = function_dict["event_expected"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, storyIx, sentIx, event, event_major, event_expected, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                # for demo
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{storyIx}'], table['{sentIx}'], table['{event}'], table['{event_major}'], table['{event_expected}'])"
                    print("CODE:"+code_string)
            elif f == "get_deception":
                smile = function_dict["smile"]
                scowl = function_dict["scowl"]
                initial_enum = set(enum)
                table, enum, description, dot =globals()[f](table, col, smile, scowl, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                # for demo
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{smile}'], table['{scowl}'])"
                    print("CODE:"+code_string)

            elif f in ["get_emotional_reaction_level", "get_exploration_level", "get_intepretation_level"]:
                column_post = function_dict["column_post"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, column_post, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                # for demo
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{column_post}'])"
                    print("CODE:"+code_string)

            elif f in ["get_event_major", "get_event_expected"]:
                event = function_dict["event"]
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, event, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                # for demo
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'], table['{event}'])"
                    print("CODE:"+code_string)
            else:
                initial_enum = set(enum)
                table, enum, description, dot = globals()[f](table, col, enum, description, verbose, dot, client, gpt4)
                new_entries = set(enum) - initial_enum
                # for demo
                if verbose:
                    code_string = ""
                    for new_col in new_entries:
                        code_string += f"table['{new_col}'], "
                    code_string = code_string[:-2] # remove the last comma
                    code_string += f" = {f}(table['{col}'])"
                    print("CODE:"+code_string)

        
        # displaying the dot graph
        if is_notebook() or is_notebook_colab():
            dot.render(output_filename, format='png', cleanup=True)
            updated_image = Image(output_filename + '.png')  
            update_display(updated_image, display_id=display_handle.display_id)        
        else:
            dot_data = dot.source
            api_url = 'https://quickchart.io/graphviz'
            params = {
                'graph': dot_data,
                'format': 'png'
            }
            response = requests.get(api_url, params=params)
            if response.status_code == 200:
                with open('static/dot_graph.png', 'wb') as image_file:
                    image_file.write(response.content)
            # dot.render(filename='static/dot_graph', format='png', cleanup=True)
    
    if verbose:
        print("VERBOSE:"+"Execution Time for Table Generation = ", time.time() - start_time)

        # for demo:
        table_demo = table.head().copy()
        for column in table_demo.columns:
            table_demo[column] = table_demo[column].apply(ensure_max_words)
        table_html = table_demo.to_html(classes='table table-stripped').replace('\n', '')
        print(f'########## AT-HTML:{table_html}')

        now = datetime.now()
        timestamp = datetime.timestamp(now)
        print(f'VER NUMBER:{timestamp}')

        # ---- saves file ----
        # table.to_csv(f'static/augmented_table{timestamp}.csv', index=False)

        # table_str = table.to_string()
        # verbose_table = "\n".join("VERBOSE: " + line for line in table_str.split('\n'))
        # print("VERBOSE: Augmented table: ")
        # print(verbose_table)

    print("Augmented table: ")
    print(table)

    status.append("table augmented")
    return table, enum, description, status, False, ""



