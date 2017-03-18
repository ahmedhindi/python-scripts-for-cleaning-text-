import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.tokenize import wordpunct_tokenize
from pandas_profiling import ProfileReport
import re
#loadin the list of stop words
stop_words = set(stopwords.words('english'))
sns.set_style('whitegrid')
IT_stop = stop_words.copy()
IT_stop.discard('it')

def fuzzy_match(city,matchs_list):
    """returns the the most likely matching value if not, it returns the city_name asis"""
    match = process.extractOne(city, matchs_list)
    if match[1] > 60:
        return match[0]
    else:
        return city


def parse_html(html_doc):
    """returns a string of parsed html with all stop words removed"""
    try:
        if html_doc == np.NaN:
            return np.NaN
        else:
            soup = BeautifulSoup(html_doc, 'html.parser')
            list_of_words = [i for i in wordpunct_tokenize(
            re.sub(r'\d+|[^\w\s]', '', (soup.text.lower()))) if i not in stop_words ]
            return ' '.join(map(lambda x: '%s' % x, list_of_words))
    except TypeError:
        return np.NaN


def clean_text(row,  tech_list):
    """returns a string of parsed html with all stop words removed"""
    row = str(row)
    try:
        if row == np.NaN:
            return np.NaN
        else:
            list_of_words = [i for i in wordpunct_tokenize(
            re.sub(r'\d+|[^\w\s]', ' ', (row.lower()))) if i in tech_list]
            astring = ' '.join(map(lambda x: '%s' % x, list_of_words))
            return astring
    except TypeError:
        return np.NaN


def mean_exper(row):
    if fuzz.partial_ratio(row,['123456789']) > 0:
        try:
            _min = list(re.findall('\d+',row))[0]
        except IndexError:
            return np.nan
        try:
            _max = list(re.findall('\d+',row))[1]
        except IndexError:
            return int(_min)
        return (int(_min)+int(_max))/2


def clean_expr_years(row):
    if fuzz.partial_ratio(row,['123456789']) > 0:
        try:
            _min = list(re.findall('\d+',row))[0]
        except IndexError:
            return np.nan
        try:
            _max = list(re.findall('\d+',row))[1]
        except IndexError:
            return _min
        return '{}-{}'.format(_min,_max)


def min_max_salary(to_mach,thresh=60):
    listo = []
    for i in data.displayed_job_title:
        if fuzz.partial_ratio(to_mach,i) > thresh:
            listo.append(i)
    sub3 = data[data.displayed_job_title.isin(listo)]
    _shape = sub3.shape
    _min = sub3.salary_min.mean()
    _max = sub3.salary_max.mean()
    return """based on {} results the min salary is {} and the max is {} for jobs the contains {} keyword""".format(_shape[0],_min,_max,to_mach)

def rec(job,num,match_list):
    matches = process.extract(query=job,limit=num, choices=match_list, scorer=fuzz.partial_ratio)
    return pd.DataFrame(matches).ix[:,0]
