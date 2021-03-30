import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
en_stopwords = set((stopwords.words('english')))
es_stopwords = set((stopwords.words('spanish')))
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import SnowballStemmer
es_stemmer = SnowballStemmer('spanish')

import re

def to_lower(word):
    '''
    word: str, string of a single word
    This function returns the lower case of the given word
    '''
    result = word.lower()
    return result

def remove_mentions(word):
    '''
    word: str, string of a single word
    Removes mentions in tweets, they are recognized via the @ character
    '''
    result = re.sub(r"@\S+","",word)
    return result

def remove_number(word):
    '''
    word: str, string of a single word
    Removes numbers in tweets
    '''
    result = re.sub(r'\d+', '', word)
    return result

def remove_punctuation(word):
    '''
    word: str, string of a single word
    Removes punctuation characters of a string
    '''
    result = re.sub('[^A-Za-z]+', ' ', word)
    return result


def clean_text(raw_text):
    emoji_pat = '[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]'
    shrink_whitespace_reg = re.compile(r'\s{2,}')
    reg = re.compile(r'({})|[^a-zA-Z]'.format(emoji_pat)) # line a
    result = reg.sub(lambda x: ' {} '.format(x.group(1)) if x.group(1) else ' ', raw_text)
    return shrink_whitespace_reg.sub(' ', result)

def remove_en_stopwords(words):
    '''
    words: list of str, list containing the str for every word in the tweet
    stop_words: List of str, list containing all stop words for a given language
    Removes stop words in the tweet
    '''
    result = [i for i in words if i not in en_stopwords]      
    return result

def remove_es_stopwords(words):
    '''
    words: list of str, list containing the str for every word in the tweet
    stop_words: List of str, list containing all stop words for a given language
    Removes stop words in the tweet
    '''
    result = [i for i in words if i not in es_stopwords]      
    return result

def remove_stopwords(words,stopwords):
    '''
    words: list of str, list containing the str for every word in the tweet
    stop_words: List of str, list containing all stop words for a given language
    Removes stop words in the tweet
    '''
    result = [i for i in words if i not in stopwords]      
    return result

def remove_hyperlink(word):
    '''
    word: str, string of a single word
    Removes URLs in the given word
    '''
    return re.sub(r"http\S+", "", word)

def remove_whitespace(word):
    '''
    word: str, string of a single word
    Removes whitespaces from a str
    '''
    result = word.strip()
    return result

def replace_newline(word):
    '''
    word: str, string of a single word
    Removes new lines from a str
    '''
    return word.replace('\n','')

def clean_up_pipeline(sentence, keep_emojis=True):
    '''
    sentence: str, string containing the tweet
    This function cleans up the tweets with previous functions
    '''
    if keep_emojis:
        cleaning_data = [remove_hyperlink,
                        replace_newline,
                        to_lower,
                        clean_text,
                        remove_mentions,
                        remove_number,
                        remove_punctuation,
                        remove_whitespace]
        for func in cleaning_data:
            
            sentence = func(sentence)
    else:
        cleaning_data = [remove_hyperlink,
                        replace_newline,
                        to_lower,
                        remove_punctuation,
                        remove_mentions,
                        remove_number,
                        remove_punctuation,
                        remove_whitespace]
        for func in cleaning_data:
            
            sentence = func(sentence) 
    return sentence


def stem_words(text,lang):  
    if lang == 'en':     
        text = [stemmer.stem(word) for word in text]
    else:
        text = [es_stemmer.stem(word) for word in text]
    return text

def create_term_index(token_docs):
    terms = [item for sublist in token_docs for item in sublist]
    terms = sorted(list(set(terms)))
    return terms

def get_topwords(df,number):
    df = df.sort_values(by=['score'],ascending=False)
    df = df.head(number).loc[:,['score']]
    return df

def get_dictionary(df):
    df = df.sort_index()
    df = df.loc[:,['score']]
    dic = df.to_dict()
    return dic