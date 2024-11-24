import os
import re
import string
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import emot
import spacy
import inflect

from .date_preprocessor import DatePreprocessor
from .utils import abbreviations, symbols, countries
from sklearn.base import TransformerMixin, BaseEstimator
from unidecode import unidecode
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning 
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class TextPreprocessor(BaseEstimator, TransformerMixin):

    _numbers_from_stopwords = {
        "first", "third", "one", "two", "three", "four", "may",
        "five", "six", "eight", "nine", "ten", "eleven", "twelve",
        "fifteen", "twenty", "forty", "fifty", "sixty", "hundred",
    }

    def __init__(self,
                 n_jobs=0):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.nlp.Defaults.stop_words -= self._numbers_from_stopwords
        self.n_jobs = n_jobs
        self.n2w = inflect.engine()
        self.dp = DatePreprocessor(self.n2w)
        self.abbreviations = {
            k.rjust(len(k)+1).ljust(len(k)+2): v.rjust(len(v)+1).ljust(len(v)+2)
            for k,v in abbreviations.items()
        }
        self.abbreviations.update(symbols)


    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        if not isinstance(X, pd.Series):
            X = pd.Series(data=X)
        X_copy = X.copy()
        if self.n_jobs == 0:
            return X_copy.apply(self._preprocess_text).values.reshape((-1,1))
        
        cores = os.cpu_count()
        if self.n_jobs > 0:
            cores = min(self.n_jobs, cores)
        cores = min(len(X_copy), cores)
        data_split = np.array_split(X_copy, cores)
        df_processed = Parallel(n_jobs=cores, max_nbytes=None)(delayed(self._preprocess_part)(chunk) for chunk in data_split)
        data = pd.concat(df_processed)
        return data.values.reshape((-1,1))

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        self.nlp.Defaults.stop_words -= self._numbers_from_stopwords
        text = self._clean_text(text)
        text = self.dp.find_and_normalise_dates(text)
        text = self._replace_common_abbreviations(text)
        text = self._replace_numbers_with_words(text)
        text = self._remove_extra_spaces(text)
        doc = self.nlp.tokenizer(text)
        clear_text = self._remove_punct_and_stop_words(doc)
        doc = self.nlp(clear_text)
        return ' '.join(t.lemma_ for t in doc)
    
    def _clean_text(self, text):
        soup = BeautifulSoup(text, "html.parser")
        cleaned_text = soup.get_text()     

        protocol_pattern = r"(?:(?:ftp|https?)://)?(?:www\.)?"
        ip_pattern = r"(?:(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?)"
        domain_pattern = r"[\da-z\.-]+\.[a-z\.]{2,6}"
        ending_pattern = r"(?:[\w#~{}\|!-_@.&+]*)"

        url_pattern = protocol_pattern \
                    + r"(?:" +  domain_pattern + r"|"+ ip_pattern +r")" \
                    + ending_pattern
        
        cleaned_text = re.sub(url_pattern, "", cleaned_text, flags=re.IGNORECASE)

        cleaned_text = self._replace_emojis_and_emoticons(cleaned_text)
        cleaned_text = self._replace_country_abbreviations(cleaned_text)
        cleaned_text = cleaned_text.lower()
        cleaned_text = unidecode(cleaned_text)
        return cleaned_text

    def _find_emojis_and_emoticons(self, text):
        emot_obj = emot.emot()
        result_emojis = emot_obj.emoji(text)
        result_emoticons = emot_obj.emoticons(text)

        emojis = dict(zip(result_emojis["value"], result_emojis["mean"]))
        emoticons = dict(zip(result_emoticons["value"], result_emoticons["mean"]))
        emojis.update(emoticons)
        return emojis.items()
    
    def _replace_emojis_and_emoticons(self, text):
        for emoji, meaning in self._find_emojis_and_emoticons(text):
            text = text.replace(emoji, meaning.rjust(len(meaning)+1).ljust(len(meaning)+2).replace(":", "").replace("_", " "))
        return text
    
    def _remove_extra_spaces(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _replace_country_abbreviations(self, text):
        pattern = r"\b("+r"\b)|(".join(countries.keys())+r"\b)"
        text = re.sub(pattern, lambda x: countries[x.group(0)], text)
        return text

    def _replace_common_abbreviations(self, text):
        for old_value, new_value in self.abbreviations.items():
            if old_value in text:
                text = text.replace(old_value, new_value)
        return text

    def _replace_numbers_with_words(self, text):
        def replacer(match):
            num = match.group(0).replace(",", "")
            num = float(num) if "." in num else int(num)
            num = self.n2w.number_to_words(num)
            return num.rjust(len(num)+1).ljust(len(num)+2)
    
        number_pattern = r"-?(?!,)[\d,]*[\.]\d+|-?(?!,)[\d,]+"
        text = re.sub(number_pattern, replacer, text)
        return text

    def _remove_punct_and_stop_words(self, doc):
        tokens = (t for t in doc if not t.is_stop)
        new_words = []
        for token in tokens:
            if token.is_punct:
                continue
            new_word = token.text.strip(string.punctuation)
            if new_word and new_word not in self.nlp.Defaults.stop_words:
                new_words.append(new_word)
        return " ".join(new_words)


if __name__ == "__main__":
    tp = TextPreprocessor(0)