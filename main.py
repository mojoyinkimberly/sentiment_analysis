import string

from sklearn.base import TransformerMixin
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('all')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

wordnet_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    #removes upper cases
    try:
        text = text.lower()
    except Exception as e:
        print("error", e)
        print("type", type(text))
        print("text:", text)

    #removes punctuation
    for char in string.punctuation:
        text = text.replace(char, " ")

    #lemmatize the words and join back into string text
    text = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text)])
    return text


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def __str__(self):
        return "DenseTransformer()"

    def __repr__(self):
        return self.__str__()
