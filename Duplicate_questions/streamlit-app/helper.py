import re
import joblib
import numpy as np
from fuzzywuzzy import fuzz
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
try:
    from distance import lcsubstrings
except ImportError:
    from difflib import SequenceMatcher

    def lcsubstrings(s1, s2):
        matcher = SequenceMatcher(None, s1, s2)
        match = matcher.find_longest_match(0, len(s1), 0, len(s2))
        if match.size == 0:
            return []
        return [s1[match.a : match.a + match.size]]


# Global tokenizer to avoid repeated fitting
global_tokenizer = None


def preprocess(q):
    """
    Preprocesses a question string by normalizing text, removing special在那

    Args:
        q (str): Input question string.

    Returns:
        str: Preprocessed question string.
    """
    if q is None or not isinstance(q, (str, bytes)) or not q.strip():
        return ""
    q = str(q).lower().strip()

    # Replace special characters
    q = q.replace("%", " percent").replace("$", " dollar ").replace("₹", " rupee ").replace("€", " euro ").replace("@", " at ")
    q = q.replace("[math]", "")

    # Replace numbers
    q = q.replace(",000,000,000 ", "b ").replace(",000,000 ", "m ").replace(",000 ", "k ")
    q = re.sub(r"([0-9]+)000000000", r"\1b", q)
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)

    # Decontract words
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
    }
    q_decontracted = [contractions.get(word, word) for word in q.split()]
    q = " ".join(q_decontracted)
    q = q.replace("'ve", " have").replace("n't", " not").replace("'re", " are").replace("'ll", " will")
    # Remove HTML tags
    q = re.sub(r"<.*?>", "", q)
    # Remove punctuation and collapse whitespace
    q = re.sub(r"\W", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def test_common_words(q1, q2):
    """
    Counts common words between two questions.

    Args:
        q1 (str): First question.
        q2 (str): Second question.

    Returns:
        int: Number of common words.
    """
    w1 = set(map(lambda word: word.lower().strip(), q1.split()))
    w2 = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(w1 & w2)


def test_total_words(q1, q2):
    """
    Calculates total unique words in both questions.

    Args:
        q1 (str): First question.
        q2 (str): Second question.

    Returns:
        int: Total number of unique words.
    """
    w1 = set(map(lambda word: word.lower().strip(), q1.split()))
    w2 = set(map(lambda word: word.lower().strip(), q2.split()))
    return len(w1) + len(w2)


def test_fetch_token_features(q1, q2):
    """
    Extracts token-based features for question pair similarity.

    Args:
        q1 (str): First question.
        q2 (str): Second question.

    Returns:
        list: List of 8 token-based features.
    """
    SAFE_DIV = 0.0001
    
    STOP_WORDS = set(['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', 'shouldn', "shouldn't", "should've", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"])

    token_features = [0.0] * 8
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set(word for word in q1_tokens if word not in STOP_WORDS)
    q2_words = set(word for word in q2_tokens if word not in STOP_WORDS)
    q1_stops = set(word for word in q1_tokens if word in STOP_WORDS)
    q2_stops = set(word for word in q2_tokens if word in STOP_WORDS)

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = (
        common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
        if q1_words and q2_words
        else 0.0
    )
    token_features[1] = (
        common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
        if q1_words and q2_words
        else 0.0
    )
    token_features[2] = (
        common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        if q1_stops and q2_stops
        else 0.0
    )
    token_features[3] = (
        common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
        if q1_stops and q2_stops
        else 0.0
    )
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(
        len(q1_tokens) > 0 and len(q2_tokens) > 0 and q1_tokens[-1] == q2_tokens[-1]
    )
    token_features[7] = int(
        len(q1_tokens) > 0 and len(q2_tokens) > 0 and q1_tokens[0] == q2_tokens[0]
    )

    return token_features


def test_fetch_length_features(q1, q2):
    """
    Extracts length-based features for question pair similarity.

    Args:
        q1 (str): First question.
        q2 (str): Second question.

    Returns:
        list: List of 3 length-based features.
    """
    length_features = [0.0] * 3
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    strs = list(lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0.0

    return length_features


def test_fetch_fuzzy_features(q1, q2):
    """
    Extracts fuzzy matching features for question pair similarity.

    Args:
        q1 (str): First question.
        q2 (str): Second question.

    Returns:
        list: List of 4 fuzzy matching features.
    """
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features


def token_padding_data(q1, q2, max_len=50, vocab_size=50000):
    """
    Converts questions to padded tokenized sequences.

    Args:
        q1 (list): List of first questions.
        q2 (list): List of second questions.
        max_len (int): Maximum length for padding.
        vocab_size (int): Maximum vocabulary size.

    Returns:
        tuple: Padded sequences for q1 and q2.
    """
    global global_tokenizer
    if global_tokenizer is None:
        global_tokenizer = Tokenizer(num_words=vocab_size)
        global_tokenizer.fit_on_texts(q1 + q2)

    q1_seq = global_tokenizer.texts_to_sequences(q1)
    q2_seq = global_tokenizer.texts_to_sequences(q2)
    q1_pad = pad_sequences(q1_seq, maxlen=max_len)
    q2_pad = pad_sequences(q2_seq, maxlen=max_len)
    return q1_pad, q2_pad


def heuristic_features(features):
    """
    Converts feature list to NumPy array.

    Args:
        features (list): List of features.

    Returns:
        np.ndarray: Array of features.
    """
    return np.array(features)


def query_point_creator(q1, q2):
    """
    Creates feature vector for a question pair.

    Args:
        q1 (str): First question.
        q2 (str): Second question.

    Returns:
        np.ndarray: Combined feature vector including padded sequences and heuristics.
    """
    if not isinstance(q1, str) or not isinstance(q2, str):
        raise ValueError("Both q1 and q2 must be strings")

    # Preprocess questions
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # Initialize feature list
    input_query = []

    # Basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split()))
    input_query.append(len(q2.split()))
    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(
        round(test_common_words(q1, q2) / (test_total_words(q1, q2) + 0.0001), 2)
    )

    # Token features
    input_query.extend(test_fetch_token_features(q1, q2))

    # Length features
    input_query.extend(test_fetch_length_features(q1, q2))

    # Fuzzy features
    input_query.extend(test_fetch_fuzzy_features(q1, q2))

    # Padded tokenized sequences
    q1_pad, q2_pad = token_padding_data([q1], [q2])

    # Heuristic features
    heuristics = heuristic_features(input_query)

    # Combine all features
    return q1_pad, q2_pad, heuristics.reshape(1, -1)