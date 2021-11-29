from re import sub
from sentence_transformers import SentenceTransformer, util
import torch

import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from gensim.models.keyedvectors import Word2VecKeyedVectors

# Import and download the most up-to-date stopwords from NLTK
# from nltk import download
# from nltk.corpus import stopwords
# download('stopwords')  # Download stopwords list.
# nltk_stop_words = set(stopwords.words("english"))
#
# Or use a hard-coded list of English stopwords


nltk_stop_words = {'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are',
                   'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
                   'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn',
                   "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
                   'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here',
                   'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't",
                   'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most',
                   'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off',
                   'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's',
                   'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so',
                   'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves',
                   'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
                   've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where',
                   'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y',
                   'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'}


class NotReadyError(Exception):
    pass


class DocSim:
    """
    Find documents that are similar to a query string.
    Calculated using word similarity (Soft Cosine Similarity) of word embedding vectors
    """

    default_model = "glove-wiki-gigaword-50"
    model_ready = False  # Only really relevant to threaded sub-class

    def __init__(self, model=None, stopwords=None, verbose=False):
        # Constructor

        self.verbose = verbose

        self._load_model(model)

        if stopwords is None:
            self.stopwords = nltk_stop_words
        else:
            self.stopwords = stopwords

    def _load_model(self, model):
        # Pass through to _setup_model (overridden in threaded)
        self._setup_model(model)

    def _setup_model(self, model):
        # Determine which model to use, download/load it, and create the similarity_index

        if isinstance(model, Word2VecKeyedVectors):
            # Use supplied model
            self.model = model
        elif isinstance(model, str):
            # Try to download named model
            if self.verbose:
                print(f'Loading word vector model: {model}')
            self.model = api.load(model)
            if self.verbose:
                print('Model loaded')
        elif model is None:
            # Download/use default GloVe model
            if self.verbose:
                print(f'Loading default GloVe word vector model: {self.default_model}')
            self.model = api.load(self.default_model)
            if self.verbose:
                print('Model loaded')
        else:
            raise ValueError('Unable to load word vector model')

        self.similarity_index = WordEmbeddingSimilarityIndex(self.model)

        self.model_ready = True

    def preprocess(self, doc: str):
        # Clean up input document string, remove stopwords, and tokenize
        doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
        doc = sub(r'<[^<>]+(>|$)', " ", doc)
        doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
        doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)

        return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if
                token not in self.stopwords]

    def _softcossim(self, query: str, documents: list):
        # Compute Soft Cosine Measure between the query and each of the documents.
        query = self.tfidf[self.dictionary.doc2bow(query)]
        index = SoftCosineSimilarity(
            self.tfidf[[self.dictionary.doc2bow(document) for document in documents]],
            self.similarity_matrix)
        similarities = index[query]

        return similarities

    def similarity_query(self, query_string: str, documents: list):

        """
        Run a new similarity ranking, for query_string against each of the documents

        """

        if self.model_ready:

            corpus = [self.preprocess(document) for document in documents]
            query = self.preprocess(query_string)

            if set(query) == set([word for document in corpus for word in document]):
                raise ValueError('query_string full overlaps content of document corpus')

            if self.verbose:
                print(f'{len(corpus)} documents loaded into corpus')

            self.dictionary = Dictionary(corpus + [query])
            self.tfidf = TfidfModel(dictionary=self.dictionary)
            self.similarity_matrix = SparseTermSimilarityMatrix(self.similarity_index,
                                                                self.dictionary, self.tfidf)

            scores = self._softcossim(query, corpus)

            return scores.tolist()

        else:
            raise NotReadyError('Word embedding model is not ready.')


def get_semantic_score(query, corpus, embedder):
    """

    We have a corpus with various sentences. Then, for a given query sentence,
    we want to find the most similar sentence in this corpus.
    This script returns documents in the corpus with their scores respect to the query.

    """

    # creating corpus embeddings
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # finding top_k documents
    top_k = len(corpus)

    # finding query embeddings
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    data = []
    ids = 0

    # Now creating response format and getting each document score
    for score, idx in zip(top_results[0], top_results[1]):
        ids += 1
        data.append({"document": idx.item() + 1,
                     "object": "search_result",
                     "score": score.item()})

    return data
