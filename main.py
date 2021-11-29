import flask
from data.gensim_model import get_semantic_score
from sentence_transformers import SentenceTransformer
from flask import request, jsonify

# Making flask app
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Reading model embedder from folder models
embedder = SentenceTransformer('models/msmarco')
embedder2 = SentenceTransformer('models/quora_ranking')


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Semantic Search</h1> <p>A search engine that matches documents with queries and return scores</p>'''


# main post api point
# function get score use our models to get score
# and return in json format

@app.route('/scorer', methods=['POST'])
def get_score():
    # getting data from json passed into data
    data_input = request.get_json()

    # getting documents from the data json
    documents = data_input['documents']
    # getting query from the data json
    query_string = data_input["query"]

    # checking if model is passed as input
    if 'model' in data_input:
        model = data_input["model"]
    # if not then model 1 is selected
    else:
        model = 1

    # selecting trained model accordingly
    if model == 1:
        trained_model = embedder
    else:
        trained_model = embedder2

    # passing query, documents and trained model to function to get results
    data = get_semantic_score(query_string, documents, trained_model)

    # creating response of data
    res = {"data": data}

    print("response:", res)
    return jsonify(res)


if __name__ == "__main__":
    app.run()
