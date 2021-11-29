# semanticsearch

This is a deep learning based semantic search platform that computes similarity scores between provided query and documents. Documents can be passed directly to the API 

## Installation

First clone the repo and download the trained models from this googledrive link:


```bash
deepAI:
https://drive.google.com/drive/folders/101IW0GGIUTYb114eQ8HNi4sCYF2eIsfO?usp=sharing

semanticML:
https://drive.google.com/drive/folders/110i2pRfN4GLBVrt-avEXlzfr-Q2ICuBW?usp=sharing
```

Create a folder with name "models" and place these trained models in the models folder. 

Hierarchy structure

```bash

-> models/deepAI

-> models/semanticML
```

After that use this command to download all required libraries


```bash
pip install -r requirements.txt
```

Now start the main.py file to run the flask app

```bash
python main.py
```
Flask app will start running now you can do semantic search 

## Usage

```python
curl --header "Content-Type: application/json"  --request POST \
--data '{
  "documents": ["Last year, I traveled to Spain","I have a dog",  " I have a close relationship to my family"],
  "query": "Do you have a dog?",
  "model": 1
}' \
http://localhost:5000/scorer

```

data json contains 3 keys 

1) documents contains list of documents passed to find score with query
2) query contain the query string 
3) model contain the model to use for making semantic score you can select number : 1 or 2

model key is optional if not passed model 1 is selected by default

if you are running this code on your local machine then use localhost else you can use your server link instead

You can pass your documents and query in a list like example shown above and select the model

