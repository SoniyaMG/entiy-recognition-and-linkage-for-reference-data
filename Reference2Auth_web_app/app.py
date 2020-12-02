
import tensorflow as tf
import pickle
import pandas as pd
import chars2vec
from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template, json, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer


chars2vec_model = chars2vec.load_model('eng_200')
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

with open('./model/index2auth.pickle','rb') as file:
  index2auth=pickle.load(file)

with open('./model/venue_fullforms','rb') as file:
  venue_fullforms=pickle.load(file)

with open('./model/url_author','rb') as file:
  url_author=pickle.load(file)


def generateEmbeddings(reference):

  authors_list = reference["authors"]
  title = reference["title"]
  venue = reference["venue"]
  reference_embedding = {}

  for i in range(len(authors_list)):
    main_author_embeddings = []
    main_author = authors_list[i]
    co_authors = [auth for auth in authors_list if auth != main_author] 

    main_author_emb = chars2vec_model.vectorize_words([main_author])[0]
    title_emb = bert_model.encode(title)
    venue_emb = bert_model.encode(venue_fullforms[venue])

    for co_author in co_authors:
       co_author_emb = chars2vec_model.vectorize_words([co_author])[0]
       main_author_embeddings.append(np.concatenate([main_author_emb,co_author_emb,title_emb,venue_emb]))
    main_author_embeddings = np.array(main_author_embeddings)
    reference_embedding[i] = main_author_embeddings
  
  return reference_embedding


def predict_authors(reference,saved_model):

  predicted_authors_dict = {}
  reference_embedding = generateEmbeddings(reference)
  for main_auth_index in reference_embedding:

    main_author_embeddings = reference_embedding[main_auth_index]
    main_author_predictions = []

    for i in range(len(main_author_embeddings)):
      emb = main_author_embeddings[i]
      emb = tf.keras.utils.normalize(emb)
      main_author_predictions.append(saved_model.predict(emb)[0])

    main_author_predictions = np.array(main_author_predictions)
    main_author_predictions_sum = np.sum(main_author_predictions,axis=0)
    index_of_main_author = np.argmax(main_author_predictions_sum)
    predicted_authors_dict[reference["authors"][main_auth_index]] = index2auth[index_of_main_author]
    
  return predicted_authors_dict


app = Flask(__name__)
run_with_ngrok(app)

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('home2.html')

@app.route('/visualisations',methods=['GET'])
def visual():
  return render_template('EntityToVecVisualisations.html')

@app.route('/testreference',methods=['GET'])
def reference():
  return render_template('testreference.html')


@app.route('/',methods=['POST'])
def login():
    form_authors = request.form['Authors']
    form_title = request.form['Title']
    form_venue = request.form['Venue']
    authors_list = form_authors.split(",")

    ref = {}
    ref["authors"] = authors_list
    ref["title"] = form_title
    ref["venue"] = form_venue

    path_to_trained_model = './model/Reference2Auth_model.h5'
    saved_model = tf.keras.models.load_model(path_to_trained_model)

    headings = ["Original Author", "Predicted Author"]
    predicted_list = predict_authors(ref,saved_model)

    dictlist = []
    for key, value in predicted_list.items():
      link = url_author[value]
      temp = [key,value,link]
      dictlist.append(temp)
    print(dictlist)

    df = pd.DataFrame(predicted_list.items()) 
    print(df)
    print(predicted_list)
    return render_template('home2.html',headings=headings,predicted_list=dictlist)                         

app.run()
