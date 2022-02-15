from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
stop_words = "english"
n_gram_range = (1, 1)
top_n = 5

def func(d1,d2):
    count1 = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([d1])
    count2 = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([d2])
    candidates1 = count1.get_feature_names()
    candidates2 = count2.get_feature_names()
    doc_embedding1 = model.encode([d1])
    doc_embedding2 = model.encode([d2])
    candidate_embeddings1 = model.encode(candidates1)
    candidate_embeddings2 = model.encode(candidates2)
    distances1 = cosine_similarity(doc_embedding1, candidate_embeddings1)
    distances2 = cosine_similarity(doc_embedding2, candidate_embeddings2)
    keywords1 = [candidates1[index] for index in distances1.argsort()[0][-top_n:]]
    keywords2 = [candidates2[index] for index in distances2.argsort()[0][-top_n:]]
    sc=0
    for i in keywords1:
        for j in keywords2:
            if i==j:
                sc+=1
    return(sc)

from flask import *
app = Flask(__name__)

@app.route('/',methods=['POST', 'GET'])
def index():
    if(request.method=='GET'):
        return render_template('upload.html')
    elif(request.method=='POST'):
        d1 = request.form['file1']
        d2 = request.form['file2']

        score=str(func(d1,d2))
        return render_template("upload.html", name = score)

if __name__ == '__main__':
    app.run(debug=True)