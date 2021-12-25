import nltk, os, sqlite3, sys, time
import numpy as np

from contextlib import closing
from flask import Flask, request
from flask_cors import CORS, cross_origin
from lexrank import degree_centrality_scores
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

DATABASE_NAME = "myelin.db"
SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"

# Create database if it doesn't exist already
if not os.path.exists(DATABASE_NAME):
    with closing(sqlite3.connect(DATABASE_NAME)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("CREATE TABLE resources (uri TEXT, "
                "title TEXT, text TEXT NOT NULL, markup TEXT NOT NULL, "
                "markup_type TEXT)")

# I want to create an abstract interface for managing and searching over the
# document collection. Ideally there are verbs that exist and different
# implementations that can be experimented with over time. The initial
# implementation wiil use sqlite to support rapid prototyping. I'm going to
# use huggingface + a vector database for the querying operations in the
# initial version

# GET - retrieve an existing document from the corpus (using key from ADD or
# more general retrieval, e.g., by URI)

# SEARCH - retrieve using vector-based full text search. This might be more
# challenging for longer articles, so it might be best to do some
# summarization and then search based on the summary?

# DELETE

# UPVOTE - upvote the relationship between resources. This is done to improve
# ranking of search results in the future. This is an active way of indicating
# interest in a topic. 

# RELATED - retrieve related resources based on what I'm reading now. This
# should use some kind of summarization technique to query the knowledgebase
# for similar documents (both notes and saved resources). Perhaps this should
# use SEARCH function but passing in the summarized version of the current
# page as input?

# I wonder if there's a good way to auto-generate tags for a topic (or at)
# least let the user define tags and then start making tag suggestions based
# on the content from other tags. This would be a huge usability boost as some
# time spent training the tagging model would be repaid by good tagging
# results in the future. There is a lot of prior art in NLP + tagging space.

# ADD - add a new document to the corpus and returns the result of summarizing
# the document.

@app.route("/add", methods=["POST"])
@cross_origin()
def add():
    """Add a resource to the corpus"""
    json_data = request.json
    with closing(sqlite3.connect(DATABASE_NAME)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("INSERT INTO resources(uri, title, text, markup, "
                "markup_type) VALUES (?, ?, ?, ?, ?)", (
                    json_data["uri"], 
                    json_data["title"],
                    json_data["text"],
                    json_data["markup"],
                    json_data["markup_type"]
            ))
            conn.commit()
    print(f"INSERTED {json_data['title']} into resources", file=sys.stderr)
    return { "status": 0 }

# SUMMARIZE - compute the summary of a document in cases where we don't want
# to add the document to the corpus as well

@app.route("/summarize", methods=["POST"])
@cross_origin()
def summarize():
    """Return a summarization of the resource"""
    json_data = request.json
    text = json_data["text"]

    # Time the transformation algorithm
    start_time = time.process_time()

    # Tokenize the text into sentences. Note that I pre-process on the client
    # into standard ASCII single and double quotes from the fancy ones that
    # are often found on web properties. The sent_tokenize algorithm fails
    # to tokenize those correctly.
    sentences = nltk.sent_tokenize(text)

    # DEBUG check out the sentences from tokenizer
    for i, sentence in enumerate(sentences):
        print(f"{i} {sentence}", file=sys.stderr)

    # Load the SENTENCE_TRANSFORMER model. This can be changed by changing the
    # constant defined earlier in this file. This one seems to run reasonably
    # well on my RTX 2080, though this is still hardly "fast"
    model = SentenceTransformer(SENTENCE_TRANSFORMER)

    # Generate vector of embeddings from the previously split sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # This is an O(N^2) compute of cosine similarities between all sentence
    # pairs in the embeddings vector
    cosine_similarities = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # Use the LEXRANK algorithm to compute the most central sentences
    # See this paper: https://arxiv.org/abs/1109.2128 for algorithm details
    centrality_scores = degree_centrality_scores(cosine_similarities, 
        threshold=None)

    # Sort centrality scores from best to worst
    most_central_sentence_indices = np.argsort(-centrality_scores)

    # Generate a summary that contains the top 5 most central sentences 
    summary = ""
    for i in most_central_sentence_indices[0:5]:
        summary += sentences[i].strip() 
    elapsed_time = time.process_time() - start_time

    return { 
        "input_sentences": len(sentences),
        "input_characters": len(text),
        "summary": summary,
        "compute_time_seconds": f"{elapsed_time:.2f}"
    }

app.run(host="127.0.0.1", port=8888)