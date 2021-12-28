from typing import Any
import nltk, os, sqlite3, sys, time
import embeddinghub as eh
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

# Tunable parameters based on model used for embedding calculations
MAX_LENGTH = 384
STRIDE = 128

# embeddinghub parameters
EH_HOST = "0.0.0.0"
EH_PORT = 7462
EH_SPACE = "kb"

# Maximum number of results to return when querying embeddinghub
MAX_RESULTS = 5

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

def insert_resource(resource: Any)-> int:
    """Insert resource into SQLite resource table

    Args:
        resource (Any): JSON object representing request from client

    Returns:
        int: rowid of newly inserted row
    """
    id = None
    with closing(sqlite3.connect(DATABASE_NAME)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("INSERT INTO resources(uri, title, text, markup, "
                "markup_type) VALUES (?, ?, ?, ?, ?)", (
                    resource["uri"], 
                    resource["title"],
                    resource["text"],
                    resource["markup"],
                    resource["markup_type"]
            ))
            id = cur.lastrowid
            conn.commit()
    print(f"INSERTED {resource['title']}, ID {id} into resources", file=sys.stderr)
    return id

def index_resource(resource: Any, id: int):
    """Add resource, id to embeddinghub

    Args:
        resource (Any): JSON object representing request from client        
        id (int): rowid of resource in SQLite database
    """
    model = SentenceTransformer(SENTENCE_TRANSFORMER)

    # Call to tokenizer() will compute N embeddings, each one representing a
    # chunk of the resource. The STRIDE parameter controls the amount of
    # token overlap between chunks, and MAX_LENGTH (which is model-dependent)
    # controls the size of each chunk.
    tokens = model.tokenizer(
        text=resource["text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_overflowing_tokens=True,
        stride=STRIDE
    )

    chunk_embeddings = []
    for chunk in tokens["input_ids"]:

        # TODO: update this as decode() will return tokens like [SEP] etc.
        # which are not needed here. Ideally what we do is use the same 
        # chunking algorithm used in the model.tokenizer() call above to 
        # generate the chunks that are encoded vs. going back to a string
        # first before re-encoding them.
        chunk_str = model.tokenizer.decode(chunk)
        chunk_embeddings.append(
            model.encode(chunk_str, convert_to_tensor=True).cpu())

    # Each chunk embedding must map to the same resource id
    hub = eh.connect(eh.Config(host=EH_HOST, port=EH_PORT))
    space = hub.get_space(EH_SPACE)
    for chunk_embedding in chunk_embeddings:
        space.set(id, chunk_embedding)

@app.route("/add", methods=["POST"])
@cross_origin()
def add():
    """Add a resource to the corpus"""
    id = insert_resource(request.json)
    index_resource(request.json, id)
    return { "status": 0 }

@app.route("/search", methods=["POST"])
@cross_origin()
def search():
    query = request.json["query"]
    model = SentenceTransformer(SENTENCE_TRANSFORMER)
    embedding = model.encode(query, convert_to_tensor=True).cpu()

    # TODO: remove this workaround once embeddinghub fixes this bug
    e = eh.embedding_store_pb2.Embedding()
    e.values[:] = embedding.tolist()

    hub = eh.connect(eh.Config(host=EH_HOST, port=EH_PORT))
    space = hub.get_space(EH_SPACE)

    # results contains a list of rowids that need to be fetched from SQLite
    # to get the actual results
    rowids = space.nearest_neighbors(MAX_RESULTS, vector=e)

    # Note that result is an id that needs to retrieve actual results from 
    print(f"QUERY: {query}", file=sys.stderr)
    for rowid in rowids:
        # TODO: lookup results in SQLite
        print(f"rowid {rowid}", file=sys.stderr)

# Two different models for summarization. One is more explicit through the use
# of the LEXRANK algorithm, the other uses BART or T5 which is explicitly 
# trained on this task.

def summarize_lexrank():
    pass

def summarize_bart():
    pass

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