from typing import Any
import nltk, os, shutil, sqlite3, sys, time
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
EH_PATH = f"metadata/{EH_SPACE}"

# Maximum number of results to return when querying embeddinghub
MAX_RESULTS = 5

# Load the SENTENCE_TRANSFORMER model. This can be changed by changing the
# constant defined earlier in this file. This one seems to run reasonably
# well on my RTX 2080, though this is still hardly "fast"
model = SentenceTransformer(SENTENCE_TRANSFORMER)

def create_sqlite(force: bool=False):
    if force:
        os.remove(DATABASE_NAME)

    print(f"CREATING SQLite database #{DATABASE_NAME}", file=sys.stderr)
    with closing(sqlite3.connect(DATABASE_NAME)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("""
CREATE TABLE resources (uri TEXT, title TEXT, markup TEXT, markup_type TEXT)
""")
            cur.execute("""
CREATE TABLE chunks (text TEXT, resource_id INTEGER NOT NULL)
""")

def create_embeddinghub(force: bool=False):
    if force and os.path.exists(EH_PATH):
        shutil.rmtree(EH_PATH)

    print(f"CREATING embeddinghub space #{EH_SPACE}", file=sys.stderr)
    hub = eh.connect(eh.Config(host=EH_HOST, port=EH_PORT))
    hub.create_space(EH_SPACE, dims=MAX_LENGTH)

def create_databases(force: bool=False):
    """Create SQLite and embeddinghub databases for use by the indexer"""

    # Create SQLite database if it doesn't exist already
    # Two tables, and a join is needed to retrieve search results
    # - resources which contains the HTML of the resource
    # - embedding which contains the chunk and a link to the resource
    if force:
        create_sqlite(force)
        create_embeddinghub(force)
    else:
        if not os.path.exists(DATABASE_NAME):
            create_sqlite()
        if not os.path.exists(EH_PATH):
            create_embeddinghub()

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

def index_resource(resource: Any):
    """Add resource to SQLite and embeddinghub 

    Args:
        resource (Any): JSON object representing request from client        
    """
    id = None
    with closing(sqlite3.connect(DATABASE_NAME)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("INSERT INTO resources(uri, title, markup, "
                "markup_type) VALUES (?, ?, ?, ?)", (
                    resource["uri"], 
                    resource["title"],
                    resource["markup"],
                    resource["markup_type"]
            ))
            resource_id = cur.lastrowid
            print(f"INSERTED {resource['title']}, ID {resource_id} INTO resources", 
                file=sys.stderr)

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

            hub = eh.connect(eh.Config(host=EH_HOST, port=EH_PORT))
            space = hub.get_space(EH_SPACE)
            for chunk in tokens["input_ids"]:

                # TODO: update this as decode() will return tokens like [SEP] etc.
                # which are not needed here. Ideally what we do is use the same 
                # chunking algorithm used in the model.tokenizer() call above to 
                # generate the chunks that are encoded vs. going back to a string
                # first before re-encoding them.
                chunk_str = model.tokenizer.decode(chunk)
                cur.execute("INSERT INTO chunks(text, resource_id) "
                    "VALUES (?, ?)", (chunk_str, resource_id))
                print(f"INSERTED chunk", file=sys.stderr)
                chunk_id = cur.lastrowid
                chunk_embedding = model.encode(chunk_str, 
                    convert_to_tensor=True).cpu()
                space.set(chunk_id, chunk_embedding)

            conn.commit()

@app.route("/add", methods=["POST"])
@cross_origin()
def add():
    """Add a resource to the corpus"""
    index_resource(request.json)
    return { "status": 0 }

@app.route("/search", methods=['GET'])
@cross_origin()
def search():
    if "q" in request.args:
        query = request.args["q"]
        print(f"SEARCHING for {query}", file=sys.stderr)
        embedding = model.encode(query, convert_to_tensor=True).cpu()

        # TODO: remove this workaround once embeddinghub fixes this bug
        e = eh.embedding_store_pb2.Embedding()
        e.values[:] = embedding.tolist()

        hub = eh.connect(eh.Config(host=EH_HOST, port=EH_PORT))
        space = hub.get_space(EH_SPACE)

        # results contains a list of rowids that need to be fetched from SQLite
        # to get the actual results
        rowids = space.nearest_neighbors(MAX_RESULTS, vector=e)
        print(f"RESULTS: {rowids}", file=sys.stderr)

        search_results = []
        seq = ",".join(['?'] * len(rowids))
        sql = f"""
SELECT resources.rowid, resources.title, resources.uri, resources.markup, 
    resources.markup_type, chunks.text
FROM resources 
INNER JOIN chunks ON chunks.resource_id = resources.rowid
WHERE chunks.rowid IN ({seq})
"""
        print(f"SQL: {sql}, rowids {rowids}")
        with closing(sqlite3.connect(DATABASE_NAME)) as conn:
            with closing(conn.cursor()) as cur:
                results = cur.execute(sql, rowids[:MAX_RESULTS])
                for result in results:
                    search_result = f"""
<a href='{result[2]}'>{result[1]}</a>
<div>{result[5]}</div>
                    """
                    search_results.append(search_result)

        return f"<html><body>{''.join(search_results)}</body></html>"

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

create_databases()
app.run(host="127.0.0.1", port=8888)