import os, sqlite3, sys

from contextlib import closing
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

DATABASE_NAME = "myelin.db"

# Create database if it doesn't exist already
if not os.path.exists(DATABASE_NAME):
    with closing(sqlite3.connect(DATABASE_NAME)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("CREATE TABLE resources (uri TEXT, "
                "title TEXT, text TEXT NOT NULL, markup TEXT NOT NULL, "
                "markup_type TEXT)")

# I need a table that contains all the entries. It has the following schema
# - 

# I want to create an abstract interface for managing and searching over the
# document collection. Ideally there are verbs that exist and different
# implementations that can be experimented with over time. The initial
# implementation wiil use sqlite to support rapid prototyping. I'm going to
# use huggingface + a vector database for the querying operations in the
# initial version

# ADD - add a new document to the corpus and returns the result of summarizing
# the document.

# SUMMARIZE - compute the summary of a document in cases where we don't want
# to add the document to the corpus as well

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

@app.route("/add", methods=["POST"])
@cross_origin()
def add():
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

app.run(host="127.0.0.1", port=8888)