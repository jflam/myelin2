import sys
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# I want to create an abstract interface for managing and searching over the
# document collection. Ideally there are verbs that exist and different
# implementations that can be experimented with over time. The initial
# implementation wiil use sqlite to support rapid prototyping. I'm going to
# use huggingface + a vector database for the querying operations in the
# initial version

# ADD - add a new document to the corpus (returns an key)

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


@app.route("/", methods=["POST"])
@cross_origin()
def index():
    json_data = request.json
    print(f"input {json_data}", file=sys.stderr)
    title = json_data['title']
    text = json_data['text']
    result = {
        "title": title,
        "length": len(text)
    }
    print(f"RESULT: {result}", file=sys.stderr)
    return result

app.run(host="127.0.0.1", port=8888)