var doc = null;

window.addEventListener("load", (event) => {
    document.getElementById("add").addEventListener("click", function() {
        if (doc !== null) {
            let xhr = new XMLHttpRequest();
            xhr.open("POST", "http://localhost:8888/add");
            xhr.setRequestHeader("Content-Type", 
                "application/json;charset=UTF-8");
            xhr.send(JSON.stringify(
            { 
                "uri": doc.uri,
                "title": doc.title, 
                "text": doc.textContent,
                "markup": doc.content,
                "markup_type": "html"
            }));
        }
        else {
            console.log("ERROR: doc is not defined on this page");
        }
    })

    document.getElementById("summarize").addEventListener("click", function() {
        if (doc !== null) {
            let xhr = new XMLHttpRequest();
            xhr.open("POST", "http://localhost:8888/summarize");
            xhr.setRequestHeader("Content-Type", 
                "application/json;charset=UTF-8");
            xhr.onreadystatechange = function() {
                if (xhr.readyState == XMLHttpRequest.DONE) {
                    let status = xhr.status;
                    if (status === 0 || (status >= 200 && status < 400)) {
                        let json = JSON.parse(xhr.responseText);
                        document.getElementById("summary").innerText = 
                            json.summary;
                    }
                    else {
                        console.log(`ERROR: ${xhr.responseText}`)
                    }
                }             }
            xhr.send(JSON.stringify(
            { 
                "text": doc.textContent
            }));
        }
        else {
            console.log("ERROR: doc is not defined on this page");
        }
    })
})

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.content) {
            doc = request;
            document.getElementById("readability").innerHTML = request.content;
            document.getElementById("title").innerText = request.title;
            document.getElementById("domain").innerText = request.siteName;
            document.getElementById("domain").setAttribute("href", request.uri);
        }
    }
)