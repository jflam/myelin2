chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.content) {
            document.getElementById("readability").innerHTML = request.content;
            document.getElementById("title").innerText = request.title;
            document.getElementById("domain").innerText = request.siteName;
            document.getElementById("domain").setAttribute("href", request.uri);
        }
    }
)