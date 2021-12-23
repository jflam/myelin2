chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.content) {
            document.getElementById("readability").innerText = request.content;
        }
    }
)