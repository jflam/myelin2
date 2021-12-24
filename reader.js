chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.content) {
            document.getElementById("readability").innerHTML = request.content;
            document.getElementById("title").innerText = request.title;
            document.getElementById("domain").innerText = request.siteName;
            document.getElementById("domain").setAttribute("href", request.uri);

            // Sending the page to the server will happen automatically for
            // now until I create some UI in the page that will let the 
            // user decide whether to send or not. This is something to test
            // with users to figure out whether the increased friction is 
            // worth it. Perhaps better to have a "submit by default" and
            // give the user an option to undo the submission?

            // Request will be a cross-origin request. Details can be found:
            // https://developer.chrome.com/docs/extensions/mv2/xhr/
            // Need to add localhost to the manifest "permissions" key
            // let xhr = new XMLHttpRequest();
            // xhr.open("POST", "http://localhost:8888/");
            // xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
            // xhr.send(JSON.stringify(
            // { 
            //     "title": request.title, 
            //     "text": request.textContent 
            // }));
        }
    }
)