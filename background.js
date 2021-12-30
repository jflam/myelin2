// Initiate search when user presses ENTER on the omnibox and myelin is the
// selected search engine
chrome.omnibox.onInputEntered.addListener((text) => {
    let url = `http://localhost:8888/search?q=${encodeURIComponent(text)}`;
    chrome.tabs.create({ url: url });
});

// User clicked on the add page to myelin button
chrome.action.onClicked.addListener(async function (tab) {
    const [t] = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    // When the user clicks on the add page to myelin button in the browser
    // toolbar, we inject the readability.js file into the page and register a
    // message handler that listens for the response from the readability.js
    // script that generates the readable form of the page and passes us a
    // parameter that contains the parsed content.

    // Dynamically inject the reader.js page into the newly created tab
    chrome.scripting.executeScript({
        target: { tabId: t.id },
        files: [ 'readability.js' ]
    });

    // Register the event handler that receives the computed results from
    // readability in the content page
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

        // Create a new tab for the reader.html page
        chrome.tabs.create({ 
            url: chrome.runtime.getURL("reader.html")
        });

        // Register a listener for the updated event that will tell us if 
        // when the newly created reader.html tab has completed initializing.
        chrome.tabs.onUpdated.addListener((tabId, info, tab) => {

            // Normalize to standard ASCII quote characters
            let splitFix = request.readable.textContent
                .replace(/[\u2018\u2019]/g, "'")
                .replace(/[\u201C\u201D]/g, '"')
                .replace(/\.([a-zA-Z])/g, ". $1");

            // Once the page is created in the tab, we send a message to 
            // an event listener on reader.html with the content to be rendered
            chrome.tabs.sendMessage(tab.id, 
            { 
                "content": request.readable.content,
                "textContent": splitFix,
                "title": request.readable.title,
                "siteName": request.readable.siteName,
                "uri": t.url
            })
        });
    });
});