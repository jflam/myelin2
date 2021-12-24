chrome.action.onClicked.addListener(async function (tab) {
    const [t] = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    // When a web page is loaded into a tab in the browser, we inject the
    // readability.js file into the page and start a message handler that
    // listens for incoming messages.

    // Here, we send a command message to that handler asking it to parse
    // the contents of the page and return to us a struct that contains the
    // parsed contents of the page. There are several pieces of information
    // that are returned: the title, the text content (with no markup) of
    // the page that is suitable for indexing, the simplified HTML content
    // of the page that is suitable for rendering, the finally the name and
    // uri of the site.
    chrome.tabs.sendMessage(t.id, { command: "readable" }, (response) => {

        // Using the information returned from the readability.js script 
        // injected in the page, construct a new web page using the template
        // in reader.html. This will be rendered as an extension UI with
        // the URI extension://{id}/reader.html, and will contain the 
        // primary UI that the user will use to interact with the page.

        // TODO: invoke DOMpurify over the page to ensure that we sanitize
        // the results before displaying them in the reader.html context.

        chrome.tabs.create({ 
            url: chrome.runtime.getURL("reader.html")
        });
        chrome.tabs.onUpdated.addListener((tabId, info, tab) => {

            // Once the page is created in the tab, we send a message to 
            // an event listener on the page with the content to be rendered
            chrome.tabs.sendMessage(tab.id, 
            { 
                "content": response.readable.content,
                "textContent": response.readable.textContent,
                "title": response.readable.title,
                "siteName": response.readable.siteName,
                "uri": t.url
            })
        });
    });
});