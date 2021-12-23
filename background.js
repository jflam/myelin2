chrome.action.onClicked.addListener(async function (tab) {
    const [t] = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });
    chrome.tabs.sendMessage(t.id, { command: "readable" }, (response) => {
        let textContent = response.readable.textContent;
        let content = response.readable.content;
        let background = chrome.runtime.getURL("reader.html");
        chrome.tabs.create({ url: background });
        chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
            chrome.tabs.sendMessage(tab.id, { "content": textContent })
        });
    });
});