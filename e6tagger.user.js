// ==UserScript==
// @name         e621 Autotagger
// @namespace    https://MeusArtis.ca
// @version      1.2.2
// @author       Meus Artis
// @description  Adds a button that automatically tags e621 images
// @icon         https://www.google.com/s2/favicons?domain=e621.net
// @match        https://e621.net/uploads/new
// @match        https://e926.net/uploads/new
// @match        https://e6ai.net/uploads/new
// @license      CC BY-NC-SA 4.0
// @grant        GM_xmlhttpRequest
// ==/UserScript==
(function() {
    function sendRequest(url, base64String, confidence, textarea, fallback) {
        GM_xmlhttpRequest({
            method: "POST",
            url: url,
            headers: { "Content-Type": "application/json" },
            data: JSON.stringify({ data: [base64String, confidence], fn_index: 0 }),
            responseType: "json",
            onload: function(response) {
                if (response.status === 200 && response.response) {
                    textarea.value = response.response.data[0];
                } else if (fallback) {
                    sendRequest(fallback, base64String, confidence, textarea, null);
                } else {
                    console.error("Error in response:", response);
                }
            },
            onerror: function(err) {
                if (fallback) {
                    sendRequest(fallback, base64String, confidence, textarea, null);
                } else {
                    console.error("Error processing image:", err);
                }
            },
            ontimeout: function() {
                if (fallback) {
                    sendRequest(fallback, base64String, confidence, textarea, null);
                } else {
                    console.error("Request timed out.");
                }
            }
        });
    }
    function processImage(button, textarea, throbber) {
        var confidence = 0.25;
        var localjtp = "http://127.0.0.1:7860/run/predict"
        let img = document.querySelector(".upload_preview_img");
        if (!img) {
            console.error("No image found!");
            return;
        }
        button.disabled = true;
        button.style.opacity = "0.5";
        textarea.parentElement.insertBefore(throbber, textarea);
        fetch(img.src)
            .then(res => res.blob())
            .then(blob => new Promise(resolve => {
                let reader = new FileReader();
                reader.onloadend = () => resolve(reader.result);
                reader.readAsDataURL(blob);
            }))
            .then(base64String => {
                sendRequest(localjtp, base64String, confidence, textarea, "https://meusartis.ca/run/predict");
            })
            .finally(() => {
                button.disabled = false;
                button.style.opacity = "1";
                throbber.remove();
            });
    }
    function addButton() {
        let textarea = document.getElementById("post_tags");
        if (!textarea || !textarea.parentElement) return;
        let button = document.createElement("button");
        button.textContent = "Tag Automatically";
        button.classList.add("toggle-button");
        button.title = "Powered By JTP Pilot², Hosted By Meus Artis";
        let warningText = document.createElement("span");
        warningText.textContent = "⚠️Manually review tags, or increase confidence setting in userscript.";
        warningText.style.color = "yellow";
        warningText.title = "Dynabird made me put this here";
        let throbber = document.createElement("div");
        throbber.textContent = "⏳ Processing...";
        throbber.style.position = "absolute";
        throbber.style.background = "rgba(0, 0, 0, 0.5)";
        throbber.style.color = "white";
        throbber.style.padding = "2vh";
        throbber.style.borderRadius = "1vh";
        throbber.style.zIndex = "1000";
        throbber.style.marginTop = "1%";
        throbber.style.marginLeft = "15%";
        button.onclick = () => processImage(button, textarea, throbber);
        textarea.parentElement.appendChild(button);
        textarea.parentElement.appendChild(warningText);
    }
    window.addEventListener("load", addButton);
})();