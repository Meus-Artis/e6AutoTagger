// ==UserScript==
// @name         e621 Autotagger
// @namespace    https://MeusArtis.ca
// @version      1.3.1
// @author       Meus Artis
// @description  Adds a button that automatically tags e621 images and videos
// @icon         https://www.google.com/s2/favicons?domain=e621.net
// @updateURL    https://meusartis.ca/e6tagger.meta.js
// @downloadURL  https://meusartis.ca/e6tagger.user.js
// @match        https://e621.net/uploads/new
// @match        https://e926.net/uploads/new
// @match        https://e6ai.net/uploads/new
// @license      CC BY-NC-SA 4.0
// @grant        GM_xmlhttpRequest
// @grant        GM_registerMenuCommand
// ==/UserScript==
(function() {
    function sendRequest(url, base64String, confidence, textarea, fallback, done) {
        GM_xmlhttpRequest({
            method: "POST",
            url: url,
            headers: { "Content-Type": "application/json" },
            data: JSON.stringify({ data: [base64String, confidence], fn_index: 0 }),
            responseType: "json",
            onload: function(response) {
                if (response.status === 200 && response.response) {
                    textarea.value = response.response.data[0];
                    done();
                } else if (fallback) {
                    sendRequest(fallback, base64String, confidence, textarea, null, done);
                } else {
                    console.error("Error in response:", response);
                    done();
                }
            },
            onerror: function(err) {
                if (fallback) {
                    sendRequest(fallback, base64String, confidence, textarea, null, done);
                } else {
                    console.error("Error processing image:", err);
                    done();
                }
            },
            ontimeout: function() {
                if (fallback) {
                    sendRequest(fallback, base64String, confidence, textarea, null, done);
                } else {
                    console.error("Request timed out.");
                    done();
                }
            }
        });
    }
    function processImage(button, textarea, throbber) {
        const confidence = 0.25;
        const localjtp = "http://127.0.0.1:7860/run/predict";
        const media = document.querySelector(".upload_preview_img");
        button.disabled = true;
        button.style.opacity = "0.5";
        textarea.parentElement.insertBefore(throbber, textarea);
        let base64Promise;
        if (media.tagName === "IMG") {
            base64Promise = fetch(media.src)
                .then(res => res.blob())
                .then(blob => new Promise(resolve => {
                    let reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result);
                    reader.readAsDataURL(blob);
                }));
        } else if (media.tagName === "VIDEO") {
            base64Promise = getBase64FromVideo(media);
        }
        base64Promise.then(base64String => {
            sendRequest(localjtp, base64String, confidence, textarea, "https://meusartis.ca/run/predict", () => {
                button.disabled = false;
                button.style.opacity = "1";
                throbber.remove();
            });
        });
    }
    function addButton() {
        const textarea = document.getElementById("post_tags");
        if (!textarea || !textarea.parentElement) return;
        const button = document.createElement("button");
        button.textContent = "Tag Automatically";
        button.classList.add("toggle-button");
        button.title = "Powered By JTP Pilot², Hosted By Meus Artis";
        const warningText = document.createElement("span");
        warningText.textContent = "⚠️Manually review tags, or increase confidence setting in userscript.";
        warningText.style.color = "yellow";
        warningText.title = "Dynabird made me put this here";
        const throbber = document.createElement("div");
        throbber.textContent = "⏳ Processing...";
        Object.assign(throbber.style, {
            position: "absolute",
            background: "rgba(0, 0, 0, 0.5)",
            color: "white",
            padding: "2vh",
            borderRadius: "1vh",
            zIndex: "1000",
            marginTop: "1%",
            marginLeft: "15%"
        });
        button.onclick = () => processImage(button, textarea, throbber);
        textarea.parentElement.appendChild(button);
        textarea.parentElement.appendChild(warningText);
        const observer = new MutationObserver(() => checkMediaAndToggle(button));
        observer.observe(document.body, { childList: true, subtree: true });
        setInterval(() => checkMediaAndToggle(button), 420);
    }
    function getBase64FromVideo(video) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const snap = () => {
                try {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const base64 = canvas.toDataURL("image/jpeg");
                    if (/A{999,}/.test(base64)) {
                        alert("Captured frame is a black square or blank.");
                        reject("Blank base64");
                    } else {
                        resolve(base64);
                    }
                } catch (err) {
                    console.warn("Video frame unreadable:", err);
                    reject("Video frame unreadable");
                }
            };
            if (video.readyState >= 2) {
                const current = video.currentTime; // This does not work properly due to a 9 year old chrome bug.
                video.currentTime = current + 0.000001; // Use of an extension such as https://chromewebstore.google.com/detail/dbcfpoaehlbfdeeaonihhkoocmjgalco is **REQUIRED** to tag anything other than the first frame of a video.
                setTimeout(snap, 150);
            } else {
                video.addEventListener("loadeddata", () => {
                    setTimeout(snap, 150);
                }, { once: true });
            }
        });
    }
    function checkMediaAndToggle(button) {
        const empty = "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==";
        const media = document.querySelector(".upload_preview_img");
        if (media.tagName === "IMG") {
            button.disabled = media.src === empty;
        } else if (media.tagName === "VIDEO") {
            button.disabled = false;
        } else {
            button.disabled = true;
        }
    }
    function donate() {
        window.open("https://ko-fi.com/meusartis", '_blank');
    }
    GM_registerMenuCommand("Donate", donate);
    window.addEventListener("load", addButton);
})();
