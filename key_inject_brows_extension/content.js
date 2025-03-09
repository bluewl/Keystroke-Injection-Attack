// content.js

document.addEventListener("keydown", function(event) {
    if (event.key >= '0' && event.key <= '9') {
        // event.preventDefault(); // Prevent the number from being typed

        function simulateKeyPress(key) {
            let delayBeforePress = Math.floor(Math.random() * 50) + 10; // Random delay between 10ms and 50ms
            let keyUpDelay = Math.floor(Math.random() * 50) + 50; // Random delay between 50ms and 100ms

            setTimeout(() => {
                console.log("Simulating key press: " + key);
                let keydownEvent = new KeyboardEvent("keydown", {
                    key: key,
                    code: "Key" + key.toUpperCase(),
                    keyCode: key.charCodeAt(0),
                    which: key.charCodeAt(0),
                    bubbles: true,
                    cancelable: true
                });

                let keyupEvent = new KeyboardEvent("keyup", {
                    key: key,
                    code: "Key" + key.toUpperCase(),
                    keyCode: key.charCodeAt(0),
                    which: key.charCodeAt(0),
                    bubbles: true,
                    cancelable: true
                });

                document.dispatchEvent(keydownEvent);

                setTimeout(() => document.dispatchEvent(keyupEvent), keyUpDelay);
            }, delayBeforePress);
        }

        simulateKeyPress("q");
    }
});
