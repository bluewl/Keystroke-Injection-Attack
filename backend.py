from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/save_keystrokes', methods=['POST'])
def save_keystrokes():
    data = request.json
    keystrokes = data.get("keystrokes", [])

    if not keystrokes:
        return jsonify({"message": "No keystrokes received"}), 400

    # save to csv
    df = pd.DataFrame(keystrokes)
    df.to_csv("keystrokes.csv", mode='a', index=False, header=False)

    return jsonify({"message": "Keystrokes saved successfully"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)