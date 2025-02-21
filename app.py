from flask import Flask, request, jsonify, render_template_string
import os
import json
from main import process_query  # We'll create this function in main.py

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>XyRo Agent Interface</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
        }
        .container { 
            margin-top: 20px; 
        }
        textarea { 
            width: 100%; 
            height: 100px; 
            margin-bottom: 10px; 
            padding: 10px;
        }
        button { 
            padding: 10px 20px; 
            background-color: #007bff; 
            color: white; 
            border: none; 
            cursor: pointer; 
        }
        #response { 
            margin-top: 20px; 
            white-space: pre-wrap; 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
        }
        .loading {
            display: none;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>XyRo Agent Interface</h1>
    <div class="container">
        <textarea id="query" placeholder="Enter your query here..."></textarea>
        <button onclick="submitQuery()">Submit Query</button>
        <div id="loading" class="loading">Processing your query...</div>
        <div id="response"></div>
    </div>
    <script>
        async function submitQuery() {
            const input = document.getElementById('query').value;
            const responseDiv = document.getElementById('response');
            const loadingDiv = document.getElementById('loading');
            
            if (!input.trim()) {
                responseDiv.textContent = 'Please enter a query';
                return;
            }
            
            loadingDiv.style.display = 'block';
            responseDiv.textContent = '';
            
            try {
                const response = await fetch('/run-agent', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ input: input })
                });
                
                const data = await response.json();
                responseDiv.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                responseDiv.textContent = 'Error: ' + error.message;
            } finally {
                loadingDiv.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/run-agent", methods=["POST"])
def run_agent():
    data = request.json
    user_input = data.get("input")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        result = process_query(user_input)
        response= {
            "response" : str(result)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "error": "Process failed",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)