"""
Flask REST API and minimal web UI for real-time phishing classification.
Input: URL. Output: Classification (Phishing/Legitimate), risk score (0-100), top features.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, render_template_string
from deployment.predictor import predict_dict

app = Flask(__name__)

# Rate limit / abuse: optional (e.g. flask-limiter). For now we rely on timeout and safe URL checks.

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phishing URL Detector</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.5rem; color: #1a1a2e; }
    .input-group { display: flex; gap: 0.5rem; margin: 1rem 0; }
    input[type="url"] { flex: 1; padding: 0.6rem; border: 1px solid #ccc; border-radius: 6px; }
    button { padding: 0.6rem 1.2rem; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .result { margin-top: 1.5rem; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb; }
    .result.phishing { background: #fef2f2; border-color: #fecaca; }
    .result.legitimate { background: #f0fdf4; border-color: #bbf7d0; }
    .result .classification { font-weight: 700; font-size: 1.1rem; }
    .result .risk { margin: 0.5rem 0; }
    .result .features { margin-top: 0.75rem; font-size: 0.9rem; color: #374151; }
    .result .features ul { margin: 0.25rem 0 0 1rem; padding: 0; }
    .error { color: #dc2626; }
  </style>
</head>
<body>
  <h1>Phishing URL Detector</h1>
  <p>Enter a URL to classify as Phishing or Legitimate. Risk score 0–100 and top contributing features are shown.</p>
  <form method="get" action="/" class="input-group">
    <input type="url" name="url" placeholder="https://example.com" value="{{ url or '' }}" required>
    <button type="submit">Check URL</button>
  </form>
  {% if result %}
  <div class="result {{ result.classification|lower }}">
    <div class="classification">Classification: {{ result.classification }}</div>
    <div class="risk">Risk score: {{ result.risk_score }}/100</div>
    {% if result.top_contributing_features %}
    <div class="features">Top contributing features:</div>
    <ul>
      {% for f in result.top_contributing_features %}
      <li>{{ f.name }}: {{ "%.4f"|format(f.contribution) }}</li>
      {% endfor %}
    </ul>
    {% endif %}
    {% if result.error %}<div class="error">{{ result.error }}</div>{% endif %}
  </div>
  {% endif %}
</body>
</html>
"""


@app.route("/")
def index():
    url = request.args.get("url", "").strip()
    result = None
    if url:
        result = predict_dict(url, fetch_content=False)
    return render_template_string(HTML_TEMPLATE, url=url, result=result)


@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    """REST: GET ?url=... or POST JSON { \"url\": \"...\" }."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
    else:
        url = (request.args.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400
    out = predict_dict(url, fetch_content=False)
    if out.get("error"):
        return jsonify(out), 500
    return jsonify(out)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
