"""XAUUSD analyzer web app: one JSON endpoint and a single-page dashboard."""
import os

from flask import Flask, jsonify, request, send_from_directory

import backtest
import engine

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/analysis")
def analysis():
    tf = request.args.get("tf", "H1")
    mode = request.args.get("mode", "standard")
    try:
        result = engine.analyze(tf, mode)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - surface feed failures to the page
        app.logger.exception("analysis failed")
        return jsonify({"error": f"Market data is unavailable right now ({exc})."}), 503
    response = jsonify(result)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/backtest")
def run_backtest():
    tf = request.args.get("tf", "H1")
    mode = request.args.get("mode", "standard")
    peek = request.args.get("peek") == "1"
    try:
        job = backtest.start(tf, mode, peek=peek)
    except ValueError as exc:
        return jsonify({"state": "unsupported", "error": str(exc)}), 200
    body = {"state": job["state"], "progress": job["progress"], "error": job["error"]}
    if job["state"] == "done":
        body["result"] = job["result"]
    response = jsonify(body)
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
