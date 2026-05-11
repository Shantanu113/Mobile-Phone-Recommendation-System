import subprocess
import sys
import os

# Auto-install all dependencies before anything else loads
_req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
if os.path.exists(_req_file):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", _req_file, "--quiet"])

from flask import Flask, render_template, request, jsonify
from recommender import recommend, get_brands, get_price_range, get_dataset_stats

app = Flask(__name__)


@app.route("/")
def index():
    brands = get_brands()
    price_min, price_max = get_price_range()
    stats = get_dataset_stats()
    return render_template(
        "index.html",
        brands=brands,
        price_min=price_min,
        price_max=price_max,
        stats=stats,
    )


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    try:
        results = recommend(
            budget_min    = float(data.get("budget_min", 100)),
            budget_max    = float(data.get("budget_max", 1000)),
            os_pref       = data.get("os_pref", "Any"),
            camera_w      = float(data.get("camera_w", 0.5)),
            battery_w     = float(data.get("battery_w", 0.5)),
            ram_w         = float(data.get("ram_w", 0.5)),
            performance_w = float(data.get("performance_w", 0.5)),
            storage_w     = float(data.get("storage_w", 0.5)),
            selfie_w      = float(data.get("selfie_w", 0.5)),
            brand_pref    = data.get("brand_pref", "Any"),
            top_n         = int(data.get("top_n", 5)),
        )
        return jsonify({"success": True, "results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    return jsonify(get_dataset_stats())


@app.route("/api/phones")
def api_phones():
    """Return all phones for dataset view."""
    from recommender import DF_RAW
    phones = DF_RAW.to_dict(orient="records")
    return jsonify(phones)


if __name__ == "__main__":
    print("Starting Mobile Phone Recommendation System...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
