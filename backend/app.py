from __future__ import annotations

import gzip

from flask import Flask, jsonify, request, send_file

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover - optional until requirements are installed
    CORS = None

try:
    from config import API_PREFIX, DEBUG, HOST, PORT
    from services.explain import build_explanation
    from services.data_generator import generate_test_data
    from services.custom_image import prepare_custom_image
    from services.model_registry import find_model_by_source_path, get_model, list_models
    from services.path_resolver import resolve_ai_test_image_path
    from services.test_runner import run_model_test
except ImportError:  # pragma: no cover - used when imported as backend.app
    from .config import API_PREFIX, DEBUG, HOST, PORT
    from .services.explain import build_explanation
    from .services.data_generator import generate_test_data
    from .services.custom_image import prepare_custom_image
    from .services.model_registry import find_model_by_source_path, get_model, list_models
    from .services.path_resolver import resolve_ai_test_image_path
    from .services.test_runner import run_model_test


def create_app() -> Flask:
    app = Flask(__name__)

    @app.after_request
    def compress_json(response):
        # Feature maps are large; compress before the serverless response limit.
        if response.is_json and "gzip" in request.headers.get("Accept-Encoding", ""):
            content = response.get_data()
            if len(content) > 1024:
                compressed = gzip.compress(content)
                response.response = (compressed[start:start + 65536] for start in range(0, len(compressed), 65536))
                response.headers.pop("Content-Length", None)
                response.headers["Content-Encoding"] = "gzip"
                response.vary.add("Accept-Encoding")
        return response

    if CORS is not None:
        CORS(app, resources={rf"{API_PREFIX}/*": {"origins": "*"}})

    @app.get("/")
    def root():
        return jsonify(
            {
                "service": "cnn-explainer-backend",
                "status": "ok",
                "apiPrefix": API_PREFIX,
            }
        )

    @app.get(f"{API_PREFIX}/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get(f"{API_PREFIX}/models")
    def models():
        return jsonify({"models": list_models()})

    @app.get(f"{API_PREFIX}/models/<model_id>")
    def model_detail(model_id: str):
        model = get_model(model_id)
        if model is None:
            return jsonify({"error": f"Unknown model id: {model_id}"}), 404

        return jsonify({"model": model})

    # PYTORCH_BACKEND_INTEGRATION:
    # Serves AI_Test_UI sample images for the Svelte image picker. If the UI is
    # switched back to bundled TensorFlow.js assets, this route can stay unused
    # or be commented out with the backend-only image path.
    @app.get(f"{API_PREFIX}/image")
    def image():
        image_path = request.args.get("path", "test_original/1_7.jpg")
        try:
            resolved_image_path = resolve_ai_test_image_path(image_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return send_file(resolved_image_path)

    @app.post(f"{API_PREFIX}/custom-image")
    def custom_image():
        payload = request.get_json(silent=True) or {}
        source = payload.get("source", "")
        try:
            result = prepare_custom_image(source)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(result)

    @app.post(f"{API_PREFIX}/explain")
    def explain():
        payload = request.get_json(silent=True) or {}
        model_id = payload.get("modelId", "cnn-net-28-ori")
        image_path = payload.get("imagePath", "test_original/1_7.jpg")

        model = get_model(model_id)
        if model is None:
            return jsonify({"error": f"Unknown model id: {model_id}"}), 404

        try:
            explanation = build_explanation(model, image_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": "explain_failed", "message": str(exc)}), 500

        return jsonify(explanation)

    @app.post(f"{API_PREFIX}/test")
    def test_model():
        payload = request.get_json(silent=True) or {}
        model, error_response, status_code = resolve_model_from_payload(payload)
        if error_response is not None:
            return jsonify(error_response), status_code

        # PYTORCH_BACKEND_INTEGRATION:
        # Clean JSON replacement for AI_Test_UI's test.py subprocess output.
        try:
            test_result = run_model_test(
                model_spec=model,
                test_data_path=payload.get("testDataPath", "accuracy"),
                test_type=payload.get("testType", "accuracy"),
                limit=payload.get("limit"),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": "test_failed", "message": str(exc)}), 500

        return jsonify(test_result)

    @app.post(f"{API_PREFIX}/generate-data")
    def generate_data():
        payload = request.get_json(silent=True) or {}
        try:
            result = generate_test_data(
                data_path=payload.get("dataPath"),
                output_path=payload.get("outputPath"),
                generation_method=payload.get("generationMethod"),
                option=payload.get("option"),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": "generation_failed", "message": str(exc)}), 500

        return jsonify(result)

    @app.post(f"{API_PREFIX}/save-result")
    def save_result():
        payload = request.get_json(silent=True) or {}
        return not_implemented(
            "Result saving will be wired after the test endpoint returns clean JSON.",
            expectedPayload={
                "modelId": "cnn-net-28-ori",
                "testType": "accuracy",
                "output": "accuracy: 0.95",
            },
            receivedPayload=payload,
        )

    return app


def not_implemented(message: str, **details):
    response = {"error": "not_implemented", "message": message}
    response.update(details)
    return jsonify(response), 501


def resolve_model_from_payload(payload: dict):
    model_id = payload.get("modelId")
    if model_id:
        model = get_model(model_id)
        if model is None:
            return None, {"error": f"Unknown model id: {model_id}"}, 404
        return model, None, 200

    # AI_TEST_UI_COMPATIBILITY:
    # The old Flask app sent modelPath. Keep this supported while the new
    # frontend uses modelId from /api/models.
    model_path = payload.get("modelPath")
    if model_path:
        model = find_model_by_source_path(model_path)
        if model is None:
            return None, {"error": f"Unknown registered modelPath: {model_path}"}, 404
        return model, None, 200

    model = get_model("cnn-net-28-ori")
    return model, None, 200


app = create_app()


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
