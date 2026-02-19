import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import mysql.connector
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from web_model import (
    CLASS_NAMES,
    build_preprocessing_visuals,
    get_care_suggestions,
    get_predictor,
)


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
VIS_DIR = BASE_DIR / "static" / "visuals"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-secret")
    app.config["MYSQL_HOST"] = os.getenv("MYSQL_HOST", "127.0.0.1")
    app.config["MYSQL_PORT"] = int(os.getenv("MYSQL_PORT", "3306"))
    app.config["MYSQL_USER"] = os.getenv("MYSQL_USER", "root")
    app.config["MYSQL_PASSWORD"] = os.getenv("MYSQL_PASSWORD", "")
    app.config["MYSQL_DATABASE"] = os.getenv("MYSQL_DATABASE", "scar_system")
    app.config["MODEL_CHECKPOINT"] = os.getenv(
        "MODEL_CHECKPOINT",
        "outputs/20260217_135548_efficientnet_b0/single_split/best.pt",
    )

    register_routes(app)
    return app


def get_db(app: Flask):
    return mysql.connector.connect(
        host=app.config["MYSQL_HOST"],
        port=app.config["MYSQL_PORT"],
        user=app.config["MYSQL_USER"],
        password=app.config["MYSQL_PASSWORD"],
        database=app.config["MYSQL_DATABASE"],
        autocommit=True,
    )


def has_split_name_columns(app: Flask) -> bool:
    conn = get_db(app)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_schema = %s AND table_name = 'users'
          AND column_name IN ('first_name', 'last_name')
        """,
        (app.config["MYSQL_DATABASE"],),
    )
    count = int(cur.fetchone()[0])
    cur.close()
    conn.close()
    return count == 2


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def require_login():
    if "user_id" not in session:
        flash("Please login first.", "warning")
        return False
    return True


def save_uploaded_file(file_storage, target_dir: Path) -> Path:
    safe_name = secure_filename(file_storage.filename or "upload.jpg")
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError("Unsupported file extension.")
    unique_name = f"{uuid.uuid4().hex}{ext}"
    out_path = target_dir / unique_name
    file_storage.save(out_path)
    return out_path


def register_routes(app: Flask) -> None:
    @app.route("/")
    def landing():
        return render_template("landing.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            first_name = request.form.get("first_name", "").strip()
            last_name = request.form.get("last_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not first_name or not last_name or not email or not password:
                flash("All fields are required.", "danger")
                return render_template("register.html")
            if len(password) < 8:
                flash("Password must be at least 8 characters.", "danger")
                return render_template("register.html")

            password_hash = generate_password_hash(password)
            conn = get_db(app)
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            exists = cur.fetchone()
            if exists:
                cur.close()
                conn.close()
                flash("Email is already registered.", "danger")
                return render_template("register.html")

            if has_split_name_columns(app):
                cur.execute(
                    "INSERT INTO users (first_name, last_name, email, password_hash) VALUES (%s, %s, %s, %s)",
                    (first_name, last_name, email, password_hash),
                )
            else:
                full_name = f"{first_name} {last_name}".strip()
                cur.execute(
                    "INSERT INTO users (full_name, email, password_hash) VALUES (%s, %s, %s)",
                    (full_name, email, password_hash),
                )
            cur.close()
            conn.close()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            conn = get_db(app)
            cur = conn.cursor(dictionary=True)
            if has_split_name_columns(app):
                cur.execute(
                    "SELECT id, first_name, last_name, password_hash FROM users WHERE email = %s",
                    (email,),
                )
            else:
                cur.execute("SELECT id, full_name, password_hash FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            cur.close()
            conn.close()

            if not user or not check_password_hash(user["password_hash"], password):
                flash("Invalid email or password.", "danger")
                return render_template("login.html")

            session["user_id"] = user["id"]
            if "first_name" in user and "last_name" in user:
                session["user_name"] = f"{user['first_name']} {user['last_name']}".strip()
            else:
                session["user_name"] = user["full_name"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("analyze"))
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("landing"))

    @app.route("/analyze", methods=["GET", "POST"])
    def analyze():
        if not require_login():
            return redirect(url_for("login"))

        result = None
        if request.method == "POST":
            if "image" not in request.files:
                flash("Please upload an image.", "danger")
                return render_template("analyze.html", result=None)
            image_file = request.files["image"]
            if not image_file.filename:
                flash("Please choose an image file.", "danger")
                return render_template("analyze.html", result=None)
            if not is_allowed_file(image_file.filename):
                flash("Unsupported image format.", "danger")
                return render_template("analyze.html", result=None)

            try:
                saved_path = save_uploaded_file(image_file, UPLOAD_DIR)
                image_bgr = cv2.imread(str(saved_path))
                if image_bgr is None:
                    raise ValueError("Could not read uploaded image.")

                predictor = get_predictor(app.config["MODEL_CHECKPOINT"])
                pred = predictor.predict_bgr(image_bgr)
                suggestions = get_care_suggestions(pred["label"])

                result = {
                    "image_path": f"/static/uploads/{saved_path.name}",
                    "label": pred["label"],
                    "probabilities": pred["probabilities"],
                    "suggestions": suggestions,
                }

                conn = get_db(app)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO analyses (
                      user_id, image_path, predicted_label,
                      prob_hypertrophic, prob_keloid, prob_atrophic, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session["user_id"],
                        str(saved_path),
                        pred["label"],
                        float(pred["probabilities"]["hypertrophic"]),
                        float(pred["probabilities"]["keloid"]),
                        float(pred["probabilities"]["atrophic"]),
                        datetime.utcnow(),
                    ),
                )
                cur.close()
                conn.close()
            except Exception as exc:
                flash(f"Analysis failed: {exc}", "danger")
                return render_template("analyze.html", result=None)

        return render_template("analyze.html", result=result)

    @app.route("/visualizer", methods=["GET", "POST"])
    def visualizer():
        if not require_login():
            return redirect(url_for("login"))

        visuals = None
        if request.method == "POST":
            if "image" not in request.files:
                flash("Please upload an image.", "danger")
                return render_template("visualizer.html", visuals=None)
            image_file = request.files["image"]
            if not image_file.filename:
                flash("Please choose an image file.", "danger")
                return render_template("visualizer.html", visuals=None)
            if not is_allowed_file(image_file.filename):
                flash("Unsupported image format.", "danger")
                return render_template("visualizer.html", visuals=None)
            try:
                saved_path = save_uploaded_file(image_file, VIS_DIR)
                image_bgr = cv2.imread(str(saved_path))
                if image_bgr is None:
                    raise ValueError("Could not read uploaded image.")
                predictor = get_predictor(app.config["MODEL_CHECKPOINT"])
                stage_paths = build_preprocessing_visuals(
                    image_bgr=image_bgr,
                    output_dir=VIS_DIR,
                    guided_radius=predictor.guided_radius,
                    guided_eps=predictor.guided_eps,
                )
                visuals = {k: f"/static/visuals/{v.name}" for k, v in stage_paths.items()}
            except Exception as exc:
                flash(f"Visualizer failed: {exc}", "danger")
                return render_template("visualizer.html", visuals=None)

        return render_template("visualizer.html", visuals=visuals)


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
