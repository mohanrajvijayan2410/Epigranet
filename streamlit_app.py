import hashlib
import html
import json
import mimetypes
import os
from textwrap import dedent
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from pipeline import OCRPredictor, ensure_dir, preprocess_image, segment_characters


BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ensure_dir(Path(os.environ.get("EPIGRANET_GENERATED_DIR", BASE_DIR / "runtime_generated")))
UPLOAD_DIR = ensure_dir(GENERATED_DIR / "uploads")
PREPROCESSED_DIR = ensure_dir(GENERATED_DIR / "preprocessed")
SEGMENT_DIR = ensure_dir(GENERATED_DIR / "segments")
BOXED_DIR = ensure_dir(GENERATED_DIR / "boxed")

DEFAULT_MODEL_PATH = BASE_DIR / "models" / "epigranet_embedding_model (1).pt"
DEFAULT_EMBEDDINGS_PATH = BASE_DIR / "models" / "reference_embeddings.pt"
DEFAULT_CLASS_MAPPING_PATH = BASE_DIR / "class_mapping_209 (1).json"
DEFAULT_DATASET_PATH = BASE_DIR / "aug_dataset"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_FILE_SIZE_MB = 20
ORDERED_STAGES = ["preprocessing", "segmentation", "prediction", "mapping", "result"]


def create_run_id() -> str:
    return uuid.uuid4().hex[:12]


def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_asset_settings() -> Dict[str, Optional[str]]:
    return {
        "hf_repo_id": os.environ.get("EPIGRANET_HF_REPO_ID"),
        "hf_revision": os.environ.get("EPIGRANET_HF_REVISION", "main"),
        "hf_token": os.environ.get("EPIGRANET_HF_TOKEN"),
        "hf_model_filename": os.environ.get("EPIGRANET_HF_MODEL_FILENAME", DEFAULT_MODEL_PATH.name),
        "hf_embeddings_filename": os.environ.get(
            "EPIGRANET_HF_EMBEDDINGS_FILENAME", DEFAULT_EMBEDDINGS_PATH.name
        ),
        "hf_class_mapping_filename": os.environ.get(
            "EPIGRANET_HF_CLASS_MAPPING_FILENAME", DEFAULT_CLASS_MAPPING_PATH.name
        ),
        "model_path": os.environ.get("EPIGRANET_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        "embeddings_path": os.environ.get("EPIGRANET_EMBEDDINGS_PATH", str(DEFAULT_EMBEDDINGS_PATH)),
        "class_mapping_path": os.environ.get("EPIGRANET_CLASS_MAPPING_PATH", str(DEFAULT_CLASS_MAPPING_PATH)),
        "dataset_path": os.environ.get("EPIGRANET_DATASET_PATH", str(DEFAULT_DATASET_PATH)),
    }


def normalize_stage(status_text: str) -> Optional[str]:
    text = (status_text or "").lower()
    if "preprocess" in text:
        return "preprocessing"
    if "segment" in text:
        return "segmentation"
    if "predict" in text:
        return "prediction"
    if "map" in text:
        return "mapping"
    if "result" in text or "recogniz" in text:
        return "result"
    return None


@st.cache_resource(show_spinner=False)
def resolve_runtime_assets(settings: Dict[str, Optional[str]]) -> Tuple[Path, Path, Path, Optional[Path]]:
    hf_repo_id = settings["hf_repo_id"]
    if hf_repo_id:
        from huggingface_hub import hf_hub_download

        model_path = Path(
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=str(settings["hf_model_filename"]),
                revision=str(settings["hf_revision"]),
                token=settings["hf_token"],
            )
        )
        embeddings_path = Path(
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=str(settings["hf_embeddings_filename"]),
                revision=str(settings["hf_revision"]),
                token=settings["hf_token"],
            )
        )
        class_mapping_path = Path(
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=str(settings["hf_class_mapping_filename"]),
                revision=str(settings["hf_revision"]),
                token=settings["hf_token"],
            )
        )
    else:
        model_path = Path(str(settings["model_path"]))
        embeddings_path = Path(str(settings["embeddings_path"]))
        class_mapping_path = Path(str(settings["class_mapping_path"]))

    dataset_candidate = Path(str(settings["dataset_path"]))
    dataset_path = dataset_candidate if dataset_candidate.exists() else None
    return model_path, embeddings_path, class_mapping_path, dataset_path


@st.cache_resource(show_spinner=False)
def load_predictor(settings: Dict[str, Optional[str]]) -> OCRPredictor:
    model_path, embeddings_path, class_mapping_path, dataset_path = resolve_runtime_assets(settings)
    return OCRPredictor(
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        embedding_cache_path=embeddings_path if embeddings_path.exists() else None,
        dataset_path=dataset_path,
    )


def run_ocr(uploaded_file) -> Dict[str, object]:
    filename = uploaded_file.name or "uploaded_image.png"
    if not is_allowed_file(filename):
        raise ValueError("Unsupported file format.")

    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.2f} MB. Limit is {MAX_FILE_SIZE_MB} MB.")

    run_id = create_run_id()
    ext = filename.rsplit(".", 1)[1].lower()
    safe_stem = Path(filename).stem.replace(" ", "_")

    upload_path = UPLOAD_DIR / f"{run_id}_{safe_stem}.{ext}"
    preprocessed_path = PREPROCESSED_DIR / f"{run_id}_preprocessed.png"
    boxed_path = BOXED_DIR / f"{run_id}_boxed.png"
    roi_dir = ensure_dir(SEGMENT_DIR / run_id)

    upload_path.write_bytes(file_bytes)

    pipeline_status = ["Image uploaded successfully."]
    warning = None

    preprocess_image(upload_path, preprocessed_path)
    pipeline_status.append("Preprocessing completed.")

    try:
        roi_paths = segment_characters(preprocessed_path, roi_dir, boxed_path)
        pipeline_status.append("Segmentation completed.")
    except Exception as exc:
        roi_paths = []
        warning = f"Segmentation skipped: {exc}"
        pipeline_status.append(warning)

    predictor = load_predictor(get_asset_settings())
    result = predictor.predict_text(roi_paths, preprocessed_path)
    segments_used = len(roi_paths) if roi_paths else len(result.tokens)
    pipeline_status.append("OCR prediction completed.")

    return {
        "run_id": run_id,
        "recognized_text": result.text,
        "confidence": round(result.confidence * 100, 2),
        "num_segments": segments_used,
        "token_predictions": result.tokens,
        "original_image": upload_path,
        "preprocessed_image": preprocessed_path,
        "segmented_overlay_image": boxed_path if boxed_path.exists() else preprocessed_path,
        "pipeline_status": pipeline_status,
        "warning": warning,
    }


def inject_theme() -> None:
    st.markdown(
        dedent(
            f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Noto+Sans+Tamil:wght@400;700&display=swap');

        :root {{
          --bg: #f3f4f9;
          --panel: #ffffff;
          --ink: #1b1f2a;
          --muted: #586178;
          --accent: #41766f;
          --accent-strong: #2f5d57;
          --line: #dbe0ea;
          --danger: #be2f4d;
        }}

        * {{
          box-sizing: border-box;
        }}

        html, body, [class*="css"] {{
          font-family: "Manrope", sans-serif;
        }}

        [data-testid="stAppViewContainer"] {{
          background:
            radial-gradient(circle at top left, rgba(119, 140, 255, 0.14), transparent 34%),
            radial-gradient(circle at top right, rgba(29, 99, 135, 0.12), transparent 38%),
            var(--bg);
        }}

        [data-testid="stHeader"] {{
          background: transparent;
        }}

        #MainMenu, footer {{
          visibility: hidden;
        }}

        .block-container {{
          max-width: 1320px;
          margin: 28px auto;
          padding: 0 16px 28px;
        }}

        .topbar {{
          background: linear-gradient(120deg, #454d67, #2f3852);
          color: #fff;
          border-radius: 14px 14px 0 0;
          padding: 20px 28px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }}

        .brand {{
          display: flex;
          align-items: center;
          gap: 12px;
        }}

        .brand h1 {{
          margin: 0;
          font-size: 34px;
          font-weight: 800;
        }}

        .brand-icon {{
          width: 38px;
          height: 38px;
          border-radius: 10px;
          display: grid;
          place-items: center;
          background: rgba(255, 255, 255, 0.2);
        }}

        .actions {{
          font-size: 20px;
          opacity: 0.95;
        }}

        .upload-box,
        .preview-grid,
        .results {{
          background: var(--panel);
          border: 1px solid var(--line);
        }}

        .selected-file {{
          background: var(--panel);
          border-left: 1px solid var(--line);
          border-right: 1px solid var(--line);
          margin: 0;
          padding: 12px 22px 0;
          color: var(--muted);
          font-size: 18px;
        }}

        div[data-testid="stFileUploader"] {{
          background: var(--panel);
          border-top: 1px solid var(--line);
          border-left: 1px solid var(--line);
          border-right: 1px solid var(--line);
          padding: 20px 20px 0;
        }}

        div[data-testid="stFileUploader"] > label {{
          display: none;
        }}

        section[data-testid="stFileUploaderDropzone"] {{
          border: 2px dashed #c7cfde;
          border-radius: 12px;
          padding: 18px;
          min-height: 116px;
          background: linear-gradient(180deg, #f8faff, #f1f5fb);
          display: flex;
          align-items: center;
          cursor: pointer;
          position: relative;
        }}

        section[data-testid="stFileUploaderDropzone"]:hover {{
          border-color: var(--accent);
          background: #edf7f5;
        }}

        section[data-testid="stFileUploaderDropzone"] button {{
          display: none;
        }}

        div[data-testid="stFileUploaderDropzoneInstructions"] {{
          display: none;
        }}

        section[data-testid="stFileUploaderDropzone"]::before {{
          content: "Drag & Drop or Click to Upload Inscription Image";
          display: block;
          font-size: 28px;
          color: var(--ink);
          margin-bottom: 8px;
        }}

        section[data-testid="stFileUploaderDropzone"]::after {{
          content: "(Max file size: {MAX_FILE_SIZE_MB} MB)";
          display: block;
          position: absolute;
          left: 18px;
          bottom: 18px;
          color: var(--muted);
          font-size: 18px;
        }}

        div[data-testid="stHorizontalBlock"] {{
          gap: 12px;
          background: var(--panel);
          border-left: 1px solid var(--line);
          border-right: 1px solid var(--line);
          border-bottom: 1px solid var(--line);
          padding: 0 20px 20px;
        }}

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {{
          border: 1px solid transparent;
          border-radius: 10px;
          padding: 10px 18px;
          font: inherit;
          font-size: 20px;
          cursor: pointer;
          width: 100%;
          min-height: 48px;
        }}

        div[data-testid="stButton"] > button[kind="primary"] {{
          background: var(--accent);
          color: #fff;
        }}

        div[data-testid="stButton"] > button[kind="primary"]:hover {{
          background: var(--accent-strong);
          border-color: var(--accent-strong);
          color: #fff;
        }}

        div[data-testid="stButton"] > button[kind="secondary"] {{
          background: #fff;
          border-color: #bcc7de;
          color: var(--ink);
        }}

        .pipeline-stage {{
          margin-top: 20px;
          padding: 20px;
          border-radius: 12px;
          border: 1px solid #cad6ee;
          background: linear-gradient(160deg, #f6f9ff, #ecf2ff);
          box-shadow: 0 10px 20px rgba(43, 70, 130, 0.08);
        }}

        .pipeline-head {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          margin-bottom: 14px;
        }}

        .pipeline-stage-label {{
          margin: 0;
          color: #39537b;
          font-size: 17px;
          font-weight: 700;
        }}

        .pipeline-flow {{
          margin: 0;
          padding: 0;
          list-style: none;
          display: grid;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          gap: 14px;
        }}

        .pipeline-step {{
          position: relative;
          min-height: 88px;
          border: 1px solid #c3cee5;
          border-radius: 12px;
          background: #fff;
          display: grid;
          justify-items: center;
          align-content: center;
          gap: 8px;
          padding: 10px 8px;
          text-align: center;
          transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }}

        .pipeline-step::after {{
          content: ">";
          position: absolute;
          right: -12px;
          top: 50%;
          transform: translateY(-50%);
          color: #8092b9;
          font-size: 22px;
          font-weight: 800;
        }}

        .pipeline-step:last-child::after {{
          display: none;
        }}

        .step-index {{
          width: 34px;
          height: 34px;
          border-radius: 50%;
          display: grid;
          place-items: center;
          font-weight: 800;
          color: #4f607f;
          background: #e8eefb;
        }}

        .step-label {{
          font-size: 18px;
          font-weight: 700;
          color: #2a3550;
        }}

        .pipeline-step.current {{
          border-color: #3d6ca8;
          box-shadow: 0 8px 18px rgba(46, 103, 173, 0.25);
          transform: translateY(-2px);
        }}

        .pipeline-step.current .step-index {{
          background: #3d6ca8;
          color: #fff;
        }}

        .pipeline-step.done {{
          border-color: #4b8a69;
          background: #eff9f4;
        }}

        .pipeline-step.done .step-index {{
          background: #4b8a69;
          color: #fff;
        }}

        .preview-grid {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 20px;
          padding: 20px;
        }}

        .card {{
          border: 1px solid var(--line);
          border-radius: 10px;
          padding: 16px;
          background: #fff;
        }}

        h2, h3 {{
          margin: 0 0 12px;
        }}

        .image-wrap {{
          border: 1px solid var(--line);
          border-radius: 8px;
          min-height: 220px;
          background: #f6f8fc;
          overflow: hidden;
          display: grid;
          place-items: center;
        }}

        .image-wrap img {{
          width: 100%;
          display: block;
          object-fit: contain;
          max-height: 420px;
        }}

        .image-wrap.empty::before {{
          content: "Waiting for image";
          color: var(--muted);
          font-size: 18px;
        }}

        @media (max-width: 980px) {{
          .topbar {{
            border-radius: 14px;
            flex-direction: column;
            align-items: flex-start;
            gap: 14px;
          }}

          .brand h1 {{
            font-size: 24px;
          }}

          section[data-testid="stFileUploaderDropzone"]::before {{
            font-size: 22px;
            max-width: 85%;
          }}

          .preview-grid {{
            grid-template-columns: 1fr;
          }}

          .pipeline-head {{
            flex-direction: column;
            align-items: flex-start;
          }}

          .pipeline-flow {{
            grid-template-columns: 1fr;
          }}

          .pipeline-step::after {{
            content: "v";
            right: 50%;
            top: auto;
            bottom: -14px;
            transform: translateX(50%);
          }}
        }}
        </style>
        """
        ),
        unsafe_allow_html=True,
    )


def path_to_data_uri(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/png"
    data = path.read_bytes()
    import base64

    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def image_to_data_uri(image: Image.Image) -> str:
    import base64

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def get_uploaded_file_signature(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha1(file_bytes).hexdigest()
    return f"{uploaded_file.name}:{len(file_bytes)}:{digest}"


def build_pipeline_markup(status_items: List[str], fallback_current: Optional[str] = None) -> str:
    seen = []
    for item in status_items:
        stage = normalize_stage(item)
        if stage and stage not in seen:
            seen.append(stage)

    last_seen = max((ORDERED_STAGES.index(stage) for stage in seen), default=-1)
    if fallback_current:
        current_stage = fallback_current
    elif last_seen >= 0 and last_seen < len(ORDERED_STAGES) - 1:
        current_stage = ORDERED_STAGES[last_seen + 1]
    else:
        current_stage = None

    if current_stage:
        label = f"Current: {current_stage.capitalize()}"
    elif "result" in seen:
        label = "Completed: Result generated"
    else:
        label = "Waiting for image"

    items = []
    for index, stage in enumerate(ORDERED_STAGES, start=1):
        classes = ["pipeline-step"]
        if stage in seen:
            classes.append("done")
        elif current_stage == stage:
            classes.append("current")
        items.append(
            f'<li class="{" ".join(classes)}" data-stage="{html.escape(stage)}">'
            f'<span class="step-index">{index}</span>'
            f'<span class="step-label">{html.escape(stage.capitalize())}</span>'
            "</li>"
        )

    return dedent(
        f"""
        <section class="pipeline-stage card">
          <div class="pipeline-head">
            <h2>Layer-wise Processing</h2>
            <p class="pipeline-stage-label">{html.escape(label)}</p>
          </div>
          <ol class="pipeline-flow">{''.join(items)}</ol>
        </section>
        """
    ).strip()


def render_preview_grid(original_uri: Optional[str], preprocessed_uri: Optional[str], overlay_uri: Optional[str]) -> None:
    preview_specs = [
        ("Original Inscription Image", original_uri, "Original image preview"),
        ("Preprocessed Inscription Image", preprocessed_uri, "Preprocessed image preview"),
        ("Segmentation Overlay", overlay_uri, "Segmentation overlay preview"),
    ]
    cards = []
    for title, image_uri, alt_text in preview_specs:
        image_markup = f'<img src="{image_uri}" alt="{html.escape(alt_text)}">' if image_uri else ""
        empty_class = " empty" if image_uri is None else ""
        cards.append(
            dedent(
                f"""
                <article class="card">
                  <h2>{html.escape(title)}</h2>
                  <div class="image-wrap{empty_class}">{image_markup}</div>
                </article>
                """
            ).strip()
        )

    st.markdown(f'<section class="preview-grid">{"".join(cards)}</section>', unsafe_allow_html=True)


def build_export_payload(result: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if result is None:
        return None
    return {
        "run_id": result["run_id"],
        "recognized_text": result["recognized_text"],
        "confidence": result["confidence"],
        "num_segments": result["num_segments"],
        "original_image": str(result["original_image"]),
        "preprocessed_image": str(result["preprocessed_image"]),
        "segmented_overlay_image": str(result["segmented_overlay_image"]),
        "token_predictions": result["token_predictions"],
        "pipeline_status": result["pipeline_status"],
        "warning": result["warning"],
    }


def render_results_section(result: Optional[Dict[str, object]], error_text: str, pipeline_status: List[str]) -> None:
    recognized_text = html.escape(str(result["recognized_text"])) if result else ""
    confidence = f"{float(result['confidence']):.2f}%" if result else "-"
    segments = str(result["num_segments"]) if result else "-"
    status_markup = "".join(f"<li>{html.escape(item)}</li>" for item in pipeline_status)
    warning = html.escape(str(result.get("warning", ""))) if result and result.get("warning") else ""
    overlay_uri = path_to_data_uri(result["segmented_overlay_image"]) if result else None
    overlay_markup = f'<img src="{overlay_uri}" alt="Segmented overlay preview">' if overlay_uri else ""
    overlay_empty_class = " empty" if overlay_uri is None else ""
    final_error = html.escape(error_text or warning)
    export_payload = json.dumps(build_export_payload(result), ensure_ascii=False)

    component_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Noto+Sans+Tamil:wght@400;700&display=swap" rel="stylesheet">
      <style>
        :root {{
          --panel: #ffffff;
          --ink: #1b1f2a;
          --line: #dbe0ea;
          --danger: #be2f4d;
        }}

        * {{
          box-sizing: border-box;
        }}

        body {{
          margin: 0;
          font-family: "Manrope", sans-serif;
          color: var(--ink);
          background: transparent;
        }}

        .card {{
          border: 1px solid var(--line);
          border-radius: 10px;
          padding: 16px;
          background: #fff;
        }}

        .results {{
          border-radius: 0 0 14px 14px;
          padding: 20px;
          background: var(--panel);
        }}

        .result-head {{
          display: flex;
          gap: 10px;
          align-items: center;
        }}

        .result-head h2 {{
          margin: 0 0 12px;
          margin-right: auto;
        }}

        .select {{
          padding: 8px 10px;
          border: 1px solid #bdc8df;
          border-radius: 8px;
          font: inherit;
        }}

        .btn {{
          border: 1px solid transparent;
          border-radius: 10px;
          padding: 10px 18px;
          font: inherit;
          font-size: 20px;
          cursor: pointer;
        }}

        .btn.secondary {{
          background: #3d6ca8;
          color: #fff;
        }}

        .recognized {{
          margin-top: 14px;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: #f8fafc;
          min-height: 72px;
          padding: 12px 16px;
          font-size: 44px;
          font-weight: 700;
          line-height: 1.35;
          max-width: 100%;
          white-space: normal;
          overflow-wrap: anywhere;
          word-break: break-word;
          hyphens: auto;
        }}

        .tamil {{
          font-family: "Noto Sans Tamil", "Manrope", sans-serif;
        }}

        .metrics {{
          display: flex;
          gap: 26px;
          color: #313747;
          font-size: 20px;
        }}

        h3 {{
          margin: 18px 0 12px;
        }}

        .image-wrap {{
          border: 1px solid var(--line);
          border-radius: 8px;
          min-height: 220px;
          background: #f6f8fc;
          overflow: hidden;
          display: grid;
          place-items: center;
        }}

        .image-wrap img {{
          width: 100%;
          display: block;
          object-fit: contain;
          max-height: 420px;
        }}

        .image-wrap.empty::before {{
          content: "Waiting for image";
          color: #586178;
          font-size: 18px;
        }}

        .status-list {{
          margin: 8px 0 0;
          padding-left: 18px;
          color: #2f6a59;
          font-size: 18px;
        }}

        .error-text {{
          color: var(--danger);
          font-weight: 700;
          margin-top: 12px;
        }}

        @media (max-width: 980px) {{
          .recognized {{
            font-size: 30px;
          }}
        }}
      </style>
    </head>
    <body>
      <section class="results card">
        <div class="result-head">
          <h2>Recognized Text</h2>
          <select id="exportType" class="select">
            <option value="txt">Export TXT</option>
            <option value="csv">Export CSV</option>
            <option value="json">Export JSON</option>
          </select>
          <button id="exportBtn" class="btn secondary" type="button">Export</button>
        </div>
        <div id="recognizedText" class="recognized tamil">{recognized_text}</div>
        <div class="metrics">
          <p><strong>Confidence:</strong> <span id="confidenceValue">{confidence}</span></p>
          <p><strong>Segments:</strong> <span id="segmentsValue">{segments}</span></p>
        </div>
        <h3>Segmented Overlay</h3>
        <div class="image-wrap{overlay_empty_class}">{overlay_markup}</div>
        <h3>Pipeline Log</h3>
        <ul id="statusList" class="status-list">{status_markup}</ul>
        <p id="errorText" class="error-text">{final_error}</p>
      </section>
      <script>
        const latestResponse = {export_payload};
        const exportBtn = document.getElementById("exportBtn");
        const exportType = document.getElementById("exportType");
        const errorText = document.getElementById("errorText");

        function downloadText(filename, content, mimeType) {{
          const blob = new Blob([content], {{ type: mimeType }});
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          a.click();
          URL.revokeObjectURL(url);
        }}

        function exportResult() {{
          if (!latestResponse) {{
            errorText.textContent = "No result to export yet.";
            return;
          }}

          const text = latestResponse.recognized_text || "";
          const confidence = latestResponse.confidence != null ? latestResponse.confidence : 0;
          const type = exportType.value;

          if (type === "txt") {{
            downloadText("epigranet_result.txt", text, "text/plain;charset=utf-8");
            return;
          }}

          if (type === "csv") {{
            const escapedText = text.split('"').join('""');
            const csv = `recognized_text,confidence\\n"${{escapedText}}",${{confidence}}\\n`;
            downloadText("epigranet_result.csv", csv, "text/csv;charset=utf-8");
            return;
          }}

          const json = JSON.stringify(latestResponse, null, 2);
          downloadText("epigranet_result.json", json, "application/json;charset=utf-8");
        }}

        exportBtn.addEventListener("click", exportResult);
      </script>
    </body>
    </html>
    """

    component_height = 620 + (28 * max(1, len(pipeline_status)))
    components.html(component_html, height=component_height, scrolling=False)


def reset_state() -> None:
    st.session_state["ocr_result"] = None
    st.session_state["pipeline_status"] = []
    st.session_state["pipeline_note"] = ""
    st.session_state["error_text"] = ""
    st.session_state["last_uploaded_signature"] = None
    st.session_state["last_processed_signature"] = None
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def main() -> None:
    st.set_page_config(page_title="EpigraNet Tamil OCR", layout="wide")
    inject_theme()

    st.session_state.setdefault("ocr_result", None)
    st.session_state.setdefault("pipeline_status", [])
    st.session_state.setdefault("pipeline_note", "")
    st.session_state.setdefault("error_text", "")
    st.session_state.setdefault("last_uploaded_signature", None)
    st.session_state.setdefault("last_processed_signature", None)
    st.session_state.setdefault("uploader_key", 0)

    settings = get_asset_settings()

    st.markdown(
        """
        <header class="topbar">
          <div class="brand">
            <div class="brand-icon">?</div>
            <h1>EpigraNet-Tamil - Tamil Epigraphical OCR</h1>
          </div>
          <div class="actions">History</div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    try:
        load_predictor(settings)
    except Exception as exc:
        st.error(f"Unable to initialize OCR predictor: {exc}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an inscription image",
        type=sorted(ALLOWED_EXTENSIONS),
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state['uploader_key']}",
    )

    current_signature = get_uploaded_file_signature(uploaded_file)
    new_upload_detected = current_signature is not None and current_signature != st.session_state["last_uploaded_signature"]
    if new_upload_detected:
        st.session_state["last_uploaded_signature"] = current_signature
        st.session_state["last_processed_signature"] = None
        st.session_state["ocr_result"] = None
        st.session_state["pipeline_status"] = []
        st.session_state["pipeline_note"] = ""
        st.session_state["error_text"] = ""

    selected_file_name = uploaded_file.name if uploaded_file is not None else ""
    if selected_file_name:
        st.markdown(f'<p class="selected-file">Selected: {html.escape(selected_file_name)}</p>', unsafe_allow_html=True)
    if st.session_state["pipeline_note"]:
        st.markdown(
            f'<p class="selected-file">{html.escape(st.session_state["pipeline_note"])}</p>',
            unsafe_allow_html=True,
        )

    action_col1, action_col2 = st.columns(2)
    primary_label = "Upload Image" if uploaded_file is None else "Run Prediction"
    run_clicked = action_col1.button(primary_label, type="primary", use_container_width=True)
    clear_clicked = action_col2.button("Clear", use_container_width=True)

    if clear_clicked:
        reset_state()
        st.rerun()

    should_process = uploaded_file is not None and (
        new_upload_detected or run_clicked or current_signature != st.session_state["last_processed_signature"]
    )

    if run_clicked and uploaded_file is None:
        st.session_state["error_text"] = "Please select an image."
        st.session_state["pipeline_note"] = "Pipeline failed."
        st.session_state["pipeline_status"] = []

    if should_process:
        try:
            st.session_state["error_text"] = ""
            st.session_state["pipeline_note"] = "Processing image..."
            with st.spinner("Running preprocessing, segmentation, and OCR..."):
                result = run_ocr(uploaded_file)
            st.session_state["ocr_result"] = result
            st.session_state["pipeline_status"] = result["pipeline_status"]
            st.session_state["pipeline_note"] = "Pipeline completed."
            st.session_state["last_processed_signature"] = current_signature
        except Exception as exc:
            st.session_state["ocr_result"] = None
            st.session_state["pipeline_status"] = []
            st.session_state["pipeline_note"] = "Pipeline failed."
            st.session_state["error_text"] = str(exc)

    result = st.session_state["ocr_result"]
    fallback_current = "preprocessing" if result is None else None
    st.markdown(
        build_pipeline_markup(st.session_state["pipeline_status"], fallback_current=fallback_current),
        unsafe_allow_html=True,
    )

    original_uri = None
    if uploaded_file is not None:
        with Image.open(BytesIO(uploaded_file.getvalue())) as original_image:
            original_uri = image_to_data_uri(original_image)

    preprocessed_uri = path_to_data_uri(result["preprocessed_image"]) if result else None
    overlay_uri = path_to_data_uri(result["segmented_overlay_image"]) if result else None

    render_preview_grid(original_uri, preprocessed_uri, overlay_uri)
    render_results_section(result, st.session_state["error_text"], st.session_state["pipeline_status"])


if __name__ == "__main__":
    main()
