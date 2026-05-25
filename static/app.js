const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const processBtn = document.getElementById("processBtn");
const clearBtn = document.getElementById("clearBtn");
const exportBtn = document.getElementById("exportBtn");
const exportType = document.getElementById("exportType");
const fileNameLabel = document.getElementById("selectedFileName");
const pipelineNote = document.getElementById("pipelineNote");
const originalImage = document.getElementById("originalImage");
const processedImage = document.getElementById("processedImage");
const overlayImage = document.getElementById("overlayImage");
const overlayImageTop = document.getElementById("overlayImageTop");
const recognizedText = document.getElementById("recognizedText");
const confidenceValue = document.getElementById("confidenceValue");
const segmentsValue = document.getElementById("segmentsValue");
const statusList = document.getElementById("statusList");
const errorText = document.getElementById("errorText");
const pipelineFlow = document.getElementById("pipelineFlow");
const pipelineStageLabel = document.getElementById("pipelineStageLabel");

let selectedFile = null;
let latestResponse = null;
const orderedStages = ["preprocessing", "segmentation", "prediction", "mapping", "result"];

function normalizeStage(statusText) {
  const text = (statusText || "").toLowerCase();
  if (text.includes("preprocess")) {
    return "preprocessing";
  }
  if (text.includes("segment")) {
    return "segmentation";
  }
  if (text.includes("predict")) {
    return "prediction";
  }
  if (text.includes("map")) {
    return "mapping";
  }
  if (text.includes("result") || text.includes("recogniz")) {
    return "result";
  }
  return null;
}

function updatePipelineFlow(items, fallbackCurrent = null) {
  if (!pipelineFlow) {
    return;
  }

  const seen = new Set((items || []).map(normalizeStage).filter(Boolean));
  const lastSeen = [...seen].reduce((latest, stage) => {
    const idx = orderedStages.indexOf(stage);
    return idx > latest ? idx : latest;
  }, -1);

  const currentStage = fallbackCurrent || (lastSeen >= 0 && lastSeen < orderedStages.length - 1
    ? orderedStages[lastSeen + 1]
    : null);

  pipelineFlow.querySelectorAll(".pipeline-step").forEach((step) => {
    const stage = step.dataset.stage;
    const idx = orderedStages.indexOf(stage);
    step.classList.remove("done", "current");

    if (seen.has(stage)) {
      step.classList.add("done");
    } else if (currentStage && stage === currentStage) {
      step.classList.add("current");
    } else if (!currentStage && idx === 0 && seen.size === 0) {
      step.classList.add("current");
    }
  });

  if (pipelineStageLabel) {
    if (currentStage) {
      pipelineStageLabel.textContent = `Current: ${currentStage[0].toUpperCase()}${currentStage.slice(1)}`;
    } else if (seen.has("result")) {
      pipelineStageLabel.textContent = "Completed: Result generated";
    } else {
      pipelineStageLabel.textContent = "Waiting for image";
    }
  }
}

function setImage(imgEl, src) {
  if (!imgEl) {
    return;
  }
  if (!src) {
    imgEl.removeAttribute("src");
    imgEl.style.display = "none";
    return;
  }
  imgEl.src = src;
  imgEl.style.display = "block";
}

function setFile(file) {
  selectedFile = file;
  fileNameLabel.textContent = file ? `Selected: ${file.name}` : "";
  processBtn.textContent = file ? "Run Prediction" : "Choose Image";
  if (file) {
    const url = URL.createObjectURL(file);
    setImage(originalImage, url);
  } else {
    setImage(originalImage, null);
  }
}

function setStatus(items) {
  statusList.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    statusList.appendChild(li);
  });
  updatePipelineFlow(items);
}

function clearAll() {
  setFile(null);
  fileInput.value = "";
  latestResponse = null;
  if (pipelineNote) {
    pipelineNote.textContent = "";
  }
  recognizedText.textContent = "";
  confidenceValue.textContent = "-";
  segmentsValue.textContent = "-";
  errorText.textContent = "";
  setStatus([]);
  updatePipelineFlow([], "preprocessing");
  setImage(originalImage, null);
  setImage(processedImage, null);
  setImage(overlayImage, null);
  setImage(overlayImageTop, null);
}

async function processImage() {
  errorText.textContent = "";
  if (pipelineNote) {
    pipelineNote.textContent = "Processing image...";
  }
  updatePipelineFlow([], "preprocessing");
  const fallbackFile = fileInput.files && fileInput.files.length > 0 ? fileInput.files[0] : null;
  const currentFile = selectedFile || fallbackFile || null;
  if (!currentFile) {
    fileInput.click();
    return;
  }

  const formData = new FormData();
  formData.append("image", currentFile);

  processBtn.disabled = true;
  processBtn.textContent = "Processing...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const raw = await response.text();
    let payload = {};
    try {
      payload = raw ? JSON.parse(raw) : {};
    } catch {
      payload = { error: raw || "Server returned a non-JSON response." };
    }
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }

    latestResponse = payload;
    recognizedText.textContent = payload.recognized_text || "";
    confidenceValue.textContent = `${payload.confidence}%`;
    segmentsValue.textContent = `${payload.num_segments}`;
    setImage(originalImage, payload.original_image);
    setImage(processedImage, payload.preprocessed_image);
    setImage(overlayImage, payload.segmented_overlay_image);
    setImage(overlayImageTop, payload.segmented_overlay_image);
    setStatus(payload.pipeline_status);
    if (pipelineNote) {
      pipelineNote.textContent = "Pipeline completed.";
    }
    updatePipelineFlow(payload.pipeline_status);
    if (payload.warning) {
      errorText.textContent = payload.warning;
    }
  } catch (error) {
    errorText.textContent = error.message;
    if (pipelineNote) {
      pipelineNote.textContent = "Pipeline failed.";
    }
    updatePipelineFlow([], "preprocessing");
  } finally {
    processBtn.disabled = false;
    processBtn.textContent = "Run Prediction";
  }
}

function downloadText(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function exportResult() {
  if (!latestResponse) {
    errorText.textContent = "No result to export yet.";
    return;
  }
  const text = latestResponse.recognized_text || "";
  const confidence = latestResponse.confidence != null ? latestResponse.confidence : 0;
  const type = exportType.value;

  if (type === "txt") {
    downloadText("epigranet_result.txt", text, "text/plain;charset=utf-8");
    return;
  }
  if (type === "csv") {
    const escapedText = text.split('"').join('""');
    const csv = `recognized_text,confidence\n"${escapedText}",${confidence}\n`;
    downloadText("epigranet_result.csv", csv, "text/csv;charset=utf-8");
    return;
  }
  const json = JSON.stringify(latestResponse, null, 2);
  downloadText("epigranet_result.json", json, "application/json;charset=utf-8");
}

dropZone.addEventListener("click", (event) => {
  if (event.target.closest(".button-row")) {
    return;
  }
  fileInput.click();
});
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length > 0
    ? event.dataTransfer.files[0]
    : null;
  if (file) {
    setFile(file);
  }
});

fileInput.addEventListener("change", (event) => {
  const file = event.target && event.target.files && event.target.files.length > 0
    ? event.target.files[0]
    : null;
  if (file) {
    setFile(file);
    processImage();
  }
});

processBtn.addEventListener("click", processImage);
clearBtn.addEventListener("click", clearAll);
exportBtn.addEventListener("click", exportResult);

setFile(null);
updatePipelineFlow([], "preprocessing");
