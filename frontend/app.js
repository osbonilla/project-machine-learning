/**
 * app.js — Lógica del visor Geo-Intent Classifier
 *
 * Responsabilidades:
 *   1. Llamar a POST /predict con el texto ingresado
 *   2. Web Speech API: micrófono → texto → predict
 *   3. Renderizar el resultado en la UI
 */

const API_URL = "http://localhost:8000";

// ─── Referencias al DOM ──────────────────────────────────
const textInput     = document.getElementById("text-input");
const charCount     = document.getElementById("char-count");
const modelSelect   = document.getElementById("model-select");
const predictBtn    = document.getElementById("predict-btn");
const micBtn        = document.getElementById("mic-btn");
const micStatus     = document.getElementById("mic-status");
const resultSection = document.getElementById("result-section");
const errorBanner   = document.getElementById("error-banner");
const errorMessage  = document.getElementById("error-message");

// ─── Resultado ───────────────────────────────────────────
const resultIntent     = document.getElementById("result-intent");
const confidenceFill   = document.getElementById("confidence-fill");
const confidenceValue  = document.getElementById("confidence-value");
const resultAgent      = document.getElementById("result-agent");
const resultModel      = document.getElementById("result-model");
const resultText       = document.getElementById("result-text");

// ─────────────────────────────────────────────────────────
//  Contador de caracteres
// ─────────────────────────────────────────────────────────
textInput.addEventListener("input", () => {
  charCount.textContent = textInput.value.length;
});

// ─────────────────────────────────────────────────────────
//  API — POST /predict
// ─────────────────────────────────────────────────────────
async function classifyText(text) {
  const model = modelSelect.value;

  predictBtn.disabled = true;
  predictBtn.textContent = "Clasificando...";
  hideError();

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || `Error ${response.status}`);
    }

    const data = await response.json();
    renderResult(data);

  } catch (err) {
    showError(err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Clasificar →";
  }
}

// ─────────────────────────────────────────────────────────
//  Renderizar resultado
// ─────────────────────────────────────────────────────────
function renderResult(data) {
  const pct = Math.round(data.confidence * 100);

  resultIntent.textContent    = data.intent;
  confidenceFill.style.width  = `${pct}%`;
  confidenceValue.textContent = `${pct}%`;
  resultAgent.textContent     = data.agent;
  resultModel.textContent     = data.model;
  resultText.textContent      = `"${data.text}"`;

  // Color de la barra según confianza
  if (pct >= 80)      confidenceFill.style.background = "#34d399";  // verde
  else if (pct >= 55) confidenceFill.style.background = "#f59e0b";  // amarillo
  else                confidenceFill.style.background = "#f87171";  // rojo

  resultSection.classList.remove("hidden");
}

// ─────────────────────────────────────────────────────────
//  Botón Clasificar
// ─────────────────────────────────────────────────────────
predictBtn.addEventListener("click", () => {
  const text = textInput.value.trim();
  if (!text) {
    showError("Escribe o di una instrucción primero.");
    return;
  }
  classifyText(text);
});

// Enter en textarea también clasifica (Shift+Enter hace salto de línea)
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    predictBtn.click();
  }
});

// ─────────────────────────────────────────────────────────
//  Web Speech API — micrófono
// ─────────────────────────────────────────────────────────
const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

if (!SpeechRecognition) {
  // Browser no soporta Web Speech API (Firefox)
  micBtn.disabled = true;
  micBtn.title    = "Tu browser no soporta reconocimiento de voz. Usa Chrome o Edge.";
} else {
  const recognition = new SpeechRecognition();
  recognition.lang       = "es-ES";
  recognition.continuous = false;
  recognition.interimResults = false;

  let isListening = false;

  micBtn.addEventListener("click", () => {
    if (isListening) {
      recognition.stop();
      return;
    }
    recognition.start();
  });

  recognition.onstart = () => {
    isListening = true;
    micBtn.classList.add("recording");
    micBtn.querySelector(".mic-label").textContent = "Parar";
    micStatus.classList.remove("hidden");
    hideError();
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.classList.remove("recording");
    micBtn.querySelector(".mic-label").textContent = "Hablar";
    micStatus.classList.add("hidden");
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    textInput.value     = transcript;
    charCount.textContent = transcript.length;

    // Clasificar automáticamente tras recibir el audio
    classifyText(transcript);
  };

  recognition.onerror = (event) => {
    showError(`Error de micrófono: ${event.error}`);
    recognition.stop();
  };
}

// ─────────────────────────────────────────────────────────
//  Helpers — error
// ─────────────────────────────────────────────────────────
function showError(msg) {
  errorMessage.textContent = `⚠ ${msg}`;
  errorBanner.classList.remove("hidden");
}

function hideError() {
  errorBanner.classList.add("hidden");
}