"use strict";


let descriptionEl, charCountEl, classifyBtn, clearBtn, resultEl, categoryNameEl,
  confidenceValueEl, confidenceFillEl, top3BodyEl, replyBlockEl, copyBtn,
  lowConfidenceNoticeEl, errorMessageEl, formEl;

function getElements() {
  descriptionEl = document.getElementById("description");
  charCountEl = document.getElementById("char-count");
  classifyBtn = document.getElementById("classify-btn");
  clearBtn = document.getElementById("clear-btn");
  resultEl = document.getElementById("result");
  categoryNameEl = document.getElementById("category-name");
  confidenceValueEl = document.getElementById("confidence-value");
  confidenceFillEl = document.getElementById("confidence-fill");
  top3BodyEl = document.getElementById("top3-body");
  replyBlockEl = document.getElementById("reply-block");
  copyBtn = document.getElementById("copy-btn");
  lowConfidenceNoticeEl = document.getElementById("low-confidence-notice");
  errorMessageEl = document.getElementById("error-message");
  formEl = document.getElementById("ticket-form");
}


function updateCounter() {
  const length = descriptionEl.value.length;
  charCountEl.textContent = length + " / 1000";
}

function showError(message) {
  errorMessageEl.textContent = message;
  errorMessageEl.style.display = "block";
  resultEl.hidden = true;
}

function clearError() {
  errorMessageEl.textContent = "";
  errorMessageEl.style.display = "none";
}

function clearResult() {
  resultEl.hidden = true;
  categoryNameEl.textContent = "";
  confidenceValueEl.textContent = "0";
  confidenceFillEl.style.width = "0%";
  top3BodyEl.textContent = "";
  replyBlockEl.textContent = "";
  lowConfidenceNoticeEl.textContent = "";
  lowConfidenceNoticeEl.style.display = "none";
  clearError();
}

function renderResult(data) {
  categoryNameEl.textContent = data.category;
  confidenceValueEl.textContent = data.confidence;
  confidenceFillEl.style.width = data.confidence + "%";
  top3BodyEl.textContent = "";
  for (const item of data.top3) {
    const row = document.createElement("tr");
    const catCell = document.createElement("td");
    const probCell = document.createElement("td");
    catCell.textContent = item.category;
    probCell.textContent = item.probability + "%";
    row.appendChild(catCell);
    row.appendChild(probCell);
    top3BodyEl.appendChild(row);
  }

  replyBlockEl.textContent = data.suggested_reply;

  if (data.low_confidence) {
    lowConfidenceNoticeEl.textContent = "Low confidence. Please review this ticket manually.";
    lowConfidenceNoticeEl.style.display = "block";
  } else {
    lowConfidenceNoticeEl.style.display = "none";
  }

  resultEl.hidden = false;
}

function copyReply() {
  const text = replyBlockEl.textContent;
  if (!text) return;
  navigator.clipboard.writeText(text).then(
    function () {
      copyBtn.textContent = "Copied";
      setTimeout(function () {
        copyBtn.textContent = "Copy reply";
      }, 1500);
    },
    function () {
      alert("Could not copy to clipboard. Please copy the text manually.");
    }
  );
}

async function classifyTicket(event) {
  event.preventDefault();
  const text = descriptionEl.value.trim();

  if (text === "") {
    showError("Please enter a ticket description.");
    return;
  }

  clearError();
  classifyBtn.disabled = true;
  classifyBtn.textContent = "Analysing...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      showError(errorData.error || "Request failed. Please try again.");
      return;
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    showError("Prediction failed. Please try again.");
  } finally {
    classifyBtn.disabled = false;
    classifyBtn.textContent = "Classify ticket";
  }
}

function clearForm() {
  descriptionEl.value = "";
  updateCounter();
  clearError();
  clearResult();
}

function init() {
  getElements();

  descriptionEl.addEventListener("input", updateCounter);
  updateCounter();

  formEl.addEventListener("submit", classifyTicket);
  clearBtn.addEventListener("click", clearForm);
  copyBtn.addEventListener("click", copyReply);

  descriptionEl.addEventListener("keydown", function (event) {
    if (event.ctrlKey && event.key === "Enter") {
      event.preventDefault();
      formEl.requestSubmit();
    }
  });

  clearResult();
}

document.addEventListener("DOMContentLoaded", init);