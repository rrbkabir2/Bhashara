/**
 * Bhashara Local Video Dubbing Studio (app.js)
 * Coordinates UI tabs, file/URL ingestion, real-time stage progress polling,
 * cancellation, media downloads, script editor, and disk management.
 */

// State
let activeJobId = null;
let pollInterval = null;
let currentJobData = null;
let inspectedUrlInfo = null;
let changedSegments = new Set();

// Stage Definitions for initial rendering
const INITIAL_STAGES = [
  { num: 1, name: "Input Demux", model: "FFmpeg 9.0.2" },
  { num: 2, name: "Stem Separation", model: "Demucs v4 (htdemucs)" },
  { num: 3, name: "Transcription", model: "faster-whisper base" },
  { num: 4, name: "Diarization", model: "Acoustic F0 Clustering" },
  { num: 5, name: "Translation", model: "MarianMT Helsinki-NLP" },
  { num: 6, name: "Voice Synthesis", model: "Edge-TTS Neural" },
  { num: 7, name: "Timing Fit", model: "WSOLA SolaStretch" },
  { num: 8, name: "Ducked Mixer", model: "Sidechain Ducking -16LUFS" },
  { num: 9, name: "Video Muxing", model: "FFmpeg Muxer" },
  { num: 10, name: "Subtitles & Output", model: "Bhashara Subtitle Engine" }
];

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initInputToggles();
  initDropZone();
  initStagesUI();
  initLogToggle();
  initEventHandlers();
  refreshHistory();
});

// --- Tabs Management ---
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.remove("hidden");
        targetContent.classList.add("active");
      }
      
      if (targetId === "tab-history") {
        refreshHistory();
      }
    });
  });
}

function switchTab(tabId) {
  const btn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  if (btn) btn.click();
}

// --- Input Mode Toggles ---
function initInputToggles() {
  const btnFile = document.getElementById("btn-mode-file");
  const btnUrl = document.getElementById("btn-mode-url");
  const secFile = document.getElementById("section-file");
  const secUrl = document.getElementById("section-url");

  btnFile.addEventListener("click", () => {
    btnFile.classList.add("active");
    btnUrl.classList.remove("active");
    secFile.classList.remove("hidden");
    secUrl.classList.add("hidden");
    clearErrors();
  });

  btnUrl.addEventListener("click", () => {
    btnUrl.classList.add("active");
    btnFile.classList.remove("active");
    secUrl.classList.remove("hidden");
    secFile.classList.add("hidden");
    clearErrors();
  });
}

// --- Drag & Drop Zone ---
function initDropZone() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const fileNameDisplay = document.getElementById("file-name-display");

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`;
      clearErrors();
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--border-focus)";
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.style.borderColor = "var(--border)";
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--border)";
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      const file = e.dataTransfer.files[0];
      fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`;
      clearErrors();
    }
  });
}

// --- Initial Stages Render ---
function initStagesUI() {
  const container = document.getElementById("stages-list");
  container.innerHTML = "";
  INITIAL_STAGES.forEach(s => {
    const el = document.createElement("div");
    el.className = "stage-item";
    el.id = `stage-row-${s.num}`;
    el.innerHTML = `
      <div>
        <div class="stage-name">${s.num}. ${s.name}</div>
        <div class="stage-model">${s.model}</div>
      </div>
      <div style="display: flex; align-items: center; gap: 0.6rem;">
        <span class="stage-duration" id="stage-dur-${s.num}">--</span>
        <span class="stage-badge badge-waiting" id="stage-badge-${s.num}">waiting</span>
      </div>
    `;
    container.appendChild(el);
  });
}

// --- Log Drawer ---
function initLogToggle() {
  const toggleBtn = document.getElementById("log-toggle-btn");
  const body = document.getElementById("log-console-body");
  toggleBtn.addEventListener("click", () => {
    const isHidden = body.style.display === "none";
    body.style.display = isHidden ? "block" : "none";
    toggleBtn.querySelector(".toggle-icon").textContent = isHidden ? "▼" : "▲";
  });
}

// --- Event Handlers & API Calls ---
function initEventHandlers() {
  // Inspect URL Info
  document.getElementById("btn-fetch-info").addEventListener("click", async () => {
    const urlInput = document.getElementById("url-input");
    const url = urlInput.value.trim();
    if (!url) {
      showError("Please enter a YouTube or video URL.");
      return;
    }

    clearErrors();
    const btn = document.getElementById("btn-fetch-info");
    btn.disabled = true;
    btn.textContent = "Inspecting...";

    try {
      const resp = await fetch("/api/info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });
      const data = await resp.json();
      if (!resp.ok) {
        showError(data.error || "Could not read this video link.");
        return;
      }

      inspectedUrlInfo = data;
      // Show Preview Card
      const card = document.getElementById("url-preview-card");
      document.getElementById("preview-thumb").src = data.thumbnail || "";
      document.getElementById("preview-title").textContent = data.title;
      document.getElementById("preview-duration").textContent = data.duration_formatted;
      document.getElementById("preview-est-time").textContent = `~${data.estimated_dub_mins} mins`;
      card.classList.remove("hidden");

      // Check Length Warning (> 5 minutes)
      const warningBanner = document.getElementById("length-warning-banner");
      if (data.needs_length_warning) {
        document.getElementById("warning-text").textContent = data.warning_message;
        warningBanner.classList.remove("hidden");
      } else {
        warningBanner.classList.add("hidden");
      }

    } catch (err) {
      showError("Network error checking link: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Inspect Link";
    }
  });

  // Start Job
  document.getElementById("btn-start-job").addEventListener("click", startDubbingJob);

  // Cancel Job
  document.getElementById("btn-cancel-job").addEventListener("click", async () => {
    if (!activeJobId) return;
    try {
      const resp = await fetch(`/api/jobs/${activeJobId}/cancel`, { method: "POST" });
      const data = await resp.json();
      appendLog("Cancellation signal sent to pipeline.");
    } catch (err) {
      console.error(err);
    }
  });

  // Clean Disk
  document.getElementById("btn-clean-disk").addEventListener("click", async () => {
    try {
      const resp = await fetch("/api/clean_temp", { method: "POST" });
      const data = await resp.json();
      alert(`Cleaned ${data.cleaned_mb} MB of temporary stem caches!`);
      refreshHistory();
    } catch (err) {
      alert("Error cleaning disk: " + err.message);
    }
  });

  // Refresh History
  document.getElementById("btn-refresh-history").addEventListener("click", refreshHistory);
}

async function startDubbingJob() {
  clearErrors();
  const isFileMode = document.getElementById("btn-mode-file").classList.contains("active");
  const targetLang = document.getElementById("select-target-lang").value;
  const sourceLang = document.getElementById("select-source-lang").value;
  
  const formData = new FormData();
  formData.append("target_lang", targetLang);
  formData.append("source_lang", sourceLang);

  if (isFileMode) {
    const fileInput = document.getElementById("file-input");
    if (!fileInput.files || fileInput.files.length === 0) {
      showError("Please select or drop a video file to dub.");
      return;
    }
    formData.append("file", fileInput.files[0]);
  } else {
    const urlInput = document.getElementById("url-input");
    const url = urlInput.value.trim();
    if (!url) {
      showError("Please enter a valid video link.");
      return;
    }
    // Check confirmation if warning is visible
    const warningBanner = document.getElementById("length-warning-banner");
    if (!warningBanner.classList.contains("hidden")) {
      const confirmed = document.getElementById("confirm-long-video").checked;
      if (!confirmed) {
        showError("Please check the box to confirm you want to dub a video longer than 5 minutes on CPU.");
        return;
      }
    }
    formData.append("url", url);
    if (inspectedUrlInfo) {
      formData.append("title", inspectedUrlInfo.title || "");
      formData.append("duration", inspectedUrlInfo.duration || 0);
    }
  }

  const startBtn = document.getElementById("btn-start-job");
  const cancelBtn = document.getElementById("btn-cancel-job");
  startBtn.disabled = true;
  startBtn.textContent = "Submitting Job...";

  try {
    const resp = await fetch("/api/jobs", {
      method: "POST",
      body: formData
    });
    const data = await resp.json();
    if (!resp.ok) {
      showError(data.detail || "Failed to start dubbing job.");
      startBtn.disabled = false;
      startBtn.textContent = "🚀 Start Dubbing";
      return;
    }

    activeJobId = data.job_id;
    startBtn.classList.add("hidden");
    cancelBtn.classList.remove("hidden");
    cancelBtn.disabled = false;

    // Start Real-time Polling
    startPolling(activeJobId);

  } catch (err) {
    showError("Could not submit job: " + err.message);
    startBtn.disabled = false;
    startBtn.textContent = "🚀 Start Dubbing";
  }
}

// --- Polling Progress ---
function startPolling(jobId) {
  if (pollInterval) clearInterval(pollInterval);

  pollInterval = setInterval(async () => {
    try {
      const resp = await fetch(`/api/jobs/${jobId}`);
      if (!resp.ok) return;
      const job = await resp.json();
      currentJobData = job;
      updateUIWithJobState(job);

      if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
        clearInterval(pollInterval);
        onJobFinished(job);
      }
    } catch (e) {
      console.warn("Polling error:", e);
    }
  }, 800);
}

function updateUIWithJobState(job) {
  // Overall Progress
  const percent = Math.min(100, Math.max(0, job.progress || 0));
  document.getElementById("overall-bar").style.width = `${percent}%`;
  document.getElementById("overall-percent").textContent = `${Math.round(percent)}%`;
  document.getElementById("overall-stage-label").textContent = `${job.current_stage || job.status}`;

  // Timers
  document.getElementById("timer-elapsed").textContent = formatTime(job.elapsed_time || 0);
  document.getElementById("timer-eta").textContent = formatTime(job.estimated_remaining || 0);

  // Stages List
  if (job.stages && job.stages.length > 0) {
    job.stages.forEach(st => {
      const badge = document.getElementById(`stage-badge-${st.num}`);
      const dur = document.getElementById(`stage-dur-${st.num}`);
      if (badge) {
        badge.className = `stage-badge badge-${st.status}`;
        badge.textContent = st.status;
      }
      if (dur && st.duration > 0) {
        dur.textContent = `${st.duration}s`;
      }
    });
  }

  // Logs
  if (job.logs && job.logs.length > 0) {
    const logBox = document.getElementById("log-content");
    logBox.textContent = job.logs.join("\n");
    const container = document.getElementById("log-console-body");
    container.scrollTop = container.scrollHeight;
  }
}

function onJobFinished(job) {
  const startBtn = document.getElementById("btn-start-job");
  const cancelBtn = document.getElementById("btn-cancel-job");

  startBtn.classList.remove("hidden");
  startBtn.disabled = false;
  startBtn.textContent = "🚀 Start Dubbing";
  cancelBtn.classList.add("hidden");

  if (job.status === "completed") {
    // Populate Video Players
    const origPlayer = document.getElementById("player-original");
    const dubbedPlayer = document.getElementById("player-dubbed");
    if (job.outputs.original_video) {
      origPlayer.src = `/media/${encodeURIComponent(job.outputs.original_video)}`;
    }
    if (job.outputs.dubbed_video) {
      dubbedPlayer.src = `/media/${encodeURIComponent(job.outputs.dubbed_video)}`;
    }

    // Populate Download Buttons
    setDownloadLink("dl-video", `/download/${job.id}/video`);
    setDownloadLink("dl-audio", `/download/${job.id}/audio`);
    setDownloadLink("dl-vocals", `/download/${job.id}/vocals`);
    setDownloadLink("dl-music", `/download/${job.id}/music`);
    setDownloadLink("dl-sub-bilingual", `/download/${job.id}/bilingual_sub`);
    setDownloadLink("dl-sub-hindi", `/download/${job.id}/translated_sub`);

    // Populate Script Editor
    renderScriptEditor(job);

    // Switch to output player view
    switchTab("tab-player");
  } else if (job.status === "failed") {
    showError(`Dubbing Failed: ${job.error || "Unknown error occurred"}`);
  }
}

function setDownloadLink(id, url) {
  const btn = document.getElementById(id);
  if (btn) {
    btn.href = url;
  }
}

// --- Script Editor View & Granular Re-dubbing ---
function renderScriptEditor(job) {
  document.getElementById("editor-active-job").textContent = `Job: ${job.title} (${job.id})`;
  const tbody = document.getElementById("editor-tbody");
  tbody.innerHTML = "";

  if (!job.segments || job.segments.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">No speech segments detected in this video.</td></tr>`;
    return;
  }

  job.segments.forEach(s => {
    const tr = document.createElement("tr");
    tr.id = `seg-row-${s.id}`;
    tr.innerHTML = `
      <td style="font-family: monospace; color: var(--text-muted);">${s.start.toFixed(1)}s - ${s.end.toFixed(1)}s</td>
      <td>
        <select class="speaker-select" id="spk-select-${s.id}">
          <option value="SPEAKER_00" ${s.speaker === 'SPEAKER_00' ? 'selected' : ''}>SPEAKER_00 (Male)</option>
          <option value="SPEAKER_01" ${s.speaker === 'SPEAKER_01' ? 'selected' : ''}>SPEAKER_01 (Female)</option>
        </select>
      </td>
      <td style="color: var(--text-muted);">${escapeHtml(s.original_text)}</td>
      <td>
        <input type="text" class="edit-input" id="edit-text-${s.id}" value="${escapeHtml(s.translated_text || '')}">
        <span class="changed-badge hidden" id="badge-changed-${s.id}">Modified</span>
      </td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="redubSingleLine(${s.id})">🔄 Re-dub Line</button>
      </td>
    `;
    tbody.appendChild(tr);

    // Mark as modified on edit
    const input = tr.querySelector(`#edit-text-${s.id}`);
    const sel = tr.querySelector(`#spk-select-${s.id}`);
    const markChanged = () => {
      changedSegments.add(s.id);
      const b = tr.querySelector(`#badge-changed-${s.id}`);
      if (b) b.classList.remove("hidden");
    };
    input.addEventListener("input", markChanged);
    sel.addEventListener("change", markChanged);
  });
}

async function redubSingleLine(segmentId) {
  if (!currentJobData || !currentJobData.id) return;
  const newText = document.getElementById(`edit-text-${segmentId}`).value.trim();
  const newSpeaker = document.getElementById(`spk-select-${segmentId}`).value;

  const btn = document.querySelector(`#seg-row-${segmentId} button`);
  btn.disabled = true;
  btn.textContent = "Re-dubbing...";

  try {
    const resp = await fetch(`/api/jobs/${currentJobData.id}/redub`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        segment_id: segmentId,
        new_text: newText,
        new_speaker: newSpeaker
      })
    });
    const res = await resp.json();
    if (!resp.ok) {
      alert("Failed to re-dub line: " + (res.detail || "Unknown error"));
      return;
    }

    // Refresh dubbed player
    const dubbedPlayer = document.getElementById("player-dubbed");
    dubbedPlayer.src = `/media/${encodeURIComponent(currentJobData.outputs.dubbed_video)}?t=${Date.now()}`;
    alert(`Line #${segmentId} re-dubbed successfully and final video updated!`);
    
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔄 Re-dub Line";
  }
}

// --- History & Disk Usage ---
async function refreshHistory() {
  try {
    const resp = await fetch("/api/jobs");
    const data = await resp.json();
    
    // Update Disk Stats
    if (data.disk) {
      document.getElementById("disk-total").textContent = `${data.disk.total_mb} MB`;
      document.getElementById("disk-cache").textContent = `${data.disk.cache_mb} MB`;
      document.getElementById("disk-output").textContent = `${data.disk.output_mb} MB`;
      document.getElementById("disk-uploads").textContent = `${data.disk.uploads_mb} MB`;
    }

    // Update History Table
    const tbody = document.getElementById("history-tbody");
    tbody.innerHTML = "";
    if (!data.jobs || data.jobs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">No dubbing history found.</td></tr>`;
      return;
    }

    data.jobs.forEach(j => {
      const tr = document.createElement("tr");
      const dateStr = new Date(j.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      tr.innerHTML = `
        <td><strong>${escapeHtml(j.title || j.id)}</strong></td>
        <td>${j.target_lang === 'hi' ? 'Hindi' : j.target_lang}</td>
        <td>${dateStr}</td>
        <td>${j.video_duration > 0 ? (j.video_duration).toFixed(1) + 's' : '--'}</td>
        <td><span class="stage-badge badge-${j.status}">${j.status}</span></td>
        <td>
          ${j.status === 'completed' ? `<button class="btn btn-secondary btn-sm" onclick="loadJobToEditor('${j.id}')">View & Edit</button>` : ''}
          ${j.status === 'failed' || j.status === 'cancelled' ? `<button class="btn btn-secondary btn-sm" onclick="resumeJob('${j.id}')">Resume</button>` : ''}
        </td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

async function loadJobToEditor(jobId) {
  try {
    const resp = await fetch(`/api/jobs/${jobId}`);
    const job = await resp.json();
    currentJobData = job;
    onJobFinished(job);
    switchTab("tab-editor");
  } catch (err) {
    alert("Could not load job: " + err.message);
  }
}

async function resumeJob(jobId) {
  try {
    const resp = await fetch(`/api/jobs/${jobId}/resume`, { method: "POST" });
    if (resp.ok) {
      activeJobId = jobId;
      switchTab("tab-dub");
      startPolling(jobId);
    }
  } catch (err) {
    alert("Could not resume job: " + err.message);
  }
}

// --- Utilities ---
function showError(msg) {
  const el = document.getElementById("input-error-alert");
  el.textContent = msg;
  el.classList.remove("hidden");
}

function clearErrors() {
  const el = document.getElementById("input-error-alert");
  el.textContent = "";
  el.classList.add("hidden");
}

function appendLog(line) {
  const logBox = document.getElementById("log-content");
  logBox.textContent += `\n[${new Date().toLocaleTimeString()}] ${line}`;
}

function formatTime(sec) {
  if (!sec || isNaN(sec) || sec <= 0) return "00:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
