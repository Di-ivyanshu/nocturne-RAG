/* Nocturne Archive — frontend logic
   Streaming answers · conversation memory · evidence inspection · doc management. */

const $ = (sel) => document.querySelector(sel);
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, reduce ? 0 : ms));

let pendingFiles = [];
let currentChunks = 0;
let history = [];                 // [{question, answer}] for multi-turn memory
const EMPTY_HTML = $("#emptyState").outerHTML;

/* ---------- stats + documents ---------- */
async function refreshStats() {
  try {
    const s = await fetch("/api/stats").then((r) => r.json());
    currentChunks = s.chunks;
    $("#chunkCount").textContent = s.chunks;
    $("#chunkCount2").textContent = s.chunks;
  } catch (_) {}
}
async function refreshDocs() {
  try {
    const { documents } = await fetch("/api/documents").then((r) => r.json());
    const list = $("#doclist");
    if (!documents.length) {
      list.innerHTML = `<li class="empty-docs">No documents indexed yet.</li>`;
      return;
    }
    list.innerHTML = documents.map((d) => `
      <li class="doc">
        <svg class="doc-icon" viewBox="0 0 24 24" width="15" height="15"><path d="M6 2h8l4 4v16H6z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M14 2v4h4" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>
        <span class="doc-name" title="${escapeHtml(d.source)}">${escapeHtml(d.source)}</span>
        <span class="doc-count">${d.chunks}</span>
        <button class="doc-del" data-src="${escapeHtml(d.source)}" title="Remove">×</button>
      </li>`).join("");
    list.querySelectorAll(".doc-del").forEach((b) =>
      b.addEventListener("click", () => deleteDoc(b.dataset.src)));
  } catch (_) {}
}
async function deleteDoc(source) {
  await fetch(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
  await refreshStats();
  await refreshDocs();
}
refreshStats();
refreshDocs();

/* ---------- generic staged loader ---------- */
function makeSteps(container, stages) {
  container.innerHTML = stages.map((s, i) => `<div class="pstep" data-i="${i}">
    <div class="pbullet">${i + 1}</div><div class="plabel">${s}</div><div class="pmeta"></div></div>`).join("");
  const steps = [...container.querySelectorAll(".pstep")];
  return {
    set(active) { steps.forEach((s, k) => { s.classList.toggle("active", k === active); s.classList.toggle("done", k < active); }); },
    meta(i, txt) { if (steps[i]) steps[i].querySelector(".pmeta").textContent = txt; },
    finishAll() { steps.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); }); },
    count: stages.length,
  };
}

/* ---------- file upload + indexing ---------- */
const dropzone = $("#dropzone");
const fileInput = $("#fileInput");
const filelist = $("#filelist");
const buildBtn = $("#buildBtn");
const indexLoader = $("#indexLoader");

function renderFiles() {
  filelist.innerHTML = "";
  pendingFiles.forEach((f) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="fdot"></span><span class="fname">${escapeHtml(f.name)}</span>`;
    filelist.appendChild(li);
  });
  buildBtn.disabled = pendingFiles.length === 0;
  buildBtn.textContent = pendingFiles.length
    ? `Build index · ${pendingFiles.length} file${pendingFiles.length > 1 ? "s" : ""}` : "Build index";
}
function addFiles(list) {
  for (const f of list) if (!pendingFiles.some((p) => p.name === f.name)) pendingFiles.push(f);
  renderFiles();
}
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", (e) => addFiles(e.target.files));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

buildBtn.addEventListener("click", async () => {
  if (!pendingFiles.length) return;
  const fileCount = pendingFiles.length;
  const fd = new FormData();
  pendingFiles.forEach((f) => fd.append("files", f));
  buildBtn.disabled = true; buildBtn.textContent = "Indexing…";

  indexLoader.hidden = false;
  indexLoader.innerHTML = `<div class="proc-title"><span class="spin"></span>Building index</div><div id="procSteps"></div>`;
  const proc = makeSteps($("#procSteps"),
    ["Reading documents", "Splitting into chunks", "Creating embeddings", "Storing in vector DB"]);
  proc.meta(0, `${fileCount} file${fileCount > 1 ? "s" : ""}`);
  let cur = 0; proc.set(0); let waiting = true;
  (async () => { while (waiting && cur < 2) { await sleep(650); cur++; proc.set(cur); } })();

  let res;
  try { res = await fetch("/api/upload", { method: "POST", body: fd }).then((r) => r.json()); }
  catch (_) { res = { error: "Upload failed — is the server running?" }; }
  waiting = false;

  if (res.error) {
    indexLoader.querySelector(".proc-title").innerHTML = `<span style="color:var(--rose)">${res.error}</span>`;
    buildBtn.disabled = false; buildBtn.textContent = "Build index"; return;
  }
  proc.set(3); proc.meta(2, "local model"); await sleep(420); proc.finishAll();
  indexLoader.querySelector(".proc-title .spin")?.remove();
  const done = document.createElement("div");
  done.className = "proc-done";
  done.textContent = `✓ ${res.chunks_added} chunks created · total ${res.chunks} · ${res.elapsed_ms} ms`;
  indexLoader.appendChild(done);

  pendingFiles = []; renderFiles();
  await refreshStats(); await refreshDocs();
  buildBtn.textContent = "Build index";
  setTimeout(() => { indexLoader.hidden = true; indexLoader.innerHTML = ""; }, 6000);
});

$("#clearBtn").addEventListener("click", async () => {
  await fetch("/api/clear", { method: "POST" });
  await refreshStats(); await refreshDocs();
});

/* ---------- conversation ---------- */
const thread = $("#thread");
const composer = $("#composer");
const qInput = $("#qInput");

function updateMemHint() {
  const n = history.length;
  $("#memHint").textContent = n ? `Conversation · ${n} turn${n > 1 ? "s" : ""}` : "New conversation";
}
$("#newChatBtn").addEventListener("click", () => {
  history = [];
  thread.innerHTML = EMPTY_HTML;
  bindSamples();
  updateMemHint();
});

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = qInput.value.trim();
  if (q) { qInput.value = ""; ask(q); }
});
function bindSamples() {
  document.querySelectorAll("#samples .chip").forEach((c) => c.addEventListener("click", () => ask(c.textContent)));
}
bindSamples();

async function ask(question) {
  $("#emptyState")?.remove();

  const q = document.createElement("div");
  q.className = "msg-q"; q.textContent = question;
  thread.appendChild(q);

  const card = document.createElement("div");
  card.className = "answer";
  card.innerHTML = `
    <div class="answer-body thinking">Searching the archive…</div>
    <div class="qloader">
      <div class="qloader-head"><span class="ql-title">Processing</span><span class="timer mono">0 ms</span></div>
      <div id="qSteps"></div>
    </div>`;
  thread.appendChild(card);
  thread.scrollTop = thread.scrollHeight;

  const proc = makeSteps(card.querySelector("#qSteps"), [
    "Embedding your question", `Searching ${currentChunks} chunks`,
    "Re-ranking top matches", "Writing the answer", "Verifying with critic",
  ]);
  const body = card.querySelector(".answer-body");

  const t0 = performance.now();
  const timerEl = card.querySelector(".timer");
  const timer = setInterval(() => { timerEl.textContent = `${Math.round(performance.now() - t0)} ms`; }, 60);

  let cur = 0; proc.set(0); let waiting = true;
  (async () => { while (waiting && cur < 2) { await sleep(560); cur++; proc.set(cur); } })();

  let answerText = "", firstToken = true, meta = null;
  try {
    const resp = await fetch("/api/ask_stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!raw.startsWith("data:")) continue;
        const ev = JSON.parse(raw.slice(5).trim());
        if (ev.type === "token") {
          if (firstToken) {
            firstToken = false; waiting = false;
            proc.set(3); body.classList.remove("thinking"); body.classList.add("streaming"); body.textContent = "";
          }
          answerText += ev.text;
          body.textContent = answerText;
          thread.scrollTop = thread.scrollHeight;
        } else if (ev.type === "meta") {
          meta = ev;
        }
      }
    }
  } catch (_) {
    meta = meta || { answer: "Couldn't reach the server. Is it running?", confidence: 0, grounded: false, citations: [], evidence: [], attempts: [] };
  }

  waiting = false; clearInterval(timer);
  proc.set(4); proc.finishAll();
  body.classList.remove("streaming");
  await sleep(reduce ? 0 : 200);

  if (meta) {
    renderAnswer(card, meta);
    history.push({ question, answer: meta.answer });
    updateMemHint();
    // bring the question to the top so the answer reads from the start
    // (instead of jumping to the bottom and hiding it)
    q.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }
}

function renderAnswer(card, res) {
  const conf = res.confidence ?? 0;
  const grounded = !!res.grounded;
  const state = grounded && conf >= 0.75 ? "high" : conf >= 0.5 ? "med" : "low";
  const isIdk = /don'?t know|do not know/i.test(res.answer || "");
  const evidence = res.evidence || [];
  const matched = [...new Set((res.citations || []).map((c) => c.source))];

  const body = card.querySelector(".answer-body");
  body.classList.remove("thinking", "streaming");
  body.innerHTML = formatAnswer(res.answer || "");

  card.querySelector(".qloader")?.remove();

  const passes = res.attempts ? res.attempts.length : 1;
  const strip = document.createElement("div");
  strip.className = "meta-strip";
  strip.innerHTML = `
    <span class="ms">${res.elapsed_ms ?? "—"} ms</span>
    <span class="ms">${res.chunks_searched ?? currentChunks} searched</span>
    <span class="ms">${passes} pass${passes > 1 ? "es" : ""}${res.self_corrected ? " · self-corrected" : ""}</span>`;
  card.appendChild(strip);

  // evidence — toggle lives inline in the meta strip; list expands below, compact
  if (evidence.length) {
    const toggle = document.createElement("button");
    toggle.className = "src-toggle";
    toggle.innerHTML =
      `<span class="src-caret">▸</span> ${evidence.length} source${evidence.length > 1 ? "s" : ""}`;
    strip.appendChild(toggle);

    const listEl = document.createElement("div");
    listEl.className = "src-list"; listEl.hidden = true;

    evidence.forEach((e, i) => {
      const row = document.createElement("div");
      row.className = "src";
      const rr = e.rerank_score != null ? ` · rr ${Number(e.rerank_score).toFixed(2)}`
        : (e.score != null ? ` · ${Number(e.score).toFixed(2)}` : "");
      row.innerHTML = `<span class="src-n">[${e.n ?? i + 1}]</span>
        <span class="src-name">${escapeHtml(e.source)}</span>
        <span class="src-meta">p${e.page}${rr}</span>
        <span class="src-caret">▶</span>`;
      const text = document.createElement("div");
      text.className = "src-text"; text.hidden = true;
      text.innerHTML = `${escapeHtml(e.text)}<span class="st-meta">${escapeHtml(e.source)} · page ${e.page}</span>`;
      row.addEventListener("click", () => { row.classList.toggle("open"); text.hidden = !text.hidden; });
      listEl.appendChild(row); listEl.appendChild(text);
    });

    toggle.addEventListener("click", () => {
      const open = listEl.hidden;
      listEl.hidden = !open;
      toggle.querySelector(".src-caret").textContent = open ? "▾" : "▸";
    });

    card.appendChild(listEl);
  }

  if (isIdk) {
    const b = document.createElement("div");
    b.className = "banner warn";
    b.textContent = "Not found in your documents — the assistant declined to guess.";
    card.appendChild(b);
  }
}

/* ---------- helpers ---------- */
function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
}
/* inline markdown: bold + citation superscripts */
function inlineFmt(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(\d+)\]/g, '<sup class="cite">[$1]</sup>');
}

/* block-level markdown: paragraphs + bullet/numbered lists.
   Also splits lists the model crammed into one line onto their own items. */
function formatAnswer(text) {
  let t = escapeHtml(text).trim();
  t = t.replace(/\s+(\d+[.)]\s)/g, "\n$1");   // " 2. " -> new line
  t = t.replace(/\s+[•]\s+/g, "\n- ");          // inline bullets -> new line

  const lines = t.split(/\n+/).map((l) => l.trim()).filter(Boolean);
  let html = "", listType = null, buf = [];
  const flush = () => {
    if (listType) html += `<${listType}>${buf.map((li) => `<li>${inlineFmt(li)}</li>`).join("")}</${listType}>`;
    listType = null; buf = [];
  };
  for (const line of lines) {
    const ol = line.match(/^\d+[.)]\s+(.*)/);
    const ul = line.match(/^[-*]\s+(.*)/);
    if (ol) { if (listType !== "ol") { flush(); listType = "ol"; } buf.push(ol[1]); }
    else if (ul) { if (listType !== "ul") { flush(); listType = "ul"; } buf.push(ul[1]); }
    else { flush(); html += `<p>${inlineFmt(line)}</p>`; }
  }
  flush();
  return html || `<p>${inlineFmt(t)}</p>`;
}
