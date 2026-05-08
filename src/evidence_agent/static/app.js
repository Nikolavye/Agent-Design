const STEP_DELAY_MS = 700;   // demo pacing — set to 0 for fast prod runs
const RAIL_WIDTH_KEY = "evidence-agent:rail-width";
const RUN_HISTORY_KEY = "evidence-agent:session-run-history";
const RAIL_MIN_WIDTH = 300;
const MAX_HISTORY_RUNS = 8;

const state = {
  runId: null,
  timer: null,
  config: null,
  sources: [],
  selected: new Set(),
  activePanel: "events",
  currentStatus: "idle",
  history: [],
};

const $ = (id) => document.getElementById(id);
const INITIAL_CHAT_HTML = $("chatLog").innerHTML;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compactSourceName(name) {
  const stem = String(name || "source").replace(/\.pdf$/i, "");
  return stem.length > 28 ? `${stem.slice(0, 26)}…` : stem;
}

function refChip(ref, runId = state.runId) {
  const trimmed = String(ref || "").trim();
  if (!trimmed) return "";
  const basename = trimmed.split("/").pop().replace(/\.[a-z]+$/i, "");
  const url = runId
    ? `/api/runs/${encodeURIComponent(runId)}/artifact?ref=${encodeURIComponent(trimmed)}`
    : "#";
  return `<a class="ref-chip" href="${url}" target="_blank" rel="noopener" title="${escapeHtml(trimmed)}">${escapeHtml(basename)}</a>`;
}

function setHtmlIfChanged(element, html) {
  if (!element || element.__lastHtml === html) return false;
  element.innerHTML = html;
  element.__lastHtml = html;
  return true;
}

function loadRunHistory() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(RUN_HISTORY_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.slice(-MAX_HISTORY_RUNS) : [];
  } catch (_error) {
    return [];
  }
}

function saveRunHistory() {
  try {
    sessionStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(state.history.slice(-MAX_HISTORY_RUNS)));
  } catch (_error) {
    // History is a UI convenience; agent runs remain persisted on the server.
  }
}

function upsertRunHistory(run) {
  const runId = run?.run_id || state.runId;
  if (!runId) return;
  const current = run?.state || {};
  const summary = {
    runId,
    goal: current.goal || $("goal").value.trim(),
    status: current.status || "created",
    final: String(run?.final || "").slice(0, 25000),
    taskCount: run?.task_count ?? run?.plan?.tasks?.length ?? 0,
    refCount: run?.ref_count ?? run?.index?.length ?? 0,
    updatedAt: current.updated_at || new Date().toISOString(),
  };
  state.history = state.history.filter((item) => item.runId !== runId);
  state.history.push(summary);
  state.history = state.history.slice(-MAX_HISTORY_RUNS);
  saveRunHistory();
}

function renderHistoryMessages(currentRunId = state.runId) {
  return state.history
    .filter((entry) => entry.runId !== currentRunId)
    .map((entry) => {
      const goal = entry.goal || "Untitled question";
      const status = entry.status || "unknown";
      const refLabel = Number(entry.refCount || 0).toLocaleString();
      const finalHtml = entry.final
        ? `<details class="history-final">
            <summary>View previous final brief · ${refLabel} refs</summary>
            <div class="brief-content">${renderMarkdown(entry.final, entry.runId)}</div>
          </details>`
        : `<p class="history-note">This run is ${escapeHtml(status)} and has no saved final brief yet.</p>`;

      return `
        <article class="message user history-message">
          <div class="avatar">U</div>
          <div class="message-body">
            <strong>Previous question</strong>
            <p>${escapeHtml(goal)}</p>
          </div>
        </article>
        <article class="message assistant history-message">
          <div class="avatar">A</div>
          <div class="message-body">
            <strong>Previous run · ${escapeHtml(status)}</strong>
            ${finalHtml}
          </div>
        </article>
      `;
    })
    .join("");
}

function renderSavedHistory() {
  if (!state.history.length) return;
  setHtmlIfChanged($("chatLog"), renderHistoryMessages(null));
}

function nextParentRunId() {
  if (state.runId) return state.runId;
  return state.history.length ? state.history[state.history.length - 1].runId : null;
}

function resetRailPanels() {
  $("eventCount").textContent = "0";
  $("taskCount").textContent = "0";
  $("refCount").textContent = "0";
  setHtmlIfChanged($("events"), '<div class="rail-empty">Live trace will appear here once the planner starts.</div>');
  setHtmlIfChanged($("plan"), '<div class="rail-empty">Approved task graph appears here.</div>');
  setHtmlIfChanged($("refs"), '<div class="rail-empty">Indexed evidence refs appear here.</div>');
  setHtmlIfChanged($("stateBox"), '<div class="rail-empty">Current state summary appears here.</div>');
  $("finalBrief").textContent = "";
}

function startNewSession() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  state.runId = null;
  state.currentStatus = "idle";
  state.history = [];
  state.selected = new Set();
  sessionStorage.removeItem(RUN_HISTORY_KEY);
  localStorage.removeItem("evidence-agent:run-history");

  $("chatLog").innerHTML = INITIAL_CHAT_HTML;
  $("chatLog").__lastHtml = INITIAL_CHAT_HTML;
  $("runStatus").textContent = "No active run";
  $("planFeedback").value = "";
  if (state.config?.sample_goal) $("goal").value = state.config.sample_goal;

  setStatusBadge("idle");
  renderStages("idle");
  setPlanModeControls("idle");
  renderSources();
  resetRailPanels();
  bindSuggestions();
}

function renderMarkdown(md, runId = state.runId) {
  if (!md) return "";
  let html = escapeHtml(md);

  // ref chips first (before other text transforms — they replace whole bracket expressions)
  html = html.replace(/\[ref:\s*([^\]]+)\]/g, (match, ref) => {
    return ref
      .split(";")
      .map((item) => refChip(item, runId))
      .filter(Boolean)
      .join(" ");
  });
  html = html.replace(/\[Evidence[:：]\s*([^\]]*artifacts\/[^\]]+)\]/gi, (match, body) => {
    const refs = body.match(/artifacts\/[^\]\s,;]+/g) || [];
    return refs.map((ref) => refChip(ref, runId)).join(" ");
  });

  // headings
  html = html.replace(/^####\s+(.+)$/gm, "<h4>$1</h4>");
  html = html.replace(/^###\s+(.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^##\s+(.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^#\s+(.+)$/gm, "<h2>$1</h2>");

  // inline
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)/g, "<em>$1</em>");
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");

  // unordered lists (- or * at line start)
  html = html.replace(/(^|\n)((?:[-*]\s+.+(?:\n|$))+)/g, (match, prefix, block) => {
    const items = block
      .trim()
      .split(/\n/)
      .map((line) => line.replace(/^[-*]\s+/, ""))
      .map((item) => `<li>${item}</li>`)
      .join("");
    return `${prefix}<ul>${items}</ul>`;
  });

  // paragraphs (double newline → split)
  html = html
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      if (/^<(h[1-6]|ul|ol|pre|blockquote)/.test(trimmed)) return trimmed;
      return `<p>${trimmed.replace(/\n/g, "<br/>")}</p>`;
    })
    .join("");

  return html;
}

async function loadConfig() {
  const response = await fetch("/api/config");
  const config = await response.json();
  state.config = config;
  if (!$("goal").value) $("goal").value = config.sample_goal;
  $("modelBadge").textContent = config.model || "gpt-5.1";
  const ready = Boolean(config.llm_configured);
  const ctxLabel = config.context_token_budget
    ? `ctx ${Math.round(config.context_token_budget / 1000)}k`
    : "model context";
  $("llmStatus").textContent = ready
    ? `${config.model} · ready · ${ctxLabel}`
    : `${config.model || "gpt-5.1"} · unavailable`;
  $("llmStatus").classList.toggle("ready", ready);
  $("llmStatus").classList.toggle("missing", !ready);
  setPlanModeControls("idle");
  renderStages("idle");
}

async function loadSources() {
  const response = await fetch("/api/sources");
  const payload = await response.json();
  state.sources = payload.sources;
  const availablePaths = new Set(state.sources.map((source) => source.path));
  state.selected = new Set(Array.from(state.selected).filter((path) => availablePaths.has(path)));
  renderSources();
}

function renderSources() {
  const pdfs = state.sources.filter((source) => source.kind === "pdf");
  if (!pdfs.length) {
    setHtmlIfChanged($("sources"), "");
    return;
  }
  const selectedPdfs = pdfs.filter((source) => state.selected.has(source.path));
  if (!selectedPdfs.length) {
    setHtmlIfChanged($("sources"), "");
    return;
  }
  const visible = selectedPdfs.slice(0, 3);
  const extra = Math.max(0, selectedPdfs.length - visible.length);
  setHtmlIfChanged($("sources"), [
    `<span class="source-count">📎 ${selectedPdfs.length} PDF${selectedPdfs.length === 1 ? "" : "s"}</span>`,
    ...visible.map(
      (source) =>
        `<span class="source-chip" title="${escapeHtml(source.path)}">${escapeHtml(compactSourceName(source.name))}</span>`
    ),
    extra ? `<span class="source-chip">+${extra}</span>` : "",
  ].join(""));
}

async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body: form });
  const uploaded = await response.json();
  if (uploaded.path) state.selected.add(uploaded.path);
  await loadSources();
}

async function startRun() {
  if (!state.config?.llm_configured) {
    $("llmStatus").textContent = "GPT-5 is not ready.";
    $("llmStatus").classList.add("missing");
    return;
  }
  const goal = $("goal").value.trim();
  if (!goal) return;
  const parentRunId = nextParentRunId();
  setPlanModeControls("planning");
  const response = await fetch("/api/plans", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      goal,
      source_paths: Array.from(state.selected),
      parent_run_id: parentRunId,
      step_delay_ms: STEP_DELAY_MS,
    }),
  });
  if (!response.ok) {
    $("runStatus").textContent = await response.text();
    setPlanModeControls(state.currentStatus);
    return;
  }
  const payload = await response.json();
  state.runId = payload.run_id;
  $("goal").value = "";
  upsertRunHistory({
    run_id: state.runId,
    state: { goal, status: "planning", updated_at: new Date().toISOString() },
    plan: null,
    index: [],
    final: "",
  });
  $("runStatus").textContent = state.runId;
  setStatusBadge("planning");
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(pollRun, 900);
  await pollRun();
}

async function submitPlanFeedback() {
  if (!state.runId) return;
  const feedback = $("planFeedback").value.trim();
  if (!feedback) return;
  setPlanModeControls("planning");
  const recoveryMode = state.currentStatus === "blocked" || state.currentStatus === "failed";
  const endpoint = recoveryMode
    ? `/api/runs/${state.runId}/manual-replan`
    : `/api/runs/${state.runId}/plan-feedback`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      feedback,
      step_delay_ms: STEP_DELAY_MS,
    }),
  });
  if (!response.ok) {
    $("runStatus").textContent = await response.text();
    setPlanModeControls(state.currentStatus);
    return;
  }
  $("planFeedback").value = "";
  setStatusBadge("planning");
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(pollRun, 900);
  await pollRun();
}

async function approvePlan() {
  if (!state.runId) return;
  setPlanModeControls("running");
  const response = await fetch(`/api/runs/${state.runId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      step_delay_ms: STEP_DELAY_MS,
    }),
  });
  if (!response.ok) {
    $("runStatus").textContent = await response.text();
    setPlanModeControls("awaiting_approval");
    return;
  }
  setStatusBadge("running");
  if (state.timer) clearInterval(state.timer);
  state.timer = setInterval(pollRun, 900);
  await pollRun();
}

function setStatusBadge(status) {
  const pill = $("statusPill");
  pill.textContent = status;
  pill.className = `badge badge-status status-${status}`;
}

function setPlanModeControls(status) {
  state.currentStatus = status;
  const llmReady = Boolean(state.config?.llm_configured);
  const waitingForReview = status === "awaiting_approval";
  const needsHumanRecovery = status === "blocked" || status === "failed";
  const busy = status === "planning" || status === "running" || status === "replanning";
  $("runButton").disabled = !llmReady || busy;
  $("feedbackButton").disabled = !llmReady || !(waitingForReview || needsHumanRecovery);
  $("approveButton").disabled = !llmReady || !waitingForReview;
  $("planFeedback").placeholder = needsHumanRecovery
    ? "Tell the planner how to recover this run: change source, avoid a failed tool, add a missing task…"
    : "Optional: send feedback to revise the plan…";
  $("feedbackButton").lastChild.textContent = needsHumanRecovery ? "Recover plan" : "Revise plan";
  $("approveButton").style.display = needsHumanRecovery ? "none" : "";
  // toggle plan-action row visibility
  $("planActions").classList.toggle("hidden", !(waitingForReview || needsHumanRecovery));
}

function renderStages(status) {
  const stageByStatus = {
    created: "planning",
    planning: "planning",
    awaiting_approval: "awaiting_approval",
    running: "running",
    replanning: "replanning",
    done: "done",
    blocked: "replanning",
    failed: "replanning",
  };
  const activeStage = stageByStatus[status] || "planning";
  document.querySelectorAll("[data-stage]").forEach((item) => {
    item.classList.toggle("active", item.dataset.stage === activeStage);
  });
}

function switchPanel(panelName) {
  state.activePanel = panelName;
  document.querySelectorAll(".rail-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.panel === panelName);
  });
  const panels = {
    events: $("events"),
    plan: $("plan"),
    refs: $("refs"),
    state: $("stateBox"),
  };
  Object.entries(panels).forEach(([name, el]) => {
    if (el) el.classList.toggle("rail-panel-active", name === panelName);
  });
}

async function pollRun() {
  if (!state.runId) return;
  const response = await fetch(`/api/runs/${state.runId}`);
  const run = await response.json();
  renderRun(run);
  const status = run.state?.status || "running";
  setStatusBadge(status);
  renderStages(status);
  setPlanModeControls(status);
  if (status === "awaiting_approval" || status === "done" || status === "blocked" || status === "failed") {
    clearInterval(state.timer);
    state.timer = null;
  }
}

function renderRun(run) {
  $("runStatus").textContent = run.run_id || state.runId || "No active run";
  upsertRunHistory(run);
  renderChat(run);

  // tasks panel
  const tasks = run.plan?.tasks || [];
  $("taskCount").textContent = String(tasks.length);
  const planHtml = tasks.length
    ? tasks
        .map(
          (task) => `
        <article class="task-card status-${escapeHtml(task.status)}">
          <div class="task-dot"></div>
          <div>
            <div class="task-title">
              <span class="task-id">${escapeHtml(task.id)}</span>
              <strong>${escapeHtml(task.title)}</strong>
              ${task.produces_final ? '<span class="task-final">final</span>' : ""}
            </div>
            <div class="task-desc">${escapeHtml(task.description)}</div>
            <div class="task-meta">
              <span>${escapeHtml(task.status)}</span>
              ${task.parallelizable ? "<span>parallel</span>" : ""}
              ${(task.requires || []).map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
            </div>
          </div>
        </article>`
        )
        .join("")
    : '<div class="rail-empty">Approved task graph appears here.</div>';
  setHtmlIfChanged($("plan"), planHtml);

  // events panel
  const events = run.events || [];
  $("eventCount").textContent = String(events.length);
  const visibleEvents = events.slice(-200);
  const eventsHtml = events.length
    ? visibleEvents
        .map(
          (event) => `
        <article class="event-card ${escapeHtml(event.type)}">
          <div class="event-dot"></div>
          <div class="event-body">
            <div class="event-head">
              <span class="event-title">${escapeHtml(event.title)}</span>
              <span class="event-type">${escapeHtml(event.type)}</span>
            </div>
            <div class="event-summary">${escapeHtml(event.summary)}</div>
            <div class="event-meta">
              <span>${escapeHtml(event.event_id)}</span>
              ${event.task_id ? `<span>${escapeHtml(event.task_id)}</span>` : ""}
              ${event.tool ? `<span>${escapeHtml(event.tool)}</span>` : ""}
              ${event.status && event.status !== "success" ? `<span>${escapeHtml(event.status)}</span>` : ""}
            </div>
          </div>
        </article>`
        )
        .join("")
    : '<div class="rail-empty">Live trace will appear here once the planner starts.</div>';
  const eventsChanged = setHtmlIfChanged($("events"), eventsHtml);
  if (eventsChanged && events.length && state.activePanel === "events") {
    $("events").scrollTop = $("events").scrollHeight;
  }

  // state panel
  const current = run.state || {};
  const stateLines = [
    `status:        ${current.status || "idle"}`,
    `next_action:   ${current.next_action || ""}`,
    `replan_count:  ${current.replan_count ?? 0}`,
    "",
    "planning_feedback:",
    ...(current.planning_feedback || []).map((item) => `  • ${item}`),
    "",
    "done:",
    ...(current.done || []).map((item) => `  ✓ ${item}`),
    "",
    "facts (recent):",
    ...(current.facts || []).slice(-6).map((item) => `  · ${item}`),
    "",
    "open_questions:",
    ...(current.open_questions || []).map((item) => `  ? ${item}`),
  ];
  setHtmlIfChanged($("stateBox"), `<pre class="state-content">${escapeHtml(stateLines.join("\n"))}</pre>`);

  // refs panel
  const refs = (run.index || []).slice(-30).reverse();
  $("refCount").textContent = String(run.index?.length || 0);
  const refsHtml = refs.length
    ? refs
        .map(
          (ref) => `
        <article class="ref-card">
          <div class="ref-head">
            <span class="ref-kind">${escapeHtml(ref.kind || "artifact")}</span>
            <span class="ref-source">${escapeHtml(ref.source || "—")}</span>
          </div>
          <div class="ref-summary">${escapeHtml(ref.summary || "")}</div>
          <div class="ref-meta">
            ${ref.page ? `<span>page ${escapeHtml(String(ref.page))}</span>` : ""}
            <span>${escapeHtml(ref.ref || "")}</span>
          </div>
        </article>`
        )
        .join("")
    : '<div class="rail-empty">Indexed evidence refs appear here.</div>';
  setHtmlIfChanged($("refs"), refsHtml);

  // hidden raw final brief
  $("finalBrief").textContent = run.final || "";
}

function renderChat(run) {
  const current = run.state || {};
  const tasks = run.plan?.tasks || [];
  const status = current.status || "idle";
  const feedback = current.planning_feedback || [];
  const openQuestions = current.open_questions || [];
  const done = current.done || [];
  const goal = current.goal || $("goal").value.trim();
  const finalBrief = run.final || "";

  // No run yet → keep hero
  if (!state.runId) return;

  const messages = [];
  const historyHtml = renderHistoryMessages(run.run_id || state.runId);
  if (historyHtml) messages.push(historyHtml);

  // user goal
  messages.push(`
    <article class="message user">
      <div class="avatar">U</div>
      <div class="message-body">
        <strong>You</strong>
        <p>${escapeHtml(goal || "—")}</p>
      </div>
    </article>
  `);

  // status narration (agent message)
  let statusBody = "";
  if (status === "planning") {
    statusBody = "Inspecting the goal, sources, and tools to draft a task graph…";
  } else if (status === "awaiting_approval") {
    const taskList = tasks
      .map(
        (task) =>
          `<div class="plan-task-row">
            <span class="task-id">${escapeHtml(task.id)}</span>
            <span>${escapeHtml(task.title)}</span>
            ${task.produces_final ? '<span class="task-final">final</span>' : ""}
            ${task.parallelizable ? '<span class="task-meta-pill">parallel</span>' : ""}
          </div>`
      )
      .join("");
    statusBody = `<p>Plan ready · ${tasks.length} tasks · v${run.plan?.version || 1}</p>
      <div class="plan-inline">${taskList}</div>
      <p class="plan-prompt-hint">Approve to execute, or send feedback to revise.</p>`;
  } else if (status === "running") {
    const doneLines = done.length
      ? done.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
      : '<li class="muted">Spawning executors…</li>';
    statusBody = `<p>Executing approved plan…</p><ul class="status-list">${doneLines}</ul>`;
  } else if (status === "replanning") {
    statusBody = "A task blocked or failed. Revising the remaining DAG…";
  } else if (status === "blocked" || status === "failed") {
    const recoveryHint =
      '<p class="plan-prompt-hint">Add a recovery instruction below. The planner will revise the remaining plan and ask for approval before execution resumes.</p>';
    statusBody = openQuestions.length
      ? `<p>Run needs attention:</p><ul class="status-list">${openQuestions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>${recoveryHint}`
      : `Run stopped before producing a final brief.${recoveryHint}`;
  } else if (status === "done") {
    statusBody = "Synthesis complete. Final brief below.";
  }

  if (statusBody) {
    messages.push(`
      <article class="message assistant">
        <div class="avatar">A</div>
        <div class="message-body">
          <strong>Agent · ${escapeHtml(status)}</strong>
          ${statusBody.startsWith("<") ? statusBody : `<p>${statusBody}</p>`}
        </div>
      </article>
    `);
  }

  // feedback history shown as separate bubbles
  feedback.forEach((item) => {
    messages.push(`
      <article class="message user">
        <div class="avatar">U</div>
        <div class="message-body">
          <strong>Plan feedback</strong>
          <p>${escapeHtml(item)}</p>
        </div>
      </article>
    `);
  });

  // final brief inline (markdown-rendered)
  if (finalBrief) {
    messages.push(`
      <article class="message assistant">
        <div class="avatar">A</div>
        <div class="message-body final">
          <strong>Final brief</strong>
          <div class="brief-content">${renderMarkdown(finalBrief, run.run_id || state.runId)}</div>
        </div>
      </article>
    `);
  }

  const chatChanged = setHtmlIfChanged($("chatLog"), messages.join(""));
  if (chatChanged) $("chatLog").scrollTop = $("chatLog").scrollHeight;
}

// suggestion chip handlers
function bindSuggestions() {
  document.querySelectorAll(".suggest-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const text = chip.dataset.suggest;
      if (text) {
        $("goal").value = text;
        $("goal").focus();
      }
    });
  });
}

// rail tabs
function bindRailTabs() {
  document.querySelectorAll(".rail-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchPanel(tab.dataset.panel));
  });
}

function railMaxWidth() {
  return Math.max(RAIL_MIN_WIDTH, Math.min(Math.round(window.innerWidth * 0.72), window.innerWidth - 420));
}

function setRailWidth(width, persist = true) {
  const nextWidth = clamp(Math.round(width), RAIL_MIN_WIDTH, railMaxWidth());
  document.documentElement.style.setProperty("--rail-width", `${nextWidth}px`);
  if (persist) localStorage.setItem(RAIL_WIDTH_KEY, String(nextWidth));
}

function initRailResize() {
  const handle = $("railResize");
  if (!handle) return;

  const stored = Number(localStorage.getItem(RAIL_WIDTH_KEY));
  if (Number.isFinite(stored) && stored > 0) setRailWidth(stored, false);

  handle.addEventListener("pointerdown", (event) => {
    if (window.innerWidth <= 1100) return;
    event.preventDefault();
    handle.setPointerCapture(event.pointerId);
    document.body.classList.add("rail-resizing");
  });

  handle.addEventListener("pointermove", (event) => {
    if (!document.body.classList.contains("rail-resizing")) return;
    setRailWidth(window.innerWidth - event.clientX);
  });

  const stopResize = (event) => {
    if (!document.body.classList.contains("rail-resizing")) return;
    document.body.classList.remove("rail-resizing");
    try {
      handle.releasePointerCapture(event.pointerId);
    } catch (_error) {
      // Pointer capture may already be released if the drag ended outside the window.
    }
  };

  handle.addEventListener("pointerup", stopResize);
  handle.addEventListener("pointercancel", stopResize);
  window.addEventListener("resize", () => {
    const current = Number.parseInt(getComputedStyle(document.documentElement).getPropertyValue("--rail-width"), 10);
    if (Number.isFinite(current)) setRailWidth(current, false);
  });
}

// ============================================================
// Bootstrap
// ============================================================
$("runButton").addEventListener("click", startRun);
$("feedbackButton").addEventListener("click", submitPlanFeedback);
$("approveButton").addEventListener("click", approvePlan);
$("newSessionButton").addEventListener("click", startNewSession);
$("upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file) await uploadPdf(file);
});

// Cmd/Ctrl+Enter submits
$("goal").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    if (!$("runButton").disabled) startRun();
  }
});

bindSuggestions();
bindRailTabs();
initRailResize();
state.history = loadRunHistory();
renderSavedHistory();
loadConfig();
loadSources();
