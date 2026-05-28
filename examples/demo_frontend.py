from __future__ import annotations

import argparse
import base64
import binascii
import html
import io
import json
import re
import sys
import threading
import time
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_DATASET = ROOT / "examples" / "data" / "sample_eval_five_examples.jsonl"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEVICES = ("cpu", "cuda")
OUTPUT_MODES = ("typed", "redacted")
EVAL_MODES = ("typed", "untyped")
DECODE_MODES = ("viterbi", "argmax")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".zsh",
    ".bash",
    ".sql",
}
SUPPORTED_FILE_EXTENSIONS = TEXT_FILE_EXTENSIONS | {".docx", ".odt", ".pdf", ".rtf"}
UPLOAD_ACCEPT = ",".join(sorted(SUPPORTED_FILE_EXTENSIONS))
XML_TEXT_NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}

_redactor_lock = threading.Lock()
_redactors: dict[tuple[str, str, str], Any] = {}


def _read_samples() -> list[dict[str, Any]]:
    """Load synthetic demo examples without requiring the model runtime."""
    samples: list[dict[str, Any]] = []
    if not SAMPLE_DATASET.exists():
        return samples
    with SAMPLE_DATASET.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = (
                record.get("info", {}).get("id")
                if isinstance(record.get("info"), dict)
                else None
            )
            samples.append(
                {
                    "id": sample_id or f"sample-{idx}",
                    "text": record.get("text", ""),
                    "spans": record.get("spans", {}),
                }
            )
    return samples


def _get_redactor(
    *,
    device: Literal["cpu", "cuda"],
    output_mode: Literal["typed", "redacted"],
    decode_mode: Literal["viterbi", "argmax"],
):
    """Return a cached OPF instance for the selected demo controls."""
    from opf import OPF

    key = (device, output_mode, decode_mode)
    with _redactor_lock:
        redactor = _redactors.get(key)
        if redactor is None:
            redactor = OPF(
                device=device,
                output_mode=output_mode,
                decode_mode=decode_mode,
                output_text_only=False,
            )
            _redactors[key] = redactor
        return redactor


def _as_choice(value: object, choices: tuple[str, ...], field: str) -> str:
    if not isinstance(value, str) or value not in choices:
        expected = ", ".join(choices)
        raise ValueError(f"{field} must be one of: {expected}")
    return value


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    device = _as_choice(payload.get("device", "cpu"), DEVICES, "device")
    output_mode = _as_choice(
        payload.get("output_mode", "typed"), OUTPUT_MODES, "output_mode"
    )
    eval_mode = _as_choice(payload.get("eval_mode", "typed"), EVAL_MODES, "eval_mode")
    decode_mode = _as_choice(
        payload.get("decode_mode", "viterbi"), DECODE_MODES, "decode_mode"
    )

    started = time.perf_counter()
    redactor = _get_redactor(
        device=device, output_mode=output_mode, decode_mode=decode_mode
    )
    result = redactor.redact(text)
    latency_ms = (time.perf_counter() - started) * 1000.0
    result_payload = result.to_dict()
    result_payload["demo"] = {
        "device": device,
        "output_mode": output_mode,
        "eval_mode": eval_mode,
        "decode_mode": decode_mode,
        "latency_ms": round(latency_ms, 2),
        "eval_mode_note": (
            "Eval mode applies to labeled datasets. Free-text redaction uses "
            "output mode; use typed eval for OPF labels and untyped eval for "
            "span-only matching against another taxonomy."
        ),
    }
    return result_payload


def _decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _normalize_extracted_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_docx_text(data: bytes) -> str:
    xml_paths = (
        "word/document.xml",
        "word/header1.xml",
        "word/header2.xml",
        "word/header3.xml",
        "word/footer1.xml",
        "word/footer2.xml",
        "word/footer3.xml",
        "word/footnotes.xml",
        "word/endnotes.xml",
    )
    parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for path in xml_paths:
            if path not in archive.namelist():
                continue
            root = ElementTree.fromstring(archive.read(path))
            for paragraph in root.findall(".//w:p", XML_TEXT_NAMESPACES):
                chunks: list[str] = []
                for node in paragraph.iter():
                    if node.tag == f"{{{XML_TEXT_NAMESPACES['w']}}}t" and node.text:
                        chunks.append(node.text)
                    elif node.tag == f"{{{XML_TEXT_NAMESPACES['w']}}}tab":
                        chunks.append("\t")
                    elif node.tag == f"{{{XML_TEXT_NAMESPACES['w']}}}br":
                        chunks.append("\n")
                text = "".join(chunks).strip()
                if text:
                    parts.append(text)
    return "\n".join(parts)


def _extract_odt_text(data: bytes) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ElementTree.fromstring(archive.read("content.xml"))
        for node in root.findall(".//text:h", XML_TEXT_NAMESPACES) + root.findall(
            ".//text:p", XML_TEXT_NAMESPACES
        ):
            text = "".join(node.itertext()).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError(
            "PDF upload support requires pypdf. Install the package with "
            "`pip install -e .` so demo dependencies are available."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return "\n\n".join(page for page in pages if page)
    except Exception as exc:
        raise ValueError("could not extract readable text from the PDF") from exc


def _extract_rtf_text(data: bytes) -> str:
    text = _decode_text_bytes(data)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\d* ?", " ", text)
    text = re.sub(r"[{}]", "", text)
    return html.unescape(text)


def _extract_upload_payload(payload: dict[str, Any]) -> dict[str, Any]:
    filename = payload.get("filename")
    encoded = payload.get("content_base64")
    content_type = payload.get("content_type")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("content_base64 must be a non-empty base64 string")
    if content_type is not None and not isinstance(content_type, str):
        raise ValueError("content_type must be a string")

    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("content_base64 is not valid base64") from exc

    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"file is too large; upload a file up to {limit_mb} MB")

    extension = Path(filename).suffix.lower()
    mime = (content_type or "").lower()
    try:
        if extension in TEXT_FILE_EXTENSIONS or mime.startswith("text/"):
            text = _decode_text_bytes(data)
        elif extension == ".docx":
            text = _extract_docx_text(data)
        elif extension == ".odt":
            text = _extract_odt_text(data)
        elif extension == ".pdf":
            text = _extract_pdf_text(data)
        elif extension == ".rtf":
            text = _extract_rtf_text(data)
        else:
            supported = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
            raise ValueError(
                f"unsupported file type {extension or '(none)'}; use {supported}"
            )
    except ValueError:
        raise
    except (ElementTree.ParseError, KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not read {extension or 'uploaded'} file") from exc

    text = _normalize_extracted_text(text)
    if not text:
        raise ValueError("no readable text was extracted from the uploaded file")

    return {
        "filename": filename,
        "content_type": content_type or "application/octet-stream",
        "size_bytes": len(data),
        "text": text,
    }


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: object) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    body: str,
    *,
    content_type: str,
) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _page() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OPF Local Demo</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-soft: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --border: #374151;
      --accent: #60a5fa;
      --accent-dark: #2563eb;
      --danger: #f87171;
      --ok: #34d399;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, var(--bg), #111827 45%, #020617);
      color: var(--text);
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 36px 0;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ margin-bottom: 8px; font-size: clamp(30px, 6vw, 52px); line-height: 1; }}
    h2 {{ margin-bottom: 14px; font-size: 18px; }}
    h3 {{ margin-bottom: 8px; font-size: 15px; }}
    p {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 20px;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel), transparent 5%);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
      padding: 20px;
    }}
    label {{
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    textarea, select, input, button {{
      border-radius: 12px;
      border: 1px solid var(--border);
      font: inherit;
    }}
    textarea, select, input {{
      width: 100%;
      background: #020617;
      color: var(--text);
      padding: 12px;
    }}
    textarea {{
      min-height: 210px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    select {{ cursor: pointer; }}
    button {{
      cursor: pointer;
      border-color: transparent;
      background: var(--accent-dark);
      color: white;
      padding: 11px 14px;
      font-weight: 700;
    }}
    button.secondary {{
      background: var(--panel-soft);
      border-color: var(--border);
      color: var(--text);
      text-align: left;
    }}
    button:disabled {{ cursor: wait; opacity: 0.7; }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin: 16px 0;
    }}
    .actions {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 14px;
    }}
    .file-upload {{
      display: grid;
      gap: 8px;
      margin-top: 14px;
      padding: 12px;
      border: 1px dashed var(--border);
      border-radius: 14px;
      background: rgba(2, 6, 23, 0.45);
    }}
    .file-upload p {{ margin-bottom: 0; font-size: 13px; }}
    .examples {{
      display: grid;
      gap: 10px;
    }}
    .example-id {{ display: block; color: var(--accent); font-size: 12px; }}
    .example-text {{
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      color: var(--muted);
      font-size: 13px;
    }}
    .output {{
      min-height: 92px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
    }}
    .output.empty {{ color: var(--muted); }}
    .spans {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    .span {{
      display: grid;
      grid-template-columns: 130px minmax(0, 1fr);
      gap: 10px;
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
    }}
    .label {{
      color: var(--ok);
      font-weight: 750;
      overflow-wrap: anywhere;
    }}
    .muted {{ color: var(--muted); }}
    .status {{ min-height: 22px; color: var(--muted); }}
    .status.error {{ color: var(--danger); }}
    .status.ok {{ color: var(--ok); }}
    details {{ margin-top: 18px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 700; }}
    pre {{
      max-height: 360px;
      overflow: auto;
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      color: var(--text);
    }}
    .note {{
      border-left: 3px solid var(--accent);
      padding: 10px 12px;
      background: rgba(96, 165, 250, 0.08);
      border-radius: 10px;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      header {{ display: block; }}
      .grid {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>OPF Local Demo</h1>
        <p>Test Privacy Filter redaction locally with free text, uploaded documents, or bundled synthetic examples.</p>
      </div>
      <div class="muted">Serving from <code>{html.escape(str(ROOT))}</code></div>
    </header>

    <div class="grid">
      <section class="panel">
        <h2>Input</h2>
        <label for="input-text">Free text</label>
        <textarea id="input-text" spellcheck="false">Alice was born on 1990-01-02 and can be reached at alice@example.com.</textarea>

        <div class="file-upload">
          <label for="file-input">Upload document</label>
          <input id="file-input" type="file" accept="{UPLOAD_ACCEPT}">
          <p class="muted">Supports text, Markdown, CSV/JSON, source files, PDF, Word .docx, OpenDocument .odt, and RTF up to 15 MB.</p>
        </div>

        <div class="controls">
          <div>
            <label for="device">Device</label>
            <select id="device">
              <option value="cpu" selected>cpu</option>
              <option value="cuda">cuda</option>
            </select>
          </div>
          <div>
            <label for="output-mode">Output mode</label>
            <select id="output-mode">
              <option value="typed" selected>typed</option>
              <option value="redacted">redacted</option>
            </select>
          </div>
          <div>
            <label for="eval-mode">Eval mode</label>
            <select id="eval-mode">
              <option value="typed" selected>typed</option>
              <option value="untyped">untyped</option>
            </select>
          </div>
          <div>
            <label for="decode-mode">Decode mode</label>
            <select id="decode-mode">
              <option value="viterbi" selected>viterbi</option>
              <option value="argmax">argmax</option>
            </select>
          </div>
        </div>

        <p id="mode-note" class="note"></p>

        <div class="actions">
          <button id="run-button">Run Redaction</button>
          <span id="status" class="status"></span>
        </div>
      </section>

      <aside class="panel">
        <h2>Examples</h2>
        <div id="examples" class="examples"></div>
      </aside>
    </div>

    <section class="panel" style="margin-top: 20px;">
      <h2>Output</h2>
      <label>Redacted text</label>
      <div id="redacted-output" class="output empty">Run the model to see output here.</div>

      <h3 style="margin-top: 18px;">Detected spans</h3>
      <div id="spans" class="spans muted">No spans yet.</div>

      <details>
        <summary>Raw JSON</summary>
        <pre id="raw-json">{{}}</pre>
      </details>
    </section>
  </main>

  <script>
    const state = {{ samples: [] }};
    const inputText = document.getElementById("input-text");
    const fileInput = document.getElementById("file-input");
    const device = document.getElementById("device");
    const outputMode = document.getElementById("output-mode");
    const evalMode = document.getElementById("eval-mode");
    const decodeMode = document.getElementById("decode-mode");
    const modeNote = document.getElementById("mode-note");
    const statusEl = document.getElementById("status");
    const runButton = document.getElementById("run-button");
    const redactedOutput = document.getElementById("redacted-output");
    const spansEl = document.getElementById("spans");
    const rawJson = document.getElementById("raw-json");
    const examplesEl = document.getElementById("examples");

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function updateModeNote() {{
      const outputText = outputMode.value === "typed"
        ? "Typed output keeps model labels such as private_person or private_date."
        : "Redacted output collapses every detected span label to redacted.";
      const evalText = evalMode.value === "typed"
        ? "Typed eval is for labeled datasets that already use the OPF taxonomy."
        : "Untyped eval ignores label identity and measures span detection against another taxonomy.";
      modeNote.textContent = `${{outputText}} ${{evalText}} Free-text testing uses output mode; eval mode is shown here so you can compare the documented modes while testing examples.`;
    }}

    function setStatus(message, kind = "") {{
      statusEl.textContent = message;
      statusEl.className = `status ${{kind}}`;
    }}

    function renderExamples(samples) {{
      examplesEl.innerHTML = "";
      if (!samples.length) {{
        examplesEl.innerHTML = '<p class="muted">No sample inputs found.</p>';
        return;
      }}
      for (const sample of samples) {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "secondary";
        button.innerHTML = `<span class="example-id">${{escapeHtml(sample.id)}}</span><span class="example-text">${{escapeHtml(sample.text)}}</span>`;
        button.addEventListener("click", () => {{
          inputText.value = sample.text;
          inputText.focus();
        }});
        examplesEl.appendChild(button);
      }}
    }}

    function readFileAsBase64(file) {{
      return new Promise((resolve, reject) => {{
        const reader = new FileReader();
        reader.addEventListener("load", () => {{
          const result = String(reader.result || "");
          resolve(result.includes(",") ? result.split(",", 2)[1] : result);
        }});
        reader.addEventListener("error", () => reject(reader.error || new Error("Could not read file.")));
        reader.readAsDataURL(file);
      }});
    }}

    async function uploadFile(file) {{
      if (!file) return;
      if (file.size > {MAX_UPLOAD_BYTES}) {{
        setStatus("File is too large. Upload a file up to 15 MB.", "error");
        fileInput.value = "";
        return;
      }}

      setStatus(`Extracting text from ${{file.name}}...`);
      fileInput.disabled = true;
      try {{
        const contentBase64 = await readFileAsBase64(file);
        const response = await fetch("/api/extract-file", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            filename: file.name,
            content_type: file.type,
            content_base64: contentBase64
          }})
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || "File extraction failed.");
        }}
        inputText.value = data.text;
        inputText.focus();
        setStatus(`Loaded ${{data.filename}} (${{data.text.length.toLocaleString()}} characters).`, "ok");
      }} catch (error) {{
        setStatus(error.message || String(error), "error");
      }} finally {{
        fileInput.disabled = false;
        fileInput.value = "";
      }}
    }}

    function renderResult(result) {{
      redactedOutput.textContent = result.redacted_text || "";
      redactedOutput.classList.toggle("empty", !result.redacted_text);

      const spans = result.detected_spans || [];
      if (!spans.length) {{
        spansEl.className = "spans muted";
        spansEl.textContent = "No spans detected.";
      }} else {{
        spansEl.className = "spans";
        spansEl.innerHTML = spans.map((span) => `
          <div class="span">
            <div>
              <div class="label">${{escapeHtml(span.label)}}</div>
              <div class="muted">${{span.start}}-${{span.end}}</div>
            </div>
            <div>
              <div>${{escapeHtml(span.text)}}</div>
              <div class="muted">${{escapeHtml(span.placeholder)}}</div>
            </div>
          </div>
        `).join("");
      }}
      rawJson.textContent = JSON.stringify(result, null, 2);
    }}

    async function loadOptions() {{
      const response = await fetch("/api/options");
      if (!response.ok) throw new Error("Could not load demo options.");
      const data = await response.json();
      state.samples = data.samples || [];
      renderExamples(state.samples);
    }}

    async function runRedaction() {{
      setStatus("Running model...");
      runButton.disabled = true;
      try {{
        const response = await fetch("/api/redact", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            text: inputText.value,
            device: device.value,
            output_mode: outputMode.value,
            eval_mode: evalMode.value,
            decode_mode: decodeMode.value
          }})
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.error || "Redaction failed.");
        }}
        renderResult(data);
        const latency = data.demo && data.demo.latency_ms ? ` in ${{data.demo.latency_ms}} ms` : "";
        setStatus(`Done${{latency}}`, "ok");
      }} catch (error) {{
        setStatus(error.message || String(error), "error");
      }} finally {{
        runButton.disabled = false;
      }}
    }}

    for (const element of [outputMode, evalMode]) {{
      element.addEventListener("change", updateModeNote);
    }}
    fileInput.addEventListener("change", () => uploadFile(fileInput.files && fileInput.files[0]));
    runButton.addEventListener("click", runRedaction);
    updateModeNote();
    loadOptions().catch((error) => setStatus(error.message || String(error), "error"));
  </script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "OPFDemo/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print(
            "%s - - [%s] %s"
            % (self.client_address[0], self.log_date_time_string(), format % args),
            file=sys.stderr,
        )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            _text_response(
                self,
                HTTPStatus.OK,
                _page(),
                content_type="text/html; charset=utf-8",
            )
            return
        if path == "/api/health":
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return
        if path == "/api/options":
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "devices": DEVICES,
                    "output_modes": OUTPUT_MODES,
                    "eval_modes": EVAL_MODES,
                    "decode_modes": DECODE_MODES,
                    "samples": _read_samples(),
                },
            )
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/redact", "/api/extract-file"}:
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            if path == "/api/extract-file":
                result = _extract_upload_payload(payload)
            else:
                result = _redact_payload(payload)
        except ValueError as exc:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            _json_response(
                self,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": str(exc),
                    "hint": (
                        "Check that the OPF package is installed, the checkpoint is "
                        "available, and the selected device can run the model."
                    ),
                },
            )
            return

        _json_response(self, HTTPStatus.OK, result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local OPF frontend demo.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"OPF demo running at {url}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
