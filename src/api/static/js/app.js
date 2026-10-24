/**
 * Semantic Search Engine - Vanilla JavaScript Client
 */

document.addEventListener("DOMContentLoaded", () => {
    const searchForm = document.getElementById("search-form");
    const searchInput = document.getElementById("search-input");
    const searchButton = document.getElementById("search-button");
    const topKSelect = document.getElementById("top-k-select");
    const extensionFilter = document.getElementById("extension-filter");

    const loadingState = document.getElementById("loading-state");
    const errorMessage = document.getElementById("error-message");
    const searchSummary = document.getElementById("search-summary");
    const emptyState = document.getElementById("empty-state");
    const resultsList = document.getElementById("results-list");

    // Tabs switching
    const tabGlobal = document.getElementById("tab-global");
    const tabUpload = document.getElementById("tab-upload");
    const globalView = document.getElementById("global-view");
    const uploadView = document.getElementById("upload-view");

    tabGlobal.addEventListener("click", () => {
        tabGlobal.classList.add("active");
        tabUpload.classList.remove("active");
        globalView.hidden = false;
        uploadView.hidden = true;
        clearMessages();
    });

    tabUpload.addEventListener("click", () => {
        tabUpload.classList.add("active");
        tabGlobal.classList.remove("active");
        uploadView.hidden = false;
        globalView.hidden = true;
        clearMessages();
    });

    // File Upload handling
    const uploadForm = document.getElementById("upload-form");
    const docFileInput = document.getElementById("doc-file-input");
    const selectedFileInfo = document.getElementById("selected-file-info");
    const uploadBtn = document.getElementById("upload-btn");
    const docSearchContainer = document.getElementById("doc-search-container");
    const activeDocName = document.getElementById("active-doc-name");
    const activeDocChunks = document.getElementById("active-doc-chunks");

    const docSearchForm = document.getElementById("doc-search-form");
    const docSearchInput = document.getElementById("doc-search-input");
    const docSearchBtn = document.getElementById("doc-search-btn");
    const docSearchMode = document.getElementById("doc-search-mode");
    const docTopK = document.getElementById("doc-top-k");

    let currentUploadSessionId = null;

    docFileInput.addEventListener("change", () => {
        if (docFileInput.files && docFileInput.files.length > 0) {
            const file = docFileInput.files[0];
            selectedFileInfo.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            selectedFileInfo.hidden = false;
            uploadBtn.disabled = false;
        } else {
            selectedFileInfo.hidden = true;
            uploadBtn.disabled = true;
        }
    });

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!docFileInput.files || docFileInput.files.length === 0) return;

        const file = docFileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);

        uploadBtn.disabled = true;
        setLoading(true);
        clearMessages();

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();

            if (!response.ok) {
                const err = (data && data.error && data.error.message) || "Upload failed.";
                showError(err);
                return;
            }

            currentUploadSessionId = data.session_id;
            activeDocName.textContent = data.filename;
            activeDocChunks.textContent = `${data.total_chunks} chunks`;
            docSearchContainer.hidden = false;

            searchSummary.textContent = data.message;
            searchSummary.hidden = false;
            emptyState.innerHTML = `
                <div class="empty-state-icon">🧠</div>
                <h2>Document Ready for Semantic Search</h2>
                <p>Type any question or concept into the search box above to retrieve relevant passages from "${escapeHtml(data.filename)}".</p>
            `;
            emptyState.hidden = false;
            resultsList.hidden = true;
        } catch (err) {
            showError("Network error while uploading document.");
        } finally {
            uploadBtn.disabled = false;
            setLoading(false);
        }
    });

    // In-doc search form handling
    docSearchForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!currentUploadSessionId) {
            showError("No document is currently active. Please upload a file first.");
            return;
        }

        const query = docSearchInput.value.trim();
        if (!query) return;

        const topK = parseInt(docTopK.value, 10) || 5;
        const mode = docSearchMode.value || "semantic";

        docSearchBtn.disabled = true;
        setLoading(true);
        clearMessages();

        try {
            const response = await fetch("/api/search-doc", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: currentUploadSessionId,
                    query: query,
                    top_k: topK,
                    mode: mode,
                }),
            });
            const data = await response.json();

            if (!response.ok) {
                const err = (data && data.error && data.error.message) || "Search within document failed.";
                showError(err);
                return;
            }

            renderResults(data);
        } catch (err) {
            showError("Network error searching inside document.");
        } finally {
            docSearchBtn.disabled = false;
            setLoading(false);
        }
    });
    searchForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const query = searchInput.value.trim();
        if (!query) return;

        const topK = parseInt(topKSelect.value, 10) || 5;
        const extension = extensionFilter.value;

        // Build payload matching existing backend contract
        const payload = {
            query: query,
            top_k: topK,
        };

        if (extension) {
            payload.filters = { extension: extension };
        }

        // Set UI to loading state
        setLoading(true);
        clearMessages();

        try {
            const response = await fetch("/api/search", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                const errorText = (data && data.error && data.error.message) || `Search failed with status ${response.status}`;
                showError(errorText);
                return;
            }

            renderResults(data);
        } catch (err) {
            showError("Network or server connection error. Please verify the service is running.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        searchButton.disabled = isLoading;
        searchInput.disabled = isLoading;
        topKSelect.disabled = isLoading;
        extensionFilter.disabled = isLoading;

        if (isLoading) {
            loadingState.hidden = false;
            emptyState.hidden = true;
            resultsList.hidden = true;
        } else {
            loadingState.hidden = true;
        }
    }

    function clearMessages() {
        errorMessage.hidden = true;
        errorMessage.textContent = "";
        searchSummary.hidden = true;
        searchSummary.textContent = "";
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.hidden = false;
        resultsList.hidden = true;
        emptyState.hidden = false;
    }

    function renderResults(data) {
        const results = data.results || [];
        const count = results.length;
        const latency = data.latency_ms !== undefined ? `${data.latency_ms} ms` : "";

        searchSummary.textContent = `Found ${count} result${count === 1 ? "" : "s"}${latency ? ` in ${latency}` : ""}.`;
        searchSummary.hidden = false;

        if (count === 0) {
            emptyState.innerHTML = `
                <div class="empty-state-icon">🔎</div>
                <h2>No results found</h2>
                <p>No matching documents found for "${escapeHtml(data.query)}". Try broader search terms or clear extension filters.</p>
            `;
            emptyState.hidden = false;
            resultsList.hidden = true;
            return;
        }

        emptyState.hidden = true;
        resultsList.innerHTML = "";

        results.forEach((item) => {
            const card = document.createElement("article");
            card.className = "result-card";

            const fileName = (item.metadata && item.metadata.file_name) || item.chunk_id;
            const scoreLabel = typeof item.score === "number" ? `Score: ${item.score.toFixed(4)}` : "Rank: #" + item.rank;

            card.innerHTML = `
                <div class="result-header">
                    <span class="result-rank-title">#${item.rank} ${escapeHtml(fileName)}</span>
                    <span class="result-score">${escapeHtml(scoreLabel)}</span>
                </div>
                <div class="result-text">${escapeHtml(item.text)}</div>
                <div class="result-meta">
                    <span class="result-meta-item"><strong>Document ID:</strong> ${escapeHtml(item.document_id)}</span>
                    <span class="result-meta-item"><strong>Chunk:</strong> ${item.chunk_index}</span>
                    ${item.metadata && item.metadata.extension ? `<span class="result-meta-item"><strong>Type:</strong> ${escapeHtml(item.metadata.extension)}</span>` : ""}
                </div>
            `;
            resultsList.appendChild(card);
        });

        resultsList.hidden = false;
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
