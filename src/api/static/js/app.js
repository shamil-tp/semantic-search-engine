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

    // Handle search submission
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
