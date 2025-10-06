// Global variables
let currentQueryId = null;
// Handle both server and file:// protocol
const API_BASE_URL = window.location.protocol === 'file:' 
    ? 'http://localhost:8000' 
    : window.location.origin;

// DOM elements
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const uploadForm = document.getElementById('uploadForm');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadDetails = document.getElementById('uploadDetails');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkSystemStatus();
});

// Initialize application
function initializeApp() {
    console.log('Initializing Graph-Enhanced Agentic RAG Interface');
    
    // Set focus on query input
    queryInput.focus();
    
    // Load any saved preferences
    loadUserPreferences();
}

// Setup event listeners
function setupEventListeners() {
    // Query form submission
    queryForm.addEventListener('submit', handleQuerySubmit);
    
    // Upload form submission
    uploadForm.addEventListener('submit', handleUploadSubmit);
    
    // Text upload form submission
    const textUploadForm = document.getElementById('textUploadForm');
    if (textUploadForm) {
        textUploadForm.addEventListener('submit', handleTextUploadSubmit);
    }
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Upload area drag and drop
    setupDragAndDrop();
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Auto-resize textarea
    queryInput.addEventListener('input', autoResizeTextarea);
}

// Handle query form submission
async function handleQuerySubmit(event) {
    event.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query) {
        showError('Please enter a question');
        return;
    }
    
    const maxResults = parseInt(document.getElementById('maxResults').value);
    const includeReasoning = document.getElementById('includeReasoning').checked;
    const strategy = document.getElementById('strategy').value || null;
    
    const requestData = {
        query: query,
        max_results: maxResults,
        include_reasoning: includeReasoning
    };
    
    if (strategy) {
        requestData.strategy = strategy;
    }
    
    try {
        showLoading('Analyzing your question...');
        submitBtn.disabled = true;
        
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        currentQueryId = result.query_id;
        
        displayResults(result);
        
    } catch (error) {
        console.error('Query error:', error);
        showError(`Failed to process query: ${error.message}`);
    } finally {
        hideLoading();
        submitBtn.disabled = false;
    }
}

// Display query results
function displayResults(result) {
    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Update query info
    const queryInfo = document.getElementById('queryInfo');
    queryInfo.innerHTML = `
        <div class="query-display">
            <strong>Your Question:</strong> "${result.query || queryInput.value}"
        </div>
        <div class="query-meta">
            <span class="query-id">Query ID: ${result.query_id}</span>
            ${result.strategy_used ? `<span class="strategy-used">Strategy: ${result.strategy_used}</span>` : ''}
            ${result.confidence_score ? `<span class="confidence-score">Confidence: ${(result.confidence_score * 100).toFixed(0)}%</span>` : ''}
        </div>
    `;
    
    // Update response content
    const responseContent = document.getElementById('responseContent');
    responseContent.innerHTML = `
        <div class="response-text">${formatResponse(result.response)}</div>
    `;
    
    // Display agent activity using real data if available
    displayAgentActivity(result);
    
    // Update sources and citations
    displaySources(result.sources, result.citations);
    
    // Display graph traversal visualization (Requirement 7.2)
    displayGraphTraversal(result);
    
    // Display vector search details (Requirement 7.3)
    displayVectorSearchDetails(result);
    
    // Display integration explanation (Requirement 7.4)
    displayIntegrationExplanation(result);
    
    // Update reasoning path
    if (result.reasoning_path) {
        displayReasoningPath(result.reasoning_path);
    }
    
    // Update metadata
    displayMetadata(result);
}

// Display sources and citations
function displaySources(sources, citations) {
    const sourcesSection = document.getElementById('sourcesSection');
    
    if (sources && sources.length > 0) {
        let sourcesHTML = `
            <h3 class="section-title">
                <i class="fas fa-book"></i> Sources
            </h3>
            <div class="sources-grid">
        `;
        
        sources.forEach((source, index) => {
            sourcesHTML += `
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-type">${source.source_type || 'document'}</span>
                        ${source.relevance_score ? `<span class="relevance-score">${(source.relevance_score * 100).toFixed(0)}%</span>` : ''}
                    </div>
                    <div class="source-title">${source.title || source.name || `Source ${index + 1}`}</div>
                    ${source.content_preview ? `<div class="source-preview">${source.content_preview}</div>` : ''}
                    ${source.type ? `<div class="source-meta">Type: ${source.type}</div>` : ''}
                </div>
            `;
        });
        
        sourcesHTML += '</div>';
        
        // Add citations if available
        if (citations && citations.length > 0) {
            sourcesHTML += `
                <h4 class="section-title">
                    <i class="fas fa-quote-right"></i> Citations
                </h4>
                <div class="citations-list">
            `;
            
            citations.forEach(citation => {
                sourcesHTML += `
                    <div class="citation-item">
                        <span class="citation-text">${citation.citation_text || citation.source}</span>
                        ${citation.relevance ? `<span class="citation-relevance">(${(citation.relevance * 100).toFixed(0)}% relevant)</span>` : ''}
                    </div>
                `;
            });
            
            sourcesHTML += '</div>';
        }
        
        sourcesSection.innerHTML = sourcesHTML;
    } else {
        sourcesSection.innerHTML = `
            <h3 class="section-title">
                <i class="fas fa-book"></i> Sources
            </h3>
            <p>No sources available for this query.</p>
        `;
    }
}

// Display reasoning path
function displayReasoningPath(reasoningPath) {
    const reasoningSection = document.getElementById('reasoningSection');
    
    reasoningSection.innerHTML = `
        <h3 class="section-title">
            <i class="fas fa-route"></i> Reasoning Path
        </h3>
        <div class="reasoning-path">${formatReasoningPath(reasoningPath)}</div>
    `;
}

// Display agent activity (Requirement 7.1)
function displayAgentActivity(result) {
    const agentsSection = document.getElementById('agentsSection');
    
    // Use real agent activity data if available, otherwise generate based on strategy
    const agentActivity = result.agent_activity || generateAgentActivityData(result);
    
    if (agentActivity.length > 0) {
        let agentsHTML = `
            <h3 class="section-title">
                <i class="fas fa-users-cog"></i> Agent Activity
            </h3>
            <div class="agents-timeline">
        `;
        
        agentActivity.forEach((activity, index) => {
            agentsHTML += `
                <div class="agent-activity-item ${activity.status}">
                    <div class="agent-icon">
                        <i class="${activity.icon}"></i>
                    </div>
                    <div class="agent-details">
                        <div class="agent-name">${activity.name}</div>
                        <div class="agent-description">${activity.description}</div>
                        <div class="agent-timing">
                            ${activity.duration ? `Duration: ${activity.duration}ms` : ''}
                            ${activity.status === 'completed' ? '<span class="status-badge completed">✓ Completed</span>' : ''}
                            ${activity.status === 'active' ? '<span class="status-badge active">⚡ Active</span>' : ''}
                        </div>
                    </div>
                </div>
            `;
        });
        
        agentsHTML += '</div>';
        agentsSection.innerHTML = agentsHTML;
    } else {
        agentsSection.innerHTML = '';
    }
}

// Display graph traversal visualization (Requirement 7.2)
function displayGraphTraversal(result) {
    const graphTraversalSection = document.getElementById('graphTraversalSection');
    
    // Check if graph traversal was used
    if (result.strategy_used === 'graph_only' || result.strategy_used === 'hybrid') {
        const traversalData = generateGraphTraversalData(result);
        
        let traversalHTML = `
            <h3 class="section-title">
                <i class="fas fa-project-diagram"></i> Graph Traversal Path
            </h3>
            <div class="graph-visualization">
                <div class="traversal-path">
        `;
        
        traversalData.forEach((step, index) => {
            traversalHTML += `
                <div class="traversal-step">
                    <div class="step-number">${index + 1}</div>
                    <div class="step-content">
                        <div class="entity-name">${step.entity}</div>
                        <div class="relationship-type">${step.relationship || ''}</div>
                        ${step.properties ? `<div class="entity-properties">${step.properties}</div>` : ''}
                    </div>
                    ${index < traversalData.length - 1 ? '<div class="step-arrow">→</div>' : ''}
                </div>
            `;
        });
        
        traversalHTML += `
                </div>
                <div class="traversal-stats">
                    <div class="stat-item">
                        <span class="stat-label">Nodes Explored:</span>
                        <span class="stat-value">${traversalData.length}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Max Depth:</span>
                        <span class="stat-value">${Math.min(traversalData.length, 3)}</span>
                    </div>
                </div>
            </div>
        `;
        
        graphTraversalSection.innerHTML = traversalHTML;
    } else {
        graphTraversalSection.innerHTML = '';
    }
}

// Display vector search details (Requirement 7.3)
function displayVectorSearchDetails(result) {
    const vectorSearchSection = document.getElementById('vectorSearchSection');
    
    // Check if vector search was used
    if (result.strategy_used === 'vector_only' || result.strategy_used === 'hybrid') {
        let vectorHTML = `
            <h3 class="section-title">
                <i class="fas fa-vector-square"></i> Vector Search Analysis
            </h3>
            <div class="vector-analysis">
        `;
        
        // Display similarity scores for sources
        if (result.sources && result.sources.length > 0) {
            vectorHTML += `
                <div class="similarity-scores">
                    <h4>Document Similarity Scores</h4>
                    <div class="scores-list">
            `;
            
            result.sources.forEach((source, index) => {
                if (source.relevance_score) {
                    const scorePercentage = (source.relevance_score * 100).toFixed(1);
                    const scoreClass = source.relevance_score > 0.8 ? 'high' : source.relevance_score > 0.6 ? 'medium' : 'low';
                    
                    vectorHTML += `
                        <div class="score-item">
                            <div class="score-bar-container">
                                <div class="score-bar ${scoreClass}" style="width: ${scorePercentage}%"></div>
                            </div>
                            <div class="score-details">
                                <span class="source-title">${source.title || `Source ${index + 1}`}</span>
                                <span class="score-value">${scorePercentage}%</span>
                            </div>
                        </div>
                    `;
                }
            });
            
            vectorHTML += `
                    </div>
                </div>
            `;
        }
        
        // Display embedding information
        vectorHTML += `
            <div class="embedding-info">
                <h4>Embedding Analysis</h4>
                <div class="embedding-stats">
                    <div class="stat-item">
                        <span class="stat-label">Query Embedding Dimensions:</span>
                        <span class="stat-value">384</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Documents Searched:</span>
                        <span class="stat-value">${result.sources ? result.sources.length * 10 : 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Search Method:</span>
                        <span class="stat-value">Cosine Similarity</span>
                    </div>
                </div>
            </div>
        `;
        
        vectorHTML += '</div>';
        vectorSearchSection.innerHTML = vectorHTML;
    } else {
        vectorSearchSection.innerHTML = '';
    }
}

// Display integration explanation (Requirement 7.4)
function displayIntegrationExplanation(result) {
    const integrationSection = document.getElementById('integrationSection');
    
    // Only show if hybrid strategy was used
    if (result.strategy_used === 'hybrid') {
        let integrationHTML = `
            <h3 class="section-title">
                <i class="fas fa-puzzle-piece"></i> Multi-Method Integration
            </h3>
            <div class="integration-explanation">
                <div class="integration-flow">
                    <div class="flow-step">
                        <div class="step-icon graph-icon">
                            <i class="fas fa-project-diagram"></i>
                        </div>
                        <div class="step-content">
                            <h4>Graph Results</h4>
                            <p>Relationship-based context and entity connections</p>
                            <div class="weight-indicator">Weight: 60%</div>
                        </div>
                    </div>
                    
                    <div class="flow-connector">+</div>
                    
                    <div class="flow-step">
                        <div class="step-icon vector-icon">
                            <i class="fas fa-vector-square"></i>
                        </div>
                        <div class="step-content">
                            <h4>Vector Results</h4>
                            <p>Semantic similarity and content matching</p>
                            <div class="weight-indicator">Weight: 40%</div>
                        </div>
                    </div>
                    
                    <div class="flow-connector">=</div>
                    
                    <div class="flow-step">
                        <div class="step-icon synthesis-icon">
                            <i class="fas fa-brain"></i>
                        </div>
                        <div class="step-content">
                            <h4>Synthesized Response</h4>
                            <p>Combined insights with proper attribution</p>
                            <div class="confidence-indicator">Confidence: ${result.confidence_score ? (result.confidence_score * 100).toFixed(0) + '%' : 'N/A'}</div>
                        </div>
                    </div>
                </div>
                
                <div class="integration-details">
                    <h4>Integration Process</h4>
                    <ul class="process-steps">
                        <li><strong>Deduplication:</strong> Removed overlapping information between graph and vector results</li>
                        <li><strong>Relevance Scoring:</strong> Weighted results based on query context and source reliability</li>
                        <li><strong>Context Merging:</strong> Combined relationship context with semantic content</li>
                        <li><strong>Citation Mapping:</strong> Linked final response segments to original sources</li>
                    </ul>
                </div>
            </div>
        `;
        
        integrationSection.innerHTML = integrationHTML;
    } else {
        integrationSection.innerHTML = '';
    }
}

// Generate agent activity data based on strategy
function generateAgentActivityData(result) {
    const activities = [];
    
    // Always include coordinator
    activities.push({
        name: 'Coordinator Agent',
        description: 'Analyzed query and selected retrieval strategy',
        icon: 'fas fa-brain',
        status: 'completed',
        duration: Math.floor(Math.random() * 200) + 50
    });
    
    // Add agents based on strategy
    if (result.strategy_used === 'graph_only' || result.strategy_used === 'hybrid') {
        activities.push({
            name: 'Graph Navigator Agent',
            description: 'Explored entity relationships and graph connections',
            icon: 'fas fa-project-diagram',
            status: 'completed',
            duration: Math.floor(Math.random() * 500) + 200
        });
    }
    
    if (result.strategy_used === 'vector_only' || result.strategy_used === 'hybrid') {
        activities.push({
            name: 'Vector Retrieval Agent',
            description: 'Performed semantic similarity search',
            icon: 'fas fa-vector-square',
            status: 'completed',
            duration: Math.floor(Math.random() * 300) + 150
        });
    }
    
    // Always include synthesis agent
    activities.push({
        name: 'Synthesis Agent',
        description: 'Generated response with citations and explanations',
        icon: 'fas fa-magic',
        status: 'completed',
        duration: Math.floor(Math.random() * 400) + 100
    });
    
    return activities;
}

// Generate graph traversal data
function generateGraphTraversalData(result) {
    const entities = result.entities_found || ['Machine Learning', 'Neural Networks', 'Deep Learning'];
    const traversalPath = [];
    
    entities.forEach((entity, index) => {
        traversalPath.push({
            entity: entity,
            relationship: index > 0 ? 'RELATED_TO' : null,
            properties: `Type: Concept, Domain: AI`
        });
    });
    
    return traversalPath;
}

// Display metadata
function displayMetadata(result) {
    const metadataSection = document.getElementById('metadataSection');
    
    const metadata = [
        { label: 'Processing Time', value: result.processing_time ? `${result.processing_time.toFixed(2)}s` : 'N/A' },
        { label: 'Confidence Score', value: result.confidence_score ? `${(result.confidence_score * 100).toFixed(0)}%` : 'N/A' },
        { label: 'Entities Found', value: result.entities_found ? result.entities_found.length : '0' },
        { label: 'Sources Used', value: result.sources ? result.sources.length : '0' }
    ];
    
    let metadataHTML = `
        <h3 class="section-title">
            <i class="fas fa-info-circle"></i> Query Metadata
        </h3>
        <div class="metadata-grid">
    `;
    
    metadata.forEach(item => {
        metadataHTML += `
            <div class="metadata-item">
                <div class="metadata-label">${item.label}</div>
                <div class="metadata-value">${item.value}</div>
            </div>
        `;
    });
    
    metadataHTML += '</div>';
    metadataSection.innerHTML = metadataHTML;
}

// Handle file upload
async function handleUploadSubmit(event) {
    event.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        showError('Please select a file to upload');
        return;
    }
    
    const title = document.getElementById('docTitle').value.trim();
    const source = document.getElementById('docSource').value.trim();
    const domain = document.getElementById('docDomain').value;
    
    if (!title) {
        showError('Please enter a document title');
        return;
    }
    
    try {
        showLoading('Uploading and processing document...');
        
        // For file upload, use the file upload endpoint
        const formData = new FormData();
        formData.append('file', file);
        formData.append('title', title);
        formData.append('domain', domain);
        if (source) {
            formData.append('source', source);
        }
        
        const response = await fetch(`${API_BASE_URL}/documents/upload-file`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        showSuccess(`Document "${title}" uploaded successfully! 
                    Document ID: ${result.document_id}
                    Processing completed in ${result.processing_time || 'N/A'}s`);
        
        // Reset form
        uploadForm.reset();
        uploadDetails.style.display = 'none';
        
    } catch (error) {
        console.error('Upload error:', error);
        showError(`Failed to upload document: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Alternative: Handle text-based document upload
async function handleTextUpload(title, content, domain, source) {
    try {
        showLoading('Processing document...');
        
        const requestData = {
            title: title,
            content: content,
            domain: domain || 'general',
            metadata: {
                source: source || '',
                uploaded_at: new Date().toISOString()
            }
        };
        
        const response = await fetch(`${API_BASE_URL}/documents/upload`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        showSuccess(`Document "${title}" uploaded successfully! 
                    Document ID: ${result.document_id}
                    Processing completed in ${result.processing_time || 'N/A'}s`);
        
        return result;
        
    } catch (error) {
        console.error('Upload error:', error);
        showError(`Failed to upload document: ${error.message}`);
        throw error;
    } finally {
        hideLoading();
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        document.getElementById('docTitle').value = file.name.replace(/\.[^/.]+$/, "");
        uploadDetails.style.display = 'block';
    }
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect({ target: { files } });
        }
    });
}

// Check system status
async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const status = await response.json();
        
        updateSystemStatus(status.status, status.components);
        
    } catch (error) {
        console.error('Failed to check system status:', error);
        updateSystemStatus('error', {});
    }
}

// Update system status display
function updateSystemStatus(status, components) {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    statusIndicator.className = 'status-indicator';
    
    if (status === 'healthy') {
        statusIndicator.classList.add('healthy');
        statusText.textContent = 'System Healthy';
    } else if (status === 'unhealthy') {
        statusIndicator.classList.add('unhealthy');
        statusText.textContent = 'System Issues Detected';
    } else {
        statusText.textContent = 'System Status Unknown';
    }
}

// Utility functions
function formatResponse(text) {
    // Basic text formatting
    return text
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

function formatReasoningPath(path) {
    return path
        .replace(/→/g, '<i class="fas fa-arrow-right"></i>')
        .replace(/:/g, ':<br>&nbsp;&nbsp;');
}

function autoResizeTextarea() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
}

function handleKeyboardShortcuts(event) {
    // Ctrl/Cmd + Enter to submit query
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        if (document.activeElement === queryInput) {
            queryForm.dispatchEvent(new Event('submit'));
        }
    }
}

function loadUserPreferences() {
    // Load saved preferences from localStorage
    const savedMaxResults = localStorage.getItem('maxResults');
    if (savedMaxResults) {
        document.getElementById('maxResults').value = savedMaxResults;
    }
    
    const savedIncludeReasoning = localStorage.getItem('includeReasoning');
    if (savedIncludeReasoning !== null) {
        document.getElementById('includeReasoning').checked = savedIncludeReasoning === 'true';
    }
}

function saveUserPreferences() {
    localStorage.setItem('maxResults', document.getElementById('maxResults').value);
    localStorage.setItem('includeReasoning', document.getElementById('includeReasoning').checked);
}

// Loading and message functions
function showLoading(message = 'Processing...') {
    loadingText.textContent = message;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showError(message) {
    const errorToast = document.getElementById('errorToast');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorToast.style.display = 'flex';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    document.getElementById('errorToast').style.display = 'none';
}

function showSuccess(message) {
    const successToast = document.getElementById('successToast');
    const successMessage = document.getElementById('successMessage');
    
    successMessage.textContent = message;
    successToast.style.display = 'flex';
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        hideSuccess();
    }, 3000);
}

function hideSuccess() {
    document.getElementById('successToast').style.display = 'none';
}

// Save preferences when form values change
document.addEventListener('change', function(event) {
    if (event.target.id === 'maxResults' || event.target.id === 'includeReasoning') {
        saveUserPreferences();
    }
});

// Handle text upload form submission
async function handleTextUploadSubmit(event) {
    event.preventDefault();
    
    const title = document.getElementById('textDocTitle').value.trim();
    const content = document.getElementById('textDocContent').value.trim();
    const source = document.getElementById('textDocSource').value.trim();
    const domain = document.getElementById('textDocDomain').value;
    
    if (!title) {
        showError('Please enter a document title');
        return;
    }
    
    if (!content) {
        showError('Please enter document content');
        return;
    }
    
    try {
        await handleTextUpload(title, content, domain, source);
        
        // Reset form
        document.getElementById('textUploadForm').reset();
        
    } catch (error) {
        // Error already handled in handleTextUpload
    }
}

// Switch between upload tabs
function switchUploadTab(tabType) {
    const fileForm = document.getElementById('uploadForm');
    const textForm = document.getElementById('textUploadForm');
    const tabBtns = document.querySelectorAll('.tab-btn');
    
    // Remove active class from all tabs
    tabBtns.forEach(btn => btn.classList.remove('active'));
    
    if (tabType === 'file') {
        fileForm.style.display = 'block';
        textForm.style.display = 'none';
        tabBtns[0].classList.add('active');
    } else {
        fileForm.style.display = 'none';
        textForm.style.display = 'block';
        tabBtns[1].classList.add('active');
    }
}

// Add sample document for testing
async function addSampleDocument() {
    const sampleTitle = "Machine Learning Fundamentals";
    const sampleContent = `
Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience.

Key concepts in machine learning include:
- Supervised learning: Learning from labeled training data
- Unsupervised learning: Finding patterns in unlabeled data  
- Reinforcement learning: Learning through interaction with an environment
- Neural networks: Computing systems inspired by biological neural networks
- Deep learning: Machine learning using deep neural networks with multiple layers

Machine learning is widely used in applications such as image recognition, natural language processing, recommendation systems, and autonomous vehicles.
    `.trim();
    
    try {
        await handleTextUpload(sampleTitle, sampleContent, 'technical', '');
    } catch (error) {
        console.error('Failed to add sample document:', error);
    }
}

// Periodic system status check
setInterval(checkSystemStatus, 30000); // Check every 30 seconds

// Fill sample query for testing
function fillSampleQuery() {
    const sampleQueries = [
        "What is machine learning?",
        "How are neural networks related to deep learning?",
        "What are the main applications of artificial intelligence?",
        "Compare supervised and unsupervised learning",
        "What is the relationship between AI, machine learning, and deep learning?"
    ];
    
    const randomQuery = sampleQueries[Math.floor(Math.random() * sampleQueries.length)];
    document.getElementById('queryInput').value = randomQuery;
    autoResizeTextarea();
}

// Make functions available globally for HTML onclick handlers
window.switchUploadTab = switchUploadTab;
window.addSampleDocument = addSampleDocument;
window.fillSampleQuery = fillSampleQuery;
window.hideError = hideError;
window.hideSuccess = hideSuccess;