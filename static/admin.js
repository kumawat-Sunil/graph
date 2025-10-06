// Simplified Admin Portal JavaScript
// Handle both server and file:// protocol
const API_BASE_URL = window.location.protocol === 'file:' 
    ? 'http://localhost:8000' 
    : window.location.origin;

// Admin credentials
const ADMIN_CREDENTIALS = {
    username: 'Sunil',
    password: 'Test0663'
};

// Global variables
let currentDatabase = null;
let databaseStats = null;
let isAuthenticated = false;

// Initialize admin portal
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Simplified Admin Portal');
    
    // Always show login modal first
    showLoginModal();
    
    // Setup login form
    setupLoginForm();
});

// Show login modal
function showLoginModal() {
    const loginModal = document.getElementById('adminLoginModal');
    const adminContent = document.getElementById('adminContent');
    
    loginModal.style.display = 'flex';
    adminContent.style.display = 'none';
    
    // Focus on username field
    setTimeout(() => {
        document.getElementById('adminUsername').focus();
    }, 100);
}

// Setup login form
function setupLoginForm() {
    const loginForm = document.getElementById('adminLoginForm');
    const loginError = document.getElementById('loginError');
    
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const username = document.getElementById('adminUsername').value;
        const password = document.getElementById('adminPassword').value;
        
        // Validate credentials
        if (username === ADMIN_CREDENTIALS.username && password === ADMIN_CREDENTIALS.password) {
            // Login successful
            isAuthenticated = true;
            hideLoginModal();
            initializeAdminPortal();
        } else {
            // Login failed
            loginError.style.display = 'flex';
            document.getElementById('adminPassword').value = '';
            document.getElementById('adminPassword').focus();
            
            // Hide error after 3 seconds
            setTimeout(() => {
                loginError.style.display = 'none';
            }, 3000);
        }
    });
    
    // Handle Enter key on password field
    document.getElementById('adminPassword').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            loginForm.dispatchEvent(new Event('submit'));
        }
    });
}

// Hide login modal and show admin content
function hideLoginModal() {
    const loginModal = document.getElementById('adminLoginModal');
    const adminContent = document.getElementById('adminContent');
    
    loginModal.style.display = 'none';
    adminContent.style.display = 'block';
}

// Initialize admin portal after successful login
function initializeAdminPortal() {
    console.log('Admin authenticated successfully');
    
    // Show access note if opened as file://
    if (window.location.protocol === 'file:') {
        const accessNote = document.getElementById('accessNote');
        if (accessNote) {
            accessNote.style.display = 'flex';
        }
    }
    
    checkSystemStatus();
    loadDatabaseStats();
}

// Check if user is authenticated (for page refreshes)
function checkAuthentication() {
    // Always require login on page load/refresh
    return false;
}

// Logout function
function logout() {
    isAuthenticated = false;
    
    // Clear form fields
    document.getElementById('adminUsername').value = '';
    document.getElementById('adminPassword').value = '';
    
    // Show login modal again
    showLoginModal();
}

// Load database statistics
async function loadDatabaseStats() {
    try {
        showLoading('Loading database statistics...');
        
        const response = await fetch(`${API_BASE_URL}/admin/stats`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const stats = await response.json();
        databaseStats = stats;
        updateDatabaseCards(stats);
        
    } catch (error) {
        console.error('Error loading database stats:', error);
        
        // Show helpful error message based on the error type
        if (error.message.includes('Failed to fetch')) {
            if (window.location.protocol === 'file:') {
                showError('Please access the admin portal via http://localhost:8000/admin for full functionality');
            } else {
                showError('API server is not running. Please start the server with: python start_api.py');
            }
        } else {
            showError('Failed to load database statistics: ' + error.message);
        }
        
        // Show placeholder data when API is not available
        updateDatabaseCardsOffline();
    } finally {
        hideLoading();
    }
}

function updateDatabaseCards(stats) {
    const neo4jStats = stats.find(s => s.name.includes('Neo4j'));
    const pineconeStats = stats.find(s => s.name.includes('Pinecone'));
    
    // Update Neo4j card
    const neo4jStatsDiv = document.getElementById('neo4j-stats');
    if (neo4jStats) {
        neo4jStatsDiv.innerHTML = `
            <div class="stat-item">📄 ${neo4jStats.total_documents} documents</div>
            <div class="stat-item">🏷️ ${neo4jStats.total_entities} entities</div>
            <div class="stat-item">🔗 ${neo4jStats.total_relationships} relationships</div>
        `;
    } else {
        neo4jStatsDiv.innerHTML = '<div class="stat-error">Connection failed</div>';
    }
    
    // Update Pinecone card
    const pineconeStatsDiv = document.getElementById('pinecone-stats');
    if (pineconeStats) {
        pineconeStatsDiv.innerHTML = `
            <div class="stat-item">🔍 ${pineconeStats.total_embeddings} vectors</div>
            <div class="stat-item">📊 ${pineconeStats.size_info.dimensions || 384} dimensions</div>
        `;
    } else {
        pineconeStatsDiv.innerHTML = '<div class="stat-error">Connection failed</div>';
    }
}

function updateDatabaseCardsOffline() {
    // Show offline status when API is not available
    const neo4jStatsDiv = document.getElementById('neo4j-stats');
    neo4jStatsDiv.innerHTML = '<div class="stat-error">API not available</div>';
    
    const pineconeStatsDiv = document.getElementById('pinecone-stats');
    pineconeStatsDiv.innerHTML = '<div class="stat-error">API not available</div>';
}

// Show database details
async function showDatabase(dbType) {
    currentDatabase = dbType;
    
    try {
        showLoading(`Loading ${dbType} database details...`);
        
        // Hide database selection and show details
        document.querySelector('.database-selection').style.display = 'none';
        document.getElementById('database-details').style.display = 'block';
        
        // Update title
        const title = dbType === 'neo4j' ? 'Neo4j Graph Database' : 'Pinecone Vector Database';
        document.getElementById('details-title').textContent = title;
        
        // Load database-specific details
        if (dbType === 'neo4j') {
            await loadNeo4jDetails();
        } else {
            await loadPineconeDetails();
        }
        
    } catch (error) {
        console.error(`Error loading ${dbType} details:`, error);
        showError(`Failed to load ${dbType} database details`);
    } finally {
        hideLoading();
    }
}

// Load Neo4j details
async function loadNeo4jDetails() {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/database/neo4j`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const details = await response.json();
        displayNeo4jDetails(details);
        
    } catch (error) {
        console.error('Error loading Neo4j details:', error);
        document.getElementById('details-content').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                Failed to load Neo4j details: ${error.message}
            </div>
        `;
    }
}

function displayNeo4jDetails(details) {
    let html = `
        <div class="database-overview">
            <h4>Database Overview</h4>
            <div class="overview-stats">
                <div class="overview-stat">
                    <span class="stat-label">Documents:</span>
                    <span class="stat-value">${details.documents?.[0]?.count || 0}</span>
                </div>
                <div class="overview-stat">
                    <span class="stat-label">Entities:</span>
                    <span class="stat-value">${details.entities?.[0]?.count || 0}</span>
                </div>
                <div class="overview-stat">
                    <span class="stat-label">Relationships:</span>
                    <span class="stat-value">${details.relationships?.[0]?.count || 0}</span>
                </div>
            </div>
        </div>
    `;
    
    // Document types
    if (details.document_types && details.document_types.length > 0) {
        html += `
            <div class="detail-section">
                <h4>Document Types</h4>
                <div class="type-list">
        `;
        details.document_types.forEach(type => {
            html += `
                <div class="type-item">
                    <span class="type-name">${type.type || 'Unknown'}</span>
                    <span class="type-count">${type.count}</span>
                </div>
            `;
        });
        html += '</div></div>';
    }
    
    // Entity types
    if (details.entity_types && details.entity_types.length > 0) {
        html += `
            <div class="detail-section">
                <h4>Entity Types</h4>
                <div class="type-list">
        `;
        details.entity_types.forEach(type => {
            html += `
                <div class="type-item">
                    <span class="type-name">${type.type || 'Unknown'}</span>
                    <span class="type-count">${type.count}</span>
                </div>
            `;
        });
        html += '</div></div>';
    }
    
    // Recent documents
    if (details.recent_documents && details.recent_documents.length > 0) {
        html += `
            <div class="detail-section">
                <h4>Recent Documents</h4>
                <div class="document-list">
        `;
        details.recent_documents.forEach(doc => {
            html += `
                <div class="document-item">
                    <div class="doc-title">${doc.title || 'Untitled'}</div>
                    <div class="doc-date">${formatDate(doc.created_at)}</div>
                </div>
            `;
        });
        html += '</div></div>';
    }
    
    document.getElementById('details-content').innerHTML = html;
}

// Load Pinecone details
async function loadPineconeDetails() {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/database/pinecone`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const details = await response.json();
        displayPineconeDetails(details);
        
    } catch (error) {
        console.error('Error loading Pinecone details:', error);
        document.getElementById('details-content').innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                Failed to load Pinecone details: ${error.message}
            </div>
        `;
    }
}

function displayPineconeDetails(details) {
    let html = `
        <div class="database-overview">
            <h4>Collection Overview</h4>
            <div class="overview-stats">
                <div class="overview-stat">
                    <span class="stat-label">Collection:</span>
                    <span class="stat-value">${details.collection_name || 'documents'}</span>
                </div>
                <div class="overview-stat">
                    <span class="stat-label">Dimensions:</span>
                    <span class="stat-value">${details.vector_dimensions || 384}</span>
                </div>
                <div class="overview-stat">
                    <span class="stat-label">Sample Results:</span>
                    <span class="stat-value">${details.sample_results_count || 0}</span>
                </div>
            </div>
        </div>
    `;
    
    // Metadata analysis
    if (details.metadata_analysis) {
        const analysis = details.metadata_analysis;
        
        html += `
            <div class="detail-section">
                <h4>Metadata Analysis</h4>
                <div class="overview-stats">
                    <div class="overview-stat">
                        <span class="stat-label">Unique Documents:</span>
                        <span class="stat-value">${analysis.unique_documents || 0}</span>
                    </div>
                </div>
        `;
        
        // Domains
        if (analysis.domains && Object.keys(analysis.domains).length > 0) {
            html += `
                <h5>Domains</h5>
                <div class="type-list">
            `;
            Object.entries(analysis.domains).forEach(([domain, count]) => {
                html += `
                    <div class="type-item">
                        <span class="type-name">${domain}</span>
                        <span class="type-count">${count}</span>
                    </div>
                `;
            });
            html += '</div>';
        }
        
        // Sources
        if (analysis.sources && Object.keys(analysis.sources).length > 0) {
            html += `
                <h5>Top Sources</h5>
                <div class="type-list">
            `;
            Object.entries(analysis.sources).forEach(([source, count]) => {
                html += `
                    <div class="type-item">
                        <span class="type-name">${source.substring(0, 40)}...</span>
                        <span class="type-count">${count}</span>
                    </div>
                `;
            });
            html += '</div>';
        }
        
        html += '</div>';
    }
    
    // Sample similarities
    if (details.sample_similarities && details.sample_similarities.length > 0) {
        html += `
            <div class="detail-section">
                <h4>Sample Similarity Scores</h4>
                <div class="similarity-list">
        `;
        details.sample_similarities.forEach((score, index) => {
            const percentage = ((1 - score) * 100).toFixed(1);
            html += `
                <div class="similarity-item">
                    <span class="similarity-label">Result ${index + 1}</span>
                    <div class="similarity-bar">
                        <div class="similarity-fill" style="width: ${percentage}%"></div>
                    </div>
                    <span class="similarity-value">${percentage}%</span>
                </div>
            `;
        });
        html += '</div></div>';
    }
    
    document.getElementById('details-content').innerHTML = html;
}

// Hide database details
function hideDetails() {
    document.querySelector('.database-selection').style.display = 'grid';
    document.getElementById('database-details').style.display = 'none';
    currentDatabase = null;
}

// Confirm and clear database
async function confirmClearDatabase() {
    if (!currentDatabase) {
        showError('No database selected');
        return;
    }
    
    const dbName = currentDatabase === 'neo4j' ? 'Neo4j Graph Database' : 'Pinecone Vector Database';
    
    if (!confirm(`Are you sure you want to clear the ${dbName}? This will permanently delete all data and cannot be undone.`)) {
        return;
    }
    
    // Second confirmation
    const confirmText = prompt(`Type "CLEAR ${currentDatabase.toUpperCase()}" to confirm this destructive action:`);
    if (confirmText !== `CLEAR ${currentDatabase.toUpperCase()}`) {
        showError('Confirmation text did not match. Operation cancelled.');
        return;
    }
    
    try {
        showLoading(`Clearing ${dbName}...`);
        
        const endpoint = currentDatabase === 'neo4j' ? '/admin/clear-neo4j' : '/admin/clear-vectors';
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        showSuccess(`${dbName} cleared successfully`);
        
        // Refresh the current view
        if (currentDatabase === 'neo4j') {
            await loadNeo4jDetails();
        } else {
            await loadPineconeDetails();
        }
        
        // Refresh database stats
        await loadDatabaseStats();
        
    } catch (error) {
        console.error(`Error clearing ${currentDatabase}:`, error);
        showError(`Failed to clear ${dbName}: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Navigation Functions (simplified)
function showSection(sectionId) {
    // Only dashboard section exists now
    console.log('Dashboard section active');
}

// System Status Functions
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

// Utility Functions
function formatDate(dateString) {
    if (!dateString) return 'Unknown';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch (error) {
        return dateString;
    }
}

// Loading and message functions
function showLoading(message = 'Loading...') {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    
    loadingText.textContent = message;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
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
    const errorToast = document.getElementById('errorToast');
    errorToast.style.display = 'none';
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
    const successToast = document.getElementById('successToast');
    successToast.style.display = 'none';
}