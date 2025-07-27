/**
 * Dialogue Tree Generator Web UI
 * JavaScript functionality for the interactive web interface
 */

class DialogueTreeApp {
    constructor() {
        this.currentTree = null;
        this.selectedNodeId = null;
        this.isGenerating = false;
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadAppStatus();
        await this.loadTree();
    }

    setupEventListeners() {
        // Save button
        document.getElementById('saveButton').addEventListener('click', () => {
            this.saveTree();
        });

        // Refresh button
        document.getElementById('refreshButton').addEventListener('click', () => {
            this.loadTree();
        });
    }

    async loadAppStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            const statusText = document.getElementById('statusText');
            
            if (status.status === 'ready') {
                statusText.textContent = `✅ Ready (${status.llm_model})`;
                statusText.className = 'status-ready';
            } else {
                statusText.textContent = `⚠️ Not Ready`;
                statusText.className = 'status-not-ready';
                
                let issues = [];
                if (!status.tree_file_exists) {
                    issues.push('Tree file not found');
                }
                if (!status.llm_available) {
                    issues.push('LLM not available');
                }
                
                this.showToast(`Issues: ${issues.join(', ')}`, 'warning');
            }
        } catch (error) {
            console.error('Error loading status:', error);
            this.showToast('Failed to load application status', 'error');
        }
    }

    async loadTree() {
        try {
            this.showLoading('Loading dialogue tree...');
            
            const response = await fetch('/api/tree');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.currentTree = data;
            
            this.updateTreeStats(data);
            this.renderTreeNavigator(data.tree.nodes);
            
            // Enable save button if tree is loaded
            document.getElementById('saveButton').disabled = false;
            
            this.showToast('Dialogue tree loaded successfully', 'success');
            
        } catch (error) {
            console.error('Error loading tree:', error);
            this.showToast(`Failed to load tree: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    updateTreeStats(data) {
        document.getElementById('totalNodes').textContent = data.total_nodes;
        document.getElementById('completedNodes').textContent = data.completed_nodes;
        document.getElementById('nullNodes').textContent = data.null_nodes.length;
    }

    renderTreeNavigator(nodes) {
        const navigator = document.getElementById('treeNavigator');
        navigator.innerHTML = '';

        // Sort nodes to show completed ones first, then null ones
        const sortedNodes = Object.entries(nodes).sort(([idA, dataA], [idB, dataB]) => {
            // Completed nodes first
            if (dataA !== null && dataB === null) return -1;
            if (dataA === null && dataB !== null) return 1;
            
            // Then sort by ID
            return idA.localeCompare(idB);
        });

        sortedNodes.forEach(([nodeId, nodeData]) => {
            const nodeElement = this.createTreeNodeElement(nodeId, nodeData);
            navigator.appendChild(nodeElement);
        });
    }

    createTreeNodeElement(nodeId, nodeData) {
        const element = document.createElement('div');
        element.className = `tree-node ${nodeData === null ? 'null-node' : ''}`;
        element.dataset.nodeId = nodeId;
        
        const isNull = nodeData === null;
        const statusText = isNull ? '⚪ Null' : '✅ Complete';
        
        element.innerHTML = `
            <div class="node-id">${nodeId}</div>
            <div class="node-status">${statusText}</div>
        `;
        
        element.addEventListener('click', () => {
            this.selectNode(nodeId);
        });
        
        return element;
    }

    async selectNode(nodeId) {
        // Update visual selection
        document.querySelectorAll('.tree-node').forEach(el => {
            el.classList.remove('active');
        });
        
        const selectedElement = document.querySelector(`[data-node-id="${nodeId}"]`);
        if (selectedElement) {
            selectedElement.classList.add('active');
        }
        
        this.selectedNodeId = nodeId;
        
        // Load node details
        try {
            const response = await fetch(`/api/node/${nodeId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const nodeInfo = await response.json();
            this.renderNodeViewer(nodeInfo);
            
        } catch (error) {
            console.error('Error loading node:', error);
            this.showToast(`Failed to load node: ${error.message}`, 'error');
        }
    }

    renderNodeViewer(nodeInfo) {
        const viewer = document.getElementById('nodeViewer');
        
        if (nodeInfo.is_null) {
            this.renderNullNodeViewer(viewer, nodeInfo);
        } else {
            this.renderCompleteNodeViewer(viewer, nodeInfo);
        }
    }

    renderNullNodeViewer(viewer, nodeInfo) {
        const parentInfo = nodeInfo.parent_info;
        let parentHtml = '';
        
        if (parentInfo) {
            const [parentId, choice] = parentInfo;
            parentHtml = `
                <div class="parent-info">
                    <h5>📍 Context</h5>
                    <p><strong>Parent Node:</strong> ${parentId}</p>
                    <p><strong>Choice Leading Here:</strong> "${choice.text}"</p>
                </div>
            `;
        }
        
        viewer.innerHTML = `
            <div class="node-content">
                <div class="node-header">
                    <div class="node-title">
                        <h2>${nodeInfo.node_id}</h2>
                        <span class="node-badge null">⚪ Null Node</span>
                    </div>
                </div>
                
                <div class="null-node-content">
                    <h3>🎯 Node Ready for Generation</h3>
                    <p>This node is waiting for AI-generated content. Click the button below to generate a new dialogue situation and choices.</p>
                    
                    ${parentHtml}
                    
                    <div class="generate-section">
                        <h4>🤖 Generate Content</h4>
                        <p>Use AI to create a new dialogue situation and choices for this node.</p>
                        <button class="btn btn-success btn-large" onclick="app.generateNode('${nodeInfo.node_id}')">
                            ✨ Generate Node Content
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    renderCompleteNodeViewer(viewer, nodeInfo) {
        const nodeData = nodeInfo.data;
        
        let choicesHtml = '';
        if (nodeData.choices && nodeData.choices.length > 0) {
            choicesHtml = nodeData.choices.map(choice => {
                let effectsHtml = '';
                if (choice.effects && Object.keys(choice.effects).length > 0) {
                    effectsHtml = Object.entries(choice.effects)
                        .map(([key, value]) => {
                            const sign = value >= 0 ? '+' : '';
                            return `<span class="effect-tag">${key}: ${sign}${value}</span>`;
                        })
                        .join('');
                }
                
                const nextNode = choice.next || 'End';
                
                return `
                    <div class="choice-item">
                        <div class="choice-text">"${choice.text}"</div>
                        <div class="choice-meta">
                            <span>→ ${nextNode}</span>
                            <div class="choice-effects">${effectsHtml}</div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        viewer.innerHTML = `
            <div class="node-content">
                <div class="node-header">
                    <div class="node-title">
                        <h2>${nodeInfo.node_id}</h2>
                        <span class="node-badge completed">✅ Complete</span>
                    </div>
                </div>
                
                <div class="situation-section">
                    <h3>📖 Situation</h3>
                    <div class="situation-text">${nodeData.situation}</div>
                </div>
                
                <div class="choices-section">
                    <h3>🎯 Player Choices</h3>
                    ${choicesHtml || '<p>No choices available.</p>'}
                </div>
            </div>
        `;
    }

    async generateNode(nodeId) {
        if (this.isGenerating) {
            this.showToast('Already generating content, please wait...', 'warning');
            return;
        }
        
        try {
            this.isGenerating = true;
            this.showLoading('Generating content with AI...');
            
            const response = await fetch(`/api/generate/${nodeId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }
            
            const result = await response.json();
            
            this.showToast('Node content generated successfully!', 'success');
            
            // Reload the tree to show updated data
            await this.loadTree();
            
            // Re-select the node to show the new content
            this.selectNode(nodeId);
            
        } catch (error) {
            console.error('Error generating node:', error);
            this.showToast(`Failed to generate node: ${error.message}`, 'error');
        } finally {
            this.isGenerating = false;
            this.hideLoading();
        }
    }

    async saveTree() {
        try {
            this.showLoading('Saving dialogue tree...');
            
            const response = await fetch('/api/tree', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }
            
            const result = await response.json();
            this.showToast('Dialogue tree saved successfully!', 'success');
            
        } catch (error) {
            console.error('Error saving tree:', error);
            this.showToast(`Failed to save tree: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    showLoading(message = 'Loading...') {
        const overlay = document.getElementById('loadingOverlay');
        const spinner = overlay.querySelector('.loading-spinner p');
        spinner.textContent = message;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = 'none';
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>${this.getToastIcon(type)}</span>
                <span>${message}</span>
            </div>
        `;
        
        container.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
        
        // Allow manual close on click
        toast.addEventListener('click', () => {
            toast.remove();
        });
    }

    getToastIcon(type) {
        switch (type) {
            case 'success': return '✅';
            case 'error': return '❌';
            case 'warning': return '⚠️';
            default: return 'ℹ️';
        }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DialogueTreeApp();
});