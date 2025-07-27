// Dialogue Tree Web UI JavaScript

class DialogueTreeApp {
    constructor() {
        this.currentNodeId = 'start';
        this.navigationHistory = [];
        this.treeData = {};
        this.treeStructure = {};
        this.gameParams = {};
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        await this.loadTree();
        await this.loadTreeStructure();
        this.renderTreeNavigation();
        this.navigateToNode(this.currentNodeId);
    }
    
    setupEventListeners() {
        // Refresh tree button
        document.getElementById('refreshTree').addEventListener('click', () => {
            this.refreshTree();
        });
        
        // Save tree button
        document.getElementById('saveTree').addEventListener('click', () => {
            this.saveTree();
        });
        
        // AI generation modal
        document.getElementById('generateAIContent').addEventListener('click', () => {
            this.generateAIContent();
        });
        
        // Modal close buttons
        document.querySelectorAll('[data-dismiss="modal"]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                if (modal) hideModal(modal.id);
            });
        });
    }
    
    async loadTree() {
        try {
            const response = await fetch('/api/tree');
            if (!response.ok) {
                throw new Error('Failed to load tree');
            }
            
            this.treeData = await response.json();
            this.gameParams = this.treeData.params || {};
            this.renderGameParams();
            
        } catch (error) {
            console.error('Error loading tree:', error);
            this.showError('Failed to load dialogue tree');
        }
    }
    
    async loadTreeStructure() {
        try {
            const response = await fetch('/api/tree/structure');
            if (!response.ok) {
                throw new Error('Failed to load tree structure');
            }
            
            this.treeStructure = await response.json();
            
        } catch (error) {
            console.error('Error loading tree structure:', error);
            this.showError('Failed to load tree structure');
        }
    }
    
    renderTreeNavigation() {
        const container = document.getElementById('treeStructure');
        container.innerHTML = '';
        
        // Create tree nodes
        for (const [nodeId, nodeData] of Object.entries(this.treeStructure)) {
            const nodeElement = this.createTreeNode(nodeId, nodeData);
            container.appendChild(nodeElement);
        }
    }
    
    createTreeNode(nodeId, nodeData) {
        const nodeDiv = document.createElement('div');
        nodeDiv.className = `tree-node ${nodeData.is_null ? 'null-node' : ''}`;
        nodeDiv.dataset.nodeId = nodeId;
        
        const nodeIdSpan = document.createElement('div');
        nodeIdSpan.className = 'tree-node-id';
        nodeIdSpan.textContent = nodeId;
        
        const nodeText = document.createElement('div');
        nodeText.className = 'tree-node-text';
        nodeText.textContent = nodeData.situation;
        
        nodeDiv.appendChild(nodeIdSpan);
        nodeDiv.appendChild(nodeText);
        
        // Add generate button for null nodes
        if (nodeData.is_null) {
            const generateBtn = document.createElement('button');
            generateBtn.className = 'btn btn-sm btn-warning generate-btn';
            generateBtn.innerHTML = '<i class="bi bi-magic"></i> Generate';
            generateBtn.onclick = (e) => {
                e.stopPropagation();
                this.showAIGenerationModal(nodeId);
            };
            nodeDiv.appendChild(generateBtn);
        }
        
        // Make node clickable
        nodeDiv.addEventListener('click', () => {
            this.navigateToNode(nodeId);
        });
        
        return nodeDiv;
    }
    
    async navigateToNode(nodeId) {
        if (this.treeStructure[nodeId]?.is_null) {
            this.showError('Cannot navigate to incomplete node. Generate content first.');
            return;
        }
        
        try {
            const response = await fetch(`/api/node/${nodeId}`);
            if (!response.ok) {
                throw new Error('Failed to load node');
            }
            
            const nodeData = await response.json();
            
            // Update current node
            this.currentNodeId = nodeId;
            document.getElementById('currentNodeId').textContent = nodeId;
            
            // Update UI
            this.renderCurrentSituation(nodeData);
            this.renderChoices(nodeData.choices);
            this.updateTreeNavigationHighlight();
            
            // Add to history
            this.addToHistory(nodeId, nodeData.situation);
            
        } catch (error) {
            console.error('Error navigating to node:', error);
            this.showError(`Failed to navigate to node: ${nodeId}`);
        }
    }
    
    renderCurrentSituation(nodeData) {
        const container = document.getElementById('currentSituation');
        container.textContent = nodeData.situation;
        container.classList.add('fade-in');
    }
    
    renderChoices(choices) {
        const container = document.getElementById('choices');
        container.innerHTML = '';
        
        if (!choices || choices.length === 0) {
            container.innerHTML = '<p class="text-muted">No choices available.</p>';
            return;
        }
        
        choices.forEach((choice, index) => {
            const choiceElement = this.createChoiceElement(choice, index);
            container.appendChild(choiceElement);
        });
    }
    
    createChoiceElement(choice, index) {
        const choiceDiv = document.createElement('div');
        const isNull = !choice.next || this.treeStructure[choice.next]?.is_null;
        choiceDiv.className = `choice-item ${isNull ? 'null-choice' : ''}`;
        
        const choiceText = document.createElement('div');
        choiceText.className = 'choice-text';
        choiceText.textContent = `${index + 1}. ${choice.text}`;
        
        const choiceNext = document.createElement('div');
        choiceNext.className = 'choice-next';
        choiceNext.textContent = choice.next ? `→ ${choice.next}` : '→ incomplete';
        
        choiceDiv.appendChild(choiceText);
        choiceDiv.appendChild(choiceNext);
        
        // Add effects if present
        if (choice.effects && Object.keys(choice.effects).length > 0) {
            const effectsDiv = document.createElement('div');
            effectsDiv.className = 'choice-effects';
            const effectsText = Object.entries(choice.effects)
                .map(([key, value]) => `${key}: ${value > 0 ? '+' : ''}${value}`)
                .join(', ');
            effectsDiv.textContent = `Effects: ${effectsText}`;
            choiceDiv.appendChild(effectsDiv);
        }
        
        // Make choice clickable
        choiceDiv.addEventListener('click', () => {
            if (choice.next) {
                if (this.treeStructure[choice.next]?.is_null) {
                    this.showAIGenerationModal(choice.next);
                } else {
                    this.navigateToNode(choice.next);
                }
            }
        });
        
        return choiceDiv;
    }
    
    updateTreeNavigationHighlight() {
        // Remove previous highlights
        document.querySelectorAll('.tree-node.active').forEach(node => {
            node.classList.remove('active');
        });
        
        // Highlight current node
        const currentNode = document.querySelector(`[data-node-id="${this.currentNodeId}"]`);
        if (currentNode) {
            currentNode.classList.add('active');
            currentNode.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
    
    addToHistory(nodeId, situation) {
        this.navigationHistory.push({ nodeId, situation });
        this.renderNavigationHistory();
    }
    
    renderNavigationHistory() {
        const container = document.getElementById('navigationHistory');
        container.innerHTML = '';
        
        this.navigationHistory.slice(-5).forEach((item, index) => {
            const historyDiv = document.createElement('div');
            historyDiv.className = 'history-item';
            
            const shortSituation = item.situation.length > 80 
                ? item.situation.substring(0, 80) + '...' 
                : item.situation;
                
            historyDiv.innerHTML = `
                <strong>${item.nodeId}</strong>
                <span class="history-arrow">→</span>
                ${shortSituation}
            `;
            
            container.appendChild(historyDiv);
        });
    }
    
    renderGameParams() {
        const container = document.getElementById('gameParams');
        container.innerHTML = '';
        
        for (const [paramName, paramValue] of Object.entries(this.gameParams)) {
            const col = document.createElement('div');
            col.className = 'col-md-2 col-sm-4 col-6 mb-2';
            
            col.innerHTML = `
                <div class="param-item">
                    <div class="param-name">${paramName}</div>
                    <div class="param-value">${paramValue}</div>
                </div>
            `;
            
            container.appendChild(col);
        }
    }
    
    showAIGenerationModal(nodeId) {
        this.pendingGenerationNodeId = nodeId;
        showModal('aiGenerationModal');
    }
    
    async generateAIContent() {
        const nodeId = this.pendingGenerationNodeId;
        if (!nodeId) return;
        
        const statusDiv = document.getElementById('generationStatus');
        const generateBtn = document.getElementById('generateAIContent');
        
        // Show loading state
        statusDiv.classList.remove('d-none');
        generateBtn.disabled = true;
        
        try {
            const response = await fetch(`/api/generate/${nodeId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Generation failed');
            }
            
            const result = await response.json();
            
            // Update tree structure and navigation
            await this.refreshTree();
            this.showSuccess('AI content generated successfully!');
            
            // Close modal
            hideModal('aiGenerationModal');
            
            // Navigate to the newly generated node
            this.navigateToNode(nodeId);
            
        } catch (error) {
            console.error('Error generating AI content:', error);
            this.showError(`Failed to generate content: ${error.message}`);
        } finally {
            statusDiv.classList.add('d-none');
            generateBtn.disabled = false;
        }
    }
    
    async refreshTree() {
        await this.loadTree();
        await this.loadTreeStructure();
        this.renderTreeNavigation();
        this.renderGameParams();
        this.showSuccess('Tree refreshed successfully!');
    }
    
    async saveTree() {
        try {
            const response = await fetch('/api/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to save tree');
            }
            
            this.showSuccess('Tree saved successfully!');
            
        } catch (error) {
            console.error('Error saving tree:', error);
            this.showError('Failed to save tree');
        }
    }
    
    showError(message) {
        this.showMessage(message, 'error');
    }
    
    showSuccess(message) {
        this.showMessage(message, 'success');
    }
    
    showMessage(message, type) {
        // Create a toast-like notification
        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 250px;';
        
        toast.innerHTML = `
            ${message}
            <button type="button" class="btn-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DialogueTreeApp();
});