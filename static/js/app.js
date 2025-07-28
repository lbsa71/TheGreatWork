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
        
        // Edit node button
        document.getElementById('editNodeBtn').addEventListener('click', () => {
            this.showEditNodeModal();
        });
        
        // AI generation modal
        document.getElementById('generateAIContent').addEventListener('click', () => {
            this.generateAIContent();
        });
        
        // Save node changes button
        document.getElementById('saveNodeChanges').addEventListener('click', () => {
            this.saveNodeChanges();
        });
        
        // Add choice button
        document.getElementById('addChoiceBtn').addEventListener('click', () => {
            this.addChoiceToEditor();
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
            
            // Show/hide edit button for complete nodes only
            const editBtn = document.getElementById('editNodeBtn');
            if (this.treeStructure[nodeId] && !this.treeStructure[nodeId].is_null) {
                editBtn.style.display = 'inline-block';
            } else {
                editBtn.style.display = 'none';
            }
            
            // Load and render situation history
            await this.loadAndRenderSituationHistory(nodeId);
            
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
    
    async loadAndRenderSituationHistory(nodeId) {
        try {
            const response = await fetch(`/api/history/${nodeId}`);
            if (!response.ok) {
                throw new Error('Failed to load history');
            }
            
            const historyData = await response.json();
            this.renderSituationHistory(historyData.history);
            
        } catch (error) {
            console.error('Error loading situation history:', error);
            // Hide history container if we can't load it
            document.getElementById('situationHistoryContainer').style.display = 'none';
        }
    }
    
    renderSituationHistory(historySteps) {
        const container = document.getElementById('situationHistory');
        const historyContainer = document.getElementById('situationHistoryContainer');
        
        // If no history, hide the container
        if (!historySteps || historySteps.length === 0) {
            historyContainer.style.display = 'none';
            return;
        }
        
        // Show the container and populate it
        historyContainer.style.display = 'block';
        container.innerHTML = '';
        
        historySteps.forEach((step, index) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'history-step mb-3';
            
            const stepNumber = index + 1;
            let stepHtml = `
                <div class="d-flex align-items-start">
                    <div class="step-number me-3">
                        <span class="badge bg-primary">${stepNumber}</span>
                    </div>
                    <div class="step-content flex-grow-1">
                        <div class="step-situation mb-2">
                            <strong>Situation:</strong> ${step.situation}
                        </div>
                        <div class="step-choice">
                            <strong>Choice made:</strong> 
                            <span class="choice-highlight">${step.choice_text}</span>
            `;
            
            // Add effects if present
            if (step.choice_effects && Object.keys(step.choice_effects).length > 0) {
                const effectsText = Object.entries(step.choice_effects)
                    .map(([key, value]) => `${key}: ${value > 0 ? '+' : ''}${value}`)
                    .join(', ');
                stepHtml += `
                    <div class="step-effects mt-1">
                        <small class="text-muted">Effects: ${effectsText}</small>
                    </div>
                `;
            }
            
            stepHtml += `
                        </div>
                    </div>
                </div>
            `;
            
            stepDiv.innerHTML = stepHtml;
            container.appendChild(stepDiv);
            
            // Add arrow connector except for the last step
            if (index < historySteps.length - 1) {
                const arrow = document.createElement('div');
                arrow.className = 'history-arrow text-center my-2';
                arrow.innerHTML = '<i class="bi bi-arrow-down text-muted"></i>';
                container.appendChild(arrow);
            }
        });
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
                let errorMessage = error.error || 'Generation failed';
                
                // Add additional details if available
                if (error.details) {
                    errorMessage += `\n\nDetails: ${error.details}`;
                }
                if (error.instructions) {
                    errorMessage += `\n\nTo fix this: ${error.instructions}`;
                }
                
                throw new Error(errorMessage);
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
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 250px; max-width: 400px; white-space: pre-line;';
        
        // Escape HTML but preserve line breaks
        const escapedMessage = message
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
        
        toast.innerHTML = `
            ${escapedMessage}
            <button type="button" class="btn-close" onclick="this.parentElement.remove()">×</button>
        `;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 8 seconds for longer messages
        const timeout = type === 'error' ? 8000 : 5000;
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, timeout);
    }
    
    async showEditNodeModal() {
        if (!this.currentNodeId) {
            this.showError('No node selected for editing');
            return;
        }
        
        try {
            const response = await fetch(`/api/node/${this.currentNodeId}`);
            if (!response.ok) {
                throw new Error('Failed to load node for editing');
            }
            
            const nodeData = await response.json();
            
            // Populate the edit form
            document.getElementById('editSituation').value = nodeData.situation || '';
            
            // Clear and populate choices
            const choicesContainer = document.getElementById('editChoices');
            choicesContainer.innerHTML = '';
            
            if (nodeData.choices && nodeData.choices.length > 0) {
                nodeData.choices.forEach((choice, index) => {
                    this.addChoiceToEditor(choice, index);
                });
            } else {
                // Add one empty choice if none exist
                this.addChoiceToEditor();
            }
            
            // Show the modal
            showModal('editNodeModal');
            
        } catch (error) {
            console.error('Error loading node for editing:', error);
            this.showError('Failed to load node for editing');
        }
    }
    
    addChoiceToEditor(choiceData = null, index = null) {
        const choicesContainer = document.getElementById('editChoices');
        const choiceIndex = index !== null ? index : choicesContainer.children.length;
        
        const choiceDiv = document.createElement('div');
        choiceDiv.className = 'edit-choice-item';
        choiceDiv.dataset.choiceIndex = choiceIndex;
        
        choiceDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <label class="form-label mb-0">Choice ${choiceIndex + 1}</label>
                <button type="button" class="btn btn-sm remove-choice-btn" onclick="this.closest('.edit-choice-item').remove()">
                    <i class="bi bi-trash"></i> Remove
                </button>
            </div>
            <div class="mb-2">
                <input type="text" class="form-control choice-text" placeholder="Choice text..." value="${choiceData?.text || ''}" />
            </div>
            <div class="row">
                <div class="col-md-6">
                    <label class="form-label">Next Node ID</label>
                    <input type="text" class="form-control choice-next" placeholder="next_node_id" value="${choiceData?.next || ''}" />
                </div>
                <div class="col-md-6">
                    <label class="form-label">Effects (JSON)</label>
                    <input type="text" class="form-control choice-effects" placeholder='{"param": 5}' value="${choiceData?.effects ? JSON.stringify(choiceData.effects) : ''}" />
                </div>
            </div>
        `;
        
        choicesContainer.appendChild(choiceDiv);
    }
    
    async saveNodeChanges() {
        if (!this.currentNodeId) {
            this.showError('No node selected for saving');
            return;
        }
        
        const statusDiv = document.getElementById('editStatus');
        const saveBtn = document.getElementById('saveNodeChanges');
        
        // Show loading state
        statusDiv.classList.remove('d-none');
        saveBtn.disabled = true;
        
        try {
            // Collect form data
            const situation = document.getElementById('editSituation').value;
            const choiceItems = document.querySelectorAll('.edit-choice-item');
            const choices = [];
            
            choiceItems.forEach((item) => {
                const text = item.querySelector('.choice-text').value;
                const next = item.querySelector('.choice-next').value;
                const effectsStr = item.querySelector('.choice-effects').value;
                
                if (text.trim()) {
                    const choice = { text: text.trim() };
                    
                    if (next.trim()) {
                        choice.next = next.trim();
                    }
                    
                    if (effectsStr.trim()) {
                        try {
                            choice.effects = JSON.parse(effectsStr);
                        } catch (e) {
                            throw new Error(`Invalid JSON in choice effects: ${effectsStr}`);
                        }
                    }
                    
                    choices.push(choice);
                }
            });
            
            if (!situation.trim()) {
                throw new Error('Situation text is required');
            }
            
            if (choices.length === 0) {
                throw new Error('At least one choice is required');
            }
            
            // Send update request
            const response = await fetch(`/api/node/${this.currentNodeId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    situation: situation.trim(),
                    choices: choices
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to save node');
            }
            
            // Success
            this.showSuccess('Node updated successfully!');
            hideModal('editNodeModal');
            
            // Refresh the UI
            await this.refreshTree();
            this.navigateToNode(this.currentNodeId);
            
        } catch (error) {
            console.error('Error saving node changes:', error);
            this.showError(`Failed to save changes: ${error.message}`);
        } finally {
            statusDiv.classList.add('d-none');
            saveBtn.disabled = false;
        }
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DialogueTreeApp();
});