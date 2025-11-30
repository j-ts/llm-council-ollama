import React, { useState, useEffect } from 'react';
import { api } from '../api';
import './SettingsModal.css';

function SettingsModal({ isOpen, onClose }) {
    const [config, setConfig] = useState(null);
    const [models, setModels] = useState({});
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [activeTab, setActiveTab] = useState('models'); // 'models', 'other', 'council'

    // Model editor state
    const [showModelEditor, setShowModelEditor] = useState(false);
    const [editingModel, setEditingModel] = useState(null); // null = new, otherwise model_id
    const [modelForm, setModelForm] = useState({
        label: '',
        type: 'ollama',
        model_name: '',
        base_url: 'http://localhost:11434',
        api_key: ''
    });
    const [apiKeyMode, setApiKeyMode] = useState('direct'); // 'direct' | 'env'
    const [envVarName, setEnvVarName] = useState('');
    const [draftModelId, setDraftModelId] = useState(null);

    const generalSettings = config?.general_settings || { use_env_for_api_keys: false };

    const generateEnvVarName = (idHint) => {
        const safeId = (idHint || 'model').toString().replace(/[^A-Za-z0-9]/g, '_').toUpperCase();
        return `MODEL_${safeId}_API_KEY`;
    };

    useEffect(() => {
        if (isOpen) {
            loadConfig();
        }
    }, [isOpen]);

    const loadConfig = async () => {
        setLoading(true);
        try {
            const data = await api.getConfig();
            const incomingGeneral = data.general_settings || { use_env_for_api_keys: false };
            setConfig({ ...data, general_settings: incomingGeneral });
            setModels(data.models || {});
        } catch (err) {
            console.error('Failed to load config:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            const updated = { ...config, models, general_settings: generalSettings };
            await api.updateConfig(updated);
            onClose();
        } catch (err) {
            console.error('Failed to save config:', err);
            alert('Failed to save settings');
        } finally {
            setSaving(false);
        }
    };

    const openModelEditor = (modelId = null) => {
        const targetId = modelId || `model_${Date.now()}`;
        const model = modelId ? models[modelId] : null;
        const apiKeyValue = model?.api_key || '';
        const isEnvReference = typeof apiKeyValue === 'string' && apiKeyValue.startsWith('env:');
        const inferredEnvName = isEnvReference ? apiKeyValue.slice(4).trim() : generateEnvVarName(targetId);
        const shouldUseEnvMode = isEnvReference || (!model && generalSettings.use_env_for_api_keys);

        setDraftModelId(targetId);
        setEditingModel(modelId);
        setApiKeyMode(shouldUseEnvMode ? 'env' : 'direct');
        setEnvVarName(inferredEnvName || generateEnvVarName(targetId));

        setModelForm({
            label: model?.label || '',
            type: model?.type || 'ollama',
            model_name: model?.model_name || '',
            base_url: model?.base_url || 'http://localhost:11434',
            api_key: isEnvReference ? '' : (apiKeyValue || '')
        });
        setShowModelEditor(true);
    };

    const closeModelEditor = () => {
        setShowModelEditor(false);
        setEditingModel(null);
        setApiKeyMode('direct');
        setEnvVarName('');
        setDraftModelId(null);
    };

    const saveModel = async () => {
        const modelId = editingModel || draftModelId || `model_${Date.now()}`;
        const requiresApiKey = modelForm.type === 'openrouter' || modelForm.type === 'openai-compatible';
        const usingEnvMode = requiresApiKey && apiKeyMode === 'env';

        // Validate
        if (!modelForm.label) {
            alert('Label is required');
            return;
        }
        if (!modelForm.model_name) {
            alert('Model name is required');
            return;
        }
        if (modelForm.type === 'ollama' && !modelForm.base_url) {
            alert('Base URL is required for Ollama models');
            return;
        }
        if (modelForm.type === 'openrouter') {
            if (usingEnvMode && !envVarName.trim()) {
                alert('Environment variable name is required for OpenRouter models');
                return;
            }
            if (!usingEnvMode && !modelForm.api_key) {
                alert('API Key is required for OpenRouter models');
                return;
            }
        }
        if (modelForm.type === 'openai-compatible') {
            if (!modelForm.base_url) {
                alert('Base URL is required for OpenAI-compatible models');
                return;
            }
            if (usingEnvMode && !envVarName.trim()) {
                alert('Environment variable name is required for OpenAI-compatible models');
                return;
            }
            if (!usingEnvMode && !modelForm.api_key) {
                alert('API Key is required for OpenAI-compatible models');
                return;
            }
        }

        const apiKeyToSave = (() => {
            if (!requiresApiKey) {
                return '';
            }
            if (usingEnvMode) {
                return `env:${envVarName.trim()}`;
            }
            return modelForm.api_key;
        })();

        const newModels = { ...models };

        if (editingModel) {
            // Update existing
            newModels[editingModel] = {
                ...modelForm,
                api_key: apiKeyToSave
            };
        } else {
            // Add new
            newModels[modelId] = {
                ...modelForm,
                api_key: apiKeyToSave
            };
        }

        setModels(newModels);
        setConfig({ ...config, models: newModels });
        closeModelEditor();
    };

    const deleteModel = (modelId) => {
        // Check if in use
        const councilModels = config?.council_models || [];
        const chairmanModel = config?.chairman_model;

        if (councilModels.includes(modelId)) {
            alert('Cannot delete: Model is in use by council');
            return;
        }
        if (chairmanModel === modelId) {
            alert('Cannot delete: Model is in use as chairman');
            return;
        }

        if (!confirm(`Delete model "${models[modelId]?.label}"?`)) {
            return;
        }

        const newModels = { ...models };
        delete newModels[modelId];
        setModels(newModels);
        setConfig({ ...config, models: newModels });
    };

    const toggleCouncilModel = (modelId) => {
        const councilModels = config?.council_models || [];
        const newCouncilModels = councilModels.includes(modelId)
            ? councilModels.filter(id => id !== modelId)
            : [...councilModels, modelId];

        setConfig({ ...config, council_models: newCouncilModels });
    };

    const setChairmanModel = (modelId) => {
        setConfig({ ...config, chairman_model: modelId });
    };

    if (!isOpen) return null;
    if (loading) return <div className="settings-modal-overlay"><div className="settings-modal">Loading...</div></div>;

    const ollamaSettings = config?.ollama_settings || { num_ctx: 4096, serialize_requests: false };
    const councilModels = config?.council_models || [];
    const chairmanModel = config?.chairman_model;

    return (
        <div className="settings-modal-overlay" onClick={onClose}>
            <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
                <div className="settings-header">
                    <h2>Settings</h2>
                    <button className="close-button" onClick={onClose}>×</button>
                </div>

                {/* Tabs */}
                <div className="settings-tabs">
                    <button
                        className={activeTab === 'models' ? 'active' : ''}
                        onClick={() => setActiveTab('models')}
                    >
                        Models
                    </button>
                    <button
                        className={activeTab === 'other' ? 'active' : ''}
                        onClick={() => setActiveTab('other')}
                    >
                        Other Settings
                    </button>
                    <button
                        className={activeTab === 'council' ? 'active' : ''}
                        onClick={() => setActiveTab('council')}
                    >
                        Council Configuration
                    </button>
                </div>

                <div className="settings-content">
                    {/* Models Tab */}
                    {activeTab === 'models' && (
                        <div className="models-section">
                            <div className="section-header">
                                <h3>Configured Models</h3>
                                <button className="add-button" onClick={() => openModelEditor()}>
                                    + Add Model
                                </button>
                            </div>

                            {Object.keys(models).length === 0 ? (
                                <p className="empty-message">No models configured. Click "Add Model" to get started.</p>
                            ) : (
                                <div className="models-list">
                                    {Object.entries(models).map(([modelId, model]) => (
                                        <div key={modelId} className="model-card">
                                            <div className="model-card-header">
                                                <span className="model-label">{model.label}</span>
                                                <span className={`model-type-badge ${model.type}`}>
                                                    {model.type === 'openai-compatible' ? 'OpenAI' :
                                                        model.type === 'openrouter' ? 'OpenRouter' : 'Ollama'}
                                                </span>
                                            </div>
                                            <div className="model-card-details">
                                                <div><strong>Model:</strong> {model.model_name}</div>
                                                {model.base_url && <div><strong>URL:</strong> {model.base_url}</div>}
                                                {model.api_key && (
                                                    <div>
                                                        <strong>API Key:</strong> {model.api_key.startsWith('env:') ? model.api_key : '*** (hidden)'}
                                                    </div>
                                                )}
                                            </div>
                                            <div className="model-card-actions">
                                                <button onClick={() => openModelEditor(modelId)}>Edit</button>
                                                <button onClick={() => deleteModel(modelId)} className="delete-btn">Delete</button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Ollama Settings Tab */}
                    {activeTab === 'other' && (
                        <div className="other-section">
                            <div className="general-section">
                                <h3>General</h3>
                                <label className="checkbox-label general-checkbox">
                                    <input
                                        type="checkbox"
                                        checked={generalSettings.use_env_for_api_keys}
                                        onChange={(e) => setConfig({
                                            ...config,
                                            general_settings: {
                                                ...generalSettings,
                                                use_env_for_api_keys: e.target.checked
                                            }
                                        })}
                                    />
                                    <span>Store API Key as ENV variable, not in json</span>
                                </label>
                                <p className="api-key-note">Keeps secrets in environment variables while preserving env: references in the UI.</p>
                            </div>

                            <div className="ollama-section">
                                <h3>Global Ollama Settings</h3>
                                <p className="section-help">These settings apply to all Ollama models.</p>

                                <div className="form-group">
                                    <label>Context Window (num_ctx)</label>
                                    <input
                                        type="number"
                                        min="512"
                                        step="512"
                                        value={ollamaSettings.num_ctx}
                                        onChange={(e) => setConfig({
                                            ...config,
                                            ollama_settings: {
                                                ...ollamaSettings,
                                                num_ctx: parseInt(e.target.value) || 4096
                                            }
                                        })}
                                    />
                                    <small>Context window size for Ollama models. Larger values require more VRAM.</small>
                                </div>

                                <div className="form-group">
                                    <label className="checkbox-label">
                                        <input
                                            type="checkbox"
                                            checked={ollamaSettings.serialize_requests}
                                            onChange={(e) => setConfig({
                                                ...config,
                                                ollama_settings: {
                                                    ...ollamaSettings,
                                                    serialize_requests: e.target.checked
                                                }
                                            })}
                                        />
                                        <span>Serialize Ollama Requests</span>
                                    </label>
                                    <small>Run Ollama models sequentially to avoid GPU thrashing.</small>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Council Configuration Tab */}
                    {activeTab === 'council' && (
                        <div className="council-section">
                            <h3>Council Models</h3>
                            <p className="section-help">Select which models participate in the council.</p>

                            {Object.keys(models).length === 0 ? (
                                <p className="empty-message">No models configured. Add models in the "Models" tab first.</p>
                            ) : (
                                <div className="model-selection-list">
                                    {Object.entries(models).map(([modelId, model]) => (
                                        <label key={modelId} className="model-checkbox">
                                            <input
                                                type="checkbox"
                                                checked={councilModels.includes(modelId)}
                                                onChange={() => toggleCouncilModel(modelId)}
                                            />
                                            <span>{model.label} ({model.model_name})</span>
                                        </label>
                                    ))}
                                </div>
                            )}

                            <hr />

                            <h3>Chairman Model</h3>
                            <p className="section-help">Select the model that synthesizes the final answer.</p>

                            {Object.keys(models).length === 0 ? (
                                <p className="empty-message">No models configured. Add models in the "Models" tab first.</p>
                            ) : (
                                <div className="chairman-selection">
                                    <select
                                        value={chairmanModel || ''}
                                        onChange={(e) => setChairmanModel(e.target.value)}
                                    >
                                        <option value="">-- Select Chairman --</option>
                                        {Object.entries(models).map(([modelId, model]) => (
                                            <option key={modelId} value={modelId}>
                                                {model.label} ({model.model_name})
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                <div className="settings-footer">
                    <button className="cancel-button" onClick={onClose}>Cancel</button>
                    <button className="save-button" onClick={handleSave} disabled={saving}>
                        {saving ? 'Saving...' : 'Save Changes'}
                    </button>
                </div>

                {/* Model Editor Modal */}
                {showModelEditor && (
                    <div className="modal-overlay" onClick={closeModelEditor}>
                        <div className="model-editor-modal" onClick={(e) => e.stopPropagation()}>
                            <h3>{editingModel ? 'Edit Model' : 'Add New Model'}</h3>

                            <div className="form-group">
                                <label>Label*</label>
                                <input
                                    type="text"
                                    value={modelForm.label}
                                    onChange={(e) => setModelForm({ ...modelForm, label: e.target.value })}
                                    placeholder="e.g., My Local Llama"
                                />
                            </div>

                            <div className="form-group">
                                <label>Type*</label>
                                <select
                                    value={modelForm.type}
                                    onChange={(e) => setModelForm({ ...modelForm, type: e.target.value })}
                                >
                                    <option value="ollama">Ollama</option>
                                    <option value="openrouter">OpenRouter</option>
                                    <option value="openai-compatible">OpenAI Compatible</option>
                                </select>
                            </div>

                            <div className="form-group">
                                <label>Model Name*</label>
                                <input
                                    type="text"
                                    value={modelForm.model_name}
                                    onChange={(e) => setModelForm({ ...modelForm, model_name: e.target.value })}
                                    placeholder={
                                        modelForm.type === 'ollama' ? 'e.g., llama2' :
                                            modelForm.type === 'openrouter' ? 'e.g., anthropic/claude-3.5-sonnet' :
                                                'e.g., gpt-4'
                                    }
                                />
                            </div>

                            {(modelForm.type === 'ollama' || modelForm.type === 'openai-compatible') && (
                                <div className="form-group">
                                    <label>Base URL*</label>
                                    <input
                                        type="text"
                                        value={modelForm.base_url}
                                        onChange={(e) => setModelForm({ ...modelForm, base_url: e.target.value })}
                                        placeholder="e.g., http://localhost:11434"
                                    />
                                </div>
                            )}

                            {(modelForm.type === 'openrouter' || modelForm.type === 'openai-compatible') && (
                                <div className={`form-group api-key-group ${apiKeyMode === 'env' ? 'env-mode' : 'direct-mode'}`}>
                                    <div className="api-key-header">
                                        <label>API Key*</label>
                                        <div className="api-key-toggle">
                                            <button
                                                type="button"
                                                className={apiKeyMode === 'direct' ? 'active' : ''}
                                                onClick={() => setApiKeyMode('direct')}
                                            >
                                                Direct
                                            </button>
                                            <button
                                                type="button"
                                                className={apiKeyMode === 'env' ? 'active' : ''}
                                                onClick={() => setApiKeyMode('env')}
                                            >
                                                Env Var
                                            </button>
                                        </div>
                                    </div>
                                    {apiKeyMode === 'env' ? (
                                        <>
                                            <input
                                                type="text"
                                                className="env-input"
                                                value={envVarName}
                                                onChange={(e) => setEnvVarName(e.target.value.replace(/[^A-Za-z0-9_]/g, '_').toUpperCase())}
                                                placeholder="e.g., MODEL_MYMODEL_API_KEY"
                                            />
                                            <small className="api-key-note">Saved as env:{envVarName || 'VAR_NAME'} so the key stays outside config.json.</small>
                                        </>
                                    ) : (
                                        <>
                                            <input
                                                type="password"
                                                value={modelForm.api_key}
                                                onChange={(e) => setModelForm({ ...modelForm, api_key: e.target.value })}
                                                placeholder="sk-..."
                                            />
                                            <small className="api-key-note">Leave as *** to keep the existing key.</small>
                                        </>
                                    )}
                                </div>
                            )}

                            <div className="modal-actions">
                                <button onClick={closeModelEditor}>Cancel</button>
                                <button onClick={saveModel} className="primary">Save Model</button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default SettingsModal;
