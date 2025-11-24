import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage1.css';

export default function Stage1({ responses, stageCost }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!responses || responses.length === 0) {
    return null;
  }

  // Calculate total stage cost
  const totalCost = stageCost || responses.reduce((sum, r) => sum + (r.cost || 0), 0);
  const getDisplayName = (resp) => resp.model_display || resp.model || 'Model';

  return (
    <div className="stage stage1">
      <h3 className="stage-title">
        Stage 1: Individual Responses
        {totalCost > 0 && <span className="stage-cost"> ${totalCost.toFixed(4)}</span>}
      </h3>

      <div className="tabs">
        {responses.map((resp, index) => (
          <button
            key={resp.model_id || resp.model || index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            <span className="tab-model">{getDisplayName(resp)}</span>
            {resp.cost > 0 && <span className="tab-cost">${resp.cost.toFixed(4)}</span>}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="model-name">{getDisplayName(responses[activeTab])}</div>
        <div className="response-text markdown-content">
          <ReactMarkdown>{responses[activeTab].response}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
