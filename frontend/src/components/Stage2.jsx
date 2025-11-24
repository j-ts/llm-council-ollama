import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage2.css';

function deAnonymizeText(text, labelToModel) {
  if (!labelToModel) return text;

  let result = text;

  const getLabelName = (label) => {
    const entry = labelToModel[label];
    if (!entry) return label;
    if (typeof entry === 'string') {
      return entry.split('/')[1] || entry;
    }
    return entry.model_display || entry.model || entry.id || label;
  };

  // Replace each "Response X" with the actual model name
  Object.entries(labelToModel).forEach(([label, model]) => {
    const modelShortName = getLabelName(label);
    result = result.replace(new RegExp(label, 'g'), `**${modelShortName}**`);
  });
  return result;
}

export default function Stage2({ rankings, labelToModel, aggregateRankings, stageCost }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!rankings || rankings.length === 0) {
    return null;
  }

  // Calculate total stage cost
  const totalCost = stageCost || rankings.reduce((sum, r) => sum + (r.cost || 0), 0);

  const getModelName = (entry) => entry.model_display || entry.model || entry.model_id || 'Model';

  return (
    <div className="stage stage2">
      <h3 className="stage-title">
        Stage 2: Peer Rankings
        {totalCost > 0 && <span className="stage-cost"> ${totalCost.toFixed(4)}</span>}
      </h3>

      <h4>Raw Evaluations</h4>
      <p className="stage-description">
        Each model evaluated all responses (anonymized as Response A, B, C, etc.) and provided rankings.
        Below, model names are shown in <strong>bold</strong> for readability, but the original evaluation used anonymous labels.
      </p>

      <div className="tabs">
        {rankings.map((rank, index) => (
          <button
            key={rank.model_id || rank.model || index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            <span className="tab-model">{getModelName(rank)}</span>
            {rank.cost > 0 && <span className="tab-cost">${rank.cost.toFixed(4)}</span>}
          </button>
        ))}
      </div>

      <div className="tab-content">
        <div className="ranking-model">
          {getModelName(rankings[activeTab])}
        </div>
        <div className="ranking-content markdown-content">
          <ReactMarkdown>
            {deAnonymizeText(rankings[activeTab].ranking, labelToModel)}
          </ReactMarkdown>
        </div>

        {rankings[activeTab].parsed_ranking &&
         rankings[activeTab].parsed_ranking.length > 0 && (
          <div className="parsed-ranking">
            <strong>Extracted Ranking:</strong>
            <ol>
              {rankings[activeTab].parsed_ranking.map((label, i) => (
                <li key={i}>
                  {(() => {
                    const entry = labelToModel && labelToModel[label];
                    if (!entry) return label;
                    if (typeof entry === 'string') {
                      return entry.split('/')[1] || entry;
                    }
                    return entry.model_display || entry.model || entry.id || label;
                  })()}
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>

      {aggregateRankings && aggregateRankings.length > 0 && (
        <div className="aggregate-rankings">
          <h4>Aggregate Rankings (Street Cred)</h4>
          <p className="stage-description">
            Combined results across all peer evaluations (lower score is better):
          </p>
          <div className="aggregate-list">
            {aggregateRankings.map((agg, index) => (
              <div key={agg.model_id || agg.model || index} className="aggregate-item">
                <span className="rank-position">#{index + 1}</span>
                <span className="rank-model">
                  {agg.model_display || agg.model}
                </span>
                <span className="rank-score">
                  Avg: {agg.average_rank.toFixed(2)}
                </span>
                <span className="rank-count">
                  ({agg.rankings_count} votes)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
