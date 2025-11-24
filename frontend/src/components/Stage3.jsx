import ReactMarkdown from 'react-markdown';
import './Stage3.css';

export default function Stage3({ finalResponse }) {
  if (!finalResponse) {
    return null;
  }

  const cost = finalResponse.cost || 0;
  const displayName = finalResponse.model_display || finalResponse.model || 'Chairman';
  const hasError = Boolean(finalResponse.error);

  return (
    <div className="stage stage3">
      <h3 className="stage-title">
        Stage 3: Final Council Answer
        {cost > 0 && <span className="stage-cost"> ${cost.toFixed(4)}</span>}
      </h3>
      <div className="final-response">
        <div className="chairman-label">
          Chairman: {displayName}
        </div>
        {hasError ? (
          <div className="stage-error">
            <strong>Failed to generate final answer.</strong>
            <p>{finalResponse.response}</p>
          </div>
        ) : (
          <div className="final-text markdown-content">
            <ReactMarkdown>{finalResponse.response}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
