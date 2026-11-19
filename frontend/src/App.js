import React, { useEffect, useState } from 'react';
import { checkHealth } from './api/client';
import Header from './components/Header';
import HistoryPanel from './components/HistoryPanel';
import ImageUploader from './components/ImageUploader';
import ResultCard from './components/ResultCard';
import SessionBar from './components/SessionBar';
import Spinner from './components/Spinner';
import useAnalysisSession from './hooks/useAnalysisSession';

export default function App() {
  const [apiOnline, setApiOnline] = useState(null); // null = checking
  const session = useAnalysisSession();

  useEffect(() => {
    checkHealth().then(setApiOnline);
  }, []);

  const { file, result, loading, error } = session;

  return (
    <div className="app">
      <Header apiOnline={apiOnline} />

      <main className="app__main">
        {apiOnline === false && (
          <div className="banner banner--error">
            Cannot reach the API or the model is not loaded. Make sure the backend is running.
          </div>
        )}

        <SessionBar photos={session.photosInSession} onNewAvocado={session.startNewAvocado} />

        {!result && (
          <ImageUploader
            previewUrl={session.previewUrl}
            disabled={loading}
            onSelect={session.selectImage}
            onClear={session.clearImage}
          />
        )}

        {file && !loading && !result && (
          <button type="button" className="button button--primary button--block" onClick={session.analyze}>
            Analyze photo
          </button>
        )}

        {loading && <Spinner title="Analyzing photo..." subtitle="Running Monte Carlo Dropout inference" />}

        {error && <div className="banner banner--error">{error}</div>}

        {result && (
          <>
            <ResultCard key={result.image_id} result={result} />
            <div className="actions">
              <button type="button" className="button button--outline" onClick={session.addAnotherPhoto}>
                Add another photo of this avocado
              </button>
              <button type="button" className="button button--ghost" onClick={session.startNewAvocado}>
                Analyze a new avocado
              </button>
            </div>
          </>
        )}

        <HistoryPanel history={session.history} onSelect={session.showResult} />
      </main>

      <footer className="app__footer">
        ResNet-50 · Monte Carlo Dropout · Multi-photo Bayesian fusion · Hass avocado
      </footer>
    </div>
  );
}
