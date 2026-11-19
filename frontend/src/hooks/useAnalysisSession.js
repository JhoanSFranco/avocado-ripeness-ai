import { useCallback, useEffect, useRef, useState } from 'react';
import { analyzeImage, clearSession } from '../api/client';
import { MAX_HISTORY_ITEMS } from '../constants';
import { newSessionId } from '../utils/format';

/**
 * State and actions for analyzing photos.
 *
 * One session = one physical avocado. Every photo analyzed in the same
 * session is fused by the API into a single, more reliable result.
 * "New avocado" starts a fresh session so different fruits never mix.
 */
export default function useAnalysisSession() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [photosInSession, setPhotosInSession] = useState(0);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  // Object URLs still referenced by history items (must not be revoked yet).
  const historyUrls = useRef(new Set());

  const releasePreview = useCallback((url) => {
    if (url && !historyUrls.current.has(url)) URL.revokeObjectURL(url);
  }, []);

  // Free the server-side session when the page closes.
  const sessionRef = useRef(sessionId);
  sessionRef.current = sessionId;
  useEffect(() => () => clearSession(sessionRef.current), []);

  const selectImage = useCallback(
    (selected) => {
      setPreviewUrl((previous) => {
        releasePreview(previous);
        return URL.createObjectURL(selected);
      });
      setFile(selected);
      setResult(null);
      setError(null);
    },
    [releasePreview]
  );

  const clearImage = useCallback(() => {
    setPreviewUrl((previous) => {
      releasePreview(previous);
      return null;
    });
    setFile(null);
  }, [releasePreview]);

  const analyze = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeImage(file, sessionId);
      setResult(data);
      setPhotosInSession(data.prediction.images_used);

      historyUrls.current.add(previewUrl);
      setHistory((previous) => {
        const next = [{ ...data, previewUrl }, ...previous];
        next.slice(MAX_HISTORY_ITEMS).forEach((item) => {
          historyUrls.current.delete(item.previewUrl);
          URL.revokeObjectURL(item.previewUrl);
        });
        return next.slice(0, MAX_HISTORY_ITEMS);
      });
    } catch (e) {
      setError(e.message || 'The image could not be analyzed.');
    } finally {
      setLoading(false);
    }
  }, [file, previewUrl, sessionId]);

  /** Keep the session: the next photo is fused with the previous ones. */
  const addAnotherPhoto = useCallback(() => {
    setResult(null);
    clearImage();
  }, [clearImage]);

  /** Forget the current avocado and start a new session. */
  const startNewAvocado = useCallback(() => {
    clearSession(sessionId);
    setSessionId(newSessionId());
    setPhotosInSession(0);
    setResult(null);
    setError(null);
    clearImage();
  }, [sessionId, clearImage]);

  return {
    sessionId,
    photosInSession,
    file,
    previewUrl,
    result,
    loading,
    error,
    history,
    selectImage,
    clearImage,
    analyze,
    addAnotherPhoto,
    startNewAvocado,
    showResult: setResult,
  };
}
