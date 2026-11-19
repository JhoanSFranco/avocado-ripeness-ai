import React, { useRef, useState } from 'react';

/**
 * Controlled image picker: drag & drop, gallery or camera.
 * The parent owns the selected file and preview URL, so resetting the
 * analysis also resets this component.
 */
export default function ImageUploader({ previewUrl, disabled, onSelect, onClear }) {
  const galleryInput = useRef(null);
  const cameraInput = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) onSelect(file);
  };

  const handleInputChange = (event) => {
    handleFile(event.target.files[0]);
    event.target.value = ''; // allow selecting the same file again
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    handleFile(event.dataTransfer.files[0]);
  };

  if (previewUrl) {
    return (
      <div className="uploader__preview">
        <img src={previewUrl} alt="Selected avocado" />
        {!disabled && (
          <button type="button" className="button button--small button--ghost" onClick={onClear}>
            Change photo
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      className={`uploader${dragging ? ' uploader--dragging' : ''}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" strokeWidth="1.5" aria-hidden="true">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
      <p className="uploader__text">Drop a photo of one avocado here, or choose a source</p>

      <div className="uploader__buttons">
        <button type="button" className="button button--primary" onClick={() => galleryInput.current?.click()}>
          Gallery
        </button>
        <button type="button" className="button button--outline" onClick={() => cameraInput.current?.click()}>
          Camera
        </button>
      </div>

      <input ref={galleryInput} type="file" accept="image/*" hidden onChange={handleInputChange} />
      <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden onChange={handleInputChange} />
    </div>
  );
}
