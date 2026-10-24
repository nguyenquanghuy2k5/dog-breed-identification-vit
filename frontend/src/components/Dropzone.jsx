import React, { useRef } from 'react';

export default function Dropzone({
  onFileSelected,
  activeFile,
  activePreview,
  onClearFile,
  onOpenRoiModal,
  onSubmitRetrieval,
  isLoading,
  cropCoords
}) {
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onFileSelected(file);
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelected(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const hasRoi = cropCoords && cropCoords.width > 0 && cropCoords.height > 0;

  return (
    <div className="rounded-3xl bg-surface-container-lowest p-6 shadow-sm border border-surface-container-high/40 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className="material-symbols-outlined text-primary text-[22px]">upload_file</span>
          <h2 className="text-lg text-on-surface font-bold">Tải lên ảnh chụp truy vấn</h2>
        </div>
        <span className="px-2.5 py-0.5 rounded-full bg-primary-fixed text-on-primary-fixed text-xs font-semibold">
          Query Input
        </span>
      </div>

      {/* Dropzone Container */}
      <div
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className="relative group cursor-pointer rounded-2xl bg-surface-container-low p-6 text-center transition-all duration-300 hover:bg-surface-container border-2 border-dashed border-outline-variant/60 hover:border-primary/80"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          className="hidden"
        />
        <div className="flex flex-col items-center justify-center space-y-3 pointer-events-none">
          <div className="w-14 h-14 rounded-2xl bg-primary-fixed-dim/40 flex items-center justify-center text-primary group-hover:scale-105 transition-transform duration-300">
            <span className="material-symbols-outlined text-[32px]">cloud_upload</span>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-on-surface font-bold">
              Kéo & thả tập tin hình ảnh vào đây
            </p>
            <p className="text-xs text-on-surface-variant">
              hoặc nhấp chuột để chọn tệp từ máy tính
            </p>
          </div>
          <div className="flex items-center gap-2 pt-1">
            <span className="px-2 py-0.5 rounded-md bg-surface-container-high text-on-surface-variant text-[11px] font-semibold">JPG</span>
            <span className="px-2 py-0.5 rounded-md bg-surface-container-high text-on-surface-variant text-[11px] font-semibold">PNG</span>
            <span className="px-2 py-0.5 rounded-md bg-surface-container-high text-on-surface-variant text-[11px] font-semibold">WEBP</span>
            <span className="text-on-surface-variant text-[11px]">• Tối đa 15MB</span>
          </div>
        </div>
      </div>

      {/* Active Ingested File Status Capsule & ROI Shortcuts */}
      {activeFile && (
        <div className="space-y-3 pt-1">
          <div className="p-3.5 rounded-2xl bg-surface-container flex items-center justify-between gap-3 shadow-sm border border-surface-container-high">
            <div className="flex items-center gap-3 min-w-0">
              <div 
                onClick={onOpenRoiModal}
                title="Nhấp để mở lại phân vùng ROI"
                className="w-12 h-12 rounded-xl bg-surface-container-highest overflow-hidden shrink-0 shadow-inner cursor-pointer hover:ring-2 hover:ring-primary transition-all relative group"
              >
                <img 
                  src={activePreview} 
                  alt="Uploaded thumbnail" 
                  className="w-full h-full object-cover" 
                />
                <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white">
                  <span className="material-symbols-outlined text-[18px]">crop_free</span>
                </div>
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-on-surface font-bold truncate max-w-[160px]">
                    {activeFile.name}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-secondary text-on-secondary text-[10px] font-bold shrink-0">
                    Đã tải lên
                  </span>
                </div>
                <p className="text-[11px] text-on-surface-variant truncate">
                  {formatFileSize(activeFile.size)} • {hasRoi ? `ROI: ${cropCoords.width}x${cropCoords.height}px` : 'Ảnh toàn cảnh'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={onOpenRoiModal}
                title="Mở cửa sổ khoanh vùng ROI"
                className="px-2.5 py-1.5 rounded-xl bg-surface-container-high hover:bg-surface-container text-xs font-semibold text-primary flex items-center gap-1 transition-colors"
              >
                <span className="material-symbols-outlined text-[16px]">crop_free</span>
                <span>{hasRoi ? 'Sửa ROI' : 'Khoanh ROI'}</span>
              </button>

              <button
                type="button"
                onClick={onClearFile}
                title="Đổi ảnh khác"
                className="p-1.5 rounded-xl text-on-surface-variant hover:text-error hover:bg-error-container/40 transition-colors"
              >
                <span className="material-symbols-outlined text-[18px]">close</span>
              </button>
            </div>
          </div>

          {/* Quick Submit Button from Section 1 */}
          <button
            type="button"
            disabled={isLoading}
            onClick={onSubmitRetrieval}
            className={`w-full py-3 rounded-2xl font-bold text-xs sm:text-sm transition-all shadow-md flex items-center justify-center gap-2 ${
              isLoading
                ? 'bg-surface-container-highest text-on-surface-variant cursor-not-allowed'
                : 'bg-primary-container text-on-primary-fixed hover:bg-amber-500 shadow-primary-container/20 active:scale-98'
            }`}
          >
            {isLoading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-on-primary-fixed" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span>Đang trích xuất vector & đối khớp FAISS...</span>
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[18px]">bolt</span>
                <span>TIẾN HÀNH TRUY XUẤT NGAY</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
