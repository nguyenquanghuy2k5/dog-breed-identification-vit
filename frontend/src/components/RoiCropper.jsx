import React, { useEffect, useRef, useState } from 'react';
import Cropper from 'cropperjs';

export default function RoiCropper({
  imageSrc,
  onCropChange,
  onSubmit,
  isLoading,
  selectedModel
}) {
  const imageRef = useRef(null);
  const cropperRef = useRef(null);
  const [aspectRatio, setAspectRatio] = useState(NaN); // NaN = Free
  const [coords, setCoords] = useState({ x: 0, y: 0, width: 0, height: 0 });

  useEffect(() => {
    if (!imageSrc || !imageRef.current) return;

    if (cropperRef.current) {
      cropperRef.current.destroy();
    }

    const cropper = new Cropper(imageRef.current, {
      aspectRatio: aspectRatio,
      viewMode: 1,
      autoCropArea: 0.85,
      responsive: true,
      crop(event) {
        const detail = event.detail;
        const newCoords = {
          x: Math.round(detail.x),
          y: Math.round(detail.y),
          width: Math.round(detail.width),
          height: Math.round(detail.height),
        };
        setCoords(newCoords);
        if (onCropChange) {
          onCropChange(newCoords);
        }
      },
    });

    cropperRef.current = cropper;

    return () => {
      cropper.destroy();
      cropperRef.current = null;
    };
  }, [imageSrc]);

  const handleAspectRatioChange = (ratio) => {
    setAspectRatio(ratio);
    if (cropperRef.current) {
      cropperRef.current.setAspectRatio(ratio);
    }
  };

  const handleZoom = (delta) => {
    if (cropperRef.current) {
      cropperRef.current.zoom(delta);
    }
  };

  const handleReset = () => {
    if (cropperRef.current) {
      cropperRef.current.reset();
    }
  };

  const handleRotate = (deg) => {
    if (cropperRef.current) {
      cropperRef.current.rotate(deg);
    }
  };

  return (
    <div className="rounded-3xl bg-surface-container-lowest p-6 shadow-sm border border-surface-container-high/40 space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-[22px]">crop_free</span>
          <h2 className="text-lg text-on-surface font-bold">
            3. Phân Vùng Quan Tâm (Canine ROI Workspace)
          </h2>
        </div>

        {/* Toolbar Controls */}
        <div className="flex items-center gap-1.5 bg-surface-container-low p-1 rounded-xl">
          <button
            type="button"
            onClick={() => handleAspectRatioChange(NaN)}
            className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
              isNaN(aspectRatio) ? 'bg-primary text-on-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            Tự do
          </button>
          <button
            type="button"
            onClick={() => handleAspectRatioChange(1)}
            className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
              aspectRatio === 1 ? 'bg-primary text-on-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            1:1
          </button>
          <button
            type="button"
            onClick={() => handleAspectRatioChange(4 / 3)}
            className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
              aspectRatio === 4 / 3 ? 'bg-primary text-on-primary shadow-sm' : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            4:3
          </button>
          <div className="w-[1px] h-4 bg-outline-variant/40 mx-1"></div>
          <button
            type="button"
            onClick={() => handleZoom(0.1)}
            title="Phóng to"
            className="p-1 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container"
          >
            <span className="material-symbols-outlined text-[18px]">zoom_in</span>
          </button>
          <button
            type="button"
            onClick={() => handleZoom(-0.1)}
            title="Thu nhỏ"
            className="p-1 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container"
          >
            <span className="material-symbols-outlined text-[18px]">zoom_out</span>
          </button>
          <button
            type="button"
            onClick={() => handleRotate(90)}
            title="Xoay 90 độ"
            className="p-1 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container"
          >
            <span className="material-symbols-outlined text-[18px]">rotate_right</span>
          </button>
          <button
            type="button"
            onClick={handleReset}
            title="Đặt lại"
            className="p-1 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container"
          >
            <span className="material-symbols-outlined text-[18px]">restart_alt</span>
          </button>
        </div>
      </div>

      {/* Canvas Viewport */}
      <div className="w-full aspect-[4/3] max-h-[460px] rounded-2xl bg-surface-container overflow-hidden relative flex items-center justify-center border border-surface-container-high shadow-inner">
        {imageSrc ? (
          <div className="w-full h-full">
            <img
              ref={imageRef}
              src={imageSrc}
              alt="Query input preview"
              className="max-w-full max-h-full block"
            />
          </div>
        ) : (
          <div className="text-center p-8 space-y-3">
            <div className="w-16 h-16 rounded-3xl bg-surface-container-high/60 flex items-center justify-center mx-auto text-on-surface-variant">
              <span className="material-symbols-outlined text-[36px]">image</span>
            </div>
            <p className="text-sm font-semibold text-on-surface">Chưa có ảnh được nạp</p>
            <p className="text-xs text-on-surface-variant max-w-sm">
              Vui lòng tải ảnh lên từ cột bên trái hoặc chọn ảnh mẫu để kích hoạt Canvas cắt vùng quan tâm (ROI).
            </p>
          </div>
        )}
      </div>

      {/* Coordinate Meta & Action Button */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-1">
        <div className="flex items-center gap-3 text-xs text-on-surface-variant font-mono">
          <span className="px-2.5 py-1 rounded-lg bg-surface-container text-on-surface font-semibold">
            ROI: {coords.width} x {coords.height} px
          </span>
          <span>(X: {coords.x}, Y: {coords.y})</span>
        </div>

        <button
          type="button"
          disabled={!imageSrc || isLoading}
          onClick={onSubmit}
          className={`w-full sm:w-auto px-8 py-3.5 rounded-2xl font-bold text-sm transition-all shadow-lg flex items-center justify-center gap-2.5 ${
            !imageSrc || isLoading
              ? 'bg-surface-container-highest text-on-surface-variant cursor-not-allowed shadow-none'
              : 'bg-primary-container text-on-primary-fixed hover:bg-amber-500 hover:shadow-primary-container/30 active:scale-98'
          }`}
        >
          {isLoading ? (
            <>
              <svg className="animate-spin h-5 w-5 text-on-primary-fixed" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              <span>Đang trích xuất vector & đối khớp FAISS...</span>
            </>
          ) : (
            <>
              <span className="material-symbols-outlined text-[20px]">bolt</span>
              <span>TIẾN HÀNH TRUY XUẤT ({selectedModel === 'vit' ? 'ViT-B/16' : 'ResNet-50'})</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}

