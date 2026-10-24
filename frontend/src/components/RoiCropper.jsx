import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Cropper from 'cropperjs';

export default function RoiCropper({
  isOpen,
  onClose,
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
    if (!isOpen || !imageSrc || !imageRef.current) return;

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
  }, [isOpen, imageSrc]);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen || !imageSrc) return null;

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

  const modalContent = (
    <div
      className="fixed inset-0 w-screen h-screen bg-black/80 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto animate-fadeIn"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 99999
      }}
    >
      <div
        className="bg-surface-container-lowest rounded-3xl max-w-4xl w-full p-6 shadow-2xl space-y-4 animate-scaleIn border border-surface-container-high my-auto relative"
        style={{ zIndex: 100000 }}
      >
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-surface-container-high/60">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-[24px]">crop_free</span>
            <div>
              <h2 className="text-lg text-on-surface font-bold">
                Phân Vùng Quan Tâm (Canine ROI Workspace)
              </h2>
              <p className="text-xs text-on-surface-variant">
                Kéo thả khung chọn để khoanh vùng khuôn mặt hoặc cơ thể chú chó cần nhận dạng
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Toolbar Controls */}
            <div className="flex items-center gap-1 bg-surface-container-low p-1 rounded-xl">
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

            {/* Close Button */}
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-xl hover:bg-surface-container text-on-surface-variant hover:text-on-surface transition-colors ml-1"
              title="Đóng cửa sổ"
            >
              <span className="material-symbols-outlined text-[22px]">close</span>
            </button>
          </div>
        </div>

        {/* Canvas Viewport */}
        <div className="w-full aspect-[4/3] max-h-[440px] rounded-2xl bg-surface-container overflow-hidden relative flex items-center justify-center border border-surface-container-high shadow-inner">
          <img
            ref={imageRef}
            src={imageSrc}
            alt="Query input preview"
            className="max-w-full max-h-full block"
          />
        </div>

        {/* Footer Meta & Actions */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2 border-t border-surface-container-high/40">
          <div className="flex items-center gap-3 text-xs text-on-surface-variant font-mono">
            <span className="px-2.5 py-1 rounded-lg bg-surface-container text-on-surface font-semibold">
              Kích thước ROI: {coords.width} x {coords.height} px
            </span>
            <span>(Tọa độ X: {coords.x}, Y: {coords.y})</span>
          </div>

          <div className="flex items-center gap-2.5 w-full sm:w-auto">
            <button
              type="button"
              onClick={onClose}
              className="px-5 py-3 rounded-2xl border border-surface-container-high text-xs font-bold text-on-surface-variant hover:bg-surface-container transition-all"
            >
              Giữ nguyên ảnh gốc
            </button>

            <button
              type="button"
              disabled={isLoading}
              onClick={onSubmit}
              className={`flex-1 sm:flex-none px-7 py-3 rounded-2xl font-bold text-xs sm:text-sm transition-all shadow-lg flex items-center justify-center gap-2 ${
                isLoading
                  ? 'bg-surface-container-highest text-on-surface-variant cursor-not-allowed shadow-none'
                  : 'bg-primary-container text-on-primary-fixed hover:bg-amber-500 hover:shadow-primary-container/30 active:scale-98'
              }`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-on-primary-fixed" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Đang xử lý...</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-[18px]">bolt</span>
                  <span>TIẾN HÀNH TRUY XUẤT ({selectedModel === 'vit' ? 'ViT-B/16' : 'ResNet-50'})</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  return createPortal(modalContent, document.body);
}
