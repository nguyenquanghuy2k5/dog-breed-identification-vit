import React from 'react';

export default function ConfigPanel({
  selectedModel,
  setSelectedModel,
  topK,
  setTopK,
  metric,
  setMetric,
  modelOptions
}) {
  return (
    <div className="rounded-3xl bg-surface-container-lowest p-6 shadow-sm border border-surface-container-high/40 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className="material-symbols-outlined text-secondary text-[22px]">tune</span>
          <h2 className="text-lg text-on-surface font-bold">Tham Số Trích Xuất & Metric</h2>
        </div>
      </div>

      <div className="space-y-4">
        {/* Search Space Dataset */}
        <div className="space-y-1.5">
          <label className="text-xs text-on-surface font-bold flex items-center justify-between">
            <span>Không gian tra cứu (Search Space):</span>
          </label>
          <div className="relative">
            <select 
              disabled
              className="w-full appearance-none rounded-xl bg-surface-container px-4 py-2.5 text-on-surface text-xs font-medium pr-10 focus:outline-none border border-transparent shadow-inner cursor-not-allowed"
            >
              <option>Stanford Dogs Dataset (120 Giống chuẩn FCI/AKC - 8.580 ảnh kiểm thử)</option>
            </select>
            <span className="material-symbols-outlined absolute right-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none text-[18px]">lock</span>
          </div>
        </div>

        {/* Feature Extractor Model Selection */}
        <div className="space-y-1.5">
          <label className="text-xs text-on-surface font-bold flex items-center justify-between">
            <span>Mô hình trích xuất đặc trưng (Backbone):</span>
            <span className="px-2 py-0.5 rounded bg-primary-fixed text-primary text-[10px] font-bold">
              {selectedModel === 'vit' ? 'Fine-Grained SOTA' : 'Baseline'}
            </span>
          </label>
          <div className="relative">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full appearance-none rounded-xl bg-surface-container px-4 py-2.5 text-on-surface text-xs font-semibold pr-10 focus:outline-none focus:ring-2 focus:ring-primary border border-surface-container-highest cursor-pointer"
            >
              <option value="vit">Vision Transformer (ViT-B/16) • 768-dim Embedding (Khuyên dùng)</option>
              <option value="resnet">ResNet-50 Baseline • 2048-dim Embedding</option>
            </select>
            <span className="material-symbols-outlined absolute right-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none text-[18px]">expand_more</span>
          </div>
        </div>

        {/* Distance Metric & Index Engine */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <label className="text-[11px] text-on-surface font-bold uppercase tracking-wide">
              Hàm Khoảng Cách:
            </label>
            <div className="relative">
              <select
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                className="w-full appearance-none rounded-xl bg-surface-container px-3.5 py-2 text-on-surface text-xs font-medium pr-8 focus:outline-none focus:ring-2 focus:ring-primary border border-surface-container-highest"
              >
                <option value="cosine">Cosine Similarity</option>
                <option value="l2">Euclidean (L2 Norm)</option>
              </select>
              <span className="material-symbols-outlined absolute right-2.5 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none text-[16px]">expand_more</span>
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-[11px] text-on-surface font-bold uppercase tracking-wide">
              Indexing Engine:
            </label>
            <div className="relative">
              <select
                disabled
                className="w-full appearance-none rounded-xl bg-surface-container px-3.5 py-2 text-on-surface text-xs font-medium pr-8 focus:outline-none border border-transparent cursor-not-allowed"
              >
                <option>FAISS FlatIP (Exact)</option>
              </select>
              <span className="material-symbols-outlined absolute right-2.5 top-1/2 -translate-y-1/2 text-on-surface-variant pointer-events-none text-[16px]">lock</span>
            </div>
          </div>
        </div>

        {/* Number of Top-K Results Slider */}
        <div className="space-y-2 pt-1">
          <div className="flex items-center justify-between">
            <label className="text-xs text-on-surface font-semibold" htmlFor="k-slider">
              Số lượng kết quả truy xuất tương đồng (Top-K):
            </label>
            <span className="px-2.5 py-0.5 rounded-full bg-primary text-on-primary text-xs font-bold shadow-sm">
              K = {topK}
            </span>
          </div>
          <input
            id="k-slider"
            type="range"
            min="1"
            max="30"
            step="1"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-full h-2 bg-surface-container-high rounded-lg appearance-none cursor-pointer accent-primary"
          />
          <div className="flex justify-between text-[11px] text-on-surface-variant px-1 font-medium">
            <span>K=1</span>
            <span className="text-primary font-bold">K=5</span>
            <span className="text-primary font-bold">K=10</span>
            <span>K=20</span>
            <span>K=30</span>
          </div>
        </div>
      </div>
    </div>
  );
}

