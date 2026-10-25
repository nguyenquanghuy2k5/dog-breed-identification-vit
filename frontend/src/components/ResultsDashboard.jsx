import React, { useState } from 'react';

export default function ResultsDashboard({ resultsData, queryPreviewUrl }) {
  const [selectedComparison, setSelectedComparison] = useState(null);

  if (!resultsData) return null;

  const { metrics, top_match, results } = resultsData;

  return (
    <div className="space-y-8 animate-fadeIn pt-4" id="results-section">
      {/* SECTION 1: Performance Metric Strip */}
      <section className="bg-surface-container-lowest rounded-3xl p-6 shadow-sm border border-surface-container-high/40">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-1">
            <h2 className="text-xl font-extrabold text-on-surface">Kết Quả Nhận Dạng & Khảo Sát Không Gian Vector</h2>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-surface-container-low p-3 rounded-2xl flex flex-col justify-center">
              <span className="text-xs text-on-surface-variant font-medium">Mô hình trích xuất</span>
              <span className="text-sm font-extrabold text-on-surface flex items-center gap-1 mt-0.5">
                <span className="material-symbols-outlined text-[16px] text-secondary">neurology</span>
                {metrics?.model_architecture?.includes('ViT') ? 'ViT-B/16' : 'ResNet-50'}
              </span>
              <span className="text-[11px] text-on-surface-variant">{metrics?.embedding_dim}-dim Vector</span>
            </div>

            <div className="bg-surface-container-low p-3 rounded-2xl flex flex-col justify-center">
              <span className="text-xs text-on-surface-variant font-medium">Thời gian FAISS</span>
              <span className="text-sm font-extrabold text-tertiary flex items-center gap-1 mt-0.5">
                <span className="material-symbols-outlined text-[16px]">hub</span>
                {metrics?.faiss_time_ms} ms
              </span>
              <span className="text-[11px] text-on-surface-variant">L2 / Cosine Flat</span>
            </div>

            <div className="bg-surface-container-low p-3 rounded-2xl flex flex-col justify-center">
              <span className="text-xs text-on-surface-variant font-medium">Tổng độ trễ</span>
              <span className="text-sm font-extrabold text-primary flex items-center gap-1 mt-0.5">
                <span className="material-symbols-outlined text-[16px]">bolt</span>
                {metrics?.total_latency_ms} ms
              </span>
              <span className="text-[11px] text-on-surface-variant">Inference & Search</span>
            </div>

            <div className="bg-surface-container-low p-3 rounded-2xl flex flex-col justify-center">
              <span className="text-xs text-on-surface-variant font-medium">Tập dữ liệu</span>
              <span className="text-sm font-extrabold text-on-surface flex items-center gap-1 mt-0.5">
                <span className="material-symbols-outlined text-[16px] text-secondary">dataset</span>
                Stanford Dogs
              </span>
              <span className="text-[11px] text-on-surface-variant">{metrics?.search_space_count?.toLocaleString()} Vectors</span>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: Top Match Hero Card */}
      {top_match && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[24px]">verified</span>
              <h3 className="text-xl font-bold text-on-surface">Định Danh Phù Hợp Cao Nhất</h3>
            </div>
            <span className="text-xs text-on-surface-variant hidden sm:inline">
              Khoảng cách vector cosine tối ưu
            </span>
          </div>

          <div className="w-full bg-surface-container-lowest rounded-3xl p-6 sm:p-8 shadow-[0_20px_40px_-8px_rgba(0,88,190,0.08)] relative overflow-hidden border border-surface-container-high/60">
            {/* Background Glow */}
            <div className="absolute -right-20 -top-20 w-80 h-80 bg-primary-fixed/30 rounded-full blur-3xl pointer-events-none"></div>

            {/* Badge Ribbon */}
            <div className="flex flex-wrap items-center justify-between gap-3 pb-6 border-b border-surface-container-high/40">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary-fixed text-on-primary-fixed text-xs font-bold shadow-sm">
                <span className="material-symbols-outlined text-[18px]">trophy</span>
                HẠNG #1 - ĐỘ TƯƠNG ĐỒNG CAO NHẤT (TOP MATCH)
              </div>
              <div className="flex items-center gap-2">
                <span className="px-3 py-1 rounded-full bg-secondary-fixed text-on-secondary-fixed text-xs font-semibold">
                  {top_match.fci_group}
                </span>
                <span className="px-3 py-1 rounded-full bg-surface-container-high text-on-surface text-xs font-semibold">
                  Xuất xứ: {top_match.origin}
                </span>
              </div>
            </div>

            {/* Split Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start pt-6">
              {/* Left Column: Reference Image */}
              <div className="lg:col-span-5 flex flex-col gap-3">
                <div className="relative w-full aspect-square rounded-2xl overflow-hidden bg-surface-container shadow-md group">
                  <img
                    src={top_match.image_url}
                    alt={top_match.breed_name}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                  />
                  {/* Overlay Tags */}
                  <div className="absolute bottom-3 left-3 bg-surface-container-lowest/90 backdrop-blur-md px-3 py-1.5 rounded-xl text-on-surface shadow-sm">
                    <p className="text-[11px] font-mono text-on-surface-variant">
                      DB Ref: <strong className="text-on-surface">#{top_match.db_ref}</strong>
                    </p>
                    <p className="text-[11px] text-on-surface-variant font-mono">
                      Class ID: #{top_match.class_id}
                    </p>
                  </div>
                </div>
              </div>

              {/* Right Column: Breed Facts & Metrics */}
              <div className="lg:col-span-7 flex flex-col justify-between space-y-6">
                <div>
                  <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 mb-1">
                    <h1 className="text-3xl font-extrabold text-on-surface tracking-tight">
                      {top_match.breed_name}
                    </h1>
                    <span className="text-3xl font-black text-primary">
                      {top_match.similarity}% Match
                    </span>
                  </div>
                  <p className="text-sm text-on-surface-variant font-medium">
                    {top_match.vietnamese_name} • <i>({top_match.scientific_name})</i>
                  </p>
                </div>

                {/* Progress Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-semibold text-on-surface-variant">
                    <span>Mức độ tự tin mô hình:</span>
                    <span className="text-primary font-bold">{top_match.similarity}%</span>
                  </div>
                  <div className="w-full h-3 bg-surface-container-high rounded-full overflow-hidden p-0.5">
                    <div
                      className="h-full bg-gradient-to-r from-amber-500 to-primary-container rounded-full transition-all duration-1000"
                      style={{ width: `${Math.min(100, top_match.similarity)}%` }}
                    ></div>
                  </div>
                </div>

                {/* Morphometrics & Facts Grid */}
                <div className="grid grid-cols-2 gap-3 pt-2">
                  <div className="p-3.5 rounded-2xl bg-surface-container-low border border-surface-container-high/60">
                    <span className="text-xs text-on-surface-variant font-medium block">Tuổi thọ trung bình</span>
                    <span className="text-sm font-bold text-on-surface">{top_match.lifespan}</span>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-surface-container-low border border-surface-container-high/60">
                    <span className="text-xs text-on-surface-variant font-medium block">Nguồn gốc</span>
                    <span className="text-sm font-bold text-on-surface">{top_match.origin}</span>
                  </div>
                  <div className="col-span-2 p-3.5 rounded-2xl bg-surface-container-low border border-surface-container-high/60">
                    <span className="text-xs text-on-surface-variant font-medium block">Đặc điểm tính cách</span>
                    <span className="text-sm font-bold text-on-surface">{top_match.temperament}</span>
                  </div>
                </div>

                <div className="pt-2">
                  <button
                    type="button"
                    onClick={() => setSelectedComparison(top_match)}
                    className="px-5 py-2.5 rounded-xl bg-surface-container-high hover:bg-surface-container text-on-surface text-xs font-bold transition-all flex items-center gap-2"
                  >
                    <span className="material-symbols-outlined text-[18px] text-secondary">compare</span>
                    So sánh trực quan song song (Side-by-Side)
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* SECTION 3: Top-K Ranked Gallery Grid */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-secondary text-[24px]">grid_view</span>
            <h3 className="text-xl font-bold text-on-surface">Danh Sách Đối Khớp Tương Đồng (Top-{results.length})</h3>
          </div>
          <span className="text-xs text-on-surface-variant">
            Nhấp chuột vào ảnh để so sánh chi tiết
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {results.map((item) => (
            <div
              key={item.rank}
              onClick={() => setSelectedComparison(item)}
              className="group cursor-pointer rounded-2xl bg-surface-container-lowest p-3 shadow-sm hover:shadow-md transition-all border border-surface-container-high/50 hover:border-primary/50 flex flex-col"
            >
              <div className={`relative w-full aspect-square rounded-xl overflow-hidden bg-surface-container mb-2 border-2 ${item.border_class}`}>
                <img
                  src={item.image_url}
                  alt={item.breed_name}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <span className="absolute top-2 left-2 px-2 py-0.5 rounded-md bg-surface-container-lowest/90 backdrop-blur-sm text-[10px] font-bold text-on-surface shadow">
                  #{item.rank}
                </span>
                <span className={`absolute top-2 right-2 px-2 py-0.5 rounded-md text-[10px] font-bold shadow ${item.badge_color}`}>
                  {item.similarity}%
                </span>
              </div>

              <div className="space-y-0.5 mt-auto">
                <p className="text-xs font-bold text-on-surface truncate" title={item.breed_name}>
                  {item.breed_name}
                </p>
                <p className="text-[10px] font-mono text-on-surface-variant truncate">
                  {item.db_ref}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Comparison Modal */}
      {selectedComparison && (
        <div className="fixed inset-0 z-[9999] w-screen h-screen bg-black/75 backdrop-blur-md flex items-center justify-center p-4 overflow-y-auto animate-fadeIn">
          <div className="bg-surface-container-lowest rounded-3xl max-w-3xl w-full p-6 shadow-2xl space-y-5 animate-scaleIn border border-surface-container-high">
            <div className="flex items-center justify-between pb-3 border-b border-surface-container-high">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-primary text-[24px]">compare</span>
                <h4 className="text-lg font-bold text-on-surface">
                  So Sánh Trực Quan: Ảnh Truy Vấn & Kết Quả #{selectedComparison.rank}
                </h4>
              </div>
              <button
                type="button"
                onClick={() => setSelectedComparison(null)}
                className="p-1.5 rounded-xl hover:bg-surface-container text-on-surface-variant"
              >
                <span className="material-symbols-outlined text-[20px]">close</span>
              </button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <span className="text-xs font-bold text-on-surface-variant uppercase tracking-wider block">
                  Ảnh Truy Vấn (Query Image)
                </span>
                <div className="aspect-square rounded-2xl bg-surface-container overflow-hidden shadow-inner border border-surface-container-high">
                  <img src={queryPreviewUrl} alt="Query" className="w-full h-full object-cover" />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-primary uppercase tracking-wider">
                    {selectedComparison.breed_name}
                  </span>
                  <span className="text-xs font-bold text-primary">
                    {selectedComparison.similarity}% Match
                  </span>
                </div>
                <div className="aspect-square rounded-2xl bg-surface-container overflow-hidden shadow-inner border-2 border-primary">
                  <img src={selectedComparison.image_url} alt="Reference" className="w-full h-full object-cover" />
                </div>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-surface-container-low text-xs text-on-surface-variant space-y-1">
              <p><strong>Tệp cơ sở dữ liệu:</strong> {selectedComparison.db_ref}</p>
              <p><strong>Mô hình trích xuất:</strong> {metrics?.model_architecture} ({metrics?.embedding_dim} chiều vector)</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

