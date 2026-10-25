import React from 'react';

export default function HistoryView({
  history,
  onSelectHistory,
  onDeleteHistoryItem,
  onClearHistory,
  onStartNewSearch
}) {
  if (!history || history.length === 0) {
    return (
      <div className="max-w-4xl mx-auto py-12 px-4">
        <div className="p-12 rounded-3xl bg-surface-container-lowest border border-surface-container-high/60 text-center space-y-6 shadow-sm">
          <div className="w-20 h-20 mx-auto rounded-3xl bg-surface-container-low flex items-center justify-center text-on-surface-variant/60 shadow-inner">
            <span className="material-symbols-outlined text-[44px]">history_toggle_off</span>
          </div>
          <div className="space-y-2 max-w-md mx-auto">
            <h2 className="text-2xl font-bold text-on-surface tracking-tight">Chưa có lịch sử tra cứu nào</h2>
            <p className="text-sm text-on-surface-variant leading-relaxed">
              Các lượt tra cứu hình ảnh, nhận dạng giống chó và kết quả độ tương đồng vector sẽ được tự động lưu lại tại đây.
            </p>
          </div>
          <button
            type="button"
            onClick={onStartNewSearch}
            className="px-6 py-3 rounded-2xl bg-primary text-on-primary font-bold text-sm shadow-md shadow-primary/20 hover:bg-primary/90 transition-all inline-flex items-center gap-2 cursor-pointer"
          >
            <span className="material-symbols-outlined text-[20px]">image_search</span>
            <span>Bắt đầu tra cứu ngay</span>
          </button>
        </div>
      </div>
    );
  }

  const totalCount = history.length;

  return (
    <div className="max-w-5xl mx-auto py-4 space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-3xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-primary-container flex items-center justify-center text-on-primary-fixed shadow-sm shrink-0">
            <span className="material-symbols-outlined text-[28px]">history</span>
          </div>
          <div>
            <h1 className="text-2xl font-extrabold text-on-surface tracking-tight">Lịch Sử Tra Cứu Ảnh (CBIR)</h1>
            <p className="text-xs text-on-surface-variant font-medium">
              Đã ghi nhận <strong className="text-primary font-bold">{totalCount}</strong> phiên nhận dạng giống chó
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onStartNewSearch}
            className="px-4 py-2.5 rounded-xl bg-primary text-on-primary font-bold text-xs shadow-sm hover:bg-primary/90 transition-all flex items-center gap-1.5 cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">add_photo_alternate</span>
            <span>Tra cứu mới</span>
          </button>

          <button
            type="button"
            onClick={onClearHistory}
            className="px-4 py-2.5 rounded-xl bg-surface-container hover:bg-error-container hover:text-on-error-container text-on-surface-variant font-semibold text-xs transition-all flex items-center gap-1.5 cursor-pointer"
            title="Xóa toàn bộ lịch sử"
          >
            <span className="material-symbols-outlined text-[18px]">delete_sweep</span>
            <span>Xóa lịch sử</span>
          </button>
        </div>
      </div>

      {/* History Cards List */}
      <div className="space-y-3">
        {history.map((item) => {
          const sim = item.topMatch?.similarity || 0;
          let badgeColor = 'bg-emerald-500/10 text-emerald-700 border-emerald-500/30';
          if (sim < 50) badgeColor = 'bg-rose-500/10 text-rose-700 border-rose-500/30';
          else if (sim < 80) badgeColor = 'bg-amber-500/10 text-amber-700 border-amber-500/30';

          return (
            <div
              key={item.id}
              className="p-4 rounded-2xl bg-surface-container-lowest border border-surface-container-high/60 hover:border-primary/50 transition-all duration-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4 group"
            >
              {/* Left: Images & Info */}
              <div className="flex items-center gap-4 min-w-0">
                {/* Query Image Thumbnail */}
                <div className="relative w-16 h-16 rounded-xl overflow-hidden bg-surface-container-high shrink-0 border border-surface-container-highest shadow-inner">
                  <img
                    src={item.queryPreviewUrl}
                    alt="Query"
                    className="w-full h-full object-cover"
                  />
                  <span className="absolute bottom-0 inset-x-0 bg-black/60 text-white text-[9px] text-center font-bold py-0.5">
                    Ảnh mẫu
                  </span>
                </div>

                <div className="text-on-surface-variant/40 hidden sm:block">
                  <span className="material-symbols-outlined text-[20px]">arrow_forward</span>
                </div>

                {/* Top Match Result Thumbnail */}
                {item.topMatch?.image_url && (
                  <div className="relative w-16 h-16 rounded-xl overflow-hidden bg-surface-container-high shrink-0 border border-surface-container-highest shadow-inner">
                    <img
                      src={item.topMatch.image_url}
                      alt={item.topMatch.breed_name}
                      className="w-full h-full object-cover"
                    />
                    <span className="absolute bottom-0 inset-x-0 bg-black/60 text-white text-[9px] text-center font-bold py-0.5 truncate px-1">
                      Khớp #1
                    </span>
                  </div>
                )}

                {/* Details */}
                <div className="space-y-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-base font-extrabold text-on-surface truncate">
                      {item.topMatch?.vietnamese_name || item.topMatch?.breed_name || 'Không xác định'}
                    </h3>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-bold border ${badgeColor}`}>
                      {sim.toFixed(1)}%
                    </span>
                  </div>

                  <p className="text-xs text-on-surface-variant truncate">
                    {item.topMatch?.scientific_name && (
                      <span className="italic">{item.topMatch.scientific_name} • </span>
                    )}
                    {item.topMatch?.fci_group || 'FCI Standard'}
                  </p>

                  <div className="flex flex-wrap items-center gap-3 text-[11px] text-on-surface-variant/80 pt-0.5">
                    <span className="flex items-center gap-1 font-medium">
                      <span className="material-symbols-outlined text-[14px]">schedule</span>
                      {item.formattedTime}
                    </span>
                    <span>•</span>
                    <span className="font-semibold px-2 py-0.5 rounded bg-surface-container-high text-on-surface text-[10px]">
                      {item.model === 'vit' ? 'ViT-B/16 (768d)' : 'ResNet-50 (2048d)'}
                    </span>
                    <span>•</span>
                    <span>{item.metrics?.total_latency_ms ? `${item.metrics.total_latency_ms} ms` : ''}</span>
                  </div>
                </div>
              </div>

              {/* Right: Actions */}
              <div className="flex items-center gap-2 shrink-0 self-end md:self-center">
                <button
                  type="button"
                  onClick={() => onSelectHistory(item)}
                  className="px-4 py-2 rounded-xl bg-primary-container text-on-primary-fixed hover:bg-primary hover:text-on-primary font-bold text-xs transition-all flex items-center gap-1.5 cursor-pointer shadow-sm"
                  title="Mở lại toàn bộ kết quả phiên này"
                >
                  <span className="material-symbols-outlined text-[16px]">visibility</span>
                  <span>Xem lại</span>
                </button>

                <button
                  type="button"
                  onClick={() => onDeleteHistoryItem(item.id)}
                  className="p-2 rounded-xl text-on-surface-variant hover:text-error hover:bg-error-container/40 transition-all cursor-pointer"
                  title="Xóa mục này khỏi lịch sử"
                >
                  <span className="material-symbols-outlined text-[18px]">delete</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

