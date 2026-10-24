import React from 'react';

export default function Navbar({ health, activeTab, setActiveTab, isRoiModalOpen }) {
  const isReady = health?.status === 'healthy';

  return (
    <header className={`fixed top-0 w-full z-40 bg-surface/90 backdrop-blur-xl border-b border-surface-container-high/60 shadow-[0_1px_8px_rgba(21,28,39,0.04)] transition-opacity duration-200 ${isRoiModalOpen ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
      <div className="h-20 max-w-7xl mx-auto px-6 lg:px-12 flex items-center justify-between gap-4">
        {/* Logo & Platform Info */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('cbir')}>
            <div className="w-10 h-10 rounded-2xl bg-primary-container flex items-center justify-center text-on-primary-fixed shadow-md shadow-primary-container/20">
              <span className="material-symbols-outlined text-[24px]">pets</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xl tracking-tight text-on-surface font-extrabold flex items-center gap-1.5">
                Breed Lens
                {/* <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-primary-fixed text-primary">AI v2.4</span> */}
              </span>
              <span className="text-xs text-on-surface-variant font-medium">Truy xuất hình ảnh loài chó ở mức độ chi tiết cao</span>
            </div>
          </div>

          {/* Model Status Pill */}
          <div className="hidden xl:flex items-center gap-2 pl-4 py-1.5 pr-3 rounded-full bg-surface-container-low shadow-inner border border-surface-container-high/50">
            <span className="relative flex h-2.5 w-2.5">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${isReady ? 'bg-emerald-400' : 'bg-amber-400'} opacity-75`}></span>
              <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${isReady ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
            </span>
            <span className="text-xs text-on-surface font-medium">
              {isReady 
                ? `ViT-B/16 & FAISS Sẵn Sàng (${health.total_vectors?.toLocaleString() || 8580} Vectors)` 
                : 'Đang kết nối Backend...'}
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="hidden lg:flex items-center gap-1">
          <button 
            onClick={() => setActiveTab('cbir')}
            className={`text-sm px-4 py-2 rounded-xl transition-all font-semibold flex items-center gap-2 ${
              activeTab === 'cbir' 
                ? 'bg-primary-container text-on-primary-fixed shadow-sm' 
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">image_search</span>
            Tra cứu ảnh / CBIR
          </button>
          
          <button 
            onClick={() => setActiveTab('model-hub')}
            className={`text-sm px-4 py-2 rounded-xl transition-all font-semibold flex items-center gap-2 ${
              activeTab === 'model-hub' 
                ? 'bg-primary-container text-on-primary-fixed shadow-sm' 
                : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">neurology</span>
            Mô hình ViT & ResNet
          </button>

          <a 
            href="/docs" 
            target="_blank" 
            rel="noreferrer"
            className="text-sm px-4 py-2 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-all font-semibold flex items-center gap-1.5"
          >
            <span className="material-symbols-outlined text-[18px]">api</span>
            FastAPI Docs
          </a>
        </nav>

        {/* User / Quick Actions */}
        <div className="flex items-center gap-3">
          <div className="text-right hidden sm:block">
            <p className="text-xs font-bold text-on-surface">Stanford Dogs</p>
            <p className="text-[11px] text-on-surface-variant">120 Giống Quốc Tế</p>
          </div>
          <div className="w-10 h-10 rounded-full bg-surface-container-high flex items-center justify-center text-primary shadow-sm border border-surface-container-highest">
            <span className="material-symbols-outlined text-[20px]">verified</span>
          </div>
        </div>
      </div>
    </header>
  );
}

