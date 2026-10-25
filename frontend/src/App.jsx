import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import PresetsBar from './components/PresetsBar';
import Dropzone from './components/Dropzone';
import ConfigPanel from './components/ConfigPanel';
import RoiCropper from './components/RoiCropper';
import ResultsDashboard from './components/ResultsDashboard';
import HistoryView from './components/HistoryView';
import ModelHub from './components/ModelHub';

export default function App() {
  const [activeTab, setActiveTab] = useState('cbir');
  const [health, setHealth] = useState(null);
  const [presets, setPresets] = useState([]);
  const [selectedPresetId, setSelectedPresetId] = useState(null);

  const [activeFile, setActiveFile] = useState(null);
  const [activePreview, setActivePreview] = useState(null);
  const [cropCoords, setCropCoords] = useState(null);
  const [isRoiModalOpen, setIsRoiModalOpen] = useState(false);

  const [selectedModel, setSelectedModel] = useState('vit');
  const [topK, setTopK] = useState(10);
  const [metric, setMetric] = useState('cosine');

  const [isLoading, setIsLoading] = useState(false);
  const [resultsData, setResultsData] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  // Search History State (persisted in localStorage)
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('breed_lens_history');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.warn('Error reading history from localStorage:', e);
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem('breed_lens_history', JSON.stringify(history));
    } catch (e) {
      console.warn('Could not save history to localStorage:', e);
    }
  }, [history]);

  // Helper to create persistent thumbnail for history
  const createThumbnail = (imageSource, maxWidth = 160) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ratio = img.height / img.width;
        canvas.width = maxWidth;
        canvas.height = Math.round(maxWidth * ratio);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        try {
          resolve(canvas.toDataURL('image/jpeg', 0.8));
        } catch {
          resolve(imageSource);
        }
      };
      img.onerror = () => resolve(imageSource);
      img.src = imageSource;
    });
  };

  // Initialize
  useEffect(() => {
    // 1. Fetch Health
    axios.get('/api/health')
      .then(res => setHealth(res.data))
      .catch(err => console.warn('Backend not ready:', err));

    // 2. Fetch Presets
    axios.get('/api/presets')
      .then(res => setPresets(res.data))
      .catch(err => console.warn('Could not fetch presets:', err));
  }, []);

  // Handle Preset Click
  const handleSelectPreset = async (preset) => {
    setSelectedPresetId(preset.id);
    setErrorMsg(null);
    try {
      const resp = await fetch(preset.image_url);
      const blob = await resp.blob();
      const file = new File([blob], `${preset.id}.jpg`, { type: 'image/jpeg' });
      setActiveFile(file);
      setActivePreview(preset.image_url);
      setIsRoiModalOpen(true);
    } catch (e) {
      console.error('Failed to load preset image:', e);
      setErrorMsg('Không thể tải ảnh mẫu này từ máy chủ.');
    }
  };

  // Handle File Upload from Dropzone
  const handleFileSelected = (file) => {
    setSelectedPresetId(null);
    setErrorMsg(null);
    setActiveFile(file);
    const objectUrl = URL.createObjectURL(file);
    setActivePreview(objectUrl);
    setIsRoiModalOpen(true);
  };

  const handleClearFile = () => {
    setActiveFile(null);
    setActivePreview(null);
    setCropCoords(null);
    setSelectedPresetId(null);
    setResultsData(null);
    setIsRoiModalOpen(false);
  };

  // Submit Retrieval
  const handleSubmitRetrieval = async () => {
    if (!activeFile) return;

    setIsLoading(true);
    setIsRoiModalOpen(false);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', activeFile);
    formData.append('model_name', selectedModel);
    formData.append('top_k', topK);
    formData.append('metric', metric);

    if (cropCoords && cropCoords.width > 0 && cropCoords.height > 0) {
      formData.append('crop_x', cropCoords.x);
      formData.append('crop_y', cropCoords.y);
      formData.append('crop_width', cropCoords.width);
      formData.append('crop_height', cropCoords.height);
    }

    try {
      const response = await axios.post('/api/retrieve', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResultsData(response.data);

      // Save to History
      try {
        const thumb = await createThumbnail(activePreview);
        const newEntry = {
          id: Date.now().toString(),
          formattedTime: new Date().toLocaleString('vi-VN', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          }),
          queryPreviewUrl: thumb,
          model: selectedModel,
          topMatch: response.data.top_match,
          metrics: response.data.metrics,
          fullResults: response.data,
          cropCoords: cropCoords,
        };
        setHistory(prev => [newEntry, ...prev.slice(0, 49)]); // keep up to 50 entries
      } catch (e) {
        console.warn('Failed to record history entry:', e);
      }

      // Smooth scroll to results
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      console.error('Retrieval error:', err);
      const msg = err.response?.data?.detail || 'Đã có lỗi xảy ra trong quá trình truy xuất.';
      setErrorMsg(msg);
    } finally {
      setIsLoading(false);
    }
  };

  // History action handlers
  const handleSelectHistory = (item) => {
    setActivePreview(item.queryPreviewUrl);
    setSelectedModel(item.model);
    setResultsData(item.fullResults);
    setActiveTab('cbir');
    setTimeout(() => {
      document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  const handleDeleteHistoryItem = (id) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  };

  const handleClearHistory = () => {
    if (window.confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử tra cứu không?')) {
      setHistory([]);
    }
  };

  return (
    <div className="min-h-screen bg-surface font-sans text-on-surface flex flex-col">
      {/* Top Navigation */}
      <Navbar 
        health={health} 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        isRoiModalOpen={isRoiModalOpen}
        historyCount={history.length}
      />

      {/* Main Container */}
      <main className="w-full pt-24 pb-16 flex-1">
        <div className="max-w-7xl mx-auto px-6 lg:px-12 space-y-8">
          
          {activeTab === 'cbir' && (
            <>
              {/* SECTION: Editorial Hero & Header */}
              <section className="space-y-4 pt-2">
                <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-2">
                  <div className="space-y-3 max-w-3xl">
                    <h1 className="text-3xl lg:text-4xl font-extrabold text-on-surface tracking-tight">
                      HỆ THỐNG TRA CỨU ẢNH DỰA VÀO NỘI DUNG
                      <span className="block text-primary text-2xl lg:text-3xl font-black mt-1">
                        Nhận Dạng Giống Chó Tự Động Qua Không Gian Vector
                      </span>
                    </h1>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        if (presets.length > 0) {
                          const rand = presets[Math.floor(Math.random() * presets.length)];
                          handleSelectPreset(rand);
                        }
                      }}
                      className="px-4 py-2.5 rounded-2xl bg-surface-container-high hover:bg-surface-container text-secondary text-xs font-bold transition-all shadow-sm flex items-center gap-2 cursor-pointer"
                    >
                      <span className="material-symbols-outlined text-[18px]">shuffle</span>
                      <span>Ngẫu nhiên mẫu thử</span>
                    </button>
                  </div>
                </div>

                {/* Error Banner */}
                {errorMsg && (
                  <div className="p-4 rounded-2xl bg-error-container text-on-error-container text-xs font-bold flex items-center gap-2 border border-error/20">
                    <span className="material-symbols-outlined text-[20px] text-error">error</span>
                    <span>{errorMsg}</span>
                  </div>
                )}
              </section>

              {/* SECTION: Workstation */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                {/* Left Column: Dropzone & Presets (7 cols) */}
                <div className="lg:col-span-7 space-y-6">
                  <Dropzone
                    onFileSelected={handleFileSelected}
                    activeFile={activeFile}
                    activePreview={activePreview}
                    onClearFile={handleClearFile}
                    onOpenRoiModal={() => setIsRoiModalOpen(true)}
                    onSubmitRetrieval={handleSubmitRetrieval}
                    isLoading={isLoading}
                    cropCoords={cropCoords}
                  />

                  <PresetsBar
                    presets={presets}
                    onSelectPreset={handleSelectPreset}
                    selectedPresetId={selectedPresetId}
                  />
                </div>

                {/* Right Column: Hyperparameters & Configuration (5 cols) */}
                <div className="lg:col-span-5 space-y-6">
                  <ConfigPanel
                    selectedModel={selectedModel}
                    setSelectedModel={setSelectedModel}
                    topK={topK}
                    setTopK={setTopK}
                    metric={metric}
                    setMetric={setMetric}
                  />
                </div>
              </div>

              {/* Pop-up Modal: Phần 3 - Phân Vùng Quan Tâm (Canine ROI Workspace) */}
              <RoiCropper
                isOpen={isRoiModalOpen}
                onClose={() => setIsRoiModalOpen(false)}
                imageSrc={activePreview}
                onCropChange={setCropCoords}
                onSubmit={handleSubmitRetrieval}
                isLoading={isLoading}
                selectedModel={selectedModel}
              />

              {/* SECTION: Inference Results Dashboard */}
              <ResultsDashboard
                resultsData={resultsData}
                queryPreviewUrl={activePreview}
              />
            </>
          )}

          {/* History Tab */}
          {activeTab === 'history' && (
            <HistoryView
              history={history}
              onSelectHistory={handleSelectHistory}
              onDeleteHistoryItem={handleDeleteHistoryItem}
              onClearHistory={handleClearHistory}
              onStartNewSearch={() => setActiveTab('cbir')}
            />
          )}

          {/* Model Hub Tab */}
          {activeTab === 'model-hub' && (
            <ModelHub health={health} />
          )}

        </div>
      </main>
    </div>
  );
}
