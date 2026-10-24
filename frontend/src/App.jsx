import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import PresetsBar from './components/PresetsBar';
import Dropzone from './components/Dropzone';
import ConfigPanel from './components/ConfigPanel';
import RoiCropper from './components/RoiCropper';
import ResultsDashboard from './components/ResultsDashboard';

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
      // Fetch preset image as blob
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

  return (
    <div className="min-h-screen bg-surface font-sans text-on-surface flex flex-col">
      {/* Top Navigation */}
      <Navbar health={health} activeTab={activeTab} setActiveTab={setActiveTab} isRoiModalOpen={isRoiModalOpen} />

      {/* Main Container */}
      <main className="w-full pt-24 pb-16 flex-1">
        <div className="max-w-7xl mx-auto px-6 lg:px-12 space-y-8">
          
          {activeTab === 'cbir' ? (
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

                    <p className="text-sm text-on-surface-variant leading-relaxed">
                      Nền tảng CBIR (Content-Based Image Retrieval) kết hợp trích xuất đặc trưng Fine-Grained từ mạng <strong className="text-on-surface font-semibold">Vision Transformer ViT-B/16</strong> và <strong className="text-on-surface font-semibold">ResNet-50</strong>. Hỗ trợ khoanh vùng ROI trọng tâm để loại bỏ tạp âm môi trường và so khớp siêu phân giải theo độ tương đồng Cosine Similarity.
                    </p>
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
                      className="px-4 py-2.5 rounded-2xl bg-surface-container-high hover:bg-surface-container text-secondary text-xs font-bold transition-all shadow-sm flex items-center gap-2"
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
          ) : (
            /* Model Hub Tab */
            <div className="space-y-6 max-w-4xl mx-auto py-6">
              <div className="p-8 rounded-3xl bg-surface-container-lowest border border-surface-container-high/60 space-y-4 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-2xl bg-primary-fixed flex items-center justify-center text-primary">
                    <span className="material-symbols-outlined text-[28px]">neurology</span>
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-on-surface">Mô Hình Thị Giác Máy Học (Deep Learning Backbones)</h2>
                    <p className="text-xs text-on-surface-variant">Chi tiết kiến trúc mạng trích xuất đặc trưng và chỉ mục FAISS</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
                  <div className="p-5 rounded-2xl bg-surface-container-low border border-primary/30 space-y-3">
                    <div className="flex justify-between items-center">
                      <h3 className="font-bold text-base text-primary">Vision Transformer (ViT-B/16)</h3>
                      <span className="px-2 py-0.5 rounded-full bg-primary-fixed text-primary text-[10px] font-bold">Khuyên Dùng</span>
                    </div>
                    <p className="text-xs text-on-surface-variant leading-relaxed">
                      Sử dụng cơ chế Self-Attention đa đầu (Multi-Head Self-Attention) phân chia ảnh thành các patch 16x16. Nắm bắt các đặc trưng hình thái tinh vi (đôi mắt, sống mũi, vân lông) của từng giống chó.
                    </p>
                    <div className="text-[11px] font-mono space-y-1 text-on-surface-variant pt-2 border-t border-surface-container-high">
                      <p>• Embedding Dimension: <strong>768 dimensions</strong></p>
                      <p>• Pretrained weights: <strong>ViT Base Patch 16 (Fine-Tuned)</strong></p>
                      <p>• Vector Normalization: <strong>L2-Normalized (Unit sphere)</strong></p>
                    </div>
                  </div>

                  <div className="p-5 rounded-2xl bg-surface-container-low border border-secondary/30 space-y-3">
                    <div className="flex justify-between items-center">
                      <h3 className="font-bold text-base text-secondary">ResNet-50 (Baseline)</h3>
                      <span className="px-2 py-0.5 rounded-full bg-secondary-fixed text-secondary text-[10px] font-bold">Cổ Điển</span>
                    </div>
                    <p className="text-xs text-on-surface-variant leading-relaxed">
                      Mạng nơ-ron tích chập 50 lớp với kết nối phần dư (Residual Skip Connections). Tốc độ tính toán nhanh, hội tụ tốt, trích xuất đặc trưng qua lớp Global Average Pooling.
                    </p>
                    <div className="text-[11px] font-mono space-y-1 text-on-surface-variant pt-2 border-t border-surface-container-high">
                      <p>• Embedding Dimension: <strong>2048 dimensions</strong></p>
                      <p>• Feature Layer: <strong>Pooler Output (Conv5_x)</strong></p>
                      <p>• Margin Adjustment: <strong>Distribution Calibrated</strong></p>
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-2xl bg-surface-container text-xs text-on-surface-variant space-y-2 mt-4">
                  <h4 className="font-bold text-on-surface">Chỉ Mục Tìm Kiếm FAISS (Facebook AI Similarity Search)</h4>
                  <p>
                    Hệ thống lưu trữ 8.580 vector đặc trưng tương ứng với 120 giống chó quốc tế trong chỉ mục <code>IndexFlatIP</code> (Inner Product tương đương Cosine Similarity với vector đã chuẩn hóa L2). Thời gian truy xuất chỉ từ <strong>2ms đến 8ms</strong>.
                  </p>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

