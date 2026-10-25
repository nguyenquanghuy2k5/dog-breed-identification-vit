import React, { useState } from 'react';

const VIT_LAYERS = [
  { name: 'Input Patch Embedding', detail: 'Ảnh 224×224 → 196 patches 16×16 → Linear Projection → 768d tokens', type: 'input' },
  { name: 'CLS Token Prepend', detail: 'Thêm 1 token phân loại [CLS] → 197 tokens tổng', type: 'special' },
  { name: 'Positional Encoding', detail: 'Cộng thêm embedding vị trí 1D học được để bảo toàn cấu trúc không gian', type: 'encode' },
  { name: 'Transformer Encoder ×12', detail: 'Multi-Head Self-Attention (12 heads, 64d/head) + MLP FFN (3072d hidden)', type: 'core' },
  { name: '[CLS] Token Output', detail: 'Lấy vector của token [CLS] tại lớp cuối cùng (768 chiều)', type: 'output' },
  { name: 'L2 Normalization', detail: 'Chuẩn hóa L2 → Unit vector trên siêu cầu, phù hợp Cosine Similarity', type: 'norm' },
];

const RESNET_LAYERS = [
  { name: 'Input Normalization', detail: 'Chuẩn hóa ảnh 224×224: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]', type: 'input' },
  { name: 'Conv1 + MaxPool', detail: 'Conv 7×7, stride=2 → BatchNorm → ReLU → MaxPool 3×3 stride=2', type: 'core' },
  { name: 'Layer 1–4 (Bottleneck ×16)', detail: '4 giai đoạn ResBlock với Residual Skip Connection, channel: 64→256→512→1024→2048', type: 'core' },
  { name: 'Global Average Pooling', detail: 'Rút gọn feature map (7×7×2048) → vector 2048 chiều qua GAP', type: 'output' },
  { name: 'Pooler Output', detail: 'Đầu ra cuối 2048d, trích xuất từ Conv5_x làm embedding CBIR', type: 'output' },
  { name: 'L2 Normalization', detail: 'Chuẩn hóa L2 + Margin Calibration ×0.965 để cân bằng phân phối similarity', type: 'norm' },
];

const layerColor = {
  input: 'bg-sky-500/10 border-sky-500/40 text-sky-700',
  special: 'bg-violet-500/10 border-violet-500/40 text-violet-700',
  encode: 'bg-indigo-500/10 border-indigo-500/40 text-indigo-700',
  core: 'bg-primary/10 border-primary/40 text-primary',
  output: 'bg-emerald-500/10 border-emerald-500/40 text-emerald-700',
  norm: 'bg-amber-500/10 border-amber-500/40 text-amber-700',
};

function MetricBar({ label, vitValue, resnetValue, unit = '' }) {
  const max = Math.max(vitValue, resnetValue);
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-[11px] font-medium text-on-surface-variant">
        <span>{label}</span>
        <span className="text-on-surface font-bold">{vitValue}{unit} vs {resnetValue}{unit}</span>
      </div>
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="text-[10px] w-16 font-semibold text-primary shrink-0">ViT-B/16</span>
          <div className="flex-1 h-2 bg-surface-container rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-500"
              style={{ width: `${(vitValue / max) * 100}%` }}
            />
          </div>
          <span className="text-[10px] font-bold text-primary w-12 text-right">{vitValue}{unit}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] w-16 font-semibold text-secondary shrink-0">ResNet-50</span>
          <div className="flex-1 h-2 bg-surface-container rounded-full overflow-hidden">
            <div
              className="h-full bg-secondary rounded-full transition-all duration-500"
              style={{ width: `${(resnetValue / max) * 100}%` }}
            />
          </div>
          <span className="text-[10px] font-bold text-secondary w-12 text-right">{resnetValue}{unit}</span>
        </div>
      </div>
    </div>
  );
}

export default function ModelHub({ health }) {
  const [activeModel, setActiveModel] = useState('vit');

  return (
    <div className="space-y-8 max-w-5xl mx-auto py-4">

      {/* === SECTION HEADER === */}
      <div className="flex items-center gap-4">
        <div className="w-14 h-14 rounded-2xl bg-primary-container flex items-center justify-center text-on-primary-fixed shadow-md shrink-0">
          <span className="material-symbols-outlined text-[30px]">neurology</span>
        </div>
        <div>
          <h1 className="text-3xl font-extrabold text-on-surface tracking-tight">Deep Learning Backbones</h1>
          <p className="text-sm text-on-surface-variant">Kiến trúc mạng trích xuất đặc trưng & chỉ mục Vector FAISS cho hệ thống CBIR</p>
        </div>
      </div>

      {/* === QUICK STATS ROW === */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Gallery Vectors', value: health?.total_vectors?.toLocaleString() || '8,580', icon: 'database', color: 'text-primary' },
          { label: 'Giống Chó', value: '120', icon: 'pets', color: 'text-amber-600' },
          { label: 'FAISS Latency', value: '~4 ms', icon: 'bolt', color: 'text-emerald-600' },
          { label: 'Dataset Split', value: 'Test Set', icon: 'dataset', color: 'text-secondary' },
        ].map(stat => (
          <div key={stat.label} className="p-4 rounded-2xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm flex items-center gap-3">
            <span className={`material-symbols-outlined text-[24px] ${stat.color}`}>{stat.icon}</span>
            <div>
              <p className="text-[11px] text-on-surface-variant font-medium">{stat.label}</p>
              <p className={`text-lg font-extrabold ${stat.color}`}>{stat.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* === MODEL SELECTOR TABS === */}
      <div className="flex gap-2 p-1.5 rounded-2xl bg-surface-container-low w-fit shadow-inner">
        <button
          onClick={() => setActiveModel('vit')}
          className={`px-5 py-2.5 rounded-xl text-sm font-bold transition-all flex items-center gap-2 cursor-pointer ${
            activeModel === 'vit' ? 'bg-surface shadow-sm text-primary' : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          <span className="material-symbols-outlined text-[18px]">auto_awesome</span>
          Vision Transformer (ViT-B/16)
          <span className="px-1.5 py-0.5 rounded-full bg-primary/10 text-primary text-[10px] font-bold">768d</span>
        </button>
        <button
          onClick={() => setActiveModel('resnet')}
          className={`px-5 py-2.5 rounded-xl text-sm font-bold transition-all flex items-center gap-2 cursor-pointer ${
            activeModel === 'resnet' ? 'bg-surface shadow-sm text-secondary' : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          <span className="material-symbols-outlined text-[18px]">layers</span>
          ResNet-50 (Baseline)
          <span className="px-1.5 py-0.5 rounded-full bg-secondary/10 text-secondary text-[10px] font-bold">2048d</span>
        </button>
      </div>

      {/* === VIT DETAIL === */}
      {activeModel === 'vit' && (
        <div className="space-y-6 animate-fadeIn">
          {/* Header card */}
          <div className="p-6 rounded-3xl bg-surface-container-lowest border border-primary/20 shadow-sm space-y-5">
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
              <div className="space-y-2 max-w-2xl">
                <div className="flex items-center gap-3">
                  <h2 className="text-2xl font-extrabold text-primary tracking-tight">Vision Transformer — ViT-B/16</h2>
                  <span className="px-3 py-0.5 rounded-full bg-primary/10 text-primary text-xs font-bold border border-primary/20">Khuyên Dùng</span>
                </div>
                <p className="text-sm text-on-surface-variant leading-relaxed">
                  Mô hình Transformer thuần túy ứng dụng vào thị giác máy tính, được giới thiệu bởi Google Brain (2020). Thay vì tích chập, ViT chia ảnh thành các patch nhỏ và xử lý chúng như một chuỗi token qua cơ chế <strong className="text-on-surface">Multi-Head Self-Attention</strong>, cho phép mô hình học các phụ thuộc toàn cục xuyên toàn bộ ảnh.
                </p>
                <p className="text-sm text-on-surface-variant leading-relaxed">
                  Sau khi fine-tuning trên Stanford Dogs Dataset bằng <strong className="text-on-surface">Triplet Margin Loss</strong>, ViT tạo ra không gian embedding kết cụm các cá thể cùng giống chó ở khoảng cách nhỏ và đẩy xa các giống khác loài.
                </p>
              </div>

              <div className="shrink-0 space-y-2 min-w-[200px]">
                {[
                  { label: 'Architecture', val: 'Transformer Encoder' },
                  { label: 'Patch Size', val: '16×16 pixels' },
                  { label: 'Embedding Dim', val: '768 dimensions' },
                  { label: 'Attention Heads', val: '12 heads' },
                  { label: 'MLP Hidden Dim', val: '3,072 (×4)' },
                  { label: 'Num Layers', val: '12 Encoder Blocks' },
                  { label: 'Training Loss', val: 'Triplet Margin (α=1.0)' },
                  { label: 'Frozen Layers', val: '10/12 đầu (tránh overfitting)' },
                ].map(row => (
                  <div key={row.label} className="flex justify-between gap-4 text-[11px]">
                    <span className="text-on-surface-variant font-medium">{row.label}</span>
                    <span className="font-bold text-on-surface text-right">{row.val}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Layer pipeline */}
            <div>
              <h3 className="text-xs font-bold text-on-surface uppercase tracking-wider mb-3">Pipeline Trích Xuất Vector (Inference)</h3>
              <div className="flex flex-wrap gap-2 items-center">
                {VIT_LAYERS.map((layer, i) => (
                  <React.Fragment key={layer.name}>
                    <div className={`px-3 py-2 rounded-xl border text-[10px] font-bold ${layerColor[layer.type]} max-w-[180px]`}>
                      <div>{layer.name}</div>
                      <div className="font-normal opacity-75 mt-0.5 leading-tight">{layer.detail}</div>
                    </div>
                    {i < VIT_LAYERS.length - 1 && (
                      <span className="material-symbols-outlined text-on-surface-variant/40 text-[18px] shrink-0">arrow_forward</span>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>

          {/* Training Details */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-5 rounded-2xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-primary text-[22px]">psychology</span>
                <h3 className="font-bold text-sm text-on-surface">Chiến Lược Huấn Luyện</h3>
              </div>
              <ul className="space-y-1.5 text-xs text-on-surface-variant">
                <li className="flex gap-2"><span className="text-primary mt-0.5">▸</span><span><strong className="text-on-surface">Triplet Loss</strong> với biên margin α = 1.0</span></li>
                <li className="flex gap-2"><span className="text-primary mt-0.5">▸</span><span>Anchor / Positive / Negative triplet sampling ngẫu nhiên theo nhãn</span></li>
                <li className="flex gap-2"><span className="text-primary mt-0.5">▸</span><span>Cosine LR Scheduler + Early Stopping</span></li>
                <li className="flex gap-2"><span className="text-primary mt-0.5">▸</span><span>Mixed Precision FP16 (GPU) / FP32 (CPU)</span></li>
                <li className="flex gap-2"><span className="text-primary mt-0.5">▸</span><span>Data Augmentation: RandomCrop, HorizontalFlip, Rotation ±15°, ColorJitter</span></li>
              </ul>
            </div>

            <div className="p-5 rounded-2xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-emerald-600 text-[22px]">verified</span>
                <h3 className="font-bold text-sm text-on-surface">Hiệu Suất CBIR</h3>
              </div>
              <ul className="space-y-1.5 text-xs text-on-surface-variant">
                <li className="flex justify-between"><span>Accuracy@5</span><strong className="text-emerald-600">~88.6%</strong></li>
                <li className="flex justify-between"><span>Precision@5</span><strong className="text-emerald-600">~86.4%</strong></li>
                <li className="flex justify-between"><span>Recall@5</span><strong className="text-emerald-600">~84.1%</strong></li>
                <li className="flex justify-between"><span>mAP@5</span><strong className="text-emerald-600">~84.2%</strong></li>
                <li className="flex justify-between"><span>FAISS Search Latency</span><strong className="text-emerald-600">~4.2 ms</strong></li>
              </ul>
            </div>

            <div className="p-5 rounded-2xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-amber-600 text-[22px]">database</span>
                <h3 className="font-bold text-sm text-on-surface">Thông Số FAISS Index</h3>
              </div>
              <ul className="space-y-1.5 text-xs text-on-surface-variant">
                <li className="flex justify-between"><span>Index Type</span><strong className="text-on-surface">IndexFlatIP</strong></li>
                <li className="flex justify-between"><span>Số Vectors</span><strong className="text-on-surface">8,580</strong></li>
                <li className="flex justify-between"><span>Dimension</span><strong className="text-on-surface">768d / FP32</strong></li>
                <li className="flex justify-between"><span>Metric</span><strong className="text-on-surface">Cosine Similarity</strong></li>
                <li className="flex justify-between"><span>File Size</span><strong className="text-on-surface">~26.3 MB</strong></li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* === RESNET DETAIL === */}
      {activeModel === 'resnet' && (
        <div className="space-y-6 animate-fadeIn">
          <div className="p-6 rounded-3xl bg-surface-container-lowest border border-secondary/20 shadow-sm space-y-5">
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
              <div className="space-y-2 max-w-2xl">
                <div className="flex items-center gap-3">
                  <h2 className="text-2xl font-extrabold text-secondary tracking-tight">ResNet-50 — Baseline</h2>
                  <span className="px-3 py-0.5 rounded-full bg-secondary/10 text-secondary text-xs font-bold border border-secondary/20">Đối Chiếu</span>
                </div>
                <p className="text-sm text-on-surface-variant leading-relaxed">
                  Mạng nơ-ron tích chập sâu 50 lớp với kiến trúc <strong className="text-on-surface">Residual Network (ResNet)</strong> của Microsoft Research (2015). Đặc điểm nổi bật là các <strong className="text-on-surface">Skip Connection</strong> kết nối tắt qua từng Bottleneck Block giúp tín hiệu gradient truyền thẳng về các lớp đầu, giải quyết hiện tượng biến mất gradient trong mạng sâu.
                </p>
                <p className="text-sm text-on-surface-variant leading-relaxed">
                  Trong hệ thống CBIR này, ResNet-50 được dùng như baseline đối chứng: trích xuất đặc trưng từ lớp <strong className="text-on-surface">Pooler Output (Conv5_x)</strong>, sau đó áp dụng điều chỉnh biên phân phối (×0.965) để cân bằng thang đo similarity so với ViT.
                </p>
              </div>

              <div className="shrink-0 space-y-2 min-w-[200px]">
                {[
                  { label: 'Architecture', val: 'CNN + Residual Blocks' },
                  { label: 'Depth', val: '50 layers (Bottleneck)' },
                  { label: 'Embedding Dim', val: '2048 dimensions' },
                  { label: 'Feature Layer', val: 'Conv5_x / Pooler Output' },
                  { label: 'Global Pooling', val: 'Global Average Pooling' },
                  { label: 'ResBlocks', val: '3+4+6+3 = 16 Bottlenecks' },
                  { label: 'Pretrained', val: 'ImageNet-1K (microsoft/resnet-50)' },
                  { label: 'Calibration', val: '×0.965 Margin Adjustment' },
                ].map(row => (
                  <div key={row.label} className="flex justify-between gap-4 text-[11px]">
                    <span className="text-on-surface-variant font-medium">{row.label}</span>
                    <span className="font-bold text-on-surface text-right">{row.val}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Layer pipeline */}
            <div>
              <h3 className="text-xs font-bold text-on-surface uppercase tracking-wider mb-3">Pipeline Trích Xuất Vector (Inference)</h3>
              <div className="flex flex-wrap gap-2 items-center">
                {RESNET_LAYERS.map((layer, i) => (
                  <React.Fragment key={layer.name}>
                    <div className={`px-3 py-2 rounded-xl border text-[10px] font-bold ${layerColor[layer.type]} max-w-[190px]`}>
                      <div>{layer.name}</div>
                      <div className="font-normal opacity-75 mt-0.5 leading-tight">{layer.detail}</div>
                    </div>
                    {i < RESNET_LAYERS.length - 1 && (
                      <span className="material-symbols-outlined text-on-surface-variant/40 text-[18px] shrink-0">arrow_forward</span>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>

          {/* ResNet performance cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-5 rounded-2xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-secondary text-[22px]">layers</span>
                <h3 className="font-bold text-sm text-on-surface">Kiến Trúc Residual Block</h3>
              </div>
              <ul className="space-y-1.5 text-xs text-on-surface-variant">
                <li className="flex gap-2"><span className="text-secondary mt-0.5">▸</span><span><strong className="text-on-surface">Bottleneck Block</strong>: 1×1 → 3×3 → 1×1 conv giảm tham số</span></li>
                <li className="flex gap-2"><span className="text-secondary mt-0.5">▸</span><span>Skip Connection: <code className="bg-surface-container-high px-1 rounded text-[10px]">F(x) + x</code> tổng hợp đặc trưng</span></li>
                <li className="flex gap-2"><span className="text-secondary mt-0.5">▸</span><span>BatchNorm + ReLU sau mỗi conv layer</span></li>
                <li className="flex gap-2"><span className="text-secondary mt-0.5">▸</span><span>Channels: 64 → 256 → 512 → 1024 → 2048</span></li>
              </ul>
            </div>

            <div className="p-5 rounded-2xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-amber-500 text-[22px]">analytics</span>
                <h3 className="font-bold text-sm text-on-surface">Hiệu Suất CBIR</h3>
              </div>
              <ul className="space-y-1.5 text-xs text-on-surface-variant">
                <li className="flex justify-between"><span>Accuracy@5</span><strong className="text-amber-600">~78.2%</strong></li>
                <li className="flex justify-between"><span>Precision@5</span><strong className="text-amber-600">~74.8%</strong></li>
                <li className="flex justify-between"><span>Recall@5</span><strong className="text-amber-600">~73.1%</strong></li>
                <li className="flex justify-between"><span>mAP@5</span><strong className="text-amber-600">~72.5%</strong></li>
                <li className="flex justify-between"><span>FAISS Search Latency</span><strong className="text-amber-600">~5.8 ms</strong></li>
              </ul>
            </div>

            <div className="p-5 rounded-2xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-amber-600 text-[22px]">database</span>
                <h3 className="font-bold text-sm text-on-surface">Thông Số FAISS Index</h3>
              </div>
              <ul className="space-y-1.5 text-xs text-on-surface-variant">
                <li className="flex justify-between"><span>Index Type</span><strong className="text-on-surface">IndexFlatIP</strong></li>
                <li className="flex justify-between"><span>Số Vectors</span><strong className="text-on-surface">8,580</strong></li>
                <li className="flex justify-between"><span>Dimension</span><strong className="text-on-surface">2048d / FP32</strong></li>
                <li className="flex justify-between"><span>Metric</span><strong className="text-on-surface">Cosine Similarity</strong></li>
                <li className="flex justify-between"><span>File Size</span><strong className="text-on-surface">~67.0 MB</strong></li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* === COMPARISON TABLE === */}
      <div className="p-6 rounded-3xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-5">
        <h2 className="text-lg font-extrabold text-on-surface">So Sánh Hiệu Suất Toàn Diện (ViT-B/16 vs ResNet-50)</h2>
        <div className="space-y-4">
          <MetricBar label="Accuracy@5 (%)" vitValue={88.6} resnetValue={78.2} unit="%" />
          <MetricBar label="mAP@5 (%)" vitValue={84.2} resnetValue={72.5} unit="%" />
          <MetricBar label="FAISS Latency (ms) — thấp hơn = tốt hơn" vitValue={4.2} resnetValue={5.8} unit=" ms" />
        </div>

        <div className="overflow-x-auto mt-2">
          <table className="w-full text-xs text-on-surface-variant border-collapse">
            <thead>
              <tr className="border-b border-surface-container-high">
                <th className="text-left py-2.5 pr-4 font-bold text-on-surface text-sm">Tiêu Chí</th>
                <th className="text-center py-2.5 px-4 font-bold text-primary">ViT-B/16</th>
                <th className="text-center py-2.5 px-4 font-bold text-secondary">ResNet-50</th>
                <th className="text-left py-2.5 pl-4 font-bold text-on-surface">Nhận Xét</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-container-high/60">
              {[
                ['Embedding Dimension', '768d', '2048d', 'ViT nhỏ gọn hơn ~2.7× → FAISS nhanh hơn'],
                ['FAISS Index Size', '~26.3 MB', '~67.0 MB', 'ViT dễ triển khai trên thiết bị hạn chế RAM'],
                ['Accuracy@5', '~88.6%', '~78.2%', 'ViT vượt trội nhờ global attention'],
                ['mAP@5', '~84.2%', '~72.5%', 'ViT gom cụm giống chó tốt hơn đáng kể'],
                ['Search Latency', '~4.2 ms', '~5.8 ms', 'Cả hai đều rất nhanh, ViT có lợi thế nhỏ'],
                ['Độ ổn định vs Nhiễu nền', '★★★★★', '★★★☆☆', 'ViT + ROI triệt tiêu hoàn toàn nhiễu phông'],
                ['Tốc độ Inference CPU', 'Trung bình', 'Nhanh hơn ~20%', 'ResNet ít tham số Attention hơn'],
                ['Dùng trong dự án', 'Mặc định (Recommended)', 'Đối chiếu Baseline', '—'],
              ].map(([crit, vit, resnet, note]) => (
                <tr key={crit} className="hover:bg-surface-container-low/40 transition-colors">
                  <td className="py-2.5 pr-4 font-semibold text-on-surface">{crit}</td>
                  <td className="py-2.5 px-4 text-center font-bold text-primary">{vit}</td>
                  <td className="py-2.5 px-4 text-center font-bold text-secondary">{resnet}</td>
                  <td className="py-2.5 pl-4 text-on-surface-variant">{note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* === FAISS SECTION === */}
      <div className="p-6 rounded-3xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center text-amber-600 shrink-0">
            <span className="material-symbols-outlined text-[22px]">database</span>
          </div>
          <div>
            <h2 className="text-lg font-extrabold text-on-surface">Chỉ Mục Tìm Kiếm Vector — FAISS (Facebook AI Similarity Search)</h2>
            <p className="text-xs text-on-surface-variant">Thư viện tìm kiếm vector láng giềng gần nhất tốc độ cao của Meta AI</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <p className="text-sm text-on-surface-variant leading-relaxed">
              FAISS lập chỉ mục toàn bộ <strong className="text-on-surface">8,580 vector đặc trưng</strong> của tập Gallery (Stanford Dogs Test Set) vào cấu trúc <code className="bg-surface-container-high px-1.5 py-0.5 rounded text-[11px] font-mono text-on-surface">IndexFlatIP</code> (Inner Product). Với vector đã được chuẩn hóa L2 (unit vector), phép tính Inner Product tương đương hoàn toàn với <strong className="text-on-surface">Cosine Similarity</strong>.
            </p>
            <p className="text-sm text-on-surface-variant leading-relaxed">
              Đây là dạng chỉ mục tìm kiếm chính xác 100% (Exact Search, không xấp xỉ), đảm bảo truy vấn luôn trả về Top-K kết quả tương đồng nhất tuyệt đối trong toàn bộ không gian vector.
            </p>
          </div>

          <div className="space-y-2">
            <h3 className="text-xs font-bold text-on-surface uppercase tracking-wider">Quy Trình Tìm Kiếm</h3>
            {[
              { step: '1', label: 'Tiền xử lý ảnh truy vấn', detail: 'ROI crop → resize → normalize → tensor' },
              { step: '2', label: 'Trích xuất embedding', detail: 'ViT/ResNet forward pass → 768d/2048d vector' },
              { step: '3', label: 'L2 Normalization', detail: 'Vector chuẩn hóa về unit length → unit sphere' },
              { step: '4', label: 'FAISS IndexFlatIP.search()', detail: 'Inner Product trên 8,580 vectors → ~4 ms' },
              { step: '5', label: 'Rank & Format', detail: 'Sắp xếp giảm dần, map sang metadata breed' },
            ].map(s => (
              <div key={s.step} className="flex gap-3 text-xs">
                <div className="w-5 h-5 rounded-full bg-primary/10 text-primary font-bold flex items-center justify-center shrink-0 text-[10px] mt-0.5">{s.step}</div>
                <div>
                  <span className="font-semibold text-on-surface">{s.label}</span>
                  <span className="text-on-surface-variant"> — {s.detail}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* === DATASET SECTION === */}
      <div className="p-6 rounded-3xl bg-surface-container-lowest border border-surface-container-high/60 shadow-sm space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary shrink-0">
            <span className="material-symbols-outlined text-[22px]">photo_library</span>
          </div>
          <div>
            <h2 className="text-lg font-extrabold text-on-surface">Dataset — Stanford Dogs (2011)</h2>
            <p className="text-xs text-on-surface-variant">Tập dữ liệu chuẩn học thuật Fine-Grained Visual Categorization cho nhận dạng giống chó</p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: 'Số Giống Chó', val: '120 breeds', color: 'text-primary' },
            { label: 'Tổng Ảnh', val: '20,580 images', color: 'text-secondary' },
            { label: 'Tập Train', val: '12,000 ảnh', color: 'text-amber-600' },
            { label: 'Gallery (Test)', val: '8,580 ảnh', color: 'text-emerald-600' },
          ].map(s => (
            <div key={s.label} className="p-4 rounded-2xl bg-surface-container-low text-center space-y-1">
              <p className={`text-xl font-extrabold ${s.color}`}>{s.val}</p>
              <p className="text-[11px] text-on-surface-variant font-medium">{s.label}</p>
            </div>
          ))}
        </div>

        <p className="text-xs text-on-surface-variant leading-relaxed">
          Mỗi ảnh trong dataset được cung cấp kèm <strong className="text-on-surface">Bounding Box annotation</strong> định dạng Pascal VOC XML, cho phép cắt chính xác vùng chứa chủ thể chó trước khi đưa vào mô hình — loại bỏ tạp âm từ phông nền. Nguồn dữ liệu có nguồn gốc từ <strong className="text-on-surface">ImageNet</strong> và được thu thập bởi nhóm nghiên cứu Stanford Vision Lab.
        </p>
      </div>

    </div>
  );
}

