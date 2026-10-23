import React from 'react';

export default function PresetsBar({ presets, onSelectPreset, selectedPresetId }) {
  if (!presets || presets.length === 0) return null;

  return (
    <div className="space-y-2 pt-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold text-on-surface-variant uppercase tracking-wider flex items-center gap-1.5">
          <span className="material-symbols-outlined text-[16px] text-primary">auto_awesome</span>
          Mẫu ảnh kiểm thử nhanh (Presets):
        </span>
        <span className="text-xs text-secondary font-semibold">1-Click Test</span>
      </div>

      <div className="grid grid-cols-5 gap-2">
        {presets.map((preset) => {
          const isSelected = selectedPresetId === preset.id;
          return (
            <button
              key={preset.id}
              type="button"
              onClick={() => onSelectPreset(preset)}
              className={`p-2 rounded-2xl transition-all text-center flex flex-col items-center group relative border ${
                isSelected 
                  ? 'bg-primary-fixed/50 border-primary shadow-sm scale-102' 
                  : 'bg-surface-container-low hover:bg-surface-container border-transparent'
              }`}
            >
              <div className="w-12 h-12 rounded-xl bg-surface-container-highest overflow-hidden mb-1.5 shadow-sm group-hover:scale-105 transition-transform">
                <img 
                  src={preset.image_url} 
                  alt={preset.title}
                  className="w-full h-full object-cover"
                />
              </div>
              <span className="text-xs font-bold text-on-surface truncate w-full">
                {preset.title.split(' ')[0]}
              </span>
              <span className="text-[10px] text-on-surface-variant">
                {preset.subtitle}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

