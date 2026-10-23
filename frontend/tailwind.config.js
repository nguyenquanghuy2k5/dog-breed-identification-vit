/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "surface": "#f9f9ff",
        "surface-dim": "#d3daea",
        "surface-bright": "#f9f9ff",
        "surface-container-lowest": "#ffffff",
        "surface-container-low": "#f0f3ff",
        "surface-container": "#e7eefe",
        "surface-container-high": "#e2e8f8",
        "surface-container-highest": "#dce2f3",
        "on-surface": "#151c27",
        "on-surface-variant": "#514538",
        "inverse-surface": "#2a313c",
        "inverse-on-surface": "#ebf1ff",
        "outline": "#867461",
        "outline-variant": "#d8c3ad",
        "surface-tint": "#855300",
        
        "primary": "#653e00",
        "on-primary": "#ffffff",
        "primary-container": "#f59e0b",
        "on-primary-container": "#613b00",
        "primary-fixed": "#ffddb8",
        "primary-fixed-dim": "#fdb965",
        "on-primary-fixed": "#2a1700",
        "on-primary-fixed-variant": "#653e00",
        "inverse-primary": "#fdb965",
        
        "secondary": "#085ac0",
        "on-secondary": "#ffffff",
        "secondary-container": "#2170e4",
        "on-secondary-container": "#fefcff",
        "secondary-fixed": "#d8e2ff",
        "secondary-fixed-dim": "#adc6ff",
        "on-secondary-fixed": "#001a42",
        "on-secondary-fixed-variant": "#004395",
        
        "tertiary": "#004c6a",
        "on-tertiary": "#ffffff",
        "tertiary-container": "#1abdff",
        "on-tertiary-container": "#004966",
        "tertiary-fixed": "#c5e7ff",
        "tertiary-fixed-dim": "#88cffa",
        
        "error": "#ba1a1a",
        "on-error": "#ffffff",
        "error-container": "#ffdad6",
        "on-error-container": "#93000a",
        
        "background": "#f9f9ff",
        "on-background": "#151c27",
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      }
    },
  },
  plugins: [],
}

