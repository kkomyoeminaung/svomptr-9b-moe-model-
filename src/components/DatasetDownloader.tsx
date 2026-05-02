import { useState } from 'react';
import { Download, Loader2 } from 'lucide-react';

export default function DatasetDownloader() {
  const [isDownloading, setIsDownloading] = useState(false);

  const downloadDataset = async () => {
    setIsDownloading(true);
    try {
      const response = await fetch('/api/download-dataset');
      if (!response.ok) throw new Error('Dataset not ready yet.');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'synthetic_1M_high_quality.jsonl';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      alert("Failed to download dataset. Ensure it has been generated.");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <button
      onClick={downloadDataset}
      disabled={isDownloading}
      className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold transition-all disabled:opacity-50"
    >
      {isDownloading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
      Download 100k Dataset
    </button>
  );
}
