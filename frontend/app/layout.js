import './globals.css';

export const metadata = {
  title: 'Federated Tamil Court ASR — Privacy-Preserving Speech Transcription',
  description:
    'Interactive simulation and deployment dashboard for federated privacy-preserving Tamil speech transcription in courtrooms. Whisper + LoRA + FedAvg with zero audio egress.',
  keywords: ['federated learning', 'Tamil ASR', 'privacy-preserving', 'Whisper', 'LoRA', 'courtroom', 'speech transcription'],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚖️</text></svg>" />
      </head>
      <body>{children}</body>
    </html>
  );
}
