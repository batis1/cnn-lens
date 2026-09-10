import { cpSync, mkdirSync } from 'node:fs';

// Vercel serves public/ separately; inference needs its own bundled image copy.
mkdirSync('backend/public/assets', { recursive: true });
cpSync('public/assets/img', 'backend/public/assets/img', { recursive: true });
