/**
 * frontend/src/services/api.ts
 * REST API Client for BoardMind AI Backend.
 */

import {
  HealthResponse,
  DetectResponse,
  ExtractTextResponse,
  GenerateNotesResponse,
  ProcessResponse
} from '../types';

const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';
const defaultApiBaseUrl = isLocalhost ? 'http://localhost:8001/api' : 'http://127.0.0.1:8001/api';

// Allow deployment-specific overrides without editing source (see .env.example).
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || defaultApiBaseUrl;

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export const api = {
  /**
   * Health check endpoint to inspect ML models, GPU/CPU, and system metrics.
   */
  async getHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (!res.ok) {
        throw new ApiError(`Health check failed with status ${res.status}`, res.status);
      }
      return await res.json();
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(`Cannot connect to BoardMind AI backend at ${API_BASE_URL}. Is FastAPI running?`);
    }
  },

  /**
   * Execute the full pipeline: Detection -> Crop -> Stroke Analysis -> AI Explanation.
   */
  async processImage(file: File, topicHint: string = '', apiKey?: string): Promise<ProcessResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (topicHint.trim()) {
      formData.append('topic_hint', topicHint.trim());
    }
    if (apiKey && apiKey.trim()) {
      formData.append('api_key', apiKey.trim());
    }

    try {
      const res = await fetch(`${API_BASE_URL}/process`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Unknown error occurred' }));
        throw new ApiError(errorData.detail || `Server returned ${res.status}`, res.status);
      }

      return await res.json();
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(err.message || 'Network error while processing image.');
    }
  },

  /**
   * Only run whiteboard detection on an uploaded image.
   */
  async detectOnly(file: File): Promise<DetectResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE_URL}/detect`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Detection failed' }));
      throw new ApiError(err.detail || `Detection error ${res.status}`, res.status);
    }
    return await res.json();
  },

  /**
   * Re-generate or structure notes and concept explanation from user-edited raw text.
   */
  async generateNotes(rawText: string, topicHint?: string, apiKey?: string): Promise<GenerateNotesResponse> {
    const res = await fetch(`${API_BASE_URL}/generate-notes`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        raw_text: rawText,
        topic_hint: topicHint,
        api_key: apiKey,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Note generation failed' }));
      throw new ApiError(err.detail || `Notes generation error ${res.status}`, res.status);
    }
    return await res.json();
  }
};
