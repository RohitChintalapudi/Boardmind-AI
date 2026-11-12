/**
 * frontend/src/types/index.ts
 * TypeScript data models for BoardMind AI frontend matching backend Pydantic schemas.
 */

export interface BoundingBox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  normalized?: [number, number, number, number];
  confidence: number;
  class_id: number;
  class_name: string;
}

export interface DetectResponse {
  success: boolean;
  num_detections: number;
  detections: BoundingBox[];
  primary_detection: BoundingBox | null;
  image_width: number;
  image_height: number;
  annotated_image: string | null;
  message: string;
}

export interface RecognizedLine {
  line_number: number;
  text: string;
  confidence: number;
  bbox?: [number, number, number, number];
}

export interface ExtractTextResponse {
  success: boolean;
  model_name: string;
  status: 'ready' | 'untrained_baseline' | 'offline';
  raw_text: string;
  lines: RecognizedLine[];
  stroke_density?: number;
  message: string;
}

export interface StructuredNotes {
  title: string;
  headings: string[];
  bullet_points: string[];
  numbered_steps: string[];
  definitions: string[];
  formulas_or_key_terms: string[];
  raw_preserved_text: string;
  markdown_content: string;
  ai_concept_explanation?: string;
  ai_engine_used?: string;
}

export interface GenerateNotesResponse {
  success: boolean;
  notes: StructuredNotes;
  message: string;
}

export interface ProcessResponse {
  success: boolean;
  detection: DetectResponse;
  cropped_whiteboard: string | null;
  text_extraction: ExtractTextResponse;
  structured_notes: StructuredNotes;
  ai_concept_explanation?: string;
  processing_time_ms: number;
  warnings: string[];
}

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  whiteboard_detector: {
    loaded: boolean;
    device: string;
    weights_path: string;
    classes: Record<string, string>;
  };
  handwriting_recognizer: {
    engine: string;
    is_trained_model_ready: boolean;
    weights_path: string | null;
    dataset_available: boolean;
    status: string;
    message: string;
  };
  hardware: {
    os: string;
    cpu_architecture: string;
    cpu_count: number;
    ram_total_gb: number;
    ram_used_percent: number;
  };
}

export type ProcessingStage = 'idle' | 'uploading' | 'detecting' | 'cropping' | 'analyzing' | 'formatting' | 'explaining' | 'complete' | 'error';
