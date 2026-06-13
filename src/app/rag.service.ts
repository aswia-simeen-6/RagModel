import { Injectable } from '@angular/core';
import { environment } from '../environments/environment';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChartData {
  labels: string[];
  values: number[];
}

export interface RagasScores {
  context_precision: number | null;
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_recall: number | null;
}

export interface AskResponse {
  answer: string;
  sources: number[];
  chartData: ChartData | null;
  scores: RagasScores | null;
  traceId: string;
  retrieval_ms: number;
  generation_ms: number;
}

export interface EvaluateResponse {
  answer: string;
  scores: RagasScores;
  sources: number[];
  traceId: string;
}

export interface BatchAverages {
  context_precision: number;
  context_recall: number;
  faithfulness: number;
  answer_relevancy: number;
}

export interface BatchDetailRow {
  question: string;
  answer: string;
  context_precision: number;
  context_recall: number;
  faithfulness: number;
  answer_relevancy: number;
}

export interface BatchEvaluateResponse {
  averages: BatchAverages;
  details: BatchDetailRow[];
}

export interface DocumentInfo {
  id: string;
  name: string;
  type: 'pdf' | 'url' | 'image';
  pages: number;
  chunks: number;
  total_chunks: number;
  ingested_at: string;
  elapsed_ms: number;
}

export interface PipelineStatus {
  ready: boolean;
  chunk_count: number;
  document_count: number;
  documents: DocumentInfo[];
  timestamp: string;
}

export interface QueryTrace {
  trace_id: string;
  question: string;
  chunks_retrieved: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
  sources: (number | string)[];
  source_names: string[];
  evaluated: boolean;
  scores: RagasScores | null;
  timestamp: string;
}

@Injectable({ providedIn: 'root' })
export class RagService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  uploadFile(file: File): Observable<any> {
    const fd = new FormData();
    fd.append('file', file);
    return this.http.post(`${this.apiUrl}/upload`, fd);
  }

  processUrl(url: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/url`, { url });
  }

  askQuestion(question: string, withEval = false): Observable<AskResponse> {
    return this.http.post<AskResponse>(`${this.apiUrl}/ask`, { question, evaluate: withEval });
  }

  evaluate(question: string, groundTruth?: string): Observable<EvaluateResponse> {
    return this.http.post<EvaluateResponse>(`${this.apiUrl}/evaluate`, {
      question,
      ground_truth: groundTruth ?? null,
    });
  }

  batchEvaluate(): Observable<BatchEvaluateResponse> {
    return this.http.post<BatchEvaluateResponse>(`${this.apiUrl}/batch-evaluate`, {});
  }

  analyzeImage(file: File, question: string): Observable<any> {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('question', question);
    return this.http.post(`${this.apiUrl}/analyze-image`, fd);
  }

  getStatus(): Observable<PipelineStatus> {
    return this.http.get<PipelineStatus>(`${this.apiUrl}/status`);
  }

  getTraces(): Observable<{ traces: QueryTrace[] }> {
    return this.http.get<{ traces: QueryTrace[] }>(`${this.apiUrl}/traces`);
  }

  clearMemory(): Observable<any> {
    return this.http.post(`${this.apiUrl}/clear`, {});
  }
}
