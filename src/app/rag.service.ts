import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChartData {
  labels: string[];
  values: number[];
}

export interface AskResponse {
  answer: string;
  sources: number[];
  chartData: ChartData | null;
}

@Injectable({
  providedIn: 'root'
})
export class RagService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  uploadFile(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.apiUrl}/upload`, formData);
  }

  askQuestion(question: string): Observable<AskResponse> {
    return this.http.post<AskResponse>(`${this.apiUrl}/ask`, { question });
  }

  processUrl(url: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/url`, { url });
  }

  clearMemory(): Observable<any> {
    return this.http.post(`${this.apiUrl}/clear`, {});
  }

  analyzeImage(file: File, question: string): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', question);
    
    return this.http.post(`${this.apiUrl}/analyze-image`, formData);
  }
  
}