import {
  Component,
  ChangeDetectionStrategy,
  signal,
  ViewChild,
  ElementRef,
  AfterViewChecked,
  OnDestroy,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import {
  RagService,
  ChartData,
  RagasScores,
  PipelineStatus,
  QueryTrace,
} from '../rag.service';
import { Chart, ChartConfiguration, registerables } from 'chart.js';

Chart.register(...registerables);

type ActiveTab = 'chat' | 'evaluate' | 'observability';

interface ChatMessage {
  id: number;
  sender: 'user' | 'bot';
  text: string;
  chartData?: ChartData | null;
  scores?: RagasScores | null;
  sources?: number[];
  traceId?: string;
  retrieval_ms?: number;
  generation_ms?: number;
  timestamp: Date;
  isTyping?: boolean;
}

@Component({
  selector: 'app-rag-component',
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './rag-component.html',
  styleUrls: ['./rag-component.css'],
  providers: [RagService],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RagComponent implements AfterViewChecked, OnDestroy {
  // ── Tab ──────────────────────────────────────────────────────────────────────
  activeTab = signal<ActiveTab>('chat');

  // ── Upload state ─────────────────────────────────────────────────────────────
  fileUploaded = signal(false);
  isLoading = signal(false);
  uploadStatus = signal('');
  uploadSuccess = signal(false);
  websiteUrl = '';
  isDragging = signal(false);

  // ── Chat state ────────────────────────────────────────────────────────────────
  userQuestion = '';
  chatHistory = signal<ChatMessage[]>([]);
  enableEval = false;
  private msgCounter = 0;
  private shouldScroll = false;

  // ── Image state ───────────────────────────────────────────────────────────────
  selectedImage: File | null = null;
  imagePreviewUrl: string | ArrayBuffer | null = null;

  // ── Evaluate state ────────────────────────────────────────────────────────────
  evalQuestion = '';
  evalGroundTruth = '';
  evalAnswer = signal('');
  evalScores = signal<RagasScores | null>(null);
  isEvaluating = signal(false);
  batchResult = signal<any>(null);
  isBatchRunning = signal(false);

  // ── Observability state ───────────────────────────────────────────────────────
  pipelineStatus = signal<PipelineStatus | null>(null);
  traces = signal<QueryTrace[]>([]);
  isLoadingObs = signal(false);

  // ── Chart registry ────────────────────────────────────────────────────────────
  private chartInstances = new Map<number, Chart>();
  private evalRadarChart: Chart | null = null;
  private batchBarChart: Chart | null = null;

  @ViewChild('messagesEnd') private messagesEnd?: ElementRef;

  constructor(private ragService: RagService) {
    this.loadStatus();
  }

  ngAfterViewChecked() {
    if (this.shouldScroll) {
      this.messagesEnd?.nativeElement?.scrollIntoView({ behavior: 'smooth' });
      this.shouldScroll = false;
    }
  }

  ngOnDestroy() {
    this.chartInstances.forEach((c) => c.destroy());
    this.evalRadarChart?.destroy();
    this.batchBarChart?.destroy();
  }

  // ── Tab navigation ────────────────────────────────────────────────────────────
  setTab(tab: ActiveTab) {
    this.activeTab.set(tab);
    if (tab === 'observability') {
      this.loadObservability();
    }
  }

  // ── Drag & Drop ───────────────────────────────────────────────────────────────
  onDragOver(e: DragEvent) {
    e.preventDefault();
    this.isDragging.set(true);
  }
  onDragLeave() {
    this.isDragging.set(false);
  }
  onDrop(e: DragEvent) {
    e.preventDefault();
    this.isDragging.set(false);
    const file = e.dataTransfer?.files[0];
    if (file?.type === 'application/pdf') this.processFile(file);
  }

  // ── File & URL ingestion ──────────────────────────────────────────────────────
  onFileSelected(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) this.processFile(file);
  }

  private processFile(file: File) {
    this.uploadStatus.set(`Indexing "${file.name}"…`);
    this.uploadSuccess.set(false);
    this.isLoading.set(true);

    this.ragService.uploadFile(file).subscribe({
      next: (res) => {
        this.fileUploaded.set(true);
        this.uploadSuccess.set(true);
        this.uploadStatus.set(`✓ "${file.name}" — ${res.chunk_count} chunks indexed`);
        this.isLoading.set(false);
        this.addBotMessage(
          `Document **${file.name}** is ready. ${res.chunk_count} chunks stored in ChromaDB. Ask me anything!`
        );
        this.loadStatus();
      },
      error: () => {
        this.uploadStatus.set('❌ Error indexing file. Check the console.');
        this.isLoading.set(false);
      },
    });
  }

  onUrlSubmitted() {
    const url = this.websiteUrl.trim();
    if (!url) return;
    this.uploadStatus.set(`Crawling ${url}…`);
    this.uploadSuccess.set(false);
    this.isLoading.set(true);

    this.ragService.processUrl(url).subscribe({
      next: (res) => {
        this.fileUploaded.set(true);
        this.uploadSuccess.set(true);
        this.uploadStatus.set(`✓ Website indexed — ${res.chunk_count} chunks`);
        this.isLoading.set(false);
        this.websiteUrl = '';
        this.addBotMessage(`Website **${url}** indexed. ${res.chunk_count} chunks ready. Ask away!`);
        this.loadStatus();
      },
      error: () => {
        this.uploadStatus.set('❌ Error processing URL.');
        this.isLoading.set(false);
      },
    });
  }

  // ── Image ─────────────────────────────────────────────────────────────────────
  onImageSelected(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;
    this.selectedImage = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      this.imagePreviewUrl = e.target?.result ?? null;
    };
    reader.readAsDataURL(file);
  }

  clearImage() {
    this.selectedImage = null;
    this.imagePreviewUrl = null;
  }

  // ── Chat ──────────────────────────────────────────────────────────────────────
  private addBotMessage(text: string, extras: Partial<ChatMessage> = {}) {
    this.chatHistory.update((h) => [
      ...h,
      { id: ++this.msgCounter, sender: 'bot', text, timestamp: new Date(), ...extras },
    ]);
    this.shouldScroll = true;
  }

  sendMessage() {
    if (this.selectedImage && this.userQuestion.trim()) {
      this.sendImageQuestion();
      return;
    }
    const question = this.userQuestion.trim();
    if (!question) return;

    this.chatHistory.update((h) => [
      ...h,
      { id: ++this.msgCounter, sender: 'user', text: question, timestamp: new Date() },
    ]);
    this.userQuestion = '';
    this.isLoading.set(true);
    this.shouldScroll = true;

    const typingId = ++this.msgCounter;
    this.chatHistory.update((h) => [
      ...h,
      { id: typingId, sender: 'bot', text: '', isTyping: true, timestamp: new Date() },
    ]);

    this.ragService.askQuestion(question, this.enableEval).subscribe({
      next: (res) => {
        this.chatHistory.update((h) => {
          const without = h.filter((m) => m.id !== typingId);
          const newMsg: ChatMessage = {
            id: ++this.msgCounter,
            sender: 'bot',
            text: res.answer,
            chartData: res.chartData,
            scores: res.scores,
            sources: res.sources,
            traceId: res.traceId,
            retrieval_ms: res.retrieval_ms,
            generation_ms: res.generation_ms,
            timestamp: new Date(),
          };
          return [...without, newMsg];
        });
        this.isLoading.set(false);
        this.shouldScroll = true;

        const history = this.chatHistory();
        const lastMsg = history[history.length - 1];
        if (lastMsg?.chartData?.labels?.length) {
          setTimeout(() => this.renderPieChart(lastMsg.id, lastMsg.chartData!), 100);
        }
      },
      error: () => {
        this.chatHistory.update((h) => [
          ...h.filter((m) => m.id !== typingId),
          { id: ++this.msgCounter, sender: 'bot', text: 'Something went wrong. Please try again.', timestamp: new Date() },
        ]);
        this.isLoading.set(false);
      },
    });
  }

  sendImageQuestion() {
    if (!this.selectedImage || !this.userQuestion.trim()) return;
    const question = this.userQuestion;
    this.chatHistory.update((h) => [
      ...h,
      { id: ++this.msgCounter, sender: 'user', text: '\u{1F4F7} ' + question, timestamp: new Date() },
    ]);
    this.userQuestion = '';
    this.isLoading.set(true);

    this.ragService.analyzeImage(this.selectedImage, question).subscribe({
      next: (res) => {
        this.addBotMessage(res.answer);
        this.isLoading.set(false);
        this.clearImage();
      },
      error: () => {
        this.addBotMessage('Sorry, I had trouble analyzing that image.');
        this.isLoading.set(false);
      },
    });
  }

  clearChat() {
    this.isLoading.set(true);
    this.ragService.clearMemory().subscribe({
      next: () => {
        this.fileUploaded.set(false);
        this.uploadStatus.set('');
        this.uploadSuccess.set(false);
        this.chatHistory.set([]);
        this.websiteUrl = '';
        this.isLoading.set(false);
        this.pipelineStatus.set(null);
        this.traces.set([]);
      },
      error: () => this.isLoading.set(false),
    });
  }

  runEval() {
    if (!this.evalQuestion.trim()) return;
    this.isEvaluating.set(true);
    this.evalAnswer.set('');
    this.evalScores.set(null);

    this.ragService.evaluate(this.evalQuestion, this.evalGroundTruth || undefined).subscribe({
      next: (res) => {
        this.evalAnswer.set(res.answer);
        this.evalScores.set(res.scores);
        this.isEvaluating.set(false);
        setTimeout(() => this.renderEvalRadar(res.scores), 150);
      },
      error: () => this.isEvaluating.set(false),
    });
  }

  runBatchEval() {
    this.isBatchRunning.set(true);
    this.batchResult.set(null);

    this.ragService.batchEvaluate().subscribe({
      next: (res) => {
        this.batchResult.set(res);
        this.isBatchRunning.set(false);
        setTimeout(() => this.renderBatchBar(res.averages), 150);
      },
      error: () => this.isBatchRunning.set(false),
    });
  }

  loadStatus() {
    this.ragService.getStatus().subscribe({
      next: (s) => this.pipelineStatus.set(s),
      error: () => {},
    });
  }

  loadObservability() {
    this.isLoadingObs.set(true);
    this.loadStatus();
    this.ragService.getTraces().subscribe({
      next: (res) => {
        this.traces.set(res.traces);
        this.isLoadingObs.set(false);
      },
      error: () => this.isLoadingObs.set(false),
    });
  }

  private renderPieChart(msgId: number, data: ChartData) {
    const canvas = document.getElementById('chart-' + msgId) as HTMLCanvasElement;
    if (!canvas) return;
    this.chartInstances.get(msgId)?.destroy();
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const chart = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: data.labels,
        datasets: [{ data: data.values, backgroundColor: ['#6366f1','#8b5cf6','#ec4899','#06b6d4','#10b981','#f59e0b','#ef4444','#84cc16'], borderWidth: 0 }],
      },
      options: { responsive: true, plugins: { legend: { position: 'bottom', labels: { color: '#cbd5e1', padding: 12, font: { size: 11 } } } } },
    });
    this.chartInstances.set(msgId, chart);
  }

  private renderEvalRadar(scores: RagasScores) {
    const canvas = document.getElementById('eval-radar') as HTMLCanvasElement;
    if (!canvas) return;
    this.evalRadarChart?.destroy();
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    this.evalRadarChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Context Precision', 'Faithfulness', 'Answer Relevancy'],
        datasets: [{ label: 'RAGAS', data: [scores.context_precision ?? 0, scores.faithfulness ?? 0, scores.answer_relevancy ?? 0], fill: true, backgroundColor: 'rgba(99,102,241,0.25)', borderColor: '#6366f1', pointBackgroundColor: '#6366f1', pointBorderColor: '#fff' }],
      },
      options: {
        scales: { r: { min: 0, max: 1, ticks: { stepSize: 0.2, color: '#94a3b8', backdropColor: 'transparent' }, grid: { color: 'rgba(148,163,184,0.15)' }, pointLabels: { color: '#e2e8f0', font: { size: 12 } } } },
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
      },
    });
  }

  private renderBatchBar(averages: any) {
    const canvas = document.getElementById('batch-bar') as HTMLCanvasElement;
    if (!canvas) return;
    this.batchBarChart?.destroy();
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    this.batchBarChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Context Precision', 'Context Recall', 'Faithfulness', 'Answer Relevancy'],
        datasets: [{ label: 'Average', data: [averages.context_precision, averages.context_recall, averages.faithfulness, averages.answer_relevancy], backgroundColor: ['#6366f1','#8b5cf6','#10b981','#06b6d4'], borderRadius: 8 }],
      },
      options: {
        scales: { y: { min: 0, max: 1, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' } }, x: { ticks: { color: '#94a3b8' }, grid: { display: false } } },
        plugins: { legend: { display: false } },
      },
    });
  }

  scoreColor(v: number | null): string {
    if (v === null) return '#64748b';
    if (v >= 0.8) return '#10b981';
    if (v >= 0.6) return '#f59e0b';
    return '#ef4444';
  }

  scoreLabel(v: number | null): string {
    if (v === null) return 'N/A';
    if (v >= 0.8) return 'Excellent';
    if (v >= 0.6) return 'Good';
    if (v >= 0.4) return 'Fair';
    return 'Poor';
  }

  pct(v: number | null): string {
    return v !== null ? Math.round(v * 100) + '%' : '\u2014';
  }

  fmtTime(d: Date): string {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  fmtTs(iso: string): string {
    return new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  docIcon(type: string): string {
    return type === 'pdf' ? '\u{1F4C4}' : type === 'url' ? '\u{1F310}' : '\u{1F5BC}\uFE0F';
  }

  uniqueSources(sources: (number | string)[] | undefined): string {
    if (!sources?.length) return '\u2014';
    return [...new Set(sources)].map((p) => (typeof p === 'number' ? 'p.' + (p + 1) : p)).join(', ');
  }
}
