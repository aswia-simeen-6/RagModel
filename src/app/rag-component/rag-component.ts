import { Component, ChangeDetectionStrategy, signal, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { RagService, ChartData } from '../rag.service';
import { Chart, ChartConfiguration, registerables } from 'chart.js';

Chart.register(...registerables);

interface ChatMessage {
  sender: 'user' | 'bot';
  text: string;
  chartData?: ChartData | null;
}

@Component({
  selector: 'app-rag-component',
  imports: [CommonModule, HttpClientModule, FormsModule],
  templateUrl: './rag-component.html',
  styleUrls: ['./rag-component.css'],
  providers: [RagService],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class RagComponent {
  fileUploaded = signal(false);
  isLoading = signal(false);
  userQuestion = signal('');
  chatHistory = signal<ChatMessage[]>([]);
  uploadStatus = signal('');
  websiteUrl = ''; // Holds the text from the input box
  selectedImage: File | null = null;
  imagePreviewUrl: string | ArrayBuffer | null = null;
  private chartInstances: Map<number, Chart> = new Map();

  constructor(private ragService: RagService) {}

  onUrlSubmitted() {
    if (!this.websiteUrl.trim()) return;
    
    this.uploadStatus.set('Reading website...');
    this.isLoading.set(true);
    
    this.ragService.processUrl(this.websiteUrl).subscribe({
      next: (res) => {
        this.fileUploaded.set(true);
        this.uploadStatus.set('Website Processed! You can now chat.');
        this.isLoading.set(false);
        this.websiteUrl = ''; // Clear the input box
        this.chatHistory.update(h => [...h, {sender: 'bot', text: 'Hello! I have read the website. Ask me anything about it.'}]);
      },
      error: (err) => {
        this.uploadStatus.set('Error processing website. Make sure it is a valid, public URL.');
        this.isLoading.set(false);
        console.error(err);
      }
    });
  }
  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    this.uploadStatus.set('Uploading and processing...');
    this.isLoading.set(true);

    this.ragService.uploadFile(file).subscribe({
      next: () => {
        this.fileUploaded.set(true);
        this.uploadStatus.set('File Processed! You can now chat.');
        this.isLoading.set(false);
        this.chatHistory.update(h => [...h, {sender: 'bot', text: 'Hello! I have read your file. Ask me anything about it.'}]);
      },
      error: (err) => {
        this.uploadStatus.set('Error uploading file.');
        this.isLoading.set(false);
        console.error(err);
      }
    });
  }

  sendMessage() {
    const question = this.userQuestion().trim();
    if (!question) return;

    this.chatHistory.update(h => [...h, { sender: 'user', text: question }]);
    this.userQuestion.set('');
    this.isLoading.set(true);

    this.ragService.askQuestion(question).subscribe({
      next: (res) => {
        this.chatHistory.update(h => [...h, { 
          sender: 'bot', 
          text: res.answer,
          chartData: res.chartData 
        }]);
        this.isLoading.set(false);
        
        // Render chart after view updates
        if (res.chartData && res.chartData.labels.length > 0) {
          setTimeout(() => this.renderChart(this.chatHistory().length - 1), 100);
        }
      },
      error: () => {
        this.chatHistory.update(h => [...h, { sender: 'bot', text: 'Sorry, something went wrong generating the answer.' }]);
        this.isLoading.set(false);
      }
    });
  }

  private renderChart(messageIndex: number) {
    const message = this.chatHistory()[messageIndex];
    if (!message.chartData) return;

    const canvasId = `chart-${messageIndex}`;
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!canvas) return;

    // Destroy existing chart if it exists
    if (this.chartInstances.has(messageIndex)) {
      this.chartInstances.get(messageIndex)?.destroy();
    }

    // Create new chart
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration = {
      type: 'pie',
      data: {
        labels: message.chartData.labels,
        datasets: [{
          data: message.chartData.values,
          backgroundColor: [
            '#FF6384',
            '#36A2EB',
            '#FFCE56',
            '#4BC0C0',
            '#9966FF',
            '#FF9F40',
            '#FF6384',
            '#C9CBCF'
          ],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 15,
              font: {
                size: 12
              }
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const value = context.parsed;
                const total = (context.dataset.data as number[]).reduce((a: number, b: number) => a + b, 0);
                const percentage = ((value / total) * 100).toFixed(1);
                return `${label}: ${value} (${percentage}%)`;
              }
            }
          }
        }
      }
    };

    const chart = new Chart(ctx, config);
    this.chartInstances.set(messageIndex, chart);
  }

  ngOnDestroy() {
    // Clean up all charts
    this.chartInstances.forEach(chart => chart.destroy());
    this.chartInstances.clear();
  }

  clearChat() {
    this.isLoading.set(true);
    this.ragService.clearMemory().subscribe({
      next: () => {
        // Reset all frontend state variables
        this.fileUploaded.set(false);
        this.uploadStatus.set('');
        this.chatHistory.set([]);
        this.websiteUrl = '';
        this.isLoading.set(false);
      },
      error: (err) => {
        console.error('Failed to clear memory', err);
        this.isLoading.set(false);
      }
    });
  }

  // Triggered when a user selects an image
  onImageSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.selectedImage = file;
      
      // Create a preview URL so the user can see what they selected
      const reader = new FileReader();
      reader.onload = (e) => this.imagePreviewUrl = e.target?.result || null;
      reader.readAsDataURL(file);
    }
  }

  // Clear the selected image
  clearImage() {
    this.selectedImage = null;
    this.imagePreviewUrl = null;
  }

  // A new function to send the image + question to the backend
  sendImageQuestion() {
    if (!this.selectedImage || !this.userQuestion().trim()) return;

    const question = this.userQuestion();
    this.chatHistory.update(h => [...h, { sender: 'user', text: `[Image Uploaded] ${question}` }]);
    
    this.isLoading.set(true);
    this.uploadStatus.set('Analyzing image...');

    this.ragService.analyzeImage(this.selectedImage, question).subscribe({
      next: (res) => {
        this.chatHistory.update(h => [...h, { sender: 'bot', text: res.answer }]);
        this.isLoading.set(false);  
        this.uploadStatus.set('');
        this.userQuestion.set('');
        this.clearImage(); // Reset the image input for the next question
      },
      error: (err) => {
        this.chatHistory.update(h => [...h, { sender: 'bot', text: 'Sorry, I had trouble analyzing that image.' }]);
        this.isLoading.set(false);
        console.error(err);
      }
    });
  }
}