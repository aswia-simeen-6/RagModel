import { Component, signal } from '@angular/core';
import { RagComponent } from "./rag-component/rag-component";

@Component({
  selector: 'app-root',
  imports: [RagComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('rag-ui');
}
