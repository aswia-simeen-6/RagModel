import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { RagComponent } from "./rag-component/rag-component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RagComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('rag-ui');
}
