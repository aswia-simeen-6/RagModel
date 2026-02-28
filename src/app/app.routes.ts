import { Routes } from '@angular/router';
import { RagComponent } from './rag-component/rag-component';

export const routes: Routes = [
  {
    path: 'rag',
    component: RagComponent
  },
  {
    path: '',
    redirectTo: 'rag',
    pathMatch: 'full'
  }
];