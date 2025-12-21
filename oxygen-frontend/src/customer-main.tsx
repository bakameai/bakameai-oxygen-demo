import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import CustomerPortal from './CustomerPortal'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <CustomerPortal />
  </StrictMode>,
)
