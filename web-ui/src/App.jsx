import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import UC1Intro from './pages/UC1Intro'
import UC1Dashboard from './pages/UC1Dashboard'
import UC2Intro from './pages/UC2Intro'
import UC2Dashboard from './pages/UC2Dashboard'
import './App.css'

export default function App() {
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Navigate to="/uc2/intro" replace />} />
          <Route path="/uc1/intro" element={<UC1Intro />} />
          <Route path="/uc1/demo" element={<UC1Dashboard />} />
          <Route path="/uc2/intro" element={<UC2Intro />} />
          <Route path="/uc2/demo" element={<UC2Dashboard />} />
        </Routes>
      </main>
    </div>
  )
}
