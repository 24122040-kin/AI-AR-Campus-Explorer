import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Login from './Login'
import Dashboard from './Dashboard'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Đường dẫn mặc định sẽ mở trang Login */}
        <Route path="/" element={<Login />} />
        {/* Đường dẫn /dashboard sẽ mở trang Dashboard */}
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App