import { useState } from 'react'
import axios from 'axios'
import { useNavigate } from 'react-router-dom' // Import công cụ chuyển trang

function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState('')
  const navigate = useNavigate() // Khởi tạo công cụ chuyển trang

  const handleLogin = async (e) => {
    e.preventDefault()
    
    try {
      const response = await axios.post('http://127.0.0.1:8000/login', {
        email: email,
        password: password
      })

      localStorage.setItem('token', response.data.access_token)
      setMessage('✅ Đăng nhập thành công! Đang vào hệ thống...')
      
      // Đợi 1 giây rồi tự động chuyển sang trang Dashboard
      setTimeout(() => {
        navigate('/dashboard')
      }, 1000)

    } catch (error) {
      setMessage('❌ Đăng nhập thất bại: Sai email hoặc mật khẩu.')
    }
  }

  return (
    <div style={{ padding: '50px', fontFamily: 'sans-serif' }}>
      <h2>Đăng nhập Hệ thống Quản trị AR</h2>
      <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', width: '300px', gap: '15px' }}>
        <input 
          type="email" 
          placeholder="Nhập email sinh viên" 
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          style={{ padding: '8px' }}
        />
        <input 
          type="password" 
          placeholder="Nhập mật khẩu" 
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          style={{ padding: '8px' }}
        />
        <button type="submit" style={{ padding: '10px', cursor: 'pointer', backgroundColor: '#4CAF50', color: 'white', border: 'none' }}>
          Đăng nhập
        </button>
      </form>
      {message && <p style={{ marginTop: '20px', fontWeight: 'bold' }}>{message}</p>}
    </div>
  )
}

export default Login