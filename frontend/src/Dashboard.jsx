import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

function Dashboard() {
  const navigate = useNavigate()
  
  const [user, setUser] = useState(null)
  const [students, setStudents] = useState([])
  const [locations, setLocations] = useState([])
  const [newLoc, setNewLoc] = useState({ name: '', description: '', latitude: '', longitude: '' })
  const [locMessage, setLocMessage] = useState('')

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (!token) { navigate('/'); return }

    const fetchData = async () => {
      try {
        const config = { headers: { Authorization: `Bearer ${token}` } }
        
        const [profileRes, studentsRes, locationsRes] = await Promise.all([
          axios.get('http://127.0.0.1:8000/users/me', config),
          axios.get('http://127.0.0.1:8000/users/', config),
          axios.get('http://127.0.0.1:8000/locations/', config)
        ])

        setUser(profileRes.data)
        setStudents(studentsRes.data)
        setLocations(locationsRes.data)
      } catch (error) {
        localStorage.removeItem('token'); navigate('/')
      }
    }
    fetchData()
  }, [navigate])

  const handleAddLocation = async (e) => {
    e.preventDefault()
    const token = localStorage.getItem('token')
    try {
      const response = await axios.post('http://127.0.0.1:8000/locations/', newLoc, {
        headers: { Authorization: `Bearer ${token}` }
      })
      setLocations([...locations, response.data])
      setNewLoc({ name: '', description: '', latitude: '', longitude: '' })
      setLocMessage('✅ Đã thêm địa điểm thành công!')
    } catch (error) {
      setLocMessage('❌ Lỗi khi thêm địa điểm. Vui lòng kiểm tra lại.')
    }
  }

  // --- HÀM MỚI: XỬ LÝ XÓA ĐỊA ĐIỂM ---
  const handleDeleteLocation = async (id) => {
    const token = localStorage.getItem('token')
    // Hỏi lại cho chắc chắn trước khi xóa
    if (!window.confirm("Bạn có chắc chắn muốn xóa địa điểm này không?")) return;

    try {
      await axios.delete(`http://127.0.0.1:8000/locations/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      // Lọc bỏ địa điểm vừa xóa ra khỏi danh sách đang hiển thị
      setLocations(locations.filter(loc => loc.id !== id))
      setLocMessage('✅ Đã xóa địa điểm thành công!')
    } catch (error) {
      setLocMessage('❌ Lỗi khi xóa địa điểm.')
    }
  }

  const handleLogout = () => {
    localStorage.removeItem('token'); navigate('/')
  }

  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', backgroundColor: '#1a1a1a', color: 'white', minHeight: '100vh' }}>
      <h1 style={{ textAlign: 'center' }}>Hệ Thống Quản Trị Campus Explorer 🌍</h1>
      {user && <p style={{ textAlign: 'center', marginBottom: '40px' }}>Xin chào, <strong>{user.full_name}</strong>!</p>}

      {/* BẢNG DANH SÁCH SINH VIÊN */}
      <section style={{ marginBottom: '50px' }}>
        <h3>👨‍🎓 Danh Sách Sinh Viên Tham Gia</h3>
        <table style={tableStyle}>
          <thead>
            <tr style={{backgroundColor: '#333'}}>
              <th style={thStyle}>ID</th>
              <th style={thStyle}>Họ và Tên</th>
              <th style={thStyle}>Email</th>
              <th style={thStyle}>Trạng thái</th>
            </tr>
          </thead>
          <tbody>
            {students.map(student => (
              <tr key={student.id} style={{borderBottom: '1px solid #444'}}>
                <td style={tdStyle}>{student.id}</td>
                <td style={tdStyle}>{student.full_name}</td>
                <td style={tdStyle}>{student.email}</td>
                <td style={tdStyle}>{student.is_active ? '✅ Hoạt động' : '❌ Đã khóa'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* FORM THÊM ĐỊA ĐIỂM */}
      <section style={{ backgroundColor: '#2a2a2a', padding: '20px', borderRadius: '8px', marginBottom: '30px' }}>
        <h3>📍 Thêm Địa Điểm AR Mới</h3>
        <form onSubmit={handleAddLocation} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          <input type="text" placeholder="Tên địa điểm (VD: Tòa nhà C)" value={newLoc.name} 
            onChange={e => setNewLoc({...newLoc, name: e.target.value})} required style={inputStyle} />
          <input type="text" placeholder="Mô tả" value={newLoc.description} 
            onChange={e => setNewLoc({...newLoc, description: e.target.value})} style={inputStyle} />
          <input type="number" step="any" placeholder="Vĩ độ (Latitude)" value={newLoc.latitude} 
            onChange={e => setNewLoc({...newLoc, latitude: e.target.value})} required style={inputStyle} />
          <input type="number" step="any" placeholder="Kinh độ (Longitude)" value={newLoc.longitude} 
            onChange={e => setNewLoc({...newLoc, longitude: e.target.value})} required style={inputStyle} />
          <button type="submit" style={btnPrimaryStyle}>Thêm Địa Điểm</button>
        </form>
        {locMessage && <p style={{fontSize: '14px', marginTop: '10px'}}>{locMessage}</p>}
      </section>

      {/* BẢNG DANH SÁCH ĐỊA ĐIỂM (Đã cập nhật Nút Xóa) */}
      <section>
        <h3>🗺️ Danh Sách Địa Điểm Đã Lưu</h3>
        <table style={tableStyle}>
          <thead>
            <tr style={{backgroundColor: '#333'}}>
              <th style={thStyle}>Tên</th>
              <th style={thStyle}>Mô tả</th>
              <th style={thStyle}>Tọa độ (Lat, Long)</th>
              <th style={thStyle}>Trạng thái AR</th>
              <th style={thStyle}>Hành động</th> {/* Cột mới */}
            </tr>
          </thead>
          <tbody>
            {locations.map(loc => (
              <tr key={loc.id} style={{borderBottom: '1px solid #444'}}>
                <td style={tdStyle}>{loc.name}</td>
                <td style={tdStyle}>{loc.description}</td>
                <td style={tdStyle}>{loc.latitude}, {loc.longitude}</td>
                <td style={tdStyle}>{loc.is_ar_active ? '🟢 Bật' : '🔴 Tắt'}</td>
                <td style={tdStyle}>
                  {/* Nút Xóa gọi hàm handleDeleteLocation */}
                  <button 
                    onClick={() => handleDeleteLocation(loc.id)}
                    style={btnDeleteStyle}
                  >
                    Xóa
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <div style={{ textAlign: 'center' }}>
        <button onClick={handleLogout} style={btnLogoutStyle}>Đăng xuất</button>
      </div>
    </div>
  )
}

// --- CSS ---
const inputStyle = { padding: '10px', borderRadius: '4px', border: 'none', backgroundColor: '#3d3d3d', color: 'white' }
const btnPrimaryStyle = { gridColumn: 'span 2', padding: '10px', backgroundColor: '#4CAF50', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }
const btnLogoutStyle = { marginTop: '40px', padding: '10px 30px', backgroundColor: '#f44336', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }
const btnDeleteStyle = { padding: '6px 12px', backgroundColor: '#ff9800', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' } // CSS cho nút Xóa
const tableStyle = { width: '100%', borderCollapse: 'collapse', marginTop: '10px' }
const thStyle = { padding: '12px', textAlign: 'left', borderBottom: '2px solid #555' }
const tdStyle = { padding: '10px', borderBottom: '1px solid #444' }

export default Dashboard