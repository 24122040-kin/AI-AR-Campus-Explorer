let ws;
const loginScreen = document.getElementById('login-screen');
const chatScreen = document.getElementById('chat-screen');
const chatBox = document.getElementById('chat-box');
const statusDot = document.getElementById('status-dot');
const messageInput = document.getElementById('message-input');

// 1. Hàm bấm nút Đăng nhập -> Kết nối WebSocket
function connectAI() {
    loginScreen.classList.add('hidden');
    chatScreen.classList.remove('hidden');
    chatScreen.classList.add('flex');

    // Kết nối tới Backend FastAPI
    ws = new WebSocket("ws://127.0.0.1:8000/ws/ar-stream");

    ws.onopen = function() {
        statusDot.textContent = "🟢 Đã kết nối";
        statusDot.classList.replace('text-yellow-300', 'text-green-300');
        appendMessage("Hệ thống", "Đã kết nối tới Trợ lý AI Campus!", "sys");
    };

    // 2. Hàm lắng nghe AI trả lời
    ws.onmessage = function(event) {
        try {
            const response = JSON.parse(event.data);
            if (response.type === "chat_response" && response.data.status === "success") {
                appendMessage("Trợ lý AI", response.data.reply, "ai");
            } else {
                appendMessage("Hệ thống", event.data, "sys");
            }
        } catch (e) {
            appendMessage("Trợ lý AI", event.data, "ai");
        }
    };

    ws.onerror = function() {
        statusDot.textContent = "🔴 Lỗi kết nối";
        appendMessage("Lỗi", "Không thể kết nối tới Server. Hãy chắc chắn backend đang chạy.", "sys");
    };
}

// 3. Hàm Gửi tin nhắn lên Server
function sendMessage() {
    const text = messageInput.value.trim();
    if (text !== "" && ws && ws.readyState === WebSocket.OPEN) {
        // Đóng gói JSON y hệt cách Unity làm
        const msgJson = JSON.stringify({ action: "chat", message: text });
        ws.send(msgJson);
        
        appendMessage("Bạn", text, "user");
        messageInput.value = ""; // Xóa trắng ô nhập
    }
}

// Hàm phụ trợ: In chữ lên màn hình cho đẹp
function appendMessage(sender, text, type) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('p-3', 'rounded-lg', 'max-w-[80%]');

    if (type === "user") {
        msgDiv.classList.add('bg-green-100', 'text-green-900', 'self-end', 'rounded-br-none');
    } else if (type === "ai") {
        msgDiv.classList.add('bg-white', 'text-gray-800', 'border', 'self-start', 'rounded-bl-none');
    } else {
        msgDiv.classList.add('bg-gray-200', 'text-gray-500', 'self-center', 'text-sm', 'italic');
    }

    msgDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Tự cuộn xuống dưới cùng
}