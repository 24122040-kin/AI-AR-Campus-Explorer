# engine/nlp_processor.py
import re
import networkx as nx
from typing import Optional
from engine.utils import MIN_KEYWORD_LENGTH


# ---------------------------------------------------------------------------
# Bảng ánh xạ ký tự tiếng Việt có dấu -> không dấu
# ---------------------------------------------------------------------------
_ACCENTED = (
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂễỆĐÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
)
_PLAIN = (
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyy"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYY"
)
_TRANS_TABLE = str.maketrans(_ACCENTED, _PLAIN)


def remove_accents(text: str) -> str:
    """Chuyển tiếng Việt có dấu thành không dấu dùng str.translate (nhanh hơn loop)."""
    return text.translate(_TRANS_TABLE)


def normalize_text(text: str) -> str:
    """Chuẩn hóa chuỗi: chữ thường, bỏ dấu, bỏ ký tự đặc biệt."""
    text = text.lower().strip()
    text = remove_accents(text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def find_node_by_keyword(graph: nx.Graph, query: str) -> Optional[str]:
    """
    Tìm node phù hợp nhất từ câu hỏi của người dùng.

    Ưu tiên:
      1. Tên node xuất hiện trực tiếp trong query.
      2. Alias của node nằm trong query (hoặc query nằm trong alias).
         - Có kiểm tra độ dài tối thiểu (MIN_KEYWORD_LENGTH) để tránh false positive.

    Returns:
        Tên node nếu tìm thấy, None nếu không.
    """
    if not query:
        return None

    norm_query = normalize_text(query)

    # 1. Khớp tên node trực tiếp
    for node in graph.nodes():
        norm_node = normalize_text(node)
        if norm_node and norm_node in norm_query:
            return node

    # 2. Khớp qua alias
    for node, data in graph.nodes(data=True):
        for alias in data.get("aliases", []):
            norm_alias = normalize_text(alias)

            # Alias của hệ thống nằm trong câu user
            if norm_alias and norm_alias in norm_query:
                return node

            # User gõ ngắn nhưng đủ dài để tránh false positive
            # VD: "atm" khớp "cay atm", nhưng "a" thì không
            if (
                len(norm_query) >= MIN_KEYWORD_LENGTH
                and norm_query in norm_alias
            ):
                return node

    return None
