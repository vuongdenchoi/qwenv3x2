# Walkthrough – Design Check AI System

## Tổng kết

Đã xây dựng thành công **hệ thống AI kiểm tra lỗi thiết kế 2D** theo đúng flow từ tài liệu, gồm đầy đủ 7 bước.

## Cấu trúc code đã tạo

```
d:\qwen3v\
├── design_rules/                    ← Knowledge Base (Step 1)
│   ├── typography.md
│   ├── color_theory.md
│   ├── layout_rules.md
│   ├── poster_design.md
│   └── logo_design.md
├── backend/
│   ├── main.py                      ← FastAPI server (Step 4 + 6)
│   ├── requirements.txt
│   ├── agents/
│   │   ├── design_check_agent.py    ← Orchestrator
│   │   ├── retrieval_agent.py       ← Step 2: TF-IDF retrieval
│   │   ├── prompt_agent.py          ← Step 3: Prompt builder
│   │   ├── qwen_agent.py            ← Step 4: Qwen VL API
│   │   └── post_process_agent.py    ← Step 6: Validate + clean
│   └── knowledge_base/
│       ├── build_index.py           ← Step 1: Build TF-IDF index
│       └── faiss_index/             ← Generated index files
│           ├── tfidf_vectorizer.pkl
│           ├── tfidf_matrix.npz
│           └── metadata.json
├── frontend/
│   └── index.html                   ← Step 7: React-style UI
├── run.py                           ← Unified runner
└── README.md
```

## Kết quả xác minh

| Thành phần | Trạng thái |
|------------|-----------|
| Knowledge Base (5 rules files) | ✅ OK |
| TF-IDF Index (14 chunks) | ✅ Build thành công |
| RetrievalAgent | ✅ Trả về rules đúng |
| FastAPI server `/health` | ✅ `{"status":"ok"}` |
| Frontend UI | ✅ Load đúng, dark theme đẹp |

## Screenshot UI

![Design Check AI UI](file:///C:/Users/vuong/.gemini/antigravity/brain/3884e83a-803c-4fdd-8071-a114820872ff/full_ui_screenshot_1773199725295.png)

## Demo recording

![Browser verification](file:///C:/Users/vuong/.gemini/antigravity/brain/3884e83a-803c-4fdd-8071-a114820872ff/design_check_ui_verify_1773199682564.webp)

## Cách chạy

```powershell
cd d:\qwen3v

# 1. Set API key
$env:DASHSCOPE_API_KEY = "sk-your-key-here"

# 2. Mở server (index đã build rồi)
$env:PYTHONPATH = "d:\qwen3v\backend"
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Mở browser: http://localhost:8000
```

> **Lưu ý kỹ thuật**: Thay vì dùng FAISS + neural embeddings (bị DLL conflict trên Anaconda), hệ thống dùng **TF-IDF + cosine similarity** (sklearn) — nhẹ hơn, không cần GPU, kết quả tốt cho văn bản quy tắc thiết kế.
