start cmd /k python -m uvicorn M6_backend_main:app --port 8000
timeout /t 5
start cmd /k python -m streamlit run M7_frontend_app.py