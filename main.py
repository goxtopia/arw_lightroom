import uvicorn
import os

if __name__ == "__main__":
    print("==========================================================")
    print("        α-Lightroom: Sony ARW RAW Photo Processor          ")
    print("==========================================================")
    print("Starting server...")
    print("Please open: http://127.0.0.1:8000 in your web browser.")
    print("==========================================================")
    
    # Run uvicorn server
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)
