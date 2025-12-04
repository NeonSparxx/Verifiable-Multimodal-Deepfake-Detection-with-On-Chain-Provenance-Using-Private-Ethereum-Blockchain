from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os
from media_detector import AIMediaDetector
import shutil

app = FastAPI(title="AI Media Detector API")

app.mount("/static", StaticFiles(directory="static"), name="static")
detector = AIMediaDetector()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict_media(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        result = detector.detect_media(temp_path)
        if len(result) == 3:
            label, conf, tx_hash = result
        else:
            label, conf = result
            tx_hash = None

        conf_percent = round(float(conf), 2)

        return {
            "filename": file.filename,
            "prediction": label.upper(),
            "confidence": conf_percent,
            "blockchain_proof": tx_hash if tx_hash and tx_hash != "failed" else None,
            "tx_link": f"http://127.0.0.1:7545/tx/{tx_hash}" if (tx_hash and tx_hash != "failed") else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)