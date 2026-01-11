from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from consistency_engine import extract_features

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    backstory: str = Form(...),
    book_name: str = Form(...)
):
    # Get calibrated BDH features
    semantic, drift = extract_features(book_name, backstory)

    # Proper calibrated score
    score = semantic - (1.2 * drift)

    print("semantic =", round(semantic,3),
          "drift =", round(drift,3),
          "score =", round(score,3))

    if score > 0.25:
        result = "Consistent"
    else:
        result = "Contradict"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "backstory": backstory,
            "score": round(score, 3)
        }
    )