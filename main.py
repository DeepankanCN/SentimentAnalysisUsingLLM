from fastapi import FastAPI
from pydantic import BaseModel
from sentiment import sentanalyse

app = FastAPI()

class SentimentInput(BaseModel):
    text: str

@app.post("/sentiment")
def analyze_sentiment(input_data: SentimentInput):
    print(input_data.model_dump())
    data=input_data.model_dump()
    value=sentanalyse(data['text'])

    


    return {"message":value}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)