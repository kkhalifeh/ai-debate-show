# FastAPI app for AI debate talk show
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'AI Debate Talk Show Backend'}

