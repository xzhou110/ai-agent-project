from fastapi import FastAPI

app = FastAPI()


@app.get("/{user_input}")
async def hello(user_input: str):
    return {"message": f"Hello, World {user_input}"}
