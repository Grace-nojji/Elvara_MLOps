from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Todo(BaseModel):
    id : int
    title: str

@app.get('/')
def homes():
    return{'message': 'Hello world'}

@app.get('/todos')
def get_todos():
    return [ {'id':1, 'title':'learn APIs'},
             {'id':2, 'title' : 'deploy model'}
    ]


@app.post('/todos')
def create_todo(todo: Todo):
    return todo