# MIT License
# Copyright (c) 2025 Anton Schreiner

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import database as db
from pathlib import Path

app = FastAPI(title="Model Graph Editor", version="1.0.0")

SCRIPT_PATH = Path(__file__).parent / "static" / "scripts"
TEMPLATE_PATH = Path(__file__).parent / "templates"
STATIC_PATH = Path(__file__).parent / "static"

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
templates = Jinja2Templates(directory=TEMPLATE_PATH)


# Pydantic models
class GraphCreate(BaseModel):
    name: str
    data: dict


class GraphUpdate(BaseModel):
    data: dict
    name: Optional[str] = None


class GraphFork(BaseModel):
    name: str
    data: dict


# API Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main editor page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/edit/{graph_hash}", response_class=HTMLResponse)
async def edit_graph_page(request: Request, graph_hash: str):
    """Serve the editor page for a specific graph."""
    graph = db.get_graph(graph_hash)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    return templates.TemplateResponse("index.html", {"request": request, "graph_hash": graph_hash})


@app.post("/api/graphs")
async def create_graph(graph: GraphCreate):
    """Create a new graph."""
    graph_hash = db.save_graph(graph.name, graph.data)
    return {"hash": graph_hash, "message": "Graph created successfully"}


@app.get("/api/graphs")
async def list_graphs(limit: int = 50, offset: int = 0):
    """List all graphs."""
    graphs = db.list_graphs(limit, offset)
    return {"graphs": graphs}


@app.get("/api/graphs/{graph_hash}")
async def get_graph(graph_hash: str):
    """Get a specific graph by hash."""
    graph = db.get_graph(graph_hash)
    if not graph:
        raise HTTPException(status_code=404, detail="Graph not found")
    return graph


@app.put("/api/graphs/{graph_hash}")
async def update_graph(graph_hash: str, graph: GraphUpdate):
    """Update an existing graph (override save)."""
    existing = db.get_graph(graph_hash)
    if not existing:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    db.update_graph(graph_hash, graph.data, graph.name)
    return {"hash": graph_hash, "message": "Graph updated successfully"}


@app.post("/api/graphs/{graph_hash}/fork")
async def fork_graph(graph_hash: str, graph: GraphFork):
    """Fork a graph (create new from existing)."""
    existing = db.get_graph(graph_hash)
    if not existing:
        raise HTTPException(status_code=404, detail="Parent graph not found")
    
    new_hash = db.save_graph(graph.name, graph.data, parent_hash=graph_hash)
    return {"hash": new_hash, "parent_hash": graph_hash, "message": "Graph forked successfully"}


@app.delete("/api/graphs/{graph_hash}")
async def delete_graph(graph_hash: str):
    """Delete a graph."""
    if not db.delete_graph(graph_hash):
        raise HTTPException(status_code=404, detail="Graph not found")
    return {"message": "Graph deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
