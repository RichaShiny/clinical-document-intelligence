"""Authenticated local API; source documents are not persisted; reviews are explicitly saved."""
from contextlib import asynccontextmanager
from datetime import datetime,timezone
import hmac
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Literal
import uuid
from fastapi import FastAPI,HTTPException,Header,Depends,Query,Request
from fastapi.responses import FileResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,ConfigDict
from .service import Engine


class AnalysisRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    text:str=Field(min_length=1,max_length=20000)


class ReviewRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    document_sha256:str=Field(pattern=r'^[a-f0-9]{64}$')
    sentence_start:int=Field(ge=0,le=20000)
    decision:Literal['relevant','not_relevant','uncertain']
    model_version:str=Field(min_length=1,max_length=64)
    rationale:str=Field(default='',max_length=500)


class BodyLimit:
    """Bound even chunked HTTP bodies before JSON parsing."""
    def __init__(self,app,max_bytes=131072):self.app=app;self.max_bytes=max_bytes
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['method'] not in ['POST','PUT','PATCH']:
            return await self.app(scope,receive,send)
        chunks=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            chunk=message.get('body',b'');size+=len(chunk)
            if size>self.max_bytes:
                return await Response(status_code=413)(scope,receive,send)
            chunks.append(chunk)
            if not message.get('more_body',False):break
        delivered=False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered=True
                return {'type':'http.request','body':b''.join(chunks),'more_body':False}
            return await receive()
        return await self.app(scope,replay,send)


def create_app(root=None,token=None,engine_factory=Engine):
    root=Path(root or os.environ.get('ADE_ROOT',Path.cwd()))
    secret=token if token is not None else os.environ.get('ADE_API_TOKEN','')
    state={'engine':None,'error':None,'requests':0,'errors':0,'seconds':0.};lock=threading.Lock()
    db_path=Path(os.environ.get('ADE_REVIEW_DB',root/'runtime/reviews.sqlite'))

    @asynccontextmanager
    async def lifespan(app):
        if not secret:raise RuntimeError('Set ADE_API_TOKEN before starting the API')
        try:state['engine']=engine_factory(root)
        except (OSError,ValueError,KeyError) as exc:state['error']=type(exc).__name__
        db_path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(db_path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, document_sha256 TEXT, sentence_start INTEGER, decision TEXT, model_version TEXT, rationale TEXT, created_at TEXT)')
        yield

    app=FastAPI(title='ADE Evidence Triage',version='1.0.0',lifespan=lifespan)
    app.add_middleware(BodyLimit)

    def authorize(authorization:str=Header(default='')):
        if not secret or not hmac.compare_digest(authorization,'Bearer '+secret):
            raise HTTPException(401,'Valid bearer token required',headers={'WWW-Authenticate':'Bearer'})

    @app.middleware('http')
    async def telemetry(request:Request,call_next):
        length=request.headers.get('content-length')
        if length:
            try:
                if int(length)>131072:return Response(status_code=413)
            except ValueError:return Response(status_code=400)
        start=time.perf_counter();response=await call_next(request)
        with lock:
            state['requests']+=1;state['errors']+=int(response.status_code>=400);state['seconds']+=time.perf_counter()-start
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store'
        return response

    @app.get('/health/live')
    def live():return {'status':'alive'}

    @app.get('/health/ready')
    def ready():
        if state['engine'] is None:raise HTTPException(503,'Model artifacts unavailable or invalid')
        return {'status':'ready','model':state['engine'].name,'version':state['engine'].version}

    @app.post('/v1/analyze',dependencies=[Depends(authorize)])
    def analyze(body:AnalysisRequest):
        if state['engine'] is None:raise HTTPException(503,'Model not ready')
        try:return state['engine'].analyze(body.text)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc

    @app.get('/v1/evaluation',dependencies=[Depends(authorize)])
    def evaluation(offset:int=Query(0,ge=0),limit:int=Query(20,ge=1,le=100)):
        file=root/'runtime/test_predictions.json'
        if not file.exists():raise HTTPException(404,'Run training first')
        rows=json.loads(file.read_text())
        return {'total':len(rows),'rows':rows[offset:offset+limit]}

    @app.get('/v1/report',dependencies=[Depends(authorize)])
    def report():
        file=root/'reports/metrics.json'
        if not file.exists():raise HTTPException(404,'No evaluated run')
        return json.loads(file.read_text())

    @app.post('/v1/reviews',dependencies=[Depends(authorize)],status_code=201)
    def review(body:ReviewRequest):
        record=dict(id=str(uuid.uuid4()),**body.model_dump(),created_at=datetime.now(timezone.utc).isoformat())
        with sqlite3.connect(db_path,timeout=10) as db:
            db.execute('INSERT INTO reviews VALUES (?,?,?,?,?,?,?)',tuple(record.values()))
        return {'id':record['id'],'created_at':record['created_at']}

    @app.get('/metrics',dependencies=[Depends(authorize)])
    def metrics():
        return Response(f"ade_http_requests_total {state['requests']}\nade_http_errors_total {state['errors']}\nade_http_request_seconds_sum {state['seconds']}\n",media_type='text/plain')

    if (root/'web').exists():
        app.mount('/assets',StaticFiles(directory=root/'web'),name='assets')
        @app.get('/')
        def index():return FileResponse(root/'web/index.html')
    return app


app=create_app()
