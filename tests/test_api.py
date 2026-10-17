import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from clinical_ade.api import create_app


class StubEngine:
    name='test-engine';version='test-v1'
    def __init__(self,root):pass
    def analyze(self,text):
        if text.isspace():raise ValueError('No sentences found')
        return dict(document_sha256=hashlib.sha256(text.encode()).hexdigest(),sentences=[],model=self.name)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path,token='unit-test-token',engine_factory=StubEngine)) as client:yield client


def test_authentication_is_required(client):
    assert client.post('/v1/analyze',json={'text':'input'}).status_code==401
    assert client.get('/v1/evaluation').status_code==401
    assert client.get('/metrics').status_code==401


def test_readiness_and_live_are_distinct(client):
    assert client.get('/health/live').status_code==200
    assert client.get('/health/ready').json()['model']=='test-engine'


def test_missing_artifacts_prevents_ready(tmp_path):
    with TestClient(create_app(tmp_path,token='test')) as c:
        assert c.get('/health/live').status_code==200
        assert c.get('/health/ready').status_code==503
        assert c.post('/v1/analyze',headers={'Authorization':'Bearer test'},json={'text':'input'}).status_code==503


def test_schema_and_empty_input(client):
    headers={'Authorization':'Bearer unit-test-token'}
    assert client.post('/v1/analyze',headers=headers,json={'text':''}).status_code==422
    assert client.post('/v1/analyze',headers=headers,json={'text':' '*3}).status_code==422
    assert client.post('/v1/analyze',headers=headers,json={'text':'x'*20001}).status_code==422


def test_analysis_is_authenticated_and_not_cached(client):
    response=client.post('/v1/analyze',headers={'Authorization':'Bearer unit-test-token'},json={'text':'input'})
    assert response.status_code==200
    assert response.headers['cache-control']=='no-store'
    assert response.json()['document_sha256']==hashlib.sha256(b'input').hexdigest()


def test_review_persists_without_source_text(client,tmp_path):
    import sqlite3
    body=dict(document_sha256='a'*64,sentence_start=0,decision='uncertain',model_version='test-v1',rationale='Needs expert review')
    response=client.post('/v1/reviews',headers={'Authorization':'Bearer unit-test-token'},json=body)
    assert response.status_code==201
    with sqlite3.connect(tmp_path/'runtime/reviews.sqlite') as db:
        record=db.execute('SELECT decision,model_version FROM reviews').fetchone()
    assert record==('uncertain','test-v1')


def test_oversized_request_rejected(client):
    response=client.post('/v1/analyze',headers={'Authorization':'Bearer unit-test-token'},content=b'x'*131073)
    assert response.status_code==413


def test_api_refuses_to_start_without_auth_secret(tmp_path):
    with pytest.raises(RuntimeError,match='ADE_API_TOKEN'):
        with TestClient(create_app(tmp_path,token='')):pass


def test_chunked_oversized_request_rejected(client):
    chunks=(b'x'*70000 for _ in range(2))
    response=client.post('/v1/analyze',headers={'Authorization':'Bearer unit-test-token'},content=chunks)
    assert response.status_code==413
