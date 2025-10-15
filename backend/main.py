# backend/main.py
import os
import uuid
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import logging
import traceback
import io
import zipfile
import json
from pathlib import Path
from enum import Enum
import asyncio
from privacy_masker import PrivacyMasker
from synthetic_generator import SyntheticDataGenerator
from quality_validator import QualityValidator
from data_processor import DataProcessor
from multi_table_processor import EnhancedSyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

os.makedirs("uploads", exist_ok=True)
os.makedirs("synthetic", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

app = FastAPI(title="Synthetic Data Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

datasets_db: Dict[str, Dict[str, Any]] = {}

# --- CI/CD In-memory Stores ---
pipelines_db: Dict[str, Dict[str, Any]] = {}
runs_db: Dict[str, Dict[str, Any]] = {}

class DatasetResponse(BaseModel):
    id: str
    filename: str
    status: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    relationships: Optional[Dict[str, Any]] = None
    relationship_summary: Optional[Dict[str, Any]] = None
    table_count: Optional[int] = 1
    created_at: str
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    privacy_config: Optional[Dict[str, Any]] = None
    progress: Optional[int] = 0

class PrivacyConfig(BaseModel):
    mask_emails: bool = True
    mask_names: bool = True
    mask_phone_numbers: bool = True
    mask_addresses: bool = True
    mask_ssn: bool = True
    custom_fields: List[str] = []
    use_gan: bool = True
    gan_epochs: int = 300
    anonymization_method: str = "faker"

class GenerateSyntheticRequest(BaseModel):
    dataset_id: str
    privacy_config: PrivacyConfig
    num_rows: Optional[int] = None

# --- CI/CD Models ---
class QualityGate(BaseModel):
    min_overall_quality: float = Field(ge=0, le=100, default=60.0)
    allow_missing_columns: bool = True

class ExportTarget(BaseModel):
    type: str = Field(default="local", description="local|s3|gcs|azure")
    path: Optional[str] = Field(default="artifacts")

class PipelineConfig(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = ""
    auto_trigger_on_upload: bool = False
    default_privacy: PrivacyConfig = PrivacyConfig()
    quality_gate: QualityGate = QualityGate()
    export_target: ExportTarget = ExportTarget()
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    active: bool = True

class PipelineRunResponse(BaseModel):
    id: str
    pipeline_id: str
    dataset_id: str
    status: str
    progress: int
    started_at: str
    finished_at: Optional[str] = None
    quality_report: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    artifact_path: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Synthetic Data Generator API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    try:
        logger.info(f"Upload request received for file: {file.filename}")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls', '.zip')):
            raise HTTPException(status_code=400, detail="Only CSV, Excel, and ZIP files are supported")

        file_id = str(uuid.uuid4())
        safe_filename = file.filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File saved to: {file_path}")

        if file.filename.lower().endswith('.zip'):
            enhanced_generator = EnhancedSyntheticDataGenerator()
            tables = enhanced_generator._extract_tables(content, file.filename)
            total_rows = sum(len(df) for df in tables.values())
            total_columns = sum(len(df.columns) for df in tables.values())
            table_count = len(tables)
            relationships = enhanced_generator._detect_relationships(tables)
            relationship_summary = enhanced_generator._create_relationship_summary(relationships, tables)
        else:
            data_processor = DataProcessor()
            if file.filename.lower().endswith('.csv'):
                df = data_processor.load_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            total_rows = len(df)
            total_columns = len(df.columns)
            table_count = 1
            relationships = {}
            relationship_summary = {
                "total_relationships": 0,
                "tables_with_primary_keys": 0,
                "tables_with_foreign_keys": 0,
                "generation_order": [Path(file.filename).stem],
                "relationship_details": []
            }

        dataset = {
            "id": file_id,
            "filename": safe_filename,
            "status": "uploaded",
            "file_path": file_path,
            "row_count": total_rows,
            "column_count": total_columns,
            "file_size": len(content),
            "table_count": table_count,
            "created_at": datetime.utcnow().isoformat(),
            "error_message": None,
            "quality_metrics": None,
            "privacy_config": None,
            "relationships": relationships,
            "relationship_summary": relationship_summary,
            "progress": 0
        }
        datasets_db[file_id] = dataset
        logger.info(f"Dataset stored with ID: {file_id}")

        # Auto-trigger pipelines configured as auto_trigger_on_upload
        for p in pipelines_db.values():
            if p.get("active") and p.get("auto_trigger_on_upload"):
                run_id = str(uuid.uuid4())
                runs_db[run_id] = {
                    "id": run_id,
                    "pipeline_id": p["id"],
                    "dataset_id": file_id,
                    "status": "queued",
                    "progress": 0,
                    "started_at": datetime.utcnow().isoformat(),
                    "finished_at": None,
                    "quality_report": None,
                    "message": "Auto-triggered on upload",
                    "artifact_path": None
                }
                # Run in background
                from fastapi import BackgroundTasks as _BT  # avoid shadow
                bt = _BT()
                bt.add_task(execute_pipeline_run, run_id)
                # Kick it off synchronously to avoid missing background task in this context
                await execute_pipeline_run(run_id)

        return DatasetResponse(**dataset)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/datasets/upload", response_model=DatasetResponse)
async def upload_dataset_legacy(file: UploadFile = File(...)):
    return await upload_dataset(file)

@app.get("/api/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset = datasets_db[dataset_id]
    return DatasetResponse(**{
        **dataset,
        "relationships": dataset.get("relationships", {}),
        "relationship_summary": dataset.get("relationship_summary", {
            "total_relationships": 0,
            "tables_with_primary_keys": 0,
            "tables_with_foreign_keys": 0,
            "generation_order": [],
            "relationship_details": []
        }),
        "progress": dataset.get("progress", 0)
    })

@app.get("/api/datasets", response_model=List[DatasetResponse])
async def list_datasets():
    return [DatasetResponse(**d) for d in datasets_db.values()]

async def generate_synthetic_data_background_enhanced(dataset_id: str, privacy_config: PrivacyConfig, num_rows: Optional[int] = None):
    try:
        if dataset_id not in datasets_db:
            return
        datasets_db[dataset_id]["status"] = "processing"
        datasets_db[dataset_id]["progress"] = 0

        enhanced_generator = EnhancedSyntheticDataGenerator(use_gan=privacy_config.use_gan, gan_epochs=privacy_config.gan_epochs)
        with open(datasets_db[dataset_id]["file_path"], "rb") as f:
            file_data = f.read()
        filename = datasets_db[dataset_id]["filename"]

        datasets_db[dataset_id]["progress"] = 10
        # Pass num_rows to process_upload
        result = await asyncio.to_thread(enhanced_generator.process_upload, file_data, filename, privacy_config, num_rows)
        datasets_db[dataset_id]["progress"] = 85
        synthetic_paths = {}
        for table_name, synthetic_df in result['tables'].items():
            synthetic_path = f"synthetic/{dataset_id}_{table_name}_synthetic.csv"
            synthetic_df.to_csv(synthetic_path, index=False)
            synthetic_paths[table_name] = synthetic_path

        datasets_db[dataset_id]["progress"] = 95
        datasets_db[dataset_id].update({
            "status": "completed",
            "synthetic_path": list(synthetic_paths.values())[0] if len(synthetic_paths) == 1 else None,
            "synthetic_paths": synthetic_paths,
            "quality_metrics": result.get('quality_metrics', {}),
            "relationship_summary": result.get('relationship_summary', {}),
            "progress": 100
        })

        logger.info(f"✅ Enhanced generation completed for {dataset_id}")
        logger.info(f"📊 Generated {len(synthetic_paths)} synthetic table(s)")
        logger.info(f"🎯 Target rows: {num_rows if num_rows else 'Same as original'}")

    except Exception as e:
        logger.error(f"Error in enhanced generation: {e}")
        logger.error(traceback.format_exc())
        if dataset_id in datasets_db:
            datasets_db[dataset_id]["status"] = "error"
            datasets_db[dataset_id]["status"] = "failed"
            datasets_db[dataset_id]["error_message"] = str(e)
            datasets_db[dataset_id]["progress"] = 100

@app.post("/api/datasets/{dataset_id}/generate-synthetic")
async def generate_synthetic_data(dataset_id: str, background_tasks: BackgroundTasks, request: Request):
    try:
        if dataset_id not in datasets_db:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if datasets_db[dataset_id]["status"] == "processing":
            raise HTTPException(status_code=400, detail="Already processing")

        try:
            body = await request.json()
            privacy_config_data = body.get("privacy_config", {})
            num_rows = body.get("num_rows")
            privacy_config = PrivacyConfig(**privacy_config_data)
        except Exception as e:
            logger.error(f"Error parsing request: {e}")
            privacy_config = PrivacyConfig()
            num_rows = None

        background_tasks.add_task(generate_synthetic_data_background_enhanced, dataset_id, privacy_config, num_rows)
        return {"message": "Synthetic data generation started", "dataset_id": dataset_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start: {str(e)}")

@app.get("/api/datasets/{dataset_id}/download")
async def download_synthetic_data(dataset_id: str):
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset = datasets_db[dataset_id]
    if dataset["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Data not ready. Status: {dataset['status']}")
    if "synthetic_path" in dataset and dataset["synthetic_path"]:
        if not os.path.exists(dataset["synthetic_path"]):
            raise HTTPException(status_code=404, detail="Synthetic data file not found")
        return FileResponse(dataset["synthetic_path"], media_type="text/csv", filename=f"{dataset['filename']}_synthetic.csv")
    if "synthetic_paths" in dataset and dataset["synthetic_paths"]:
        first_path = list(dataset["synthetic_paths"].values())[0]
        if os.path.exists(first_path):
            return FileResponse(first_path, media_type="text/csv", filename=f"{Path(first_path).name}")
    raise HTTPException(status_code=404, detail="Synthetic data not found")

@app.get("/api/datasets/{dataset_id}/download-zip")
async def download_synthetic_data_zip(dataset_id: str):
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset = datasets_db[dataset_id]
    if dataset["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Data not ready. Status: {dataset['status']}")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if "synthetic_path" in dataset and dataset["synthetic_path"]:
            if os.path.exists(dataset["synthetic_path"]):
                zip_file.write(dataset["synthetic_path"], f"{dataset['filename']}_synthetic.csv")
        if "synthetic_paths" in dataset:
            for table_name, path in dataset["synthetic_paths"].items():
                if os.path.exists(path):
                    zip_file.write(path, f"{table_name}_synthetic.csv")
        if "relationship_summary" in dataset and dataset["relationship_summary"]["total_relationships"] > 0:
            relationship_json = json.dumps(dataset["relationship_summary"], indent=2)
            zip_file.writestr("relationship_summary.json", relationship_json)
    zip_buffer.seek(0)
    return StreamingResponse(io.BytesIO(zip_buffer.read()), media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={dataset['filename']}_synthetic.zip"})

@app.get("/api/datasets/{dataset_id}/relationships")
async def get_dataset_relationships(dataset_id: str):
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset = datasets_db[dataset_id]
    table_count = dataset.get("table_count", 1)
    relationships = dataset.get("relationships", {})
    relationship_summary = dataset.get("relationship_summary", {
        "total_relationships": 0,
        "tables_with_primary_keys": 0,
        "tables_with_foreign_keys": 0,
        "generation_order": [Path(dataset["filename"]).stem] if "filename" in dataset else [],
        "relationship_details": []
    })
    return {
        "dataset_id": dataset_id,
        "table_count": table_count,
        "relationships": relationships,
        "relationship_summary": relationship_summary,
        "status": dataset.get("status", "unknown")
    }

@app.get("/api/datasets/{dataset_id}/preview")
async def preview_data(dataset_id: str, synthetic: bool = False, table_name: str = None):
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset = datasets_db[dataset_id]
    try:
        if synthetic:
            if "synthetic_paths" in dataset and dataset["synthetic_paths"]:
                if table_name:
                    if table_name not in dataset["synthetic_paths"]:
                        raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
                    synthetic_path = dataset["synthetic_paths"][table_name]
                else:
                    synthetic_path = list(dataset["synthetic_paths"].values())[0]
                if not os.path.exists(synthetic_path):
                    raise HTTPException(status_code=404, detail="Synthetic data file not found")
                df = pd.read_csv(synthetic_path)
            elif "synthetic_path" in dataset and os.path.exists(dataset["synthetic_path"]):
                df = pd.read_csv(dataset["synthetic_path"])
            else:
                raise HTTPException(status_code=404, detail="Synthetic data not found")
        else:
            if not os.path.exists(dataset["file_path"]):
                raise HTTPException(status_code=404, detail="Original data file not found")
            file_path = dataset["file_path"]
            if file_path.lower().endswith('.zip'):
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                enhanced_generator = EnhancedSyntheticDataGenerator()
                tables = enhanced_generator._extract_tables(file_data, dataset["filename"])
                if not tables:
                    raise HTTPException(status_code=404, detail="No tables found in ZIP file")
                if table_name and table_name in tables:
                    df = tables[table_name]
                else:
                    first = list(tables.keys())[0]
                    df = tables[first]
            elif file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type for preview")
        preview_df = df.head(5)
        return {
            "columns": df.columns.tolist(),
            "data": preview_df.to_dict('records'),
            "total_rows": len(df),
            "preview_rows": len(preview_df),
            "table_name": table_name or "default"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading data preview: {str(e)}")

@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    dataset = datasets_db[dataset_id]
    try:
        if os.path.exists(dataset["file_path"]):
            os.remove(dataset["file_path"])
        if "synthetic_path" in dataset and os.path.exists(dataset["synthetic_path"]):
            os.remove(dataset["synthetic_path"])
        if "synthetic_paths" in dataset:
            for path in dataset["synthetic_paths"].values():
                if os.path.exists(path):
                    os.remove(path)
    except Exception as e:
        logger.warning(f"Error deleting files for dataset {dataset_id}: {e}")
    del datasets_db[dataset_id]
    return {"message": "Dataset deleted successfully"}

# ----------------- CI/CD Endpoints -----------------

@app.get("/api/cicd/pipelines")
async def list_pipelines():
    return list(pipelines_db.values())

@app.post("/api/cicd/pipelines")
async def upsert_pipeline(config: PipelineConfig):
    now = datetime.utcnow().isoformat()
    pid = config.id or str(uuid.uuid4())
    record = {
        "id": pid,
        "name": config.name,
        "description": config.description,
        "auto_trigger_on_upload": config.auto_trigger_on_upload,
        "default_privacy": config.default_privacy.dict(),
        "quality_gate": config.quality_gate.dict(),
        "export_target": config.export_target.dict(),
        "active": config.active,
        "created_at": config.created_at or now,
        "updated_at": now
    }
    pipelines_db[pid] = record
    return record

@app.post("/api/cicd/pipelines/{pipeline_id}/run")
async def run_pipeline(pipeline_id: str, request: Request, background_tasks: BackgroundTasks):
    if pipeline_id not in pipelines_db:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    body = await request.json()
    dataset_id = body.get("dataset_id")
    if not dataset_id or dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    run_id = str(uuid.uuid4())
    runs_db[run_id] = {
        "id": run_id,
        "pipeline_id": pipeline_id,
        "dataset_id": dataset_id,
        "status": "queued",
        "progress": 0,
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
        "quality_report": None,
        "message": "Queued",
        "artifact_path": None
    }
    background_tasks.add_task(execute_pipeline_run, run_id)
    return {"message": "Pipeline run started", "run_id": run_id}

@app.get("/api/cicd/runs")
async def list_runs():
    # newest first
    return sorted(list(runs_db.values()), key=lambda x: x["started_at"], reverse=True)

@app.get("/api/cicd/runs/{run_id}")
async def get_run(run_id: str):
    if run_id not in runs_db:
        raise HTTPException(status_code=404, detail="Run not found")
    return runs_db[run_id]

async def execute_pipeline_run(run_id: str):
    try:
        run = runs_db[run_id]
        pipeline = pipelines_db[run["pipeline_id"]]
        dataset = datasets_db[run["dataset_id"]]
        runs_db[run_id]["status"] = "running"
        runs_db[run_id]["message"] = "Validating dataset"
        runs_db[run_id]["progress"] = 10

        # Step 1: Validation
        if not os.path.exists(dataset["file_path"]):
            raise RuntimeError("Original file missing")
        runs_db[run_id]["message"] = "Generating synthetic data"
        runs_db[run_id]["progress"] = 30

        # Step 2: Generation with pipeline's default privacy
        privacy_cfg = PrivacyConfig(**pipeline["default_privacy"])
        await generate_synthetic_data_background_enhanced(run["dataset_id"], privacy_cfg, None)

        # Step 3: Quality Gate
        runs_db[run_id]["message"] = "Evaluating quality gates"
        runs_db[run_id]["progress"] = 70
        ds = datasets_db[run["dataset_id"]]
        q = ds.get("quality_metrics") or {}
        min_q = float(pipeline["quality_gate"]["min_overall_quality"])
        score = float(q.get("overall_quality_score", 0.0))
        if score < min_q:
            runs_db[run_id]["status"] = "failed"
            runs_db[run_id]["progress"] = 100
            runs_db[run_id]["finished_at"] = datetime.utcnow().isoformat()
            runs_db[run_id]["quality_report"] = q
            runs_db[run_id]["message"] = f"Quality gate failed: {score:.2f} < {min_q:.2f}"
            return

        # Step 4: Export Artifact
        runs_db[run_id]["message"] = "Exporting artifacts"
        runs_db[run_id]["progress"] = 85
        export_dir = pipeline["export_target"]["path"] or "artifacts"
        os.makedirs(export_dir, exist_ok=True)
        artifact_name = f"{dataset['id']}_synthetic_export_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.zip"
        artifact_path = os.path.join(export_dir, artifact_name)

        # Package outputs (CSV(s) + relationship summary + quality report)
        with zipfile.ZipFile(artifact_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if "synthetic_path" in ds and ds["synthetic_path"] and os.path.exists(ds["synthetic_path"]):
                zipf.write(ds["synthetic_path"], Path(ds["synthetic_path"]).name)
            if "synthetic_paths" in ds and ds["synthetic_paths"]:
                for tname, path in ds["synthetic_paths"].items():
                    if os.path.exists(path):
                        zipf.write(path, Path(path).name)
            if "relationship_summary" in ds:
                zipf.writestr("relationship_summary.json", json.dumps(ds["relationship_summary"], indent=2))
            if ds.get("quality_metrics"):
                zipf.writestr("quality_report.json", json.dumps(ds["quality_metrics"], indent=2))

        runs_db[run_id]["artifact_path"] = artifact_path
        runs_db[run_id]["status"] = "succeeded"
        runs_db[run_id]["message"] = "Pipeline completed"
        runs_db[run_id]["quality_report"] = datasets_db[run["dataset_id"]].get("quality_metrics")
        runs_db[run_id]["progress"] = 100
        runs_db[run_id]["finished_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        logger.error(f"Pipeline run error: {e}")
        logger.error(traceback.format_exc())
        if run_id in runs_db:
            runs_db[run_id]["status"] = "failed"
            runs_db[run_id]["message"] = str(e)
            runs_db[run_id]["finished_at"] = datetime.utcnow().isoformat()
            runs_db[run_id]["progress"] = 100

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    logger.error(traceback.format_exc())
    return HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")