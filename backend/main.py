import os
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uvicorn

# Import custom modules
from data_processor import DataProcessor
from synthetic_generator import SyntheticDataGenerator
from privacy_masker import PrivacyMasker
from quality_validator import QualityValidator
from utils import setup_logging, validate_file_type, sanitize_filename

# Setup logging
logger = setup_logging()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./synthetic_data.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    original_path = Column(String, nullable=False)
    synthetic_path = Column(String, nullable=True)
    status = Column(String, default="uploaded")  # uploaded, processing, completed, failed
    privacy_config = Column(JSON)
    quality_metrics = Column(JSON)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    file_size = Column(Integer)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class PrivacyConfig(BaseModel):
    mask_emails: bool = True
    mask_names: bool = True
    mask_phone_numbers: bool = True
    mask_addresses: bool = True
    mask_ssn: bool = True
    custom_fields: List[str] = Field(default_factory=list)
    anonymization_method: str = "faker"  # faker, hash, redact

class DatasetResponse(BaseModel):
    id: str
    filename: str
    status: str
    privacy_config: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    file_size: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None

class GenerateSyntheticRequest(BaseModel):
    dataset_id: str
    privacy_config: PrivacyConfig
    num_rows: Optional[int] = None  # If None, generates same number as original

# FastAPI app
app = FastAPI(
    title="Synthetic Data Generator API",
    description="Privacy-safe synthetic data generation API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Ensure upload directory exists
os.makedirs("uploads", exist_ok=True)
os.makedirs("synthetic", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Synthetic Data Generator API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/api/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    """Upload a dataset for synthetic data generation"""
    try:
        # Validate file type
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only CSV files are supported."
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        safe_filename = sanitize_filename(file.filename)
        file_path = f"uploads/{file_id}_{safe_filename}"

        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Basic file analysis
        try:
            df = pd.read_csv(file_path)
            row_count = len(df)
            column_count = len(df.columns)
        except Exception as e:
            logger.error(f"Error analyzing uploaded file: {e}")
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

        # Create database record
        dataset = Dataset(
            id=file_id,
            filename=safe_filename,
            original_path=file_path,
            file_size=len(content),
            row_count=row_count,
            column_count=column_count
        )

        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        logger.info(f"Dataset uploaded successfully: {file_id}")

        return DatasetResponse(
            id=dataset.id,
            filename=dataset.filename,
            status=dataset.status,
            privacy_config=dataset.privacy_config,
            quality_metrics=dataset.quality_metrics,
            error_message=dataset.error_message,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            file_size=dataset.file_size,
            row_count=dataset.row_count,
            column_count=dataset.column_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_dataset: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Get dataset information"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetResponse(
        id=dataset.id,
        filename=dataset.filename,
        status=dataset.status,
        privacy_config=dataset.privacy_config,
        quality_metrics=dataset.quality_metrics,
        error_message=dataset.error_message,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        file_size=dataset.file_size,
        row_count=dataset.row_count,
        column_count=dataset.column_count
    )

@app.get("/api/datasets", response_model=List[DatasetResponse])
async def list_datasets(db: Session = Depends(get_db), limit: int = 50):
    """List all datasets"""
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).limit(limit).all()

    return [
        DatasetResponse(
            id=dataset.id,
            filename=dataset.filename,
            status=dataset.status,
            privacy_config=dataset.privacy_config,
            quality_metrics=dataset.quality_metrics,
            error_message=dataset.error_message,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            file_size=dataset.file_size,
            row_count=dataset.row_count,
            column_count=dataset.column_count
        ) for dataset in datasets
    ]

async def generate_synthetic_data_task(
        dataset_id: str,
        privacy_config: PrivacyConfig,
        num_rows: Optional[int] = None
):
    """Background task to generate synthetic data"""
    db = SessionLocal()
    try:
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            logger.error(f"Dataset not found: {dataset_id}")
            return

        # Update status to processing
        dataset.status = "processing"
        dataset.privacy_config = privacy_config.dict()
        dataset.updated_at = datetime.utcnow()
        db.commit()

        logger.info(f"Starting synthetic data generation for dataset: {dataset_id}")

        # Load original data
        processor = DataProcessor()
        df = processor.load_csv(dataset.original_path)

        # Apply privacy masking
        masker = PrivacyMasker()
        df_masked = masker.apply_privacy_masks(df, privacy_config)

        # Generate synthetic data
        generator = SyntheticDataGenerator()
        synthetic_df = generator.generate_synthetic_data(
            df_masked,
            num_rows=num_rows or len(df)
        )

        # Save synthetic data
        synthetic_path = f"synthetic/{dataset_id}_synthetic.csv"
        synthetic_df.to_csv(synthetic_path, index=False)

        # Validate quality
        validator = QualityValidator()
        quality_metrics = validator.compare_distributions(df, synthetic_df)

        # Update database
        dataset.status = "completed"
        dataset.synthetic_path = synthetic_path
        dataset.quality_metrics = quality_metrics
        dataset.updated_at = datetime.utcnow()
        db.commit()

        logger.info(f"Synthetic data generation completed for dataset: {dataset_id}")

    except Exception as e:
        logger.error(f"Error in synthetic data generation: {e}")
        # Update status to failed
        dataset.status = "failed"
        dataset.error_message = str(e)
        dataset.updated_at = datetime.utcnow()
        db.commit()

    finally:
        db.close()

@app.post("/api/datasets/{dataset_id}/generate-synthetic")
async def generate_synthetic_data(
        dataset_id: str,
        request: GenerateSyntheticRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    """Generate synthetic data for a dataset"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.status == "processing":
        raise HTTPException(status_code=400, detail="Dataset is already being processed")

    # Start background task
    background_tasks.add_task(
        generate_synthetic_data_task,
        dataset_id,
        request.privacy_config,
        request.num_rows
    )

    return {"message": "Synthetic data generation started", "dataset_id": dataset_id}

@app.get("/api/datasets/{dataset_id}/download")
async def download_synthetic_data(dataset_id: str, db: Session = Depends(get_db)):
    """Download synthetic dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.status != "completed" or not dataset.synthetic_path:
        raise HTTPException(status_code=400, detail="Synthetic data not ready for download")

    if not os.path.exists(dataset.synthetic_path):
        raise HTTPException(status_code=404, detail="Synthetic data file not found")

    return FileResponse(
        dataset.synthetic_path,
        filename=f"{dataset.filename}_synthetic.csv",
        media_type="text/csv"
    )

@app.get("/api/datasets/{dataset_id}/preview")
async def preview_data(dataset_id: str, synthetic: bool = False, db: Session = Depends(get_db)):
    """Preview original or synthetic data (first 10 rows)"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        if synthetic:
            if not dataset.synthetic_path or not os.path.exists(dataset.synthetic_path):
                raise HTTPException(status_code=404, detail="Synthetic data not found")
            df = pd.read_csv(dataset.synthetic_path)
        else:
            df = pd.read_csv(dataset.original_path)

        # Return first 10 rows
        preview_df = df.head(10)

        return {
            "columns": df.columns.tolist(),
            "data": preview_df.to_dict('records'),
            "total_rows": len(df),
            "preview_rows": len(preview_df)
        }

    except Exception as e:
        logger.error(f"Error previewing data: {e}")
        raise HTTPException(status_code=500, detail="Error loading data preview")

@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset and its files"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete files
    try:
        if os.path.exists(dataset.original_path):
            os.remove(dataset.original_path)
        if dataset.synthetic_path and os.path.exists(dataset.synthetic_path):
            os.remove(dataset.synthetic_path)
    except Exception as e:
        logger.warning(f"Error deleting files for dataset {dataset_id}: {e}")

    # Delete database record
    db.delete(dataset)
    db.commit()

    return {"message": "Dataset deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )