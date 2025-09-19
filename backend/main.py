import os
import uuid
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import time
import logging
import traceback

# Import our custom classes
from privacy_masker import PrivacyMasker
from synthetic_generator import SyntheticDataGenerator
from quality_validator import QualityValidator
from data_processor import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("synthetic", exist_ok=True)
os.makedirs("logs", exist_ok=True)

app = FastAPI(title="Synthetic Data Generator API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for datasets
datasets_db = {}

class DatasetResponse(BaseModel):
    id: str
    filename: str
    status: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    created_at: str
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    privacy_config: Optional[Dict[str, Any]] = None

class PrivacyConfig(BaseModel):
    mask_emails: bool = True
    mask_names: bool = True
    mask_phone_numbers: bool = True
    mask_addresses: bool = True
    mask_ssn: bool = True
    custom_fields: List[str] = []
    anonymization_method: str = "faker"

class GenerateSyntheticRequest(BaseModel):
    dataset_id: str
    privacy_config: PrivacyConfig
    num_rows: Optional[int] = None

@app.get("/")
async def root():
    return {"message": "Synthetic Data Generator API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset endpoint - matches frontend expectation"""
    try:
        logger.info(f"Upload request received for file: {file.filename}")

        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

        # Generate unique ID
        file_id = str(uuid.uuid4())
        safe_filename = file.filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        # Save file
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File saved to: {file_path}")

        # Analyze file
        try:
            # Use DataProcessor for better file handling
            data_processor = DataProcessor()

            if file.filename.lower().endswith('.csv'):
                df = data_processor.load_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            row_count = len(df)
            column_count = len(df.columns)

            if row_count == 0:
                raise ValueError("Dataset is empty")

            if column_count == 0:
                raise ValueError("Dataset has no columns")

            logger.info(f"Dataset analyzed: {row_count} rows, {column_count} columns")

        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")

        # Store in memory database
        dataset = {
            "id": file_id,
            "filename": safe_filename,
            "status": "uploaded",
            "file_path": file_path,
            "row_count": row_count,
            "column_count": column_count,
            "file_size": len(content),
            "created_at": datetime.utcnow().isoformat(),
            "error_message": None,
            "quality_metrics": None,
            "privacy_config": None
        }

        datasets_db[file_id] = dataset
        logger.info(f"Dataset stored with ID: {file_id}")

        return DatasetResponse(**dataset)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/datasets/upload", response_model=DatasetResponse)
async def upload_dataset_legacy(file: UploadFile = File(...)):
    """Legacy upload endpoint for backwards compatibility"""
    return await upload_dataset(file)

@app.get("/api/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    logger.info(f"Getting dataset: {dataset_id}")

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets_db[dataset_id]
    return DatasetResponse(**dataset)

@app.get("/api/datasets", response_model=List[DatasetResponse])
async def list_datasets():
    logger.info("Listing all datasets")
    datasets_list = []
    for dataset in datasets_db.values():
        datasets_list.append(DatasetResponse(**dataset))
    return datasets_list

async def generate_synthetic_data_background(dataset_id: str, privacy_config: PrivacyConfig, num_rows: Optional[int] = None):
    """Background task to generate synthetic data with proper error handling"""
    try:
        logger.info(f"Starting synthetic data generation for dataset: {dataset_id}")

        # Verify dataset exists
        if dataset_id not in datasets_db:
            logger.error(f"Dataset {dataset_id} not found in background task")
            return

        dataset = datasets_db[dataset_id]
        datasets_db[dataset_id]["status"] = "processing"
        logger.info(f"Dataset {dataset_id} status set to processing")

        # Load original data
        original_df = pd.read_csv(dataset["file_path"])
        logger.info(f"Loaded original data: {len(original_df)} rows, {len(original_df.columns)} columns")

        # Step 1: Apply privacy masking
        logger.info("Step 1: Applying privacy masks...")
        await asyncio.sleep(1)  # Simulate processing time

        privacy_masker = PrivacyMasker()
        masked_df = privacy_masker.apply_privacy_masks(original_df, privacy_config)
        logger.info("Privacy masking completed")

        # Step 2: Generate synthetic data
        logger.info("Step 2: Generating synthetic data...")
        await asyncio.sleep(2)

        synthetic_generator = SyntheticDataGenerator(method="statistical")
        target_rows = num_rows if num_rows else len(original_df)
        synthetic_df = synthetic_generator.generate_synthetic_data(masked_df, target_rows)
        logger.info(f"Synthetic data generated: {len(synthetic_df)} rows")

        # Step 3: Quality validation
        logger.info("Step 3: Validating quality...")
        await asyncio.sleep(1)

        quality_validator = QualityValidator()
        quality_metrics = quality_validator.compare_distributions(original_df, synthetic_df)
        logger.info("Quality validation completed")

        # Step 4: Save synthetic data
        logger.info("Step 4: Saving synthetic data...")
        synthetic_path = f"synthetic/{dataset_id}_synthetic.csv"
        synthetic_df.to_csv(synthetic_path, index=False)
        logger.info(f"Synthetic data saved to: {synthetic_path}")

        # Update dataset with completion
        datasets_db[dataset_id].update({
            "status": "completed",
            "synthetic_path": synthetic_path,
            "quality_metrics": quality_metrics,
            "privacy_config": privacy_config.dict()
        })

        logger.info(f"Dataset {dataset_id} generation completed successfully")

    except Exception as e:
        logger.error(f"Error in background generation for dataset {dataset_id}: {e}")
        logger.error(traceback.format_exc())

        if dataset_id in datasets_db:
            datasets_db[dataset_id]["status"] = "failed"
            datasets_db[dataset_id]["error_message"] = str(e)

@app.post("/api/datasets/{dataset_id}/generate-synthetic")
async def generate_synthetic_data(
        dataset_id: str,
        background_tasks: BackgroundTasks,
        request: Request
):
    """Generate synthetic data endpoint with proper request handling"""
    try:
        logger.info(f"Generate synthetic data requested for dataset: {dataset_id}")

        if dataset_id not in datasets_db:
            raise HTTPException(status_code=404, detail="Dataset not found")

        dataset = datasets_db[dataset_id]

        if dataset["status"] == "processing":
            raise HTTPException(status_code=400, detail="Dataset is already being processed")

        # Parse request body
        try:
            body = await request.json()
            logger.info(f"Request body: {body}")

            # Extract privacy config and num_rows
            privacy_config_data = body.get("privacy_config", {})
            num_rows = body.get("num_rows")

            # Create PrivacyConfig object
            privacy_config = PrivacyConfig(**privacy_config_data)

        except Exception as e:
            logger.error(f"Error parsing request body: {e}")
            # Use default config if parsing fails
            privacy_config = PrivacyConfig()
            num_rows = None

        logger.info(f"Using privacy config: {privacy_config.dict()}")
        logger.info(f"Target rows: {num_rows}")

        # Start background task
        background_tasks.add_task(
            generate_synthetic_data_background,
            dataset_id,
            privacy_config,
            num_rows
        )

        logger.info(f"Background task started for dataset: {dataset_id}")
        return {"message": "Synthetic data generation started", "dataset_id": dataset_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting generation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

@app.get("/api/datasets/{dataset_id}/download")
async def download_synthetic_data(dataset_id: str):
    logger.info(f"Download requested for dataset: {dataset_id}")

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets_db[dataset_id]

    if dataset["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Synthetic data not ready. Status: {dataset['status']}")

    if "synthetic_path" not in dataset or not os.path.exists(dataset["synthetic_path"]):
        raise HTTPException(status_code=404, detail="Synthetic data file not found")

    return FileResponse(
        dataset["synthetic_path"],
        filename=f"{dataset['filename']}_synthetic.csv",
        media_type="text/csv"
    )

@app.get("/api/datasets/{dataset_id}/preview")
async def preview_data(dataset_id: str, synthetic: bool = False):
    logger.info(f"Preview requested for dataset: {dataset_id}, synthetic: {synthetic}")

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets_db[dataset_id]

    try:
        if synthetic:
            if "synthetic_path" not in dataset or not os.path.exists(dataset["synthetic_path"]):
                raise HTTPException(status_code=404, detail="Synthetic data not found")
            df = pd.read_csv(dataset["synthetic_path"])
        else:
            if not os.path.exists(dataset["file_path"]):
                raise HTTPException(status_code=404, detail="Original data file not found")
            df = pd.read_csv(dataset["file_path"])

        # Return first 5 rows for preview
        preview_df = df.head(5)

        response = {
            "columns": df.columns.tolist(),
            "data": preview_df.to_dict('records'),
            "total_rows": len(df),
            "preview_rows": len(preview_df)
        }

        logger.info(f"Preview data prepared: {len(response['data'])} rows, {len(response['columns'])} columns")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading data preview: {str(e)}")

@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    logger.info(f"Delete requested for dataset: {dataset_id}")

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets_db[dataset_id]

    # Delete files
    try:
        if os.path.exists(dataset["file_path"]):
            os.remove(dataset["file_path"])
            logger.info(f"Deleted original file: {dataset['file_path']}")

        if "synthetic_path" in dataset and os.path.exists(dataset["synthetic_path"]):
            os.remove(dataset["synthetic_path"])
            logger.info(f"Deleted synthetic file: {dataset['synthetic_path']}")

    except Exception as e:
        logger.warning(f"Error deleting files for dataset {dataset_id}: {e}")

    # Remove from memory database
    del datasets_db[dataset_id]
    logger.info(f"Dataset {dataset_id} deleted from database")

    return {"message": "Dataset deleted successfully"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error reporting"""
    logger.error(f"Global exception: {exc}")
    logger.error(traceback.format_exc())

    return HTTPException(
        status_code=500,
        detail=f"Internal server error: {str(exc)}"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")