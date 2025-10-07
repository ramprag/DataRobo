import os
import uuid
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import time
import logging
import traceback
import io
import zipfile
import json
from pathlib import Path

# Import our custom classes
from privacy_masker import PrivacyMasker
from synthetic_generator import SyntheticDataGenerator
from quality_validator import QualityValidator
from data_processor import DataProcessor
from multi_table_processor import EnhancedSyntheticDataGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("synthetic", exist_ok=True)
os.makedirs("logs", exist_ok=True)

app = FastAPI(title="Synthetic Data Generator API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    table_count: Optional[int] = 1
    created_at: str
    file_size: Optional[int] = None
    error_message: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    privacy_config: Optional[Dict[str, Any]] = None
    generation_method: Optional[str] = None

class PrivacyConfig(BaseModel):
    mask_emails: bool = True
    mask_names: bool = True
    mask_phone_numbers: bool = True
    mask_addresses: bool = True
    mask_ssn: bool = True
    custom_fields: List[str] = []
    anonymization_method: str = "faker"
    use_gan: bool = True
    gan_model: str = "ctgan"

class GenerateSyntheticRequest(BaseModel):
    dataset_id: str
    privacy_config: PrivacyConfig
    num_rows: Optional[int] = None

@app.get("/")
async def root():
    return {
        "message": "Synthetic Data Generator API with GAN Support",
        "version": "2.0.0",
        "features": ["GAN Models (CTGAN, TVAE)", "Statistical Fallback", "Multi-Table Support"]
    }

@app.get("/health")
async def health_check():
    # Check if GAN libraries are available
    try:
        from sdv.single_table import CTGANSynthesizer
        gan_available = True
    except ImportError:
        gan_available = False

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gan_support": gan_available,
        "methods_available": ["statistical", "ctgan", "tvae"] if gan_available else ["statistical"]
    }

@app.get("/api/methods")
async def get_available_methods():
    """Get list of available generation methods"""
    generator = SyntheticDataGenerator()
    return {
        "methods": generator.get_available_methods(),
        "default": "ctgan" if generator.gan_available else "statistical",
        "gan_available": generator.gan_available
    }

@app.post("/api/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset endpoint - supports CSV, Excel, and ZIP files"""
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

        # For ZIP files, analyze content with error handling
        if file.filename.lower().endswith('.zip'):
            try:
                enhanced_generator = EnhancedSyntheticDataGenerator()
                tables = enhanced_generator._extract_tables(content, file.filename)

                if not tables or len(tables) == 0:
                    raise ValueError("ZIP file contains no valid CSV or Excel files")

                total_rows = sum(len(df) for df in tables.values())
                total_columns = sum(len(df.columns) for df in tables.values())
                table_count = len(tables)

                logger.info(f"ZIP contains {table_count} tables with {total_rows} total rows")
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid ZIP file format")
            except Exception as e:
                logger.error(f"Error processing ZIP file: {e}")
                raise HTTPException(status_code=400, detail=f"Error processing ZIP file: {str(e)}")
        else:
            # Single file analysis with error handling
            try:
                data_processor = DataProcessor()

                if file.filename.lower().endswith('.csv'):
                    df = data_processor.load_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

                if df is None or df.empty:
                    raise ValueError("File contains no data")

                total_rows = len(df)
                total_columns = len(df.columns)
                table_count = 1

            except pd.errors.EmptyDataError:
                raise HTTPException(status_code=400, detail="File is empty or contains no data")
            except pd.errors.ParserError as e:
                raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

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
            "generation_method": None
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

async def generate_synthetic_data_background_enhanced(
    dataset_id: str,
    privacy_config: PrivacyConfig,
    num_rows: Optional[int] = None
):
    """Enhanced background task with GAN support for multi-table datasets"""
    try:
        logger.info(f"🚀 Starting enhanced synthetic data generation for dataset: {dataset_id}")

        if dataset_id not in datasets_db:
            logger.error(f"Dataset {dataset_id} not found")
            return

        dataset = datasets_db[dataset_id]
        datasets_db[dataset_id]["status"] = "processing"

        enhanced_generator = EnhancedSyntheticDataGenerator()

        with open(dataset["file_path"], "rb") as f:
            file_data = f.read()

        filename = dataset["filename"]

        # Pass GAN configuration to the processor with error handling
        try:
            result = enhanced_generator.process_upload(
                file_data,
                filename,
                privacy_config,
                use_gan=privacy_config.use_gan,
                gan_model=privacy_config.gan_model
            )
        except Exception as e:
            logger.error(f"Error in synthetic generation: {e}")
            logger.error(traceback.format_exc())
            datasets_db[dataset_id]["status"] = "failed"
            datasets_db[dataset_id]["error_message"] = f"Generation failed: {str(e)}"
            return

        synthetic_paths = {}
        for table_name, synthetic_df in result['tables'].items():
            synthetic_path = f"synthetic/{dataset_id}_{table_name}_synthetic.csv"
            synthetic_df.to_csv(synthetic_path, index=False)
            synthetic_paths[table_name] = synthetic_path

        datasets_db[dataset_id].update({
            "status": "completed",
            "synthetic_path": list(synthetic_paths.values())[0] if len(synthetic_paths) == 1 else None,
            "synthetic_paths": synthetic_paths,
            "quality_metrics": result['quality_metrics'],
            "privacy_config": privacy_config.dict(),
            "table_count": result['table_count'],
            "relationships": result.get('relationships', {}),
            "relationship_summary": result.get('relationship_summary', {}),
            "generation_method": result.get('generation_method', 'unknown')
        })

        logger.info(f"✅ Dataset {dataset_id} generation completed successfully using {result.get('generation_method', 'unknown')} method")

    except Exception as e:
        logger.error(f"❌ Error in enhanced generation: {e}")
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
    """Generate synthetic data - now supports GAN models and multi-table"""
    try:
        logger.info(f"Generate synthetic requested for: {dataset_id}")

        if dataset_id not in datasets_db:
            raise HTTPException(status_code=404, detail="Dataset not found")

        dataset = datasets_db[dataset_id]

        if dataset["status"] == "processing":
            raise HTTPException(status_code=400, detail="Already processing")

        try:
            body = await request.json()
            privacy_config_data = body.get("privacy_config", {})
            num_rows = body.get("num_rows")

            # Set defaults for GAN if not specified
            if "use_gan" not in privacy_config_data:
                privacy_config_data["use_gan"] = True
            if "gan_model" not in privacy_config_data:
                privacy_config_data["gan_model"] = "ctgan"

            privacy_config = PrivacyConfig(**privacy_config_data)

            logger.info(f"🔧 Generation config: use_gan={privacy_config.use_gan}, model={privacy_config.gan_model}")

        except Exception as e:
            logger.error(f"Error parsing request: {e}")
            privacy_config = PrivacyConfig()
            num_rows = None

        background_tasks.add_task(
            generate_synthetic_data_background_enhanced,
            dataset_id,
            privacy_config,
            num_rows
        )

        logger.info(f"Background task started for: {dataset_id}")
        return {
            "message": "Synthetic data generation started",
            "dataset_id": dataset_id,
            "method": "gan" if privacy_config.use_gan else "statistical",
            "model": privacy_config.gan_model if privacy_config.use_gan else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting generation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to start: {str(e)}")

@app.get("/api/datasets/{dataset_id}/download")
async def download_synthetic_data(dataset_id: str):
    """Download single table synthetic data"""
    logger.info(f"Download requested for: {dataset_id}")

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets_db[dataset_id]

    if dataset["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Data not ready. Status: {dataset['status']}")

    if "synthetic_path" not in dataset or not os.path.exists(dataset["synthetic_path"]):
        raise HTTPException(status_code=404, detail="Synthetic data file not found")

    return FileResponse(
        dataset["synthetic_path"],
        media_type="text/csv",
        filename=f"{dataset['filename']}_synthetic.csv"
    )

@app.get("/api/datasets/{dataset_id}/download-zip")
async def download_synthetic_data_zip(dataset_id: str):
    """Download all synthetic tables as ZIP"""
    logger.info(f"ZIP download requested for: {dataset_id}")

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets_db[dataset_id]

    if dataset["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Data not ready. Status: {dataset['status']}")

    zip_buffer = io.BytesIO()

    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add single synthetic file if exists
            if "synthetic_path" in dataset and dataset["synthetic_path"]:
                if os.path.exists(dataset["synthetic_path"]):
                    zip_file.write(dataset["synthetic_path"], f"{dataset['filename']}_synthetic.csv")

            # Add multiple synthetic files if exists
            if "synthetic_paths" in dataset:
                for table_name, path in dataset["synthetic_paths"].items():
                    if os.path.exists(path):
                        zip_file.write(path, f"{table_name}_synthetic.csv")

            # Add relationship summary if exists
            if "relationship_summary" in dataset and dataset["relationship_summary"].get("total_relationships", 0) > 0:
                relationship_json = json.dumps(dataset["relationship_summary"], indent=2)
                zip_file.writestr("relationship_summary.json", relationship_json)

            # Add generation metadata
            metadata = {
                "dataset_id": dataset_id,
                "filename": dataset["filename"],
                "generation_method": dataset.get("generation_method", "unknown"),
                "generated_at": datetime.utcnow().isoformat(),
                "table_count": dataset.get("table_count", 1),
                "total_rows": dataset.get("row_count", 0),
                "privacy_config": dataset.get("privacy_config", {})
            }
            zip_file.writestr("generation_metadata.json", json.dumps(metadata, indent=2))

        zip_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={dataset['filename']}_synthetic.zip"}
        )

    except Exception as e:
        logger.error(f"Error creating ZIP file: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")


@app.get("/api/datasets/{dataset_id}/relationships")
async def get_dataset_relationships(dataset_id: str):
    """Get relationship information"""
    logger.info(f"Relationships requested for: {dataset_id}")

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets_db[dataset_id]

    return {
        "dataset_id": dataset_id,
        "table_count": dataset.get("table_count", 1),
        "relationships": dataset.get("relationships", {}),
        "relationship_summary": dataset.get("relationship_summary", {
            "total_relationships": 0,
            "tables_with_primary_keys": 0,
            "tables_with_foreign_keys": 0,
            "generation_order": [Path(dataset["filename"]).stem],
            "relationship_details": []
        })
    }

@app.get("/api/datasets/{dataset_id}/preview")
async def preview_data(dataset_id: str, synthetic: bool = False):
    logger.info(f"Preview requested for dataset: {dataset_id}, synthetic: {synthetic}")

    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets_db[dataset_id]

    try:
        if synthetic:
            path_to_preview = dataset.get("synthetic_path")
            if not path_to_preview and dataset.get("synthetic_paths"):
                path_to_preview = next(iter(dataset["synthetic_paths"].values()), None)

            if not path_to_preview or not os.path.exists(path_to_preview):
                raise HTTPException(status_code=404, detail="Synthetic data not found")
            df = pd.read_csv(path_to_preview)
        else:
            if not os.path.exists(dataset["file_path"]):
                raise HTTPException(status_code=404, detail="Original data file not found")

            # Check if it's a ZIP file (multi-table)
            if dataset["file_path"].lower().endswith('.zip'):
                # For ZIP files, extract and preview the first table
                try:
                    enhanced_generator = EnhancedSyntheticDataGenerator()
                    with open(dataset["file_path"], "rb") as f:
                        file_data = f.read()

                    tables = enhanced_generator._extract_tables(file_data, dataset["filename"])

                    if not tables or len(tables) == 0:
                        raise HTTPException(status_code=404, detail="No valid tables found in ZIP file")

                    # Get the first table for preview
                    first_table_name = list(tables.keys())[0]
                    df = tables[first_table_name]

                    logger.info(f"Previewing first table '{first_table_name}' from ZIP file")

                except Exception as e:
                    logger.error(f"Error extracting ZIP for preview: {e}")
                    raise HTTPException(status_code=500, detail=f"Error reading ZIP file: {str(e)}")

            elif dataset["file_path"].lower().endswith('.csv'):
                df = pd.read_csv(dataset["file_path"])
            elif dataset["file_path"].lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(dataset["file_path"])
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format for preview")

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

    try:
        if os.path.exists(dataset["file_path"]):
            os.remove(dataset["file_path"])
            logger.info(f"Deleted original file: {dataset['file_path']}")

        if "synthetic_path" in dataset and os.path.exists(dataset["synthetic_path"]):
            os.remove(dataset["synthetic_path"])
            logger.info(f"Deleted synthetic file: {dataset['synthetic_path']}")

        if "synthetic_paths" in dataset:
            for path in dataset["synthetic_paths"].values():
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Deleted synthetic file: {path}")

    except Exception as e:
        logger.warning(f"Error deleting files for dataset {dataset_id}: {e}")

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