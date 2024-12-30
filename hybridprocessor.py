# hybrid_processor.py

from typing import Dict, Any, List, Union
from contentscraper import ContentScraper
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote
import logging
from fastapi.exceptions import RequestValidationError
import traceback
import json
from fastapi.middleware.cors import CORSMiddleware
import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AssetMetadata(BaseModel):
    file_path: str
    file_type: str
    category: str
    tags: List[str]
    size: int
    last_modified: str
    summary: str = None

    class Config:
        json_encoders = {
            Path: str
        }
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "file_path": "path/to/file",
                "file_type": ".png",
                "category": "image",
                "tags": ["tag1", "tag2"],
                "size": 1000,
                "last_modified": "timestamp",
                "summary": None
            }
        }

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "category": self.category,
            "tags": self.tags,
            "size": self.size,
            "last_modified": self.last_modified,
            "summary": self.summary
        }

class HybridProcessor:
    def __init__(self):
        # Setup logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initialize components
        self.app = FastAPI()
        self.text_model = genai.GenerativeModel('gemini-1.5-pro')
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Add template and static file support BEFORE routes
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.templates = Jinja2Templates(directory="templates")
        
        # Add custom Jinja2 filters
        self.templates.env.filters["basename"] = os.path.basename
        
        # Initialize other components
        self.scraper = ContentScraper(initialize_models=False)
        self.asset_index: Dict[str, AssetMetadata] = {}
        self.category_mapping = {
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.pdf': 'document',
            '.doc': 'document',
            '.docx': 'document',
            '.xls': 'spreadsheet',
            '.xlsx': 'spreadsheet',
            '.mp4': 'video',
            '.mp3': 'audio'
        }
        
        # Setup routes
        self.setup_routes()
        self.index_files(".")  # Index files in current directory
        
        # Add middleware
        @self.app.middleware("http")
        async def add_error_handling(request: Request, call_next):
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                self.logger.error(f"Request failed: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "detail": traceback.format_exc()}
                )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/assets/")
        async def list_assets(category: str = None, 
                            file_type: str = None, 
                            tags: str = None):
            tag_list = tags.split(',') if tags else None
            return self.query_assets(category, file_type, tag_list)

        @self.app.get("/assets/{file_path:path}")
        async def get_asset(file_path: str):
            asset = self.get_asset(file_path)
            if not asset:
                raise HTTPException(status_code=404, detail="Asset not found")
            return FileResponse(asset["file_path"])

        @self.app.post("/analyze")
        async def analyze_file(request: Request, file: UploadFile = File(...)):
            try:
                # Log file details
                logger.info(f"Received file: {file.filename}")
                logger.info(f"Content type: {file.content_type}")
                logger.info(f"File size: {len(await file.read())} bytes")
                await file.seek(0)  # Reset file pointer

                # Validate file type
                if not file.filename.endswith(('.xlsx', '.xls')):
                    logger.warning(f"Invalid file type: {file.filename}")
                    return error_response("Only Excel files supported")

                # Process file
                logger.info("Starting file processing")
                result = await self._process_uploaded_file(file)
                logger.info("File processing complete")
                logger.debug(f"Analysis result: {json.dumps(result, indent=2)}")

                return self.templates.TemplateResponse(
                    "analysis.html",
                    {
                        "request": request,
                        "result": result,
                        "filename": file.filename,
                        "debug": True  # Enable debug output
                    }
                )

            except Exception as e:
                logger.exception("Analysis failed")
                return error_response(str(e))

        @self.app.get("/categories")
        async def get_categories():
            return list(set(meta.category for meta in self.asset_index.values()))

        @self.app.get("/tags")
        async def get_tags():
            all_tags = set()
            for meta in self.asset_index.values():
                all_tags.update(meta.tags)
            return list(all_tags)

        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            try:
                assets = {
                    path: metadata 
                    for path, metadata in self.asset_index.items()
                }
                
                return self.templates.TemplateResponse(
                    "index.html", 
                    {
                        "request": request,
                        "asset_index": assets,
                        "basename": os.path.basename
                    }
                )
            except Exception as e:
                print(f"Template Error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/view/{file_path:path}")
        async def view_analysis(request: Request, file_path: str):
            try:
                abs_path = os.path.abspath(file_path)
                result = self.process_asset(abs_path)
                
                return self.templates.TemplateResponse(
                    "analysis.html",
                    {
                        "request": request,
                        "result": result,
                        "filename": os.path.basename(file_path)
                    }
                )
            except Exception as e:
                return self.templates.TemplateResponse(
                    "error.html",
                    {
                        "request": request,
                        "error": str(e)
                    }
                )

        @self.app.get("/debug")
        async def debug(request: Request):
            return JSONResponse({
                "templates_dir": os.path.abspath('templates'),
                "templates_files": os.listdir('templates'),
                "static_dir": os.path.abspath('static'),
                "static_files": os.listdir('static') if os.path.exists('static') else [],
                "asset_index": len(self.asset_index)
            })

        @self.app.get("/debug/last-analysis")
        async def debug_last_analysis(request: Request):
            """View last analysis details"""
            return JSONResponse({
                "last_file": self._last_processed_file,
                "last_result": self._last_analysis_result,
                "logs": self._get_recent_logs()
            })

    def index_files(self, root_dir: str):
        """Index all files in the directory"""
        if not os.path.exists(root_dir):
            print(f"Warning: Directory not found: {root_dir}")
            return
        
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._add_to_index(file_path)

    def _add_to_index(self, file_path: str) -> None:
        """Add a file to the asset index with explicit type checking"""
        try:
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                return

            file_stat = os.stat(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Skip if not a supported file type
            if file_ext not in self.category_mapping:
                return
            
            metadata = AssetMetadata(
                file_path=file_path,
                file_type=file_ext,
                category=self.category_mapping.get(file_ext, 'other'),
                tags=self._generate_tags(file_path),
                size=file_stat.st_size,
                last_modified=str(file_stat.st_mtime)
            )
            
            self.asset_index[file_path] = metadata.to_dict()
        except Exception as e:
            print(f"Error in _add_to_index: {str(e)}")
            # Don't raise, just log and continue
            pass

    def _generate_tags(self, file_path: str) -> List[str]:
        """Generate tags from file path"""
        tags = []
        parent_dir = os.path.basename(os.path.dirname(file_path))
        tags.append(parent_dir)
        
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        tags.extend(file_name.split('_'))
        
        return [tag.lower() for tag in tags if tag]

    def query_assets(self, 
                    category: str = None, 
                    file_type: str = None, 
                    tags: List[str] = None) -> List[Dict]:
        """Query assets based on filters"""
        results = []
        
        for metadata in self.asset_index.values():
            if category and metadata.category != category:
                continue
            if file_type and metadata.file_type != file_type:
                continue
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            results.append(metadata.to_dict())
        
        return results

    def get_asset(self, file_path: str) -> Dict:
        """Get a specific asset by path"""
        metadata = self.asset_index.get(file_path)
        if metadata:
            if isinstance(metadata, AssetMetadata):
                return metadata.to_dict()
            return metadata
        return None

    def process_asset(self, file_path: str) -> Dict[str, Any]:
        """Process an asset with both indexing and content analysis"""
        try:
            self._add_to_index(file_path)
            metadata = self.get_asset(file_path)
            analysis = self.scraper.process_input(file_path)
            
            if isinstance(metadata, AssetMetadata):
                metadata = metadata.to_dict()
            
            return {
                "metadata": metadata,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error processing asset: {str(e)}")
            return {
                "metadata": metadata if metadata else {},
                "analysis": {
                    "type": "error",
                    "error": str(e)
                }
            }

    async def _process_uploaded_file(self, file: UploadFile) -> Dict:
        """Process uploaded file safely"""
        try:
            # Create temp directory
            os.makedirs("temp", exist_ok=True)
            temp_path = f"temp/{file.filename}"
            logger.debug(f"Created temp path: {temp_path}")
            
            # Save file
            content = await file.read()
            logger.debug(f"Read {len(content)} bytes")
            
            with open(temp_path, "wb") as buffer:
                buffer.write(content)
            logger.debug(f"Saved file to {temp_path}")
            
            # Process
            logger.debug("Starting file processing")
            result = self.process_asset(temp_path)
            logger.debug(f"Processing complete: {json.dumps(result, indent=2)}")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug("Cleaned up temp file")
            
            return result
            
        except Exception as e:
            logger.exception("File processing failed")
            raise
