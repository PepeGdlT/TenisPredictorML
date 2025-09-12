"""
Main entry point for the Tennis Prediction API server.
Run this file to start the FastAPI server.
"""
import uvicorn
import logging
from pathlib import Path
import sys

# Agregar el directorio del proyecto al path para imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Tennis Prediction API server...")

    # Configuración del servidor
    config = {
        "app": "backend.api:app",  # Módulo y aplicación
        "host": "0.0.0.0",         # Permitir conexiones externas
        "port": 8000,              # Puerto del servidor
        "reload": True,            # Auto-reload en desarrollo
        "log_level": "info"        # Nivel de logging
    }

    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
