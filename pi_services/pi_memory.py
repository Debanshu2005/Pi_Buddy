"""
Minimal Memory for Pi - ONLY Face Database Access
"""

import psycopg2
import numpy as np
import os
import time
from typing import Optional, Tuple

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("DEBUG: Loaded .env file")
except ImportError:
    print("DEBUG: python-dotenv not installed, using system env vars")

def get_db_connection():
    """Get database connection to Neon DB"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'), 
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT', '5432'),
        sslmode='require'  # Neon requires SSL
    )

def get_face(name: str) -> Optional[np.ndarray]:
    """Get face embedding from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT embedding FROM faces WHERE name = %s", (name,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            return np.array(result[0])
        return None
    except:
        return None

def save_face(name: str, embedding: np.ndarray, confidence: float = 0.95) -> bool:
    """Save face embedding to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure embedding is 1D and convert to list
        embedding_flat = embedding.flatten().tolist()
        
        # Convert to JSON string to prevent corruption
        import json
        embedding_json = json.dumps(embedding_flat)
        
        print(f"DEBUG: Saving {name} with embedding length: {len(embedding_flat)}")
        
        cursor.execute(
            "INSERT INTO faces (name, embedding) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET embedding = %s",
            (name, embedding_json, embedding_json)
        )
        
        conn.commit()
        rows_affected = cursor.rowcount
        print(f"DEBUG: Database rows affected: {rows_affected}")
        
        cursor.close()
        conn.close()
        
        print(f"SUCCESS: Face saved to database for {name}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save face for {name}: {e}")
        return False

def get_all_faces() -> dict:
    """Get all faces from database with fallback"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, embedding FROM faces")
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print(f"DEBUG: Found {len(results)} face records in database")
        
        faces = {}
        for name, embedding in results:
            try:
                import json
                
                print(f"DEBUG: Processing {name}, embedding type: {type(embedding)}")
                
                # Handle JSON string (new format)
                if isinstance(embedding, str):
                    try:
                        embedding = json.loads(embedding)
                        print(f"DEBUG: {name} - JSON parsed successfully, length: {len(embedding)}")
                    except json.JSONDecodeError:
                        print(f"DEBUG: {name} - JSON parse failed, trying ast.literal_eval")
                        # Fallback to ast.literal_eval for old format
                        import ast
                        embedding = ast.literal_eval(embedding)
                elif isinstance(embedding, list):
                    print(f"DEBUG: {name} - Already a list, length: {len(embedding)}")
                elif isinstance(embedding, (set, tuple)):
                    # Skip corrupted data
                    print(f"DEBUG: Skipping corrupted embedding for {name} ({type(embedding).__name__} type)")
                    continue
                else:
                    print(f"DEBUG: Unknown embedding type for {name}: {type(embedding)}")
                    continue
                
                # Convert to numpy array
                embedding_array = np.array(embedding, dtype=np.float32)
                
                # Ensure it's 1D
                embedding_array = embedding_array.flatten()
                
                # Skip if embedding is empty or invalid
                if embedding_array.size == 0:
                    print(f"DEBUG: Skipping {name} - empty embedding")
                    continue
                    
                faces[name] = embedding_array
                print(f"DEBUG: Successfully loaded {name} with {embedding_array.shape[0]} dimensions")
                
            except Exception as e:
                print(f"DEBUG: Error processing {name}: {e}")
                print(f"DEBUG: Embedding data preview: {str(embedding)[:100]}...")
                continue
        
        print(f"DEBUG: Successfully loaded {len(faces)} faces")
        return faces
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("INFO: Running without face recognition for now.")
        return {}