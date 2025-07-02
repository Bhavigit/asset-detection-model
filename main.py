# Import necessary libraries and modules
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from model import detect_objects  # Make sure model.py is in the same folder and contains the detect_objects function


# --- FastAPI Application Setup ---
# Create an instance of the FastAPI application.
app = FastAPI()

# --- File Upload Configuration ---
# Define the directory where uploaded files will be temporarily stored.
UPLOAD_DIR = "uploads"
# Create the directory if it doesn't already exist.
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- API Endpoint Definition ---
# Define a POST endpoint at the "/predict" path.
# This endpoint accepts a file upload.
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    This endpoint accepts an image file, saves it, runs object detection
    and color extraction on it, and returns the results.

    Args:
        file (UploadFile): The uploaded image file from the request.

    Returns:
        JSONResponse: A JSON object containing the detection results
                      or an error message.
    """
    try:
        # --- File Handling ---
        # Create a full path for the uploaded file in the uploads directory.
        file_location = os.path.join(UPLOAD_DIR, file.filename)

        # Save the uploaded image to the disk.
        # This is a standard way to handle file uploads in FastAPI by
        # reading chunks from the uploaded file and writing them to a local file.
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- Model Inference ---
        # Run the object detection and color extraction logic from the 'model.py' file.
        # This function takes the file path and returns the predictions.
        predictions = detect_objects(file_location)

        # --- Format the Results ---
        # Format the detection results into the requested JSON structure.
        formatted = []
        for pred in predictions:
            formatted.append({
                # Map the detected class name to the 'o/p' key.
                "o/p": pred["class"],
                # Map the detected primary color to the 'P' key.
                "P": pred["primary"],
                # Join the list of secondary colors into a comma-separated string for the 'S' key.
                "S": ", ".join(pred["secondary"])
            })

        # --- Cleanup ---
        # Optionally, you can delete the temporary file after processing to save disk space.
        # Uncomment the line below if you want to enable file deletion.
        # os.remove(file_location)

        # Return the formatted results as a JSON response.
        return {"results": formatted}

    except Exception as e:
        # --- Error Handling ---
        # If any error occurs during the process, return a 500 status code
        # and a JSON response with the error message.
        return JSONResponse(status_code=500, content={"error": str(e)})