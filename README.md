## Lightweight REST API for Document-based Q&A powered by FAISS and OpenAI, enabling efficient PDF querying and intelligent answers

Follow these steps to run the project on your local machine using Conda.

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine.

## Setup Instructions

1. **Clone the repository**

   Please rename the env.example file to .env, then add your OpenAI key in the appropriate field.
   

   **Important: How to set TESSDATA_PREFIX:**

   To set the TESSDATA_PREFIX, locate the path to the tessdata folder. 
   
   For example:

   `TESSDATA_PREFIX=/opt/homebrew/Cellar/tesseract/5.4.1_1/share/tessdata`
      
    ```sh
    brew install tesseract
    brew info tesseract
    ```

    Look for the `tessdata` folder path in the information provided above.

2. **Create a new Conda environment:**
    ```sh
    conda create --name document-qa python=3.12
    conda activate document-qa
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the project:**
    ```sh
    python main.py
    ```

5. **API Documentation**

    This section provides information on how to access the API documentation.
    To learn how to use the API, open the following URL in your web browser:
    http://localhost:8000/docs


## Additional Notes

- Replace `document-qa` with your preferred environment name.
- Ensure `requirements.txt` is up to date with all necessary dependencies.
- If the app doesn’t automatically create a vectorstore folder, please create it manually
  
That's it! Your project should now be running on your local machine.
