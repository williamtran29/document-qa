## Lightweight REST API for Document-based Q&A powered by FAISS and OpenAI, enabling efficient PDF querying and intelligent answers

Follow these steps to run the project on your local machine using Conda.

## Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine.

## Setup Instructions

1. **Clone the repository**

   Please rename the env.example file to .env, then add your OpenAI key in the appropriate field.

   ## Important: If you want to enable text parsing from images within PDFs, please check the following branch: [PDF OCR Branch](https://github.com/williamtran29/document-qa/tree/pdf-ocr).
   
3. **Create a new Conda environment:**
    ```sh
    conda create --name document-qa python=3.12
    conda activate document-qa
    ```

4. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

5. **Run the project:**
    ```sh
    python main.py
    ```

6. **API Documentation**

    This section provides information on how to access the API documentation.
    To learn how to use the API, open the following URL in your web browser:
    http://localhost:8000/docs


## Additional Notes

- Replace `document-qa` with your preferred environment name.
- Ensure `requirements.txt` is up to date with all necessary dependencies.
- If the app doesn’t automatically create a vectorstore folder, please create it manually
  
That's it! Your project should now be running on your local machine.
