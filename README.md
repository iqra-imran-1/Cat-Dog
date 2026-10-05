# Cat vs Dog Classifier

A CNN that classifies an image as a cat or a dog, with a FastAPI page to upload images.

## Files

- `cnn_catdog.py`: Colab script that trains the CNN, saves it and evaluates it on the test set
- `catdof_fast.py`: FastAPI app, loads the model and serves the upload page and `/predict`
- `catdog.keras`: the trained model
- `requirements.txt`: Python packages needed to run the app
- `.gitignore`: files git should ignore

## How to run

1. Clone the repo and move into it
   ```
   git clone https://github.com/<your-username>/catdog-classifier.git
   cd catdog-classifier
   ```
2. Create and activate an environment
   ```
   conda create -n catdog python=3.11 -y
   conda activate catdog
   ```
3. Install the packages
   ```
   pip install -r requirements.txt
   ```
4. Start the app from inside the repo folder
   ```
   uvicorn catdof_fast:app --reload
   ```
5. Open http://127.0.0.1:8000, upload an image and click Classify

## Notes

- Greyscale and RGBA images are converted to RGB before prediction.
- A score between 0.3 and 0.7 is returned as "uncertain".
- To retrain, run `cnn_catdog.py` in Google Colab with the dataset zip in Drive.
