# Machine Failure Prediction using AI4I 2020 Dataset

This project focuses on predictive maintenance by identifying potential machine failures using the AI4I 2020 dataset. It implements a complete machine learning workflow along with a FastAPI backend, Streamlit interface, and React frontend for real-time prediction.

---

## Project Overview

Unexpected machine failures can lead to high operational costs and downtime. This project aims to predict machine failure in advance using operational sensor data and machine learning models.

The project includes:

- Data preprocessing and resampling
- Machine learning model training
- FastAPI backend for predictions
- Interactive web interfaces

---

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.  
The dataset contains machine operational parameters such as air temperature, process temperature, rotational speed, torque, and tool wear, along with failure labels.

A resampled version of the dataset is used to handle class imbalance.

---

## Project Structure

**Machine_failure_ai4i2020_dataset**

- `Dataset/` – contains the processed dataset  
- `api/` – FastAPI backend for inference  
- `myreact/` – React frontend application  
- `artifacts/` – trained model outputs (e.g., LightGBM models, feature importance), usually ignored in version control via `.gitignore`  
- `training scripts/` – model training and preprocessing logic  
- `streamlit_app/` – Streamlit-based user interface  
- `requirements.txt` – project dependencies  
- `.gitignore` – excludes virtual environments, large binaries, and model files  

> **Note:** If you want to include the trained models in GitHub, you can override `.gitignore`, but for large models it is recommended to use external storage (e.g., Google Drive, S3, or GitHub Releases).

---

## Technologies Used

- Python  
- Scikit-learn  
- LightGBM  
- FastAPI  
- Streamlit  
- React (Vite)  
- Git and GitHub  

---

## Machine Learning Pipeline

- Data cleaning and preprocessing  
- Handling class imbalance using resampling techniques  
- Model training using LightGBM and other algorithms  
- Model evaluation and selection  
- Saving trained models for inference  

---

## Application Components

**FastAPI Backend**  
Provides prediction endpoints using FastAPI and serves trained models for inference.

**Streamlit Application**  
Provides a simple interface for users to input machine parameters and view prediction results.

**React Frontend**  
Offers a modern user interface for interacting with the prediction system.

---

## Output

The system predicts whether a machine is likely to fail based on input operational parameters. The prediction can be used to support preventive maintenance decisions.

---

## Best Practices Followed

- Virtual environments are excluded from version control  
- Large model files are ignored  
- Clear project structure and modular code  
- Version control with meaningful commits  

---

## Future Enhancements

- Model explainability and visualization  
- Integration with real-time sensor data  
- Docker-based deployment  
- Cloud deployment and CI/CD  
- User authentication and dashboards  

---

## Author

Adeeba Noor  
GitHub: [https://github.com/adeebanoorr](https://github.com/adeebanoorr)

---

## License

This project is intended for educational and research purposes.
