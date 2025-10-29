# MLOps Full Project

This project demonstrates a **complete MLOps workflow**, integrating data ingestion, preprocessing, custom model training, configuration management, automated testing, and coverage reporting.  
It also includes **CI/CD pipelines powered by GitHub Actions**, enabling continuous integration and delivery for reliable and automated model lifecycle management.

---

## Project Overview

The main objective of this repository is to create a **modular, production-ready machine learning system**.  
Each stage of the pipeline — from data loading to deployment — is designed to reflect real-world MLOps principles, emphasizing reproducibility, scalability, and maintainability.

### Key Components
- **Data Loading:** Handles structured and semi-structured input data efficiently.  
- **Data Preprocessing:** Includes cleaning, feature engineering, and transformation steps.  
- **Configuration Management:** Uses YAML/JSON configuration files for flexible environment setup.  
- **Model Training:** Supports multiple algorithms with custom hyperparameter tuning.  
- **Automated Testing:** Implements pytest-based unit tests and continuous integration checks.  
- **Coverage Reporting:** Measures test completeness and code reliability.  
- **CI/CD Integration:** Automatically runs tests, builds, and deployment steps through GitHub Actions.  

---

## Project Structure


```bash
MLOps_Full_Project/
│
├── src/
│ ├── data_preprocessing/
│ ├── model_training/
│ ├── config/
│ └── utils/
│
├── tests/
│ └── test_model_training.py
│
├── requirements.txt
├── setup.py
├── .github/
│ └── workflows/
│ └── ci.yml
└── README.md
```

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/SelahattinNazli/MLOps_Full_Project.git
cd MLOps_Full_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (on Windows)

# Install dependencies
pip install -r requirements.txt

# Running the Project
# Run Model Training
python src/model_training/train_model.py

# Run Tests
pytest --cov=src tests/

# Check Code Coverage
coverage report -m
```

## CI/CD Workflow (GitHub Actions)

**The CI/CD pipeline automatically:**

Runs linting and unit tests on each commit.

Checks code coverage thresholds.

Builds the project artifacts.

Optionally deploys model outputs or reports.

You can find the workflow configuration under:
```bash
.github/workflows/ci.yml
```

## License

This project is licensed under the MIT License.
