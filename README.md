# OpenSource-ML-HousingAPI

```
boston-housing-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_baseline.py
│   ├── train_advanced.py
│   ├── model_comparator.py
│   └── utils.py
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── streamlit_app.py
├── monitoring/
│   ├── __init__.py
│   ├── drift_monitor.py
│   └── setup_monitoring.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_models.py
├── models/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── monitoring_data/
│   └── .gitkeep
├── .env.example
├── .gitignore
├── .dvcignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.sh
├── pytest.ini
└── README.md
```