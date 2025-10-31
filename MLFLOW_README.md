# MLflow Integration - Quick Reference

## 🎯 What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle, including:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage models
- **Model Serving**: Deploy models to production
- **Project Management**: Package and reproduce experiments

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start MLflow Server
python start_mlflow.py

# 3. Open Browser
# Navigate to: http://localhost:5000
```

## 📊 What's Being Tracked?

### Training Pipeline
- ✅ All model hyperparameters
- ✅ Training and test metrics (MAE, MSE, R2)
- ✅ Model artifacts (saved models)
- ✅ Data preprocessing steps
- ✅ Model comparison results

### API Endpoints
- ✅ Optimization requests and results
- ✅ Health check predictions
- ✅ Response times
- ✅ Success/failure rates

## 📁 Files Added

```
BMS-AI/
├── start_mlflow.py                          # Start MLflow server
├── MLFLOW_GUIDE.md                          # Comprehensive guide
├── MLFLOW_QUICKSTART.md                     # Quick start commands
├── MLFLOW_INTEGRATION_SUMMARY.md            # Integration summary
├── MLFLOW_ARCHITECTURE.txt                  # Architecture diagram
├── requirements.txt                         # Updated with MLflow
└── src/bms_ai/
    ├── components/
    │   └── model_trainer.py                 # ✨ MLflow integrated
    ├── api/routers/
    │   ├── optimize.py                      # ✨ MLflow tracking
    │   └── heatlh_check.py                  # ✨ MLflow tracking
    └── utils/
        ├── mlflow_config.py                 # 🆕 MLflow configuration
        └── mlflow_tracking.py               # 🆕 Tracking decorators
```

## 🎓 Usage Examples

### View All Training Runs
```python
import mlflow
from src.bms_ai.utils.mlflow_config import MLflowConfig

MLflowConfig.setup_mlflow()

runs = mlflow.search_runs(
    experiment_names=["BMS-AI-Training"],
    order_by=["metrics.test_r2_score DESC"]
)

print(runs[['metrics.test_r2_score', 'params.best_model']].head())
```

### Load Best Model
```python
import mlflow.sklearn

# Get best run ID from UI or search_runs()
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Or load from Model Registry
model = mlflow.sklearn.load_model("models:/BMS-Fan-Power-Model/Production")
```

### Track Custom Metrics
```python
from src.bms_ai.utils.mlflow_tracking import MLflowContextManager

with MLflowContextManager("BMS-AI-Training", "my_experiment") as mlflow_ctx:
    mlflow_ctx.log_params({"learning_rate": 0.01})
    mlflow_ctx.log_metrics({"accuracy": 0.95})
    mlflow_ctx.log_tags({"version": "1.0"})
```

## 🔍 MLflow UI Features

### Experiments View
- Compare multiple runs side-by-side
- Filter and search runs
- Visualize metrics over time

### Model Registry
- Register and version models
- Transition models between stages (Staging → Production)
- Track model lineage

### Artifacts
- View saved models
- Download preprocessors
- Access training logs

## ⚙️ Configuration

### Change Tracking URI
Edit `src/bms_ai/utils/mlflow_config.py`:

```python
# Local (default)
TRACKING_URI = "file:./mlruns"

# Remote server
TRACKING_URI = "http://your-mlflow-server.com:5000"
```

### Customize Experiment Names
```python
class MLflowConfig:
    TRAINING_EXPERIMENT_NAME = "My-Custom-Training"
    OPTIMIZATION_EXPERIMENT_NAME = "My-Optimization"
```

### Enable/Disable Auto-logging
```python
class MLflowConfig:
    AUTOLOG_SKLEARN = True   # Auto-log sklearn models
    AUTOLOG_XGBOOST = True   # Auto-log XGBoost models
    LOG_MODELS = True        # Log model artifacts
```

## 📈 Benefits

| Feature | Benefit |
|---------|---------|
| **Experiment Tracking** | Never lose model results |
| **Model Versioning** | Track all model versions |
| **Reproducibility** | Reproduce any experiment |
| **Collaboration** | Share experiments with team |
| **Performance Monitoring** | Track API performance |
| **Model Registry** | Manage production models |

## 🎯 Common Tasks

### Compare Models
1. Train multiple models
2. Open MLflow UI
3. Navigate to "BMS-AI-Training"
4. Select runs to compare
5. Click "Compare" button

### Register Best Model
1. Find best run in UI
2. Click on run
3. Go to "Artifacts" → "model"
4. Click "Register Model"
5. Enter model name
6. Click "Register"

### Deploy Model to Production
1. Go to "Models" tab
2. Select your model
3. Choose version
4. Click "Transition to" → "Production"

## 📚 Documentation

- **Complete Guide**: [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)
- **Quick Start**: [MLFLOW_QUICKSTART.md](MLFLOW_QUICKSTART.md)
- **Integration Details**: [MLFLOW_INTEGRATION_SUMMARY.md](MLFLOW_INTEGRATION_SUMMARY.md)
- **Architecture**: [MLFLOW_ARCHITECTURE.txt](MLFLOW_ARCHITECTURE.txt)

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
netstat -ano | findstr :5000  # Windows
lsof -i :5000                 # Linux/Mac

# Use different port
python start_mlflow.py --port 5001
```

### Can't See Experiments
```bash
# Verify tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Check mlruns directory exists
ls mlruns/
```

### MLflow Not Installed
```bash
pip install --upgrade mlflow
```

## ✅ Next Steps

1. ✅ Start MLflow server: `python start_mlflow.py`
2. ✅ Run training: `python -m src.bms_ai.pipelines.training_pipeline`
3. ✅ Open UI: http://localhost:5000
4. ✅ Explore experiments
5. ✅ Register best model
6. ✅ Make API calls (auto-tracked)

---

## 🎓 Learning Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)

---

**Happy Tracking! 🚀**

For detailed information, see:
- 📖 [Complete MLflow Guide](MLFLOW_GUIDE.md)
- ⚡ [Quick Start Commands](MLFLOW_QUICKSTART.md)
- 📋 [Integration Summary](MLFLOW_INTEGRATION_SUMMARY.md)
