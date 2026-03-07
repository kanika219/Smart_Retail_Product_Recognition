# RetailVision AI

RetailVision AI is a production-style computer vision platform for grocery stores. It helps store managers and operations teams to monitor product categories, detect shelf inventory patterns, and support inventory decisions.

## Features
- **Overview**: Executive dashboard with high-level KPIs.
- **Product Scanner**: Core feature for identifying products from images using AI.
- **Operations Analytics**: Detailed analytics for retail decision-making.
- **Model Intelligence**: Information on AI model performance and reliability.
- **Inventory Insights**: Business recommendations based on AI predictions.
- **Reports**: Exportable prediction logs and performance summaries.

## Getting Started

### Prerequisites
- Python 3.8+

### Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
To run the Streamlit application:
```bash
streamlit run app.py
```

## AI Model
The application uses a CNN classifier trained on the Grocery Store Dataset. The model is saved as `model/grocery_model.h5` and labels are encoded in `model/label_encoder.pkl`.

## Project Structure
```
retailvision-ai/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   ├── grocery_model.h5
│   └── label_encoder.pkl
├── pages/
│   ├── overview.py
│   ├── scanner.py
│   ├── analytics.py
│   ├── model_intelligence.py
│   ├── insights.py
│   └── reports.py
├── utils/
│   ├── image_processing.py
│   ├── prediction.py
│   └── metrics_loader.py
└── assets/
    └── logo.png
```
