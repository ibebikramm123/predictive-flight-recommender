# Predictive Flight Recommendation System

A professional AI-powered flight recommendation system that predicts flight prices using XGBoost gradient boosting and ranks flights based on user preferences.

## AI Features

- **XGBoost Gradient Boosting**: Advanced machine learning algorithm for accurate price prediction (MAE: 85.97, RMSE: 133.87)
- **Feature Engineering**: Processes date, duration, stops, airline, and other factors
- **Smart Recommendations**: Balances price, duration, stops, and user airline preferences using weighted scoring
- **Real-time Predictions**: Computes predicted prices for future dates based on historical patterns

## Setup

1. Install dependencies:
   ```bash
   pip install flask polars pandas scikit-learn xgboost shap joblib
   ```

2. Process data:
   ```bash
   python src/make_features.py
   ```

3. Train model:
   ```bash
   python src/train.py
   ```

4. Run server:
   ```bash
   python backend/server.py
   ```

5. Open http://127.0.0.1:5001

## Deployment

1. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app locally:
   ```bash
   python backend/server.py
   ```

3. Deploy to a hosting service such as Render, Heroku, or Railway.

Render example:
1. Go to https://dashboard.render.com and connect your GitHub repository.
2. Create a new Web Service.
3. Select the `main` branch.
4. Use the default build command and start command from `render.yaml`:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn backend.server:app --log-file -`

Heroku example:
```bash
git init
heroku create
git add .
git commit -m "Deploy flight recommender"
git push heroku main
```

The repository includes a `Procfile`, `runtime.txt`, and `render.yaml` for deployment.

## Real-Life Deployment

To make this system production-ready:

### 1. Real Flight Data Integration
- Replace static dataset with live APIs:
  - **Amadeus API**: For flight search and pricing
  - **Sabre API**: Travel industry API
  - **Google Flights API** (unofficial, use responsibly)
- Update `filter_candidates` and `recommend` to fetch live data

### 2. Real Payment Processing
- Integrate Stripe for secure payments:
  ```python
  import stripe
  stripe.api_key = 'your_stripe_secret_key'

  # In confirmBooking equivalent
  charge = stripe.Charge.create(
      amount=int(total * 100),  # cents
      currency='usd',
      source=token,  # from frontend
      description='Flight booking'
  )
  ```
- Get Stripe publishable key for frontend

### 3. Database
- Use PostgreSQL or MongoDB for user data, bookings
- Store real flight data if caching

### 4. Authentication
- Add user login with Flask-Login or JWT
- Secure API endpoints

### 5. Deployment
- Deploy to Heroku, AWS, or DigitalOcean
- Use Gunicorn for production server
- Add HTTPS with SSL certificate

### 6. Security
- Validate all inputs
- Use environment variables for secrets
- Implement rate limiting

### 7. Scalability
- Cache predictions with Redis
- Use async processing for heavy computations

For a full production system, consider partnering with travel agencies or using white-label solutions.