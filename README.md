# Ship Management System

## Table of Contents
- [Architecture](#architecture)
- [Routes](#routes)
- [Setup & Installation](#setup--installation)
- [API Documentation](#api-documentation)
- [AI Models](#ai-models)
- [Deployment Steps](#deployment-steps)

---

## Architecture

The Ship Management System is a modular Node.js application built with the following components:
- **Backend Framework**: Express.js for routing and middleware.
- **Database**: MongoDB for storing ship, route, maintenance, and fuel data.
- **Caching**: Redis for caching frequently accessed data.
- **Scheduler**: Node-Schedule for periodic tasks like model training.
- **AI Models**: TensorFlow.js for predictive analytics and route optimization.
- **Logger**: Winston-based custom logger for structured logging.

### Folder Structure
```
Ship Management System/
├── controllers/       # Business logic for routes
├── models/            # Mongoose schemas for database
├── routes/            # API route definitions
├── services/          # External services (e.g., weather API)
├── utils/             # Utility functions (e.g., logger)
├── config/            # Configuration files (e.g., database, Redis)
├── public/            # Static files (if applicable)
├── server.js          # Entry point of the application
└── README.md          # Documentation
```

---

## Routes

The API is structured under `/api/v1` and includes the following endpoints:

| Route                  | Method | Description                          |
|------------------------|--------|--------------------------------------|
| `/api/v1/routes`       | GET    | Fetch all routes                    |
| `/api/v1/fuel`         | GET    | Fetch fuel logs                     |
| `/api/v1/maintenance`  | GET    | Fetch maintenance records           |
| `/api/v1/analytics`    | GET    | Fetch analytics data                |
| `/api/v1/admin`        | POST   | Admin-specific operations           |

---

## Setup & Installation

### Prerequisites
- Node.js (v16+)
- MongoDB
- Redis
- `.env` file with the following variables:
  ```
  MONGO_URI=<your-mongodb-uri>
  OPENWEATHERMAP_API_KEY=<your-weather-api-key>
  PORT=5001
  HISTORICAL_DATA_ENABLED=true
  ```

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Ship-Management-System
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the application:
   ```bash
   npm run dev
   ```

---

## API Documentation

### 1. Fetch Analytics Data
**Endpoint**: `GET /api/v1/analytics`

**Request**:
```bash
curl -X GET http://localhost:5001/api/v1/analytics
```

**Response**:
```json
{
  "overallStats": { "totalFuel": 5000, "avgFuelPerNM": 2.5 },
  "statsByShipType": [ { "shipType": "Cargo", "totalFuel": 3000 } ],
  "monthlyTrends": [ { "year": 2023, "month": 9, "totalFuel": 1000 } ],
  "mostEfficientRoutes": [ { "shipName": "Voyager", "fuelPerNM": 1.2 } ],
  "leastEfficientRoutes": [ { "shipName": "Titan", "fuelPerNM": 3.5 } ],
  "generatedAt": "2023-10-01T12:00:00Z"
}
```

---

## AI Models

### 1. Predictive Analytics
- **Model**: Universal Sentence Encoder (TensorFlow.js)
- **Purpose**: Predict route efficiency based on historical data.
- **Training**: Data is aggregated from MongoDB and processed using TensorFlow.js.

### 2. Deployment
- **Environment**: Node.js with TensorFlow.js.
- **Steps**:
  1. Install TensorFlow.js:
     ```bash
     npm install @tensorflow/tfjs @tensorflow/tfjs-node
     ```
  2. Train the model:
     - Use the `predictionService.js` to train and save the model.
  3. Load the model in the application for inference.

---

## Deployment Steps

1. **Prepare Environment**:
   - Ensure MongoDB and Redis are running.
   - Set up `.env` file with required variables.

2. **Build & Start**:
   ```bash
   npm install
   npm start
   ```

3. **Monitor Logs**:
   - Use the custom logger to monitor application health.

4. **Deploy to Cloud**:
   - Use platforms like AWS, Azure, or Heroku for deployment.
   - Configure environment variables in the cloud environment.

---

## Contributors
- **Author**: Vaibhav Gosavi