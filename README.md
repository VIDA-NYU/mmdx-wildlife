# HILTS: Wildlife Data Labeling & Search Platform

HILTS is a web-based platform for interactive labeling, searching, and managing wildlife datasets. It combines a Svelte/TypeScript frontend with a Python backend and supports scalable deployment via Docker and Kubernetes. 

The system is described in 
"HILTS: Human-LLM Collaboration for Effective Data Labeling"
To appear in Information Systems Journal, 2026.


## Environment Variables

Set these in your deployment on Kubernets or Locally:

- `ACCESS_KEY`: The access key for your S3 or MinIO storage bucket.
- `SECRET_KEY`: The secret key for your S3 or MinIO storage bucket.
- `ENDPOINT_URL`: The URL endpoint for your S3 or MinIO service where your images are stored.
- `ENV`: The environment mode for the backend (`prod` for production, `dev` for development).
- `DATA_SOURCE`: Source of your data, e.g., `S3` for cloud storage or `local` for local files.
- `DATA_PATH`: Path to your data directory (bucket name).
- `DB_DELETE_EXISTING`: If set to `true`, deletes the existing database on startup.
- `DB_PATH`: Filesystem path where the database is stored inside the container/Local file.
- `CSV_FILENAME`: Name of the CSV file to use for labeling if CSV is on S3/Minio.
- `CSV_PATH`: Path to the CSV file for local development.

**Tip:**
For local development, add these variables to a `.env` file in the `hilts/` directory. For Kubernetes, set them in your deployment manifest

## Features

- **Run locally with minimal setup:**
    1. Install prerequisites: Python 3.9+, Node.js 18+, Poetry, and optionally Docker.
    2. Set up the backend:
         ```bash
         cd hilts
         poetry install
         poetry run python server.py
         ```
    3. Set up the frontend:
         ```bash
         cd hilts/client
         npm install
         npm run dev
         ```
    4. Access the frontend at [http://localhost:5173](http://localhost:5173) and the backend at [http://localhost:5000](http://localhost:5000).


- **Interactive CSV data loading and labeling**
- **Keyword, random, and image-based search**
- **Label management and export**
- **Integrated backend API for data processing and model inference**
- **Dockerized deployment**
- **Poetry-based Python environment**
- **S3/MinIO support for cloud storage**
- **Frontend built with Svelte + Vite**

---

## Folder Structure

```
hilts/
├── client/           # Svelte + TypeScript frontend
├── mmdx/             # Python backend modules
├── server.py         # Flask backend entrypoint
├── Dockerfile        # Multi-stage Docker build
├── pyproject.toml    # Poetry dependencies
├── poetry.lock
├── README.md         # This file
└── tests/            # Backend unit tests
```

---

## Local Development

### Prerequisites

- Python 3.9+
- Node.js 18+
- Poetry (`pip install poetry`)
- Docker (optional, for containerized development)

### Backend Setup

```bash
cd hilts
poetry install
poetry run python server.py
```

### Frontend Setup

```bash
cd hilts/client
npm install
npm run dev
```

The frontend will be available at [http://localhost:5173](http://localhost:5173) and the backend at [http://localhost:5000](http://localhost:5000).

---

## Docker Deployment

Build and run the full stack using Docker:

```bash
cd hilts
docker build -t hilts-app .
docker run -p 5000:5000 hilts-app
```

This will serve both the backend and the built frontend.

---

## Kubernetes Deployment

### 1. Build and Push Docker Image

```bash
docker build -t <your-dockerhub-username>/hilts-app:latest .
docker push <your-dockerhub-username>/hilts-app:latest
```

### 2. Example Kubernetes Manifests

#### `hilts-deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hilts-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hilts-app
  template:
    metadata:
      labels:
        app: hilts-app
    spec:
      containers:
      - name: hilts-app
        image: <your-dockerhub-username>/hilts-app:latest
        ports:
        - containerPort: 5000
        env:
        - name: ENV
          value: "prod"
        # Add S3/MinIO credentials here if needed
```

#### `hilts-service.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hilts-service
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 5000
  selector:
    app: hilts-app
```

### 3. Deploy to Cluster

```bash
kubectl apply -f hilts-deployment.yaml
kubectl apply -f hilts-service.yaml
```

Access the app via the external IP of the service.


---

## API Endpoints

See `server.py` for available endpoints, including:

- `/api/v1/keyword_search`
- `/api/v1/image_search`
- `/api/v1/add_label`
- `/api/v1/remove_label`
- `/api/v1/labels`
- `/api/v1/label_counts`
- `/api/v1/download/binary_labeled_data`
- `/api/v1/load/csv_data`

---

## Frontend Customization

- Edit `client/src/lib/Descriptions.js` to update animal/keyword lists.
- Modify Svelte components in `client/src/lib/` for UI changes.



---

## Troubleshooting

- **Port conflicts:** Change exposed ports in Docker/K8s manifests.
- **S3/MinIO issues:** Check credentials and endpoint URLs.
- **Frontend not loading:** Ensure `npm run build` in `client/` before Docker build.


## References

- [ACHE crawler](https://github.com/ViDA-NYU/ache)
- [Svelte](https://svelte.dev/)
- [Poetry](https://python-poetry.org/)
- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/)
