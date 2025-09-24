from datetime import datetime
import os
import pandas as pd
from flask import Flask, send_from_directory, send_file, request, jsonify
from mmdx.search import VectorDB
from mmdx.model import ClipModel
from mmdx.settings import (
    DATA_PATH,
    DB_PATH,
    DB_DELETE_EXISTING,
    DB_BATCH_LOAD,
    ACCESS_KEY,
    ENDPOINT_URL,
    SECRET_KEY,
    DATA_SOURCE,
    LOAD_DATA,
    DEMO,
)
from mmdx.s3_client import S3Client
from mmdx.LTS_manager import LTSManager
from io import BytesIO
import json



app = Flask(__name__)
# socketio = SocketIO(app) #cors_allowed_origins="*"


# Path for our main Svelte app. All routes in the app must be added
# here to allow refreshing to work correctly.
@app.route("/")
@app.route("/csv-loader")
@app.route("/search/seller")
@app.route("/search/random")
@app.route("/search/image")
@app.route("/dashboard")
@app.route("/labels")
@app.route("/bootstrap")
def base():
    return send_from_directory("client/dist/", "index.html")

# @socketio.on('my event')
# def test_message(message):
#     socketio.emit('my response', {'data': f'{random.random()}' })

@app.route("/api/v1/start_training", methods=['POST'])
def start_training():
    print("start")
    data = request.get_json()
    args = data.get('argsDict')
    project_id = args["projectId"]
    label_hilts = args.get("labeling")
    global ltsmanager
    ltsmanager = LTSManager(project_id, db, DEMO)
    training_process = ltsmanager.start_training(label_hilts)
    return jsonify({
        'message': 'Training started!',
        # 'process_id': training_process.pid
    })


@app.route("/api/v1/restart_training", methods=['POST'])
def start_retraining():
    data = request.get_json()
    args = data.get('argsDict')
    label_hilts = args.get("labeling")
    db.create_hilts_data()
    training_process = ltsmanager.start_training(label_hilts)
    return jsonify({
        'message': 'Training started!',
    })

@app.route("/api/v1/stop_training", methods=['POST'])
def stop_training():
    ltsmanager.stop_LTS()
    return {'message': 'LTS Stop message sent'}

@app.route("/api/v1/save_counts", methods=["POST"])
def save_counts():
    try:
        # Get the counts and project_id from the request body
        data = request.get_json()
        url_click_count = data.get("urlClickCount", 0)
        project_id = data.get("project_id")

        if not project_id:
            return jsonify({"success": False, "message": "Project ID is required"}), 400

        # Define the directory and file path for saving the counts
        project_dir = os.path.join("data", project_id)
        os.makedirs(project_dir, exist_ok=True)
        counts_file_path = os.path.join(project_dir, "click_counts.json")

        # Load existing counts if the file exists
        if os.path.exists(counts_file_path):
            with open(counts_file_path, "r") as file:
                counts_data = json.load(file)
        else:
            counts_data = {"urlClickCount": 0}

        # Update the counts
        counts_data["urlClickCount"] += url_click_count

        # Save the updated counts back to the file
        with open(counts_file_path, "w") as file:
            json.dump(counts_data, file, indent=4)

        return jsonify({"success": True, "message": "Counts saved successfully", "data": counts_data})

    except Exception as e:
        print(f"Error saving counts: {e}")
        return jsonify({"success": False, "message": f"Error saving counts: {e}"})

@app.route("/api/v1/set_db", methods=['POST'])
def set_labels_db():
    data = request.get_json()
    project_id = data.get('projectId')
    if not os.path.exists(f"data/{project_id}"):
        os.makedirs(f"data/{project_id}")
    if not os.path.exists(f"data/{project_id}/hilts"):
        os.makedirs(f"data/{project_id}/hilts")
    db.set_label_db(project_id)
    return {'message': 'db set'}

@app.route("/api/v1/connect_db", methods=['POST'])
def connect_labels_db():
    data = request.get_json()
    project_id = data.get('projectId')
    if not os.path.exists(f"data/{project_id}"):
        return {'message': 'db not found'}
    db.set_label_db(project_id)
    return {'message': 'db set'}

@app.route("/api/v1/get_status/", methods=['GET'])
def get_training_results() -> pd.DataFrame:
    # manager = LTSManager(projectId) ## should make it global?
    try:
        status = ltsmanager.get_status()
        return status
    except NameError as e:
        print(f"No manager created {e}")
        return {}

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def assets(path):
    return send_from_directory("client/dist/", path)


@app.route("/images/<path:path>")
def images(path):
    if DATA_SOURCE.upper() == "S3":
        image_data = S3_Client.get_obj(DATA_PATH, path)
        image_buffer = BytesIO(image_data.read())
        return send_file(
            image_buffer,
            as_attachment=False,
            download_name=path,
        )
    else:
        return send_from_directory(DATA_PATH, path)

@app.route("/api/v1/random")
def random_search():
    limit: int = request.args.get("limit", 12, type=int)
    hits = db.random_search(limit=limit)
    return {"total": len(hits.index), "hits": hits.to_dict("records")}

@app.route("/api/v1/random_hilts")
def random_hilts_search():
    projectId: str = request.args.get("projectId", "default", type=str)
    limit: int = request.args.get("limit", 12, type=int)
    print(f"getting data from project: {projectId}")
    hits = db.random_hilts_search(limit=limit, s3_client=S3_Client)
    return {"total": len(hits.index), "hits": hits.to_dict("records")}


@app.route("/api/v1/keyword_search")
def keyword_search():
    query: str = request.args.get("q")
    exclude_labeled: bool = request.args.get("exclude_labeled", "false") == "true"
    limit: int = request.args.get("limit", 12, type=int)
    hits = db.search_by_text(query_string=query, limit=limit, exclude_labeled=exclude_labeled)
    return {"total": len(hits.index), "hits": hits.to_dict("records")}


@app.route("/api/v1/image_search")
def image_search():
    query: str = request.args.get("q")
    exclude_labeled: bool = request.args.get("exclude_labeled", "false") == "true"
    limit: int = request.args.get("limit", 12, type=int)
    hits = db.search_by_image_path(
        image_path=query, limit=limit, S3_Client=S3_Client, exclude_labeled=exclude_labeled
    )
    return {"total": len(hits.index), "hits": hits.to_dict("records")}


@app.route("/api/v1/seller_search")
def seller_search():
    query: str = request.args.get("q")
    exclude_labeled: bool = request.args.get("exclude_labeled", "false") == "true"
    limit: int = request.args.get("limit", 12, type=int)
    hits = db.search_by_seller(query_string=query, limit=limit, exclude_labeled=exclude_labeled)
    return {"total": len(hits.index), "hits": hits.to_dict("records")}

@app.route("/api/v1/labeled")
def labeled_search():
    limit: int = request.args.get("limit", 12, type=int)
    hits = db.search_labeled_data(limit=limit)
    return {"total": len(hits.index), "hits": hits.to_dict("records")}

@app.route("/api/v1/add_label")
def add_label():
    image_path: str = request.args.get("image_path", None, type=str)
    label: str = request.args.get("label", None, type=str)
    table: str = request.args.get("table", None, type=str)
    db.add_label(image_path=image_path, label=label, table=table)
    return {"success": True}


@app.route("/api/v1/remove_label")
def remove_label():
    image_path: str = request.args.get("image_path", None, type=str)
    label: str = request.args.get("label", None, type=str)
    table: str = request.args.get("table", None, type=str)
    db.remove_label(image_path=image_path, label=label, table=table)
    return {"success": True}


@app.route("/api/v1/labels")
def labels():
    table: str = request.args.get("table", None, type=str)
    labels = db.get_labels(table)
    return {"labels": labels}


@app.route("/api/v1/label_counts")
def label_counts():
    counts = db.get_label_counts()
    print(counts)
    return {"counts": counts}


@app.route("/api/v1/download/binary_labeled_data")
def download_binary_labeled_data():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"labeled_data_{current_time}.zip"
    output_zip_file = db.create_zip_labeled_binary_data(
        output_dir=os.path.join(DB_PATH, "downloads"),
        filename=filename
    )
    print("Created zip file: ", output_zip_file)
    return send_file(output_zip_file, as_attachment=True)

@app.route("/api/v1/load/create_lts_config", methods=['POST'])
def start_lts_generation():
    try:
        data = request.get_json()
        # Extract the prompt text
        args = data.get('argsDict')
        project_id = data.get('ProjectId')
        if not args:
            return {'message': 'No prompt provided'}
        if not os.path.exists(f"data/{project_id}"):
            os.makedirs(f"data/{project_id}")
        with open(f"data/{project_id}/config_file.json", "w") as file:
            json.dump(args, file)
        return {'message': 'config create successfully'}
        # main(sampler, data, sample_size, filter_label, trainer, labeler, filename, balance, metric, baseline, labeling, retrain, idx, id)

    except Exception as e:
        return {'message': f'Config file not create {e}'}

@app.route("/api/v1/get_data/<projectId>", methods=['GET'])
def create_products_data(projectId) -> pd.DataFrame:
    df = pd.read_csv(f"data/{projectId}/filename.csv")
    df = df[["seller", "image_path", "products", "animal_name_match"]]
    df.rename(columns={"animal_name_match": "animalName"}, inplace=True)
    data = df.to_json(orient='records')
    return data


@app.route("/api/v1/load/csv_data", methods=['POST'])
def create_database():
    # try:
    if 'file' not in request.files:
        return {'error': 'No file part'}

    file = request.files['file']

    if file.filename == '':
        raise ValueError('No selected file')

    project_id = request.form.get('projectId')
    datatype = request.form.get('datatype')
    if project_id =='':
        return {'error': 'Please add a project ID'}
    # Get the CSV file from the form data

    filename = file.filename #TODO change train name
    # filepath = os.path.join(project_id, filename)
    if datatype == "test":
        filename = "test.csv"
    elif datatype == "validation":
        filename = "validation.csv"


    project_dir = os.path.join("data", project_id)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    file_path = os.path.join(project_dir, filename)

    # file.save(filepath)
    file.save(file_path)
    # global db
    # db = create_db_for_data_path(S3_Client)

    return {'message': 'CSV data received and processed successfully'}


def create_db_for_data_path(S3_Client):
    data_path = "images-february24" #DATA_PATH
    db_path = DB_PATH

    if DATA_SOURCE.upper() == "S3":
        data_path_msg = f" - Data Bucket: {data_path}"
    else:
        S3_Client = None
        data_path_msg = f" - Raw data path: {os.path.abspath(data_path)}"

    print("Loading embedding model...")
    model = ClipModel()

    print(f"Loading vector database:")
    print(f" - DB path: {os.path.abspath(db_path)}")
    print(data_path_msg)
    print(f" - Delete existing?: {DB_DELETE_EXISTING}")
    print(f" - Batch load?: {DB_BATCH_LOAD}")
    vectordb = VectorDB.from_data_path(
        data_path,
        db_path,
        S3_Client,
        model,
        delete_existing=DB_DELETE_EXISTING,
        batch_load=DB_BATCH_LOAD,
    )
    print("Finished DB initialization.")
    return vectordb


if DATA_SOURCE.upper() == "S3":
    S3_Client = S3Client(
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
    )
else:
    S3_Client=None

if LOAD_DATA:
    db: VectorDB = None
else:
    db: VectorDB = create_db_for_data_path(S3_Client)

if __name__ == "__main__":
    if os.environ.get("ENV") == "prod":
        app.run(debug=False, host="0.0.0.0")
        # socketio.run(app, debug=False, host="0.0.0.0", port=5000)
    else:
        app.run(debug=True, host="127.0.0.1")
