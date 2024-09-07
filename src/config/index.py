from flask import Blueprint, render_template, request, jsonify, send_from_directory, abort
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import pandas as pd

config = Blueprint('config', __name__, template_folder='templates')

UPLOAD_FOLDER = 'storage/'

# Allowed file extensions for dataset
ALLOWED_EXTENSIONS_DATASET = {'xlsx'}
SAVED_FILENAME = 'dataset.xlsx'
REQUIRED_COLUMNS = ['date', 'cabai_merah_besar', 'cabai_merah_keriting', 'cabai_rawit_hijau', 'cabai_rawit_merah', 'bawang_merah']
MINIMUM_ROWS = 30

# Allowed file extensions for model and scaler
ALLOWED_EXTENSIONS_MODEL_SKALAR = {'h5', 'pkl'}
ALLOWED_COMODITY = {'bawang_merah', 'cabai_merah_besar', 'cabai_merah_keriting', 'cabai_rawit_hijau', 'cabai_rawit_merah'}

# Membuat folder untuk menyimpan file
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# Fungsi untuk mengecek ekstensi file
def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Fungsi untuk validasi file
def validate_dataset_file(file_path):
    try:
        df = pd.read_excel(file_path)
        # Check for required columns
        if list(df.columns) != REQUIRED_COLUMNS:
            return False, 'Invalid columns. The file must contain the following columns: ' + ', '.join(REQUIRED_COLUMNS)
        
        # Check for minimum number of rows
        if len(df) < MINIMUM_ROWS:
            return False, 'The file must contain at least {} rows of data excluding the header.'.format(MINIMUM_ROWS)
        
        return True, 'File is valid'
    except Exception as e:
        return False, str(e)

# Upload Route          
@config.route('/config/upload-xlsx', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # If the file is valid
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_DATASET):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Validate the file
        is_valid, message = validate_dataset_file(file_path)
        if not is_valid:
            os.remove(file_path)
            return jsonify({'error': message}), 400
        
        # Rename the file
        final_path = os.path.join(UPLOAD_FOLDER, SAVED_FILENAME)
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(file_path, final_path)
        return jsonify({'message': 'File uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Allowed file type is xlsx'}), 400

# Download Example Dataset Route
@config.route('/config/get-example-dataset', methods=['GET'])
def download_example_dataset():
    try:
        # Ensure the file exists in the storage folder
        if not os.path.isfile(os.path.join(UPLOAD_FOLDER, 'example-dataset.xlsx')):
            abort(404)
        
        # Send the file from the storage folder
        return send_from_directory(UPLOAD_FOLDER, 'example-dataset.xlsx', as_attachment=True)
    except Exception as e:
        return str(e), 500
    
# Get Metadata Route
## Returns the metadata of model and pickle file

## format time
def format_time(timestamp):
    months = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember"
    ]
    dt = datetime.fromtimestamp(timestamp)  # Convert the timestamp to a datetime object
    formatted_time = dt.strftime(f"%d {months[dt.month - 1]} %Y %H:%M")  # Custom format with month names
    return formatted_time

## Fungsi untuk mengambil metadata file
def get_file_metadata(file_path):
    # Mengecek apakah file ada
    if os.path.exists(file_path):
        # Mendapatkan waktu modifikasi file
        modification_time = os.path.getmtime(file_path)  # Get the Unix timestamp directly
        
        # Mengembalikan data dalam bentuk dictionary
        return {
            "file_name": os.path.basename(file_path),
            "uploaded_time": format_time(modification_time),  # Pass timestamp to format_time
        }
    else:
        return {
            "error": f"File {file_path} tidak ditemukan."
        }

## Metadata Route
@config.route('/config/get-file-metadata', methods=['GET'])
def get_files_metadata():
    # Daftar file yang akan diambil metadatanya
    files = [
        './storage/model_bawang_merah.h5',
        './storage/scaler_bawang_merah.pkl',
        './storage/model_cabai_merah_besar.h5',
        './storage/scaler_cabai_merah_besar.pkl',
        './storage/model_cabai_merah_keriting.h5',
        './storage/scaler_cabai_merah_keriting.pkl',
        './storage/model_cabai_rawit_hijau.h5',
        './storage/scaler_cabai_rawit_hijau.pkl',
        './storage/model_cabai_rawit_merah.h5',
        './storage/scaler_cabai_rawit_merah.pkl'
    ]

    # Ambil metadata dari setiap file
    files_metadata = [get_file_metadata(file) for file in files]

    # Return metadata dalam format JSON
    return jsonify(files_metadata)

# Upload Model and Scaler Route
@config.route('/config/upload-model-scaler', methods=['POST'])
def upload_model_scaler():
    # Ambil data dari form
    
    nama_komoditas = request.form.get('nama_komoditas')
    file_model_h5 = request.files.get('file_model_h5')
    file_scaler_pkl = request.files.get('file_scaler_pkl')

    # Validasi komoditas
    if nama_komoditas not in ALLOWED_COMODITY:
        return jsonify({"error": "Nama komoditas tidak valid."}), 400

    # Validasi file .h5 dan .pkl
    if not (file_model_h5 and allowed_file(file_model_h5.filename, 'h5')):
        return jsonify({"error": "File model harus berekstensi .h5"}), 400
    if not (file_scaler_pkl and allowed_file(file_scaler_pkl.filename, 'pkl')):
        return jsonify({"error": "File scaler harus berekstensi .pkl"}), 400

    # Nama file disesuaikan dengan nama komoditas
    model_filename = f"model_{nama_komoditas}.h5"
    scaler_filename = f"scaler_{nama_komoditas}.pkl"

    # Path lengkap untuk file yang akan disimpan
    model_path = os.path.join(UPLOAD_FOLDER, model_filename)
    scaler_path = os.path.join(UPLOAD_FOLDER, scaler_filename)

    # Simpan atau replace file .h5 dan .pkl
    file_model_h5.save(model_path)
    file_scaler_pkl.save(scaler_path)

    return jsonify({"message": "Files uploaded successfully", 
                    "model_file": model_filename, 
                    "scaler_file": scaler_filename})