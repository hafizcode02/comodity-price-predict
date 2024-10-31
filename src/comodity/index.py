from flask import Blueprint, jsonify
import pandas as pd
import os

comodity = Blueprint('comodity', __name__)

df = None
last_modified_time = None

@comodity.before_request
def load_and_process_excel():
    """
    This function will run before every request in this blueprint.
    It loads and processes the Excel file.
    """
    global df, last_modified_time
    file_path = 'storage/dataset.xlsx'
    
    if os.path.exists(file_path):
        # Get the last modified time of the file
        modified_time = os.path.getmtime(file_path)
        
        # Check if the file has been modified since the last load
        if df is None or last_modified_time is None or modified_time > last_modified_time:
            # Reload the Excel file
            df = pd.read_excel(file_path, parse_dates=['date'])
            df['unix_timestamp'] = df['date'].apply(lambda x: int(x.timestamp() * 1000))
            last_modified_time = modified_time  # Update the last modified time
    else:
        return jsonify({'error': 'File not found'}), 404

@comodity.route('/comodity-data/cabai-merah-besar')
def data_cabai_merah_besar():
    # Select the required columns and convert to a list of lists
    data_list = df[['unix_timestamp', 'cabai_merah_besar']].values.tolist()
    
    # Return JSON response
    return jsonify(data=data_list)

@comodity.route('/comodity-data/cabai-merah-keriting')
def data_cabai_merah_keriting():
    # Select the required columns and convert to a list of lists
    data_list = df[['unix_timestamp', 'cabai_merah_keriting']].values.tolist()
    
    # Return JSON response
    return jsonify(data=data_list)

@comodity.route('/comodity-data/cabai-rawit-hijau')
def data_cabai_rawit_hijau():
    # Select the required columns and convert to a list of lists
    data_list = df[['unix_timestamp', 'cabai_rawit_hijau']].values.tolist()
    
    # Return JSON response
    return jsonify(data=data_list)

@comodity.route('/comodity-data/cabai-rawit-merah')
def data_cabai_rawit_merah():
    # Select the required columns and convert to a list of lists
    data_list = df[['unix_timestamp', 'cabai_rawit_merah']].values.tolist()
    
    # Return JSON response
    return jsonify(data=data_list)

@comodity.route('/comodity-data/bawang-merah')
def data_bawang_merah():
    # Select the required columns and convert to a list of lists
    data_list = df[['unix_timestamp', 'bawang_merah']].values.tolist()
    
    # Return JSON response
    return jsonify(data=data_list)
