from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import os

comodity = Blueprint('comodity', __name__)

df = None

@comodity.before_request
def load_and_process_excel():
    """
    This function will run before every request in this blueprint.
    It loads and processes the Excel file.
    """
    global df
    if df is None:
        file_path = 'storage/dataset.xlsx'
        if os.path.exists(file_path):
            # Load the Excel file
            df = pd.read_excel(file_path, parse_dates=['date'])
            # Convert the 'date' to Unix timestamp in milliseconds
            df['unix_timestamp'] = df['date'].apply(lambda x: int(x.timestamp() * 1000))
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
