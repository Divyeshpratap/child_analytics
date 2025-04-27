import os
import spacy
from flask import Flask, render_template, request, jsonify, url_for, send_file
from collections import Counter
import logging
from pathlib import Path
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import re
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder (if needed) and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Define the base directory for the CHILDES data.
# Expected structure: ChildesData/LT/ and ChildesData/TD/
CHILDES_ROOT = os.path.join(os.getcwd(), 'ChildesData')

logging.info("Loading SpaCy models...")
nlpSpacy = spacy.load('en_core_web_trf')
nlpResultManner = spacy.load('taggerModels/resultManner/output/model-best')
nlpAction = spacy.load('taggerModels/action/model-best')
logging.info("All SpaCy models loaded successfully.")

# -----------------------
# Global in-memory cache to store file analysis results.
# Keys are file paths (e.g. "LT/30ec/11005.cha") and values are analysis dictionaries.
analysis_cache = {}
text_analysis = {}


# Bel-specific config

BEL_UPLOAD_FOLDER = 'bel_uploads'
Path(BEL_UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
bel_analysis = {}            # single-file cache like text_analysis
BEL_ALLOWED_EXT = {'.cha'}  
# -----------------------
# Utility Functions
# -----------------------

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_text(original_text):
    """Process text for POS tagging and linguistic annotations."""
    try:
        # Process with the original SpaCy model (preserve casing)
        doc_spacy = nlpSpacy(original_text)
        spacy_tags = [(token.text, token.pos_, token.tag_, token.idx) for token in doc_spacy if token.is_alpha]

        # Process with custom models on lowercased text
        lower_text = original_text.lower()
        # Process Result-Manner verbs
        doc_resultManner = nlpResultManner(lower_text)
        result_verbs = {token.idx: token.text.lower() for token in doc_resultManner if token.tag_ == 'result' and token.is_alpha}
        manner_verbs = {token.idx: token.text.lower() for token in doc_resultManner if token.tag_ == 'manner' and token.is_alpha}
        # Process Action verbs
        doc_action = nlpAction(lower_text)
        stative_verbs = {token.idx: token.text.lower() for token in doc_action if token.tag_ == 'STATIVE' and token.is_alpha}
        dynamic_verbs = {token.idx: token.text.lower() for token in doc_action if token.tag_ == 'DYNAMIC' and token.is_alpha}

        processed_data = {
            'spacy_tags': spacy_tags,
            'result_verbs': result_verbs,
            'manner_verbs': manner_verbs,
            'stative_verbs': stative_verbs,
            'dynamic_verbs': dynamic_verbs,
        }

        df = create_dataframe(processed_data)
        logging.info(f"DataFrame created with {len(df)} tokens.")
        return df

    except Exception as e:
        logging.error(f"Error processing text: {e}")
        raise

def create_dataframe(processed_data):
    """Create a pandas DataFrame based on processed data."""
    try:
        tokens = []
        pos_tags = []
        action = []
        result_manner = []

        spacy_tags = processed_data['spacy_tags']
        result_verbs = processed_data['result_verbs']
        manner_verbs = processed_data['manner_verbs']
        stative_verbs = processed_data['stative_verbs']
        dynamic_verbs = processed_data['dynamic_verbs']

        action_lookup = {}
        for idx, verb in stative_verbs.items():
            action_lookup[idx] = 'Stative'
        for idx, verb in dynamic_verbs.items():
            action_lookup[idx] = 'Dynamic'

        result_manner_lookup = {}
        for idx, verb in result_verbs.items():
            result_manner_lookup[idx] = 'Result'
        for idx, verb in manner_verbs.items():
            result_manner_lookup[idx] = 'Manner'

        for token_text, pos, tag, idx in spacy_tags:
            tokens.append(token_text)
            pos_tags.append(pos)
            if pos == 'VERB':
                if idx in dynamic_verbs:
                    action.append(action_lookup.get(idx, None))
                    result_manner.append(result_manner_lookup.get(idx, None))
                elif idx in stative_verbs:
                    action.append(action_lookup.get(idx, None))
                    result_manner.append('Stative')
                else:
                    action.append(None)
                    result_manner.append(None)
            else:
                action.append(None)
                result_manner.append(None)

        positions = list(range(1, len(tokens) + 1))
        df = pd.DataFrame({
            'Position': positions,
            'Token': tokens,
            'POS Tag': pos_tags,
            'Action': action,
            'Result/Manner': result_manner
        })
        return df

    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        raise

# -----------------------
# CHILDES-Specific Utility Functions
# -----------------------

def generate_childes_tree():
    """
    Generate a tree structure representing the CHILDES directory.
    Returns a list of dictionaries with keys "id", "text", "children", and "type".
    """
    tree = []
    for folder in ['LT', 'TD']:
        folder_path = os.path.join(CHILDES_ROOT, folder)
        if os.path.exists(folder_path):
            subdirs = []
            fixed_subdirs = ["30ec", "30pc", "42ec", "42pc", "54ec", "54int", "66conv"]
            for sub in fixed_subdirs:
                sub_path = os.path.join(folder_path, sub)
                if os.path.exists(sub_path) and os.path.isdir(sub_path):
                    file_children = []
                    for f in sorted(os.listdir(sub_path)):
                        if f.endswith(".cha"):
                            file_children.append({
                                "id": f"{folder}/{sub}/{f}",
                                "text": f,
                                "type": "file"
                            })
                    subdirs.append({
                        "id": f"{folder}/{sub}",
                        "text": sub,
                        "children": file_children,
                        "type": "directory"
                    })
            tree.append({
                "id": folder,
                "text": folder,
                "children": subdirs,
                "type": "directory"
            })
    return tree

# def parse_childes_file(file_relative_path):
#     """
#     Given a file relative path (e.g., "LT/30ec/11005.cha"), open the file and extract all lines
#     beginning with "*CHI:" (the child's speech). Returns a list of strings.
#     """
#     try:
#         file_path = os.path.join(CHILDES_ROOT, file_relative_path)
#         with open(file_path, encoding='utf-8') as f:
#             lines = f.readlines()
#         chi_lines = []
#         for line in lines:
#             if line.startswith("*CHI:"):
#                 text = line.strip().replace("*CHI:", "").strip()
#                 text = re.sub(r'[\x15\x14\x16]\d+_\d+[\x15\x14\x16]', '', text)
#                 chi_lines.append(text)
#         return chi_lines
#     except Exception as e:
#         logging.error(f"Error parsing file {file_relative_path}: {e}")
#         return []
    
def parse_childes_file(file_relative_path, speaker_option="both"):
    """
    speaker_option ∈ {"child", "investigator", "both"}
      child        → only *CHI:
      investigator → *MOT: or *INV:  (whichever appears)
      both         → all three tags
    """
    speaker_tags = {
        "child": ["*CHI:"],
        "investigator": ["*MOT:", "*INV:"],
        "both": ["*CHI:", "*MOT:", "*INV:"]
    }
    target_tags = speaker_tags.get(speaker_option, speaker_tags["both"])
    chi_lines = []
    try:
        file_path = os.path.join(CHILDES_ROOT, file_relative_path)
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if any(line.startswith(tag) for tag in target_tags):
                text = line.split(":", 1)[1].strip()
                # text = line.strip().replace("*CHI:", "").strip()
                # Remove time-alignment markers
                text = re.sub(r'[\x15\x14\x16]\d+_\d+[\x15\x14\x16]', '', text)
                # 2. Remove non-verbal expressions: Anything starting with &= or &+
                text = re.sub(r'&=[a-zA-Z]+', '', text)  # Removes &=whisper, &=gasp, etc.
                text = re.sub(r'&\+[a-zA-Z]+', '', text)  # Removes &+welco, &+like, etc.
                # Remove non-verbal cues like < > and disfluencies like [ ]
                text = re.sub(r'<[^>]+>', '', text)  # Remove everything between < > (non-verbal vocalizations)
                text = re.sub(r'\[[^\]]+\]', '', text)  # Remove everything between [ ] (disfluencies)
                # Normalize and clean the text
                text = ' '.join(text.split())
                text = text.strip()
                if text and text[-1] not in ['.', '!', '?']:
                    text += '.'
                if text:
                    chi_lines.append(text)
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    return chi_lines

def process_child_file(file_relative_path, speaker_option="both"):
    """
    Process an individual CHILDES file:
      - Extract the *CHI conversation.
      - Run the NLP analysis.
      - Compute overall counts and breakdowns for Result and Manner verbs.
      - Also extract child metadata (group and gender).
    Returns a dictionary with file info, counts, breakdowns, parsed data, group, and gender.
    """
    chi_lines = parse_childes_file(file_relative_path, speaker_option)
    full_text = "\n".join(chi_lines)
    df = process_text(full_text)
    result_count = int((df['Result/Manner'] == 'Result').sum())
    manner_count = int((df['Result/Manner'] == 'Manner').sum())
    result_list = df[df["Result/Manner"]=="Result"]["Token"].tolist()
    manner_list = df[df["Result/Manner"]=="Manner"]["Token"].tolist()
    result_breakdown = dict(Counter(result_list))
    manner_breakdown = dict(Counter(manner_list))
    
    # Determine group from the file path (first part before the slash: LT or TD)
    group = file_relative_path.split('/')[0] if '/' in file_relative_path else "Unknown"
    
    # Extract gender from the filename.
    # Example: "11005.cha" --> second character indicates gender (1: female, 2: male)
    base = os.path.basename(file_relative_path)
    file_id = os.path.splitext(base)[0]  # e.g., "11005"
    if len(file_id) >= 2:
        gender_code = file_id[1]
        gender = "female" if gender_code == '1' else ("male" if gender_code == '2' else "unknown")
    else:
        gender = "unknown"
    
    return {
        "file": file_relative_path,
        "group": group,
        "gender": gender,
        "result_count": result_count,
        "manner_count": manner_count,
        "result_breakdown": result_breakdown,
        "manner_breakdown": manner_breakdown,
        "parsed_data": chi_lines
    }

# -----------------------
# Routes for Existing Functionality
# -----------------------

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text_input = request.form.get('text_input', '').strip()
    try:
        df = process_text(text_input)
        df = df.where(pd.notnull(df), None)

        # Calculate summary counts.
        result_count = int((df['Result/Manner'] == 'Result').sum())
        manner_count = int((df['Result/Manner'] == 'Manner').sum())
        # Detailed token-level results.
        results = df.to_dict(orient='records')
        result_list = df[df["Result/Manner"] == "Result"]["Token"].tolist()
        manner_list = df[df["Result/Manner"] == "Manner"]["Token"].tolist()
        result_breakdown = dict(Counter(result_list))
        manner_breakdown = dict(Counter(manner_list))
        # Save summary analysis in the global variable.
        global text_analysis
        text_analysis = {
            'original_text': text_input,
            'result_count': result_count,
            'manner_count': manner_count,
            'parsed_data': text_input.splitlines(),  # Change as needed to represent the details
            'result_breakdown': result_breakdown,
            'manner_breakdown': manner_breakdown
        }
        logging.info(f"Text Analysis - Result Count: {result_count}, Manner Count: {manner_count}")
        return jsonify({
            'status': 'success',
            'original_text': text_input,
            'results': results,
            'summary': {
                'result_count': result_count,
                'manner_count': manner_count,
                'result_breakdown': result_breakdown,
                'manner_breakdown': manner_breakdown
            }
        }), 200
    except Exception as e:
        logging.error(f"Processing error: {e}")
        return jsonify({'status': 'error', 'message': "An error occurred during processing."}), 500


@app.route('/download_text_analysis', methods=['GET'])
def download_text_analysis():
    """
    Generate an Excel file containing the analysis for the text input:
    - A summary sheet with overall counts.
    - A details sheet with token-level parsed data and/or additional breakdowns.
    """
    if not text_analysis:
        return "No text analysis available. Please process some input first.", 400

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Summary sheet.
    summary_df = pd.DataFrame([{
        "Original Text": text_analysis.get("original_text", ""),
        "Result Count": text_analysis.get("result_count", 0),
        "Manner Count": text_analysis.get("manner_count", 0),
        "Unique Result Breakdown": str(text_analysis.get("result_breakdown", {})),
        "Unique Manner Breakdown": str(text_analysis.get("manner_breakdown", {}))
    }])
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    # Details sheet: for example, the raw text split by lines.
    details_df = pd.DataFrame({"Parsed Text": text_analysis.get("parsed_data", [])})
    details_df.to_excel(writer, sheet_name="Details", index=False)
    
    # writer.save()
    writer.close()
    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="text_analysis.xlsx"
    )    

# buffalo early lab
def parse_bel_file(file_path, speaker_option="all"):
    """
    speaker_option ∈ {"child", "parent", "sibling", "all"}
      child   → *CHI:
      parent  → *PAR:
      sibling → *SIB:
      all     → the three tags above
    """
    tag_map = {
        "child":   ["*CHI:"],
        "parent":  ["*PAR:"],
        "sibling": ["*SIB:"],
        "all":     ["*CHI:", "*PAR:", "*SIB:"]
    }
    target_tags = tag_map.get(speaker_option, tag_map["all"])

    lines_out = []
    try:
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                if any(line.startswith(t) for t in target_tags):
                    text = line.split(":", 1)[1].strip()
                    # same clean-up rules you used for CHILDES
                    # text = re.sub(r'[\x15\x14\x16]\d+_\d+[\x15\x14\x16]', '', text)
                    text = re.sub(r'[\x14\x15\x16][^\x14\x15\x16]*[\x14\x15\x16]', '', text)
                    text = re.sub(r'&=[a-zA-Z]+', '', text)
                    text = re.sub(r'&\+[a-zA-Z]+', '', text)
                    text = re.sub(r'<[^>]+>', '', text)
                    text = re.sub(r'\[[^\]]+\]', '', text)
                    text = ' '.join(text.split()).strip()
                    if text and text[-1] not in '.!?':
                        text += '.'
                    if text:
                        lines_out.append(text)
    except Exception as e:
        logging.error(f"Error parsing BEL file {file_path}: {e}")

    return lines_out

# -----------------------
# NEW Routes for CHILDES Analysis
# -----------------------

@app.route('/childes', methods=['GET'])
def childes_page():
    return render_template('childes.html')

@app.route('/get_childes_tree', methods=['GET'])
def get_childes_tree():
    try:
        tree = generate_childes_tree()
        return jsonify(tree)
    except Exception as e:
        logging.error(f"Error generating CHILDES tree: {e}")
        return jsonify({'error': 'Unable to generate directory tree'}), 500

@app.route('/process_childes', methods=['POST'])
def process_childes():
    """
    Process the selected CHILDES files.
    Expected JSON payload: { "selected_files": [ "LT/30ec/11005.cha", ... ] }
    Returns analysis results per file, caches them, and also includes overall totals.
    """
    try:
        analysis_cache.clear()
        data = request.get_json()
        selected_files = data.get("selected_files", [])
        speaker_option = data.get("speaker_option", "both")
        results = []
        total_result = 0
        total_manner = 0
        for file_rel in selected_files:
            analysis = process_child_file(file_rel, speaker_option)
            analysis_cache[file_rel] = analysis
            results.append(analysis)
            total_result += analysis["result_count"]
            total_manner += analysis["manner_count"]
        response = {
            'status': 'success',
            'results': results,
            'overall_totals': {
                'total_result': total_result,
                'total_manner': total_manner
            }
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error processing CHILDES files: {e}")
        return jsonify({'status': 'error', 'message': "An error occurred during CHILDES processing."}), 500

@app.route('/view_parsed_data', methods=['GET'])
def view_parsed_data():
    file_rel = request.args.get("file")
    if not file_rel:
        return "No file specified", 400
    chi_lines = parse_childes_file(file_rel)
    html_content = "<html><head><title>Parsed Data - {}</title></head><body>".format(file_rel)
    html_content += "<h2>Parsed *CHI Data for {}</h2><pre>".format(file_rel)
    html_content += "\n".join(chi_lines)
    html_content += "</pre></body></html>"
    return html_content

@app.route('/view_verb_details', methods=['GET'])
def view_verb_details():
    """
    Display the breakdown for a given CHILDES file and verb type.
    Expects query parameters: 'file' (e.g., LT/30ec/11005.cha) and 'type' (result or manner).
    Retrieves the breakdown from the cache.
    """
    file_rel = request.args.get('file')
    verb_type = request.args.get('type')  # expected: "result" or "manner"
    if not file_rel or not verb_type:
        return "Missing parameters", 400

    if file_rel in analysis_cache:
        analysis = analysis_cache[file_rel]
    else:
        analysis = process_child_file(file_rel)
        analysis_cache[file_rel] = analysis

    if verb_type.lower() == "result":
        breakdown = analysis.get("result_breakdown", {})
        verb_label = "Result Verbs"
    elif verb_type.lower() == "manner":
        breakdown = analysis.get("manner_breakdown", {})
        verb_label = "Manner Verbs"
    else:
        return "Invalid verb type", 400

    html_content = "<html><head><title>Verb Breakdown: {}</title></head><body>".format(file_rel)
    html_content += "<h2>Breakdown for {} (file: {})</h2>".format(verb_label, file_rel)
    html_content += "<ul>"
    if breakdown:
        for verb, count in breakdown.items():
            html_content += "<li><strong>{}</strong> : {}</li>".format(verb, count)
    else:
        html_content += "<li>No {} found.</li>".format(verb_label)
    html_content += "</ul></body></html>"
    return html_content

@app.route('/download_analysis', methods=['GET'])
def download_analysis():
    """
    Generate an Excel file containing the analysis:
    - A summary sheet listing each file with group, gender, result and manner counts.
    - Separate sheets for each file with the complete parsed transcript and breakdown dictionaries.
    """
    if not analysis_cache:
        return "No analysis available. Please process some CHILDES files first.", 400

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Create a summary DataFrame
    summary_rows = []
    for file_rel, analysis in analysis_cache.items():
        summary_rows.append({
            "File": file_rel,
            "Group": analysis.get("group", ""),
            "Gender": analysis.get("gender", ""),
            "Result Count": analysis.get("result_count", 0),
            "Manner Count": analysis.get("manner_count", 0),
            "Result Verbs Breakdown": analysis.get("result_breakdown", {}),
            "Manner Verbs Breakdown": analysis.get("manner_breakdown", {})
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    # Create one sheet per file with detailed data.
    for file_rel, analysis in analysis_cache.items():
        sheet_name = file_rel.replace("/", "_")
        sheet_name = sheet_name.split('.')[0]

        # Build a DataFrame for the parsed transcript.
        transcript_df = pd.DataFrame({"Parsed Transcript": analysis.get("parsed_data", [])})
        # Build DataFrames for breakdowns.
        result_breakdown_df = pd.DataFrame(list(analysis.get("result_breakdown", {}).items()),
                                           columns=["Result Verb", "Count"])
        manner_breakdown_df = pd.DataFrame(list(analysis.get("manner_breakdown", {}).items()),
                                           columns=["Manner Verb", "Count"])
        # Write transcript and breakdowns in the same sheet, separated by a blank row.
        # transcript_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        # result_breakdown_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(transcript_df)+2, header=True)
        # manner_breakdown_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(transcript_df)+len(result_breakdown_df)+4, header=True)

        result_breakdown_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0, header=True)
        manner_breakdown_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(result_breakdown_df)+4, header=True)
        transcript_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(result_breakdown_df)+ len(manner_breakdown_df) + 8, header = True)


    # writer.save()
    writer.close()
    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="childes_analysis.xlsx"  # Updated keyword argument
    )


@app.route('/bel', methods=['GET'])
def bel_page():
    return render_template('bel.html')

@app.route('/process_bel', methods=['POST'])
def process_bel():
    global bel_analysis
    bel_analysis.clear()

    # 1. pull speaker option
    speaker_option = request.form.get('speaker_option', 'all')

    # 2. pull uploaded file
    f = request.files.get('cha_file')
    if f is None or f.filename == '':
        return jsonify({'status': 'error', 'message': 'No file uploaded.'}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in BEL_ALLOWED_EXT:
        return jsonify({'status': 'error', 'message': 'Invalid file type.'}), 400

    filename = secure_filename(f.filename)
    saved_path = Path(BEL_UPLOAD_FOLDER) / filename
    f.save(saved_path)

    # 3. parse + analyse
    speaker_lines = parse_bel_file(saved_path, speaker_option)
    full_text = "\n".join(speaker_lines)
    df = process_text(full_text)

    # 4. summary
    result_count  = int((df['Result/Manner'] == 'Result').sum())
    manner_count  = int((df['Result/Manner'] == 'Manner').sum())
    result_break  = dict(Counter(df[df["Result/Manner"]=="Result"]["Token"]))
    manner_break  = dict(Counter(df[df["Result/Manner"]=="Manner"]["Token"]))

    bel_analysis.update({
        'original_text': full_text,
        'result_count': result_count,
        'manner_count': manner_count,
        'parsed_data': speaker_lines,
        'result_breakdown': result_break,
        'manner_breakdown': manner_break
    })

    return jsonify({
        'status': 'success',
        'original_text': full_text,
        'results': df.to_dict(orient='records'),
        'summary': {
            'result_count': result_count,
            'manner_count': manner_count,
            'result_breakdown': result_break,
            'manner_breakdown': manner_break
        }
    }), 200


@app.route('/download_bel_analysis', methods=['GET'])
def download_bel_analysis():
    if not bel_analysis:
        return "No analysis available.", 400

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    summary_df = pd.DataFrame([{
        "Result Count": bel_analysis['result_count'],
        "Manner Count": bel_analysis['manner_count'],
        "Unique Result Breakdown": str(bel_analysis['result_breakdown']),
        "Unique Manner Breakdown": str(bel_analysis['manner_breakdown'])
    }])
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    details_df = pd.DataFrame({"Parsed Text": bel_analysis['parsed_data']})
    details_df.to_excel(writer, sheet_name="Details", index=False)

    writer.close()
    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="bel_analysis.xlsx"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
