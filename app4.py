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
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# ---------------------------------------------------------------------
# Paths & upload folders
# ---------------------------------------------------------------------
UPLOAD_FOLDER       = 'uploads'
BEL_UPLOAD_FOLDER   = 'bel_uploads'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(BEL_UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CHILDES_ROOT = os.path.join(os.getcwd(), 'ChildesData')
ALLOWED_EXTENSIONS   = {'wav', 'mp3'}
BEL_ALLOWED_EXT      = {'.cha'}

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
logging.info("Loading SpaCy models …")
nlpSpacy        = spacy.load('en_core_web_trf')
nlpResultManner = spacy.load('taggerModels/resultManner/output/model-best')
nlpAction       = spacy.load('taggerModels/action/model-best')
logging.info("All SpaCy models loaded successfully.")

# ---------------------------------------------------------------------
# Global caches
# ---------------------------------------------------------------------
analysis_cache = {}   # CHILDES multi-file
text_analysis   = {}   # free-text
bel_analysis    = {}   # single BEL file

# ---------------------------------------------------------------------
# ──────────── shared REGEXES  (compiled once) ────────────
# ---------------------------------------------------------------------
TIMESTAMP_CHILDES_RE = re.compile(r'[\x15\x14\x16]\d+_\d+[\x15\x14\x16]')
TIMESTAMP_BEL_RE     = re.compile(r'[\x14\x15\x16][^\x14\x15\x16]*[\x14\x15\x16]')

NOISE_RE        = re.compile(r'\s*&[=+][A-Za-z]+')
META_BRACKET_RE = re.compile(r'\s*\[=! [^\]]+]')
STD_REPL_RE     = re.compile(r'<[^>]+>\s*\[:\s*([^\]]+)]')
ANGLE_RE        = re.compile(r'<([^>]+)>')
LANG_UTT_TAG_RE = re.compile(r'\s*\[-\s*[a-z]{3}]')
INLINE_SUFFIX_RE= re.compile(r'(@s:[a-z]{3}|@c|@l)\b')
START_HYPHEN_RE = re.compile(r'(^|(?<=\s))-(?=[A-Za-z])')
END_HYPHEN_RE   = re.compile(r'(?<=[A-Za-z])-(?=\s|\.|$)')
UNDERSCORE_RE   = re.compile(r'_')
MULTISPACE_RE   = re.compile(r'\s{2,}')

# ---------------------------------------------------------------------
# Shared cleaning helper
# ---------------------------------------------------------------------
def _clean_utterance(text, timestamp_re=None):
    """
    Apply corpus-independent cleaning rules + optional timestamp rule.
    Returns cleaned utterance or '' if nothing remains.
    """
    if timestamp_re:
        text = timestamp_re.sub('', text)

    text = NOISE_RE.sub('',           text)
    text = META_BRACKET_RE.sub('',    text)
    text = STD_REPL_RE.sub(r'\1',     text)
    text = ANGLE_RE.sub(r'\1',        text)
    text = LANG_UTT_TAG_RE.sub('',    text)
    text = INLINE_SUFFIX_RE.sub('',   text)
    text = START_HYPHEN_RE.sub(r'\1', text)
    text = END_HYPHEN_RE.sub('',      text)
    text = UNDERSCORE_RE.sub(' ',     text)
    text = MULTISPACE_RE.sub(' ',     text).strip()

    if text and text[-1] not in '.!?':
        text += '.'
    return text

# ---------------------------------------------------------------------
# speaker-level analysis helper
# ---------------------------------------------------------------------
def _analyse_utterances(lines: list[str]) -> dict:
    """
    Combine utterances → run POS pipeline → return counts & breakdowns.
    """
    text = "\n".join(lines)
    if not text:
        return {
            "result_count": 0,
            "manner_count": 0,
            "result_breakdown": {},
            "manner_breakdown": {},
            "text": ""
        }
    df = process_text(text)
    res_cnt = int((df['Result/Manner'] == 'Result').sum())
    man_cnt = int((df['Result/Manner'] == 'Manner').sum())
    res_bd  = dict(Counter(df[df['Result/Manner'] == 'Result']['Token']))
    man_bd  = dict(Counter(df[df['Result/Manner'] == 'Manner']['Token']))
    return {
        "result_count": res_cnt,
        "manner_count": man_cnt,
        "result_breakdown": res_bd,
        "manner_breakdown": man_bd,
        "text": text
    }

# -----------------------
# Utility Functions
# -----------------------

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

def clean_text(raw: str) -> str:
    return _clean_utterance(raw)

# ---------------------------------------------------------------------
# CHILDES utils
# ---------------------------------------------------------------------
def parse_childes_file(file_rel, speaker_option="both"):
    tag_map = {
        "child":        ["*CHI:"],
        "investigator": ["*MOT:", "*INV:"],
        "both":         ["*CHI:", "*MOT:", "*INV:"]
    }
    target_tags = tag_map.get(speaker_option, tag_map["both"])
    utterances  = {"CHI": [], "MOT": [], "INV": []}
    sequence   = []
    file_path = os.path.join(CHILDES_ROOT, file_rel)
    try:
        with open(file_path, encoding='utf-8') as fh:
            for line in fh:
                tag = next((t for t in target_tags if line.startswith(t)), None)
                if not tag:
                    continue
                speaker = tag.strip("*:")  # "*CHI:" → "CHI"
                text = line.split(":", 1)[1].strip()
                text = _clean_utterance(text, TIMESTAMP_CHILDES_RE)
                if text:
                    utterances[speaker].append(text)
                    sequence.append((speaker, text))
    except Exception as e:
        logging.error(f"Error parsing {file_rel}: {e}")

    return utterances, sequence

def process_child_file(file_relative_path, speaker_option="both"):
    """
    Process an individual CHILDES file:
      - Extract the *CHI conversation.
      - Run the NLP analysis.
      - Compute overall counts and breakdowns for Result and Manner verbs.
      - Also extract child metadata (group and gender).
    Returns a dictionary with file info, counts, breakdowns, parsed data, group, and gender.
    """
    speaker_lines, seq = parse_childes_file(file_relative_path, speaker_option)
    speaker_stats  = {spk: _analyse_utterances(v) for spk, v in speaker_lines.items() if v}

    total_result   = sum(v["result_count"] for v in speaker_stats.values())
    total_manner   = sum(v["manner_count"] for v in speaker_stats.values())

    # aggregated verb lists for legacy routes
    all_results = Counter()
    all_manners = Counter()
    for st in speaker_stats.values():
        all_results.update(st["result_breakdown"])
        all_manners.update(st["manner_breakdown"])

    # meta info (unchanged)
    group  = file_relative_path.split('/')[0] if '/' in file_relative_path else "Unknown"
    base   = os.path.basename(file_relative_path)
    gender = "female" if len(base) >= 2 and base[1] == '1' else ("male" if len(base) >= 2 and base[1] == '2' else "unknown")

    return {
        "file": file_relative_path,
        "group": group,
        "gender": gender,
        "total_result": total_result,
        "total_manner": total_manner,
        "speakers": speaker_stats,
        # legacy fields used elsewhere
        "result_breakdown": dict(all_results),
        "manner_breakdown": dict(all_manners),
        "parsed_data": speaker_lines,          # dict{spk:[lines]}
        "sequence": seq
    }




# ---------------------------------------------------------------------
# BEL parser
# ---------------------------------------------------------------------
def parse_bel_file(file_path, speaker_option):
    tag_map = {
        "child":   ["*CHI:"],
        "parent":  ["*PAR:"],
        "sibling": ["*SIB:"],
        "all":     ["*CHI:", "*PAR:", "*SIB:"]
    }
    target_tags = tag_map.get(speaker_option, tag_map["all"])
    utterances  = {"CHI": [], "PAR": [], "SIB": []}
    sequence = []
    try:
        with open(file_path, encoding='utf-8') as fh:
            for line in fh:
                tag = next((t for t in target_tags if line.startswith(t)), None)
                if not tag:
                    continue
                speaker = tag.strip("*:")      # "*PAR:" → "PAR"
                text = line.split(":", 1)[1].strip()
                text = _clean_utterance(text, TIMESTAMP_BEL_RE)
                if text:
                    utterances[speaker].append(text)
                    sequence.append((speaker, text))
    except Exception as e:
        logging.error(f"Error parsing BEL file {file_path}: {e}")

    return utterances, sequence

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
        cleaned_text = clean_text(text_input)
        df = process_text(cleaned_text)
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
            'original_text': cleaned_text,
            'result_count': result_count,
            'manner_count': manner_count,
            'parsed_data': text_input.splitlines(),  # Change as needed to represent the details
            'result_breakdown': result_breakdown,
            'manner_breakdown': manner_breakdown
        }
        logging.info(f"Text Analysis - Result Count: {result_count}, Manner Count: {manner_count}")
        return jsonify({
            'status': 'success',
            'original_text': cleaned_text,
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
            total_result += analysis["total_result"]    # was "result_count"
            total_manner += analysis["total_manner"]    # was "manner_count"
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
    # chi_dict = parse_childes_file(file_rel)        # now returns dict
    # ---------- unpack tuple ----------
    utterances, sequence = parse_childes_file(file_rel)

    # rebuild text in original order  (sequence = [(spk, utterance), …])
    flat = [f"{spk}: {utt}" for spk, utt in sequence]
    # flat = []
    # for spk, lines in chi_dict.items():
    #     flat.extend([f"{spk}: {u}" for u in lines])

    html_content = (
        f"<html><head><title>Parsed Data - {file_rel}</title></head><body>"
        f"<h2>Parsed Utterances for {file_rel}</h2><pre>"
        + "\n".join(flat) +
        "</pre></body></html>"
    )
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
    
    # ---------- Summary sheet ----------
    summary_rows = []
    for file_rel, a in analysis_cache.items():
        row = {
            "File": file_rel,
            "Group": a.get("group", ""),
            "Gender": a.get("gender", ""),
            "Total Result": a.get("total_result", 0),
            "Total Manner": a.get("total_manner", 0)
        }
        for spk in ("CHI", "MOT", "INV"):
            stats = a.get("speakers", {}).get(spk, {})
            row[f"{spk}_Result"] = stats.get("result_count", 0)
            row[f"{spk}_Manner"] = stats.get("manner_count", 0)
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

    # ---------- One sheet per file ----------
    for file_rel, a in analysis_cache.items():
        sheet_name = file_rel.replace("/", "_").split('.')[0]

        # flattened transcript "TAG: utterance"
        # flat_lines = []
        flat_lines = [f"{spk}: {utt}" for spk, utt in a["sequence"]]
        for spk, lines in a["parsed_data"].items():
            flat_lines.extend([f"{spk}: {u}" for u in lines])

        pd.DataFrame({"Transcript": flat_lines}).to_excel(
            writer, sheet_name=sheet_name, index=False
        )

    writer.close()
    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="childes_analysis.xlsx"
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

    # 3. parse & analyse per speaker
    speaker_lines, seq = parse_bel_file(saved_path, speaker_option)           # dict{CHI/PAR/SIB : [lines]}
    speaker_stats = {spk: _analyse_utterances(lines)                    # dict{…: {result_count, …}}
                     for spk, lines in speaker_lines.items() if lines}

    total_result = sum(v['result_count'] for v in speaker_stats.values())
    total_manner = sum(v['manner_count'] for v in speaker_stats.values())

    # token-level table (combined text – still handy for the UI)
    combined_text = "\n".join(
        line for lines in speaker_lines.values() for line in lines
    )
    df = process_text(combined_text)

    # 4. cache
    bel_analysis.update({
        'speakers':        speaker_stats,
        'total_result':    total_result,
        'total_manner':    total_manner,
        'parsed_data':     speaker_lines,      # keep raw utterances per speaker
        'sequence':        seq
    })

    return jsonify({
        'status': 'success',
        'speakers': speaker_stats,             # ← JSON now exposes per-speaker counts
        'overall': {
            'result_count': total_result,
            'manner_count': total_manner
        },
        'results': df.to_dict(orient='records'),
        'sequence': seq
    }), 200


@app.route('/download_bel_analysis', methods=['GET'])
def download_bel_analysis():
    if not bel_analysis:
        return "No analysis available.", 400

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # ---------- Summary ----------
    speakers      = bel_analysis.get("speakers", {})
    total_result  = bel_analysis.get("total_result", 0)
    total_manner  = bel_analysis.get("total_manner", 0)

    row = {
        "Total Result":  total_result,
        "Total Manner":  total_manner,
        # per-speaker counts
        "CHI_Result":    speakers.get("CHI", {}).get("result_count", 0),
        "CHI_Manner":    speakers.get("CHI", {}).get("manner_count", 0),
        "PAR_Result":    speakers.get("PAR", {}).get("result_count", 0),
        "PAR_Manner":    speakers.get("PAR", {}).get("manner_count", 0),
        "SIB_Result":    speakers.get("SIB", {}).get("result_count", 0),
        "SIB_Manner":    speakers.get("SIB", {}).get("manner_count", 0),
    }
    pd.DataFrame([row]).to_excel(writer, sheet_name="Summary", index=False)

    # ---------- Details ----------
    # flatten to "TAG: utterance" so reviewers can see who said what
    # flat_lines = []
    flat_lines = [f"{spk}: {utt}" for spk, utt in bel_analysis["sequence"]]
    for spk, lines in bel_analysis.get("parsed_data", {}).items():
        flat_lines.extend([f"{spk}: {u}" for u in lines])

    pd.DataFrame({"Transcript": flat_lines}).to_excel(
        writer, sheet_name="Details", index=False
    )

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
