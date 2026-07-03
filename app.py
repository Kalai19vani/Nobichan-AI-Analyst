from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
import io
import os
import base64
import requests
import traceback
from scipy import stats
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

current_df = None
current_filename = None
analysis_cache = {}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

def ask_ollama(prompt, context=""):
    full_prompt = f"{context}\n\nUser question: {prompt}" if context else prompt
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False, "options": {"num_ctx": 512}
        }, timeout=300)
        if response.status_code == 200:
            return response.json().get("response", "No response from Ollama.")
        return f"Ollama error: status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ Ollama is not running. Please open a terminal and run: ollama serve"
    except requests.exceptions.Timeout:
        return "⏳ Nobichan is thinking... Ollama took too long to respond. Try a shorter question or restart Ollama with: ollama serve"
    except Exception as e:
        return f"Error: {str(e)}"

def df_to_context(df, max_rows=3):
    if df is None:
        return ""
    ctx = f"Dataset: {current_filename}, {df.shape[0]} rows, {df.shape[1]} cols\n"
    ctx += f"Columns: {list(df.columns)}\n"
    ctx += f"Missing values: {df.isnull().sum().to_dict()}\n"
    ctx += f"Sample (3 rows):\n{df.head(3).to_string()}\n"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().loc[['mean','min','max']]
        ctx += f"Stats:\n{desc.to_string()}\n"
    return ctx

def analyze_dataframe(df):
    result = {}
    result['shape'] = {'rows': int(df.shape[0]), 'cols': int(df.shape[1])}
    result['columns'] = list(df.columns)
    result['dtypes'] = {col: str(df[col].dtype) for col in df.columns}
    result['missing'] = {col: int(df[col].isnull().sum()) for col in df.columns}
    result['total_missing'] = int(df.isnull().sum().sum())
    result['duplicates'] = int(df.duplicated().sum())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    result['numeric_cols'] = numeric_cols
    result['cat_cols'] = cat_cols
    result['date_cols'] = date_cols
    
    if numeric_cols:
        desc = df[numeric_cols].describe()
        result['numeric_stats'] = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            skewness = float(col_data.skew()) if len(col_data) > 2 else 0
            kurt = float(col_data.kurtosis()) if len(col_data) > 3 else 0
            q1 = float(col_data.quantile(0.25))
            q3 = float(col_data.quantile(0.75))
            iqr = q3 - q1
            outliers = int(((col_data < (q1 - 1.5*iqr)) | (col_data > (q3 + 1.5*iqr))).sum())
            result['numeric_stats'][col] = {
                'mean': float(desc[col]['mean']),
                'std': float(desc[col]['std']),
                'min': float(desc[col]['min']),
                'max': float(desc[col]['max']),
                'median': float(col_data.median()),
                'skewness': skewness,
                'kurtosis': kurt,
                'outliers': outliers
            }
    
    if cat_cols:
        result['cat_stats'] = {}
        for col in cat_cols:
            vc = df[col].value_counts()
            result['cat_stats'][col] = {
                'unique': int(df[col].nunique()),
                'top': str(vc.index[0]) if len(vc) > 0 else 'N/A',
                'top_count': int(vc.iloc[0]) if len(vc) > 0 else 0,
                'top_values': {str(k): int(v) for k, v in vc.head(10).items()}
            }
    
    # Correlation
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        strong_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong_corr.append({
                        'col1': corr.columns[i],
                        'col2': corr.columns[j],
                        'value': round(float(val), 3)
                    })
        result['strong_correlations'] = strong_corr
    
    # Problems detected
    problems = []
    if result['total_missing'] > 0:
        miss_pct = (result['total_missing'] / (df.shape[0] * df.shape[1])) * 100
        problems.append(f"Missing values: {result['total_missing']} cells ({miss_pct:.1f}% of data)")
    if result['duplicates'] > 0:
        problems.append(f"Duplicate rows: {result['duplicates']} rows")
    for col in numeric_cols:
        if result['numeric_stats'][col]['outliers'] > 0:
            problems.append(f"Outliers in '{col}': {result['numeric_stats'][col]['outliers']} values")
        if abs(result['numeric_stats'][col]['skewness']) > 1:
            direction = 'right' if result['numeric_stats'][col]['skewness'] > 0 else 'left'
            problems.append(f"High skewness in '{col}': {result['numeric_stats'][col]['skewness']:.2f} ({direction}-skewed)")
    result['problems'] = problems
    
    # Actions
    actions = []
    if result['total_missing'] > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if col in numeric_cols:
                    actions.append(f"Fill missing values in '{col}' with median/mean")
                else:
                    actions.append(f"Fill missing values in '{col}' with mode or 'Unknown'")
    if result['duplicates'] > 0:
        actions.append("Remove duplicate rows")
    for col in numeric_cols:
        if result['numeric_stats'][col]['outliers'] > 0:
            actions.append(f"Investigate/handle outliers in '{col}'")
    if not actions:
        actions.append("Data looks clean! No immediate action required.")
    result['actions'] = actions
    
    return result

@app.route('/')
def index():
    import os; return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global current_df, current_filename, analysis_cache
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = file.filename
    ext = filename.rsplit('.', 1)[-1].lower()
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        if ext == 'csv':
            # Try different encodings
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(filepath)
        elif ext == 'json':
            df = pd.read_json(filepath)
        elif ext == 'parquet':
            df = pd.read_parquet(filepath)
        elif ext == 'tsv':
            df = pd.read_csv(filepath, sep='\t')
        else:
            return jsonify({'error': f'Unsupported file type: {ext}'}), 400
        
        # Auto-parse dates
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        current_df = df
        current_filename = filename
        analysis_cache = analyze_dataframe(df)
        
        preview = df.head(10).to_dict(orient='records')
        for row in preview:
            for k, v in row.items():
                if pd.isna(v) if not isinstance(v, str) else False:
                    row[k] = None
                elif hasattr(v, 'item'):
                    row[k] = v.item()
                else:
                    try:
                        json.dumps(v)
                    except:
                        row[k] = str(v)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'rows': int(df.shape[0]),
            'cols': int(df.shape[1]),
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'missing': int(df.isnull().sum().sum()),
            'columns': list(df.columns),
            'preview': preview,
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'analysis': analysis_cache
        })
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/clean', methods=['POST'])
def clean_data():
    global current_df, analysis_cache
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.json or {}
    df = current_df.copy()
    actions_done = []
    
    if data.get('remove_duplicates', True):
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            actions_done.append(f"Removed {removed} duplicate rows")
    
    if data.get('fill_missing', True):
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    fill_val = df[col].median()
                    df[col] = df[col].fillna(fill_val)
                    actions_done.append(f"Filled {missing} missing values in '{col}' with median ({fill_val:.2f})")
                else:
                    mode_val = df[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    df[col] = df[col].fillna(fill_val)
                    actions_done.append(f"Filled {missing} missing values in '{col}' with '{fill_val}'")
    
    if data.get('strip_whitespace', True):
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        actions_done.append("Stripped whitespace from text columns")
    
    if data.get('standardize_case', False):
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.lower()
        actions_done.append("Standardized text to lowercase")
    
    current_df = df
    analysis_cache = analyze_dataframe(df)
    
    if not actions_done:
        actions_done.append("Data was already clean — no changes needed!")
    
    return jsonify({
        'success': True,
        'actions': actions_done,
        'new_shape': {'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
        'remaining_missing': int(df.isnull().sum().sum()),
        'analysis': analysis_cache
    })

@app.route('/api/visualize', methods=['POST'])
def visualize():
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    data = request.json or {}
    chart_type = data.get('chart_type', 'histogram')
    x_col = data.get('x_col')
    y_col = data.get('y_col')
    color_col = data.get('color_col')
    
    df = current_df
    
    try:
        if chart_type == 'histogram' and x_col:
            fig = px.histogram(df, x=x_col, color=color_col,
                               template='plotly_dark',
                               color_discrete_sequence=['#EAB308'])
        elif chart_type == 'bar' and x_col and y_col:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                         template='plotly_dark',
                         color_discrete_sequence=['#EAB308'])
        elif chart_type == 'line' and x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, color=color_col,
                          template='plotly_dark',
                          color_discrete_sequence=['#EAB308'])
        elif chart_type == 'scatter' and x_col and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             template='plotly_dark',
                             color_discrete_sequence=['#EAB308'])
        elif chart_type == 'box' and x_col:
            fig = px.box(df, y=x_col, color=color_col,
                         template='plotly_dark',
                         color_discrete_sequence=['#EAB308'])
        elif chart_type == 'pie' and x_col:
            vc = df[x_col].value_counts().head(10)
            fig = px.pie(values=vc.values, names=vc.index,
                         template='plotly_dark',
                         color_discrete_sequence=px.colors.sequential.YlOrBr_r)
        elif chart_type == 'heatmap':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return jsonify({'error': 'Need at least 2 numeric columns for heatmap'}), 400
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, template='plotly_dark',
                            color_continuous_scale='YlOrBr', text_auto=True)
        elif chart_type == 'violin' and x_col:
            fig = px.violin(df, y=x_col, box=True, template='plotly_dark',
                            color_discrete_sequence=['#EAB308'])
        elif chart_type == 'area' and x_col and y_col:
            fig = px.area(df, x=x_col, y=y_col, template='plotly_dark',
                          color_discrete_sequence=['#EAB308'])
        else:
            # Auto chart
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0], template='plotly_dark',
                                   color_discrete_sequence=['#EAB308'])
            else:
                col = df.columns[0]
                vc = df[col].value_counts().head(10)
                fig = px.bar(x=vc.index, y=vc.values, template='plotly_dark',
                             color_discrete_sequence=['#EAB308'])
        
        fig.update_layout(
            paper_bgcolor='rgba(15,10,40,0)',
            plot_bgcolor='rgba(15,10,40,0)',
            font_color='#e2e8f0',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return jsonify({'chart': json.dumps(fig, cls=PlotlyJSONEncoder)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto_visualize', methods=['GET'])
def auto_visualize():
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    df = current_df
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Distribution charts for numeric
    for col in numeric_cols[:3]:
        fig = px.histogram(df, x=col, template='plotly_dark',
                           color_discrete_sequence=['#EAB308'],
                           title=f'Distribution of {col}')
        fig.update_layout(paper_bgcolor='rgba(15,10,40,0.95)', plot_bgcolor='rgba(10,7,30,0.8)',
                          font_color='#f1f0ff')
        charts.append({'title': f'Distribution: {col}', 'chart': json.dumps(fig, cls=PlotlyJSONEncoder)})
    
    # Bar for categorical
    for col in cat_cols[:2]:
        vc = df[col].value_counts().head(10)
        fig = px.bar(x=vc.index, y=vc.values, template='plotly_dark',
                     color_discrete_sequence=['#EAB308'],
                     title=f'Top values in {col}',
                     labels={'x': col, 'y': 'Count'})
        fig.update_layout(paper_bgcolor='rgba(15,10,40,0.95)', plot_bgcolor='rgba(10,7,30,0.8)',
                          font_color='#f1f0ff')
        charts.append({'title': f'Category: {col}', 'chart': json.dumps(fig, cls=PlotlyJSONEncoder)})
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, template='plotly_dark', color_continuous_scale='YlOrBr',
                        title='Correlation Heatmap', text_auto=True)
        fig.update_layout(paper_bgcolor='rgba(15,10,40,0.95)', plot_bgcolor='rgba(10,7,30,0.8)',
                          font_color='#f1f0ff')
        charts.append({'title': 'Correlation Heatmap', 'chart': json.dumps(fig, cls=PlotlyJSONEncoder)})
    
    # Scatter for top 2 numeric
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], template='plotly_dark',
                         color_discrete_sequence=['#FDE047'],
                         title=f'{numeric_cols[0]} vs {numeric_cols[1]}')
        fig.update_layout(paper_bgcolor='rgba(15,10,40,0.95)', plot_bgcolor='rgba(10,7,30,0.8)',
                          font_color='#f1f0ff')
        charts.append({'title': f'Scatter: {numeric_cols[0]} vs {numeric_cols[1]}', 'chart': json.dumps(fig, cls=PlotlyJSONEncoder)})
    
    return jsonify({'charts': charts})

@app.route('/api/ask', methods=['POST'])
def ask_ai():
    if current_df is None:
        return jsonify({'error': 'Please upload a dataset first'}), 400
    
    data = request.json or {}
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    context = f"""You are Nobichan, a data analyst. Answer briefly in 3-5 sentences max.
{df_to_context(current_df)}
Problems: {analysis_cache.get('problems', [])}"""
    
    answer = ask_ollama(question, context)
    return jsonify({'answer': answer})

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    global current_df, analysis_cache, current_filename
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    df = current_df
    analysis = analysis_cache
    
    # Build instant AI report from analysis data (no Ollama wait)
    num_stats = analysis.get('numeric_stats', {})
    findings = []
    for col, s in num_stats.items():
        findings.append(f"{col}: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}, outliers={s['outliers']}")
    
    ai_report = f"""EXECUTIVE SUMMARY
This dataset contains {analysis['shape']['rows']} rows and {analysis['shape']['cols']} columns. 
Total missing values: {analysis['total_missing']}. Duplicate rows: {analysis['duplicates']}.

KEY FINDINGS
""" + "\n".join(f"- {f}" for f in findings) + f"""

PROBLEMS DETECTED
""" + "\n".join(f"- {p}" for p in analysis.get('problems', ['No major problems detected.'])) + f"""

RECOMMENDED ACTIONS
""" + "\n".join(f"{i+1}. {a}" for i,a in enumerate(analysis.get('actions', []))) + """

CONCLUSION
Review the problems and actions above. Clean the data before using it for analysis or modelling."""
    
    # Build PDF
    report_filename = f"Nobichan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = os.path.join(REPORTS_FOLDER, report_filename)
    
    doc = SimpleDocTemplate(report_path, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=24, textColor=colors.HexColor('#CA8A04'),
                                  spaceAfter=6, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                     fontSize=11, textColor=colors.HexColor('#EAB308'),
                                     spaceAfter=20, alignment=TA_CENTER)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading1'],
                                    fontSize=14, textColor=colors.HexColor('#CA8A04'),
                                    spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.HexColor('#1a1a2e'),
                                 spaceAfter=6, leading=16)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
                                   fontSize=10, textColor=colors.HexColor('#1a1a2e'),
                                   spaceAfter=4, leftIndent=20, leading=16)
    
    story = []
    
    # Header
    story.append(Paragraph("🤖 NOBICHAN AI ANALYST", title_style))
    story.append(Paragraph(f"Data Analysis Report — {datetime.now().strftime('%B %d, %Y %H:%M')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#CA8A04')))
    story.append(Spacer(1, 0.2*inch))
    
    # Dataset info table
    story.append(Paragraph("📊 Dataset Overview", heading_style))
    table_data = [
        ['Property', 'Value'],
        ['Filename', str(current_filename)],
        ['Total Rows', str(analysis['shape']['rows'])],
        ['Total Columns', str(analysis['shape']['cols'])],
        ['Numeric Columns', str(len(analysis.get('numeric_cols', [])))],
        ['Categorical Columns', str(len(analysis.get('cat_cols', [])))],
        ['Missing Values', str(analysis['total_missing'])],
        ['Duplicate Rows', str(analysis['duplicates'])],
    ]
    table = Table(table_data, colWidths=[2.5*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#CA8A04')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FEFCE8')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FEFCE8'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FDE047')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2*inch))
    
    # Numeric stats
    if analysis.get('numeric_stats'):
        story.append(Paragraph("📈 Numeric Column Statistics", heading_style))
        num_headers = ['Column', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Outliers', 'Skewness']
        num_data = [num_headers]
        for col, s in analysis['numeric_stats'].items():
            num_data.append([
                col,
                f"{s['mean']:.2f}", f"{s['median']:.2f}", f"{s['std']:.2f}",
                f"{s['min']:.2f}", f"{s['max']:.2f}",
                str(s['outliers']), f"{s['skewness']:.2f}"
            ])
        num_table = Table(num_data, repeatRows=1)
        num_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#CA8A04')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FEFCE8'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FDE047')),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(num_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Problems
    story.append(Paragraph("⚠️ Problems Detected", heading_style))
    problems = analysis.get('problems', [])
    if problems:
        for p in problems:
            story.append(Paragraph(f"• {p}", bullet_style))
    else:
        story.append(Paragraph("✅ No major problems detected. Data appears clean.", body_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Actions
    story.append(Paragraph("🔧 Recommended Actions", heading_style))
    for i, action in enumerate(analysis.get('actions', []), 1):
        story.append(Paragraph(f"{i}. {action}", bullet_style))
    story.append(Spacer(1, 0.1*inch))
    
    # AI Insights
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#EAB308')))
    story.append(Paragraph("🤖 AI-Generated Insights (Nobichan)", heading_style))
    # Split by lines and add paragraphs
    for line in ai_report.split('\n'):
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))
    
    # Footer
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CA8A04')))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
                                   fontSize=8, textColor=colors.HexColor('#EAB308'),
                                   alignment=TA_CENTER, spaceBefore=8)
    story.append(Paragraph(f"Generated by Nobichan AI Analyst v3.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by Ollama LLaMA3", footer_style))
    
    doc.build(story)
    
    return send_file(report_path, as_attachment=True, download_name=report_filename,
                     mimetype='application/pdf')

@app.route('/api/export_clean', methods=['GET'])
def export_clean():
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    fmt = request.args.get('format', 'csv')
    if fmt == 'csv':
        output = io.StringIO()
        current_df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"cleaned_{current_filename.rsplit('.', 1)[0]}.csv"
        )
    elif fmt == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            current_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True,
                         download_name=f"cleaned_{current_filename.rsplit('.', 1)[0]}.xlsx")

@app.route('/api/status', methods=['GET'])
def status():
    ollama_ok = False
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except:
        pass
    
    return jsonify({
        'ollama': ollama_ok,
        'data_loaded': current_df is not None,
        'filename': current_filename,
        'rows': int(current_df.shape[0]) if current_df is not None else 0,
        'cols': int(current_df.shape[1]) if current_df is not None else 0
    })

@app.route('/api/analysis', methods=['GET'])
def get_analysis():
    if current_df is None:
        return jsonify({'error': 'No data loaded'}), 400
    return jsonify(analysis_cache)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  🤖 NOBICHAN AI ANALYST v3.0")
    print("  Starting server...")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
