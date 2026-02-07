from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import joblib

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_data = {
    'model': joblib.load(os.path.join(MODEL_DIR, "model.pkl")),
    'le_dict': joblib.load(os.path.join(MODEL_DIR, "le_dict.pkl")),
    'le_crop': joblib.load(os.path.join(MODEL_DIR, "le_crop.pkl")),
    'crop_averages': joblib.load(os.path.join(MODEL_DIR, "crop_averages.pkl")),
    'feature_names': joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl")),
    'metrics': joblib.load(os.path.join(MODEL_DIR, "metrics.pkl")),
    'feature_importances': joblib.load(os.path.join(MODEL_DIR, "feature_importances.pkl")),
}


@app.route('/')
def index():
    le_dict = model_data.get('le_dict', {})
    mech_classes = []

    if 'Mexanik tarkib' in le_dict:
        mech_classes = le_dict['Mexanik tarkib'].classes_.tolist()

    return render_template('index.html', mech_classes=mech_classes)

@app.route('/predict', methods=['POST'])
def predict():
    """Bashorat qilish"""
    try:
        data = request.get_json()
        
        layer_str = data.get('layer', '0-20')
        layer_value = sum(map(float, layer_str.split('-'))) / 2
        
        mech_comp_str = data.get('mech_comp', '')
        mech_comp_encoded = 0
        if 'Mexanik tarkib' in model_data['le_dict'] and mech_comp_str:
            mech_comp_encoded = model_data['le_dict']['Mexanik tarkib'].transform([mech_comp_str])[0]
        
        input_dict = {
            'Qatlam (sm)': layer_value,
            'Mexanik tarkib': mech_comp_str,
            'DNS (%)': float(data.get('dns', 25)),
            'Tuproq zichligi (g/cm³)': float(data.get('density', 1.4)),
            'pH': float(data.get('ph', 6.5)),
            'EC (mS/cm)': float(data.get('ec', 1.0)),
            'N (mg/kg)': float(data.get('nitrogen', 100)),
            'P (mg/kg)': float(data.get('phosphorus', 50)),
            'K (mg/kg)': float(data.get('potassium', 250)),
            'Gumus (%)': float(data.get('humus', 2.0)),
            'Mg (mg/kg)': float(data.get('mg', 150)),
            'S (mg/kg)': float(data.get('s', 50)),
            'Zn (mg/kg)': float(data.get('zn', 5)),
            'Mn (mg/kg)': float(data.get('mn', 25)),
            'B (mg/kg)': float(data.get('b', 2)),
            'Fe (mg/kg)': float(data.get('fe', 100)),
            'Cu (mg/kg)': float(data.get('cu', 5)),
            'Mikroorganizmlar(CFU/g)': float(data.get('microorg', 1e8))
        }
        
        input_for_pred = {
            'Qatlam (sm)': layer_value,
            'Mexanik tarkib': mech_comp_encoded,
            'DNS (%)': input_dict['DNS (%)'],
            'Tuproq zichligi (g/cm³)': input_dict['Tuproq zichligi (g/cm³)'],
            'pH': input_dict['pH'],
            'EC (mS/cm)': input_dict['EC (mS/cm)'],
            'N (mg/kg)': input_dict['N (mg/kg)'],
            'P (mg/kg)': input_dict['P (mg/kg)'],
            'K (mg/kg)': input_dict['K (mg/kg)'],
            'Gumus (%)': input_dict['Gumus (%)'],
            'Mg (mg/kg)': input_dict['Mg (mg/kg)'],
            'S (mg/kg)': input_dict['S (mg/kg)'],
            'Zn (mg/kg)': input_dict['Zn (mg/kg)'],
            'Mn (mg/kg)': input_dict['Mn (mg/kg)'],
            'B (mg/kg)': input_dict['B (mg/kg)'],
            'Fe (mg/kg)': input_dict['Fe (mg/kg)'],
            'Cu (mg/kg)': input_dict['Cu (mg/kg)'],
            'Mikroorganizmlar(CFU/g)': input_dict['Mikroorganizmlar(CFU/g)']
        }
        
        input_data = [input_for_pred.get(col, 0.0) for col in model_data['feature_names']]
        input_df = pd.DataFrame([input_data], columns=model_data['feature_names'])
        
        probs = model_data['model'].predict_proba(input_df)[0]
        crop_probs = {
            model_data['le_crop'].classes_[i]: min(probs[i] * 100, 100) 
            for i in range(len(model_data['le_crop'].classes_))
        }
        
        df_probs = pd.DataFrame(
            list(crop_probs.items()), 
            columns=['Ekin', 'Moslik (%)']
        ).sort_values('Moslik (%)', ascending=False)
        
        best_crop = df_probs.iloc[0]['Ekin']
        best_prob = df_probs.iloc[0]['Moslik (%)']
        
        # Tabiat ranglari bilan grafik
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#f5f5dc')
        ax.set_facecolor('#faf8f3')
        
        colors = ['#2d5016', '#4a7c2c', '#6b8e23', '#8fbc8f', '#9dc183', '#c8d5b9', '#d4e7c5']
        bar_colors = [colors[i % len(colors)] for i in range(len(df_probs))]
        
        bars = ax.bar(df_probs['Ekin'], df_probs['Moslik (%)'], color=bar_colors, 
                      edgecolor='#2d5016', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Ekin', fontsize=14, fontweight='bold', color='#2d5016', labelpad=10)
        ax.set_ylabel('Moslik (%)', fontsize=14, fontweight='bold', color='#2d5016', labelpad=10)
        ax.set_title('Ekin Mosligi Diagrammasi', fontsize=16, fontweight='bold', 
                     color='#2d5016', pad=20)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, color='#6b8e23', linestyle='--', linewidth=1)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold', color='#2d5016', fontsize=11)
        
        ax.spines['top'].set_color('#6b8e23')
        ax.spines['right'].set_color('#6b8e23')
        ax.spines['bottom'].set_color('#2d5016')
        ax.spines['left'].set_color('#2d5016')
        
        plt.xticks(rotation=45, ha='right', fontsize=11, color='#2d5016')
        plt.yticks(fontsize=11, color='#2d5016')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        graph_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Taqqoslash grafigi
        top_features = model_data['feature_importances'].head(6).index.tolist()
        input_values = [input_for_pred.get(f, 0) for f in top_features]
        avg_values = model_data['crop_averages'].loc[best_crop, top_features].tolist() \
            if best_crop in model_data['crop_averages'].index else [0] * len(top_features)
        
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        fig2.patch.set_facecolor('#f5f5dc')
        ax2.set_facecolor('#faf8f3')
        
        x = np.arange(len(top_features))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, input_values, width, label='Kiritilgan qiymatlar', 
                        color='#6b8e23', edgecolor='#2d5016', linewidth=1.5, alpha=0.8)
        bars2 = ax2.bar(x + width/2, avg_values, width, label=f"{best_crop} O'rtacha", 
                        color='#8fbc8f', edgecolor='#2d5016', linewidth=1.5, alpha=0.8)
        
        ax2.set_xlabel('Xususiyatlar', fontsize=14, fontweight='bold', color='#2d5016', labelpad=10)
        ax2.set_ylabel('Qiymat', fontsize=14, fontweight='bold', color='#2d5016', labelpad=10)
        ax2.set_title(f"{best_crop}ga muhim xususiyatlar", fontsize=16, 
                      fontweight='bold', color='#2d5016', pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_features, rotation=45, ha='right', fontsize=11, color='#2d5016')
        ax2.legend(fontsize=11, framealpha=0.9, edgecolor='#2d5016')
        ax2.grid(axis='y', alpha=0.3, color='#6b8e23', linestyle='--', linewidth=1)
        
        ax2.spines['top'].set_color('#6b8e23')
        ax2.spines['right'].set_color('#6b8e23')
        ax2.spines['bottom'].set_color('#2d5016')
        ax2.spines['left'].set_color('#2d5016')
        
        plt.yticks(fontsize=11, color='#2d5016')
        plt.tight_layout()
        
        buffer2 = BytesIO()
        plt.savefig(buffer2, format='png', dpi=100, bbox_inches='tight')
        buffer2.seek(0)
        comparison_base64 = base64.b64encode(buffer2.read()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'best_crop': best_crop,
            'best_prob': float(best_prob),
            'probabilities': df_probs.to_dict('records'),
            'graph': graph_base64,
            'comparison_graph': comparison_base64,
            'input_dict': input_dict
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/stats')
def get_stats():
    """Statistika"""
    try:
        stats = model_data['df'].describe().to_dict()
        class_counts = model_data['df']['Ekin'].value_counts().to_dict()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#f5f5dc')
        ax.set_facecolor('#faf8f3')
        
        crops = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ['#2d5016', '#4a7c2c', '#6b8e23', '#8fbc8f', '#9dc183', '#c8d5b9', '#d4e7c5']
        bar_colors = [colors[i % len(colors)] for i in range(len(crops))]
        
        bars = ax.bar(crops, counts, color=bar_colors, edgecolor='#2d5016', 
                      linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Ekin', fontsize=14, fontweight='bold', color='#2d5016', labelpad=10)
        ax.set_ylabel('Miqdor', fontsize=14, fontweight='bold', color='#2d5016', labelpad=10)
        ax.set_title('Siniflar Taqsimoti', fontsize=16, fontweight='bold', 
                     color='#2d5016', pad=20)
        ax.grid(axis='y', alpha=0.3, color='#6b8e23', linestyle='--', linewidth=1)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', color='#2d5016', fontsize=11)
        
        ax.spines['top'].set_color('#6b8e23')
        ax.spines['right'].set_color('#6b8e23')
        ax.spines['bottom'].set_color('#2d5016')
        ax.spines['left'].set_color('#2d5016')
        
        plt.xticks(rotation=45, ha='right', fontsize=11, color='#2d5016')
        plt.yticks(fontsize=11, color='#2d5016')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        class_graph = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'class_counts': class_counts,
            'class_graph': class_graph,
            'metrics': model_data['metrics']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8800)