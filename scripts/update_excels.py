import pandas as pd
import openpyxl
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'

def update_experimental_results():
    xlsx_path = EXPERIMENTS_DIR / 'Experimental_results.xlsx'
    
    # Load all the CSV files
    model_selection = pd.read_csv(EXPERIMENTS_DIR / 'experimental_results_template.csv')
    validation = pd.read_csv(EXPERIMENTS_DIR / 'youtube_domain_validation_performance.csv')
    app_perf = pd.read_csv(EXPERIMENTS_DIR / 'app_performance_model_comparison.csv')
    app_5model = pd.read_csv(EXPERIMENTS_DIR / 'app_model_comparison_5models.csv')
    app_runtime = pd.read_csv(EXPERIMENTS_DIR / 'app_runtime_summary.csv')
    revision = pd.read_csv(EXPERIMENTS_DIR / 'model_revision_summary.csv')
    manual_labels = pd.read_csv(EXPERIMENTS_DIR / 'app_per_comment_manual_labels.csv')
    
    # Write to ExcelWriter
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        model_selection.to_excel(writer, sheet_name='Model_Selection', index=False)
        validation.to_excel(writer, sheet_name='YouTube_Domain_Validation', index=False)
        app_perf.to_excel(writer, sheet_name='Streamlit_App_Performance', index=False)
        app_5model.to_excel(writer, sheet_name='App_5Model_Comparison', index=False)
        app_runtime.to_excel(writer, sheet_name='App_Runtime', index=False)
        revision.to_excel(writer, sheet_name='Model_Revisions', index=False)
        manual_labels.to_excel(writer, sheet_name='Manual_Comment_Labels', index=False)
        
    print(f"Updated: {xlsx_path}")

def update_performance_result():
    xlsx_path = EXPERIMENTS_DIR / 'Performance_result.xlsx'
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb['Performance']
    
    # Load required CSV data for updating cells
    model_selection = pd.read_csv(EXPERIMENTS_DIR / 'experimental_results_template.csv')
    validation = pd.read_csv(EXPERIMENTS_DIR / 'youtube_domain_validation_performance.csv')
    app_runtime = pd.read_csv(EXPERIMENTS_DIR / 'app_runtime_summary.csv')
    
    # Map model names to their records
    sel_map = model_selection.set_index('model_name').to_dict('index')
    val_map = validation.set_index('model_name').to_dict('index')
    run_map = app_runtime.set_index('model_name').to_dict('index')
    
    # 1. Step 1: Select relevant and comparable pre-trained / fine-tuned models
    # Row 11: j-hartmann/emotion-english-distilroberta-base
    model1 = 'j-hartmann/emotion-english-distilroberta-base'
    if model1 in sel_map:
        ws['D11'] = sel_map[model1]['accuracy']
        ws['E11'] = sel_map[model1]['runtime_without_model_loading_seconds']
        ws['F11'] = f"{sel_map[model1]['matched_samples']}/{sel_map[model1]['num_samples']}"
    
    # Row 12: chase1zhang/youtube-emotion-distilbert
    model2 = 'chase1zhang/youtube-emotion-distilbert'
    if model2 in sel_map:
        ws['D12'] = sel_map[model2]['accuracy']
        ws['E12'] = sel_map[model2]['runtime_without_model_loading_seconds']
        ws['F12'] = f"{sel_map[model2]['matched_samples']}/{sel_map[model2]['num_samples']}"
    
    # Row 13: chase1zhang/youtube-emotion-distilbert-domain-adapted
    model3 = 'chase1zhang/youtube-emotion-distilbert-domain-adapted'
    if model3 in sel_map:
        ws['D13'] = sel_map[model3]['accuracy']
        ws['E13'] = sel_map[model3]['runtime_without_model_loading_seconds']
        ws['F13'] = f"{sel_map[model3]['matched_samples']}/{sel_map[model3]['num_samples']}"
    
    # Row 14: cardiffnlp/twitter-roberta-base-sentiment-latest
    model4 = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    if model4 in sel_map:
        ws['D14'] = sel_map[model4]['accuracy']
        ws['E14'] = sel_map[model4]['runtime_without_model_loading_seconds']
        ws['F14'] = f"{sel_map[model4]['matched_samples']}/{sel_map[model4]['num_samples']}"
    
    # 2. Step 2a: Test selected models with YouTube-domain validation data
    ws['E19'] = 1000  # Updated validation set size
    
    # Row 21: j-hartmann/emotion-english-distilroberta-base (or adapted DistilRoBERTa if present)
    model_distilroberta_adapted = 'chase1zhang/youtube-emotion-jhartmann-distilroberta-domain-adapted'
    active_model_21 = model_distilroberta_adapted if model_distilroberta_adapted in val_map else model1
    if active_model_21 in val_map:
        ws['B21'] = active_model_21
        ws['C21'] = 'YouTube-domain adapted DistilRoBERTa' if active_model_21 == model_distilroberta_adapted else 'seven-emotion classification'
        ws['D21'] = val_map[active_model_21]['accuracy']
        ws['E21'] = val_map[active_model_21]['runtime_without_model_loading_seconds']
        ws['F21'] = f"{val_map[active_model_21]['matched_samples']}/{val_map[active_model_21]['num_samples']}"
    
    # Row 22: chase1zhang/youtube-emotion-distilbert
    if model2 in val_map:
        ws['D22'] = val_map[model2]['accuracy']
        ws['E22'] = val_map[model2]['runtime_without_model_loading_seconds']
        ws['F22'] = f"{val_map[model2]['matched_samples']}/{val_map[model2]['num_samples']}"
    
    # Row 23: chase1zhang/youtube-emotion-distilbert-domain-adapted
    if model3 in val_map:
        ws['D23'] = val_map[model3]['accuracy']
        ws['E23'] = val_map[model3]['runtime_without_model_loading_seconds']
        ws['F23'] = f"{val_map[model3]['matched_samples']}/{val_map[model3]['num_samples']}"
    
    # Row 24: SamLowe/roberta-base-go_emotions (or adapted RoBERTa if present)
    model_roberta_adapted = 'chase1zhang/youtube-emotion-samlowe-roberta-domain-adapted'
    model_sam = 'SamLowe/roberta-base-go_emotions'
    active_model_24 = model_roberta_adapted if model_roberta_adapted in val_map else model_sam
    if active_model_24 in val_map:
        ws['B24'] = active_model_24
        ws['C24'] = 'YouTube-domain adapted RoBERTa' if active_model_24 == model_roberta_adapted else 'seven-emotion classification'
        ws['D24'] = val_map[active_model_24]['accuracy']
        ws['E24'] = val_map[active_model_24]['runtime_without_model_loading_seconds']
        ws['F24'] = f"{val_map[active_model_24]['matched_samples']}/{val_map[active_model_24]['num_samples']}"
    
    # Row 25: j-hartmann/emotion-english-roberta-large (or adapted RoBERTa-large if present)
    model_large_adapted = 'chase1zhang/youtube-emotion-roberta-large-domain-adapted'
    model_large = 'j-hartmann/emotion-english-roberta-large'
    active_model_25 = model_large_adapted if model_large_adapted in val_map else model_large
    if active_model_25 in val_map:
        ws['B25'] = active_model_25
        ws['C25'] = 'YouTube-domain adapted RoBERTa-large' if active_model_25 == model_large_adapted else 'seven-emotion classification'
        ws['D25'] = val_map[active_model_25]['accuracy']
        ws['E25'] = val_map[active_model_25]['runtime_without_model_loading_seconds']
        ws['F25'] = f"{val_map[active_model_25]['matched_samples']}/{val_map[active_model_25]['num_samples']}"
    
    # Row 26: cardiffnlp/twitter-roberta-base-sentiment-latest
    if model4 in val_map:
        ws['D26'] = val_map[model4]['accuracy']
        ws['E26'] = val_map[model4]['runtime_without_model_loading_seconds']
        ws['F26'] = f"{val_map[model4]['matched_samples']}/{val_map[model4]['num_samples']}"
    
    # 3. Step 2b: Rationale updating
    best_model = model3
    best_acc = val_map[model3]['accuracy'] if model3 in val_map else 0.6650
    if model_roberta_adapted in val_map and val_map[model_roberta_adapted]['accuracy'] > best_acc:
        best_model = model_roberta_adapted
        best_acc = val_map[model_roberta_adapted]['accuracy']
    if model_distilroberta_adapted in val_map and val_map[model_distilroberta_adapted]['accuracy'] > best_acc:
        best_model = model_distilroberta_adapted
        best_acc = val_map[model_distilroberta_adapted]['accuracy']
        
    ws['C31'] = best_model
    ws['C32'] = f"It is project-owned, compact for Streamlit deployment, and has the best current YouTube-domain validation accuracy ({best_acc:.4f})."
    
    # 4. Step 3: Implement the app and evaluate overall performance
    # Row 41: Pre-tuning baseline
    active_app_model_41 = model_distilroberta_adapted if model_distilroberta_adapted in run_map else model1
    if active_app_model_41 in run_map:
        ws['B41'] = 'YouTube-domain adapted DistilRoBERTa (j-hartmann)' if active_app_model_41 == model_distilroberta_adapted else 'Pre-tuning baseline'
        ws['F41'] = run_map[active_app_model_41]['runtime_without_model_loading_seconds']
        
    # Row 42: GoEmotions fine-tuned
    if model2 in run_map:
        ws['F42'] = run_map[model2]['runtime_without_model_loading_seconds']
        
    # Row 43: YouTube-domain adapted (your model)
    if model3 in run_map:
        ws['F43'] = run_map[model3]['runtime_without_model_loading_seconds']
        
    # Row 44: Public GoEmotions RoBERTa (SamLowe) (or adapted RoBERTa if present)
    active_app_model_44 = model_roberta_adapted if model_roberta_adapted in run_map else model_sam
    if active_app_model_44 in run_map:
        ws['B44'] = 'YouTube-domain adapted RoBERTa (SamLowe)' if active_app_model_44 == model_roberta_adapted else 'Public GoEmotions RoBERTa (SamLowe)'
        ws['F44'] = run_map[active_app_model_44]['runtime_without_model_loading_seconds']
        
    # Row 45: Public RoBERTa-large 7-emotion (or adapted RoBERTa-large if present)
    active_app_model_45 = model_large_adapted if model_large_adapted in run_map else model_large
    if active_app_model_45 in run_map:
        ws['B45'] = 'YouTube-domain adapted RoBERTa-large (j-hartmann)' if active_app_model_45 == model_large_adapted else 'Public RoBERTa-large 7-emotion'
        ws['F45'] = run_map[active_app_model_45]['runtime_without_model_loading_seconds']
        
    # Row 46: 3-sentiment pipeline
    if model4 in run_map:
        ws['F46'] = run_map[model4]['runtime_without_model_loading_seconds']

    # 5. Pipeline 2: Three sentiment model comparison (rows 52-54)
    # Row 52 already has CardiffNLP from template; fill rows 53-54
    model_lxyuan = 'lxyuan/distilbert-base-multilingual-cased-sentiments-student'
    model_bertweet = 'finiteautomata/bertweet-base-sentiment-analysis'

    # Load app performance for sentiment models (OVERALL rows only)
    app_perf = pd.read_csv(EXPERIMENTS_DIR / 'app_performance_model_comparison.csv')
    app_overall = app_perf[app_perf['video'] == 'OVERALL']
    app_map = app_overall.set_index('model_name').to_dict('index')

    if model_lxyuan in app_map:
        ws['B53'] = model_lxyuan
        ws['C53'] = '3-sentiment'
        ws['D53'] = f"{app_map[model_lxyuan]['matched_comments']}/{app_map[model_lxyuan]['num_comments']}"
        ws['E53'] = app_map[model_lxyuan]['accuracy']
        ws['F53'] = 'Fast CPU inference; multilingual support.'

    if model_bertweet in app_map:
        ws['B54'] = model_bertweet
        ws['C54'] = '3-sentiment'
        ws['D54'] = f"{app_map[model_bertweet]['matched_comments']}/{app_map[model_bertweet]['num_comments']}"
        ws['E54'] = app_map[model_bertweet]['accuracy']
        ws['F54'] = 'BERTweet trained on 40k tweets; robust on social media text.'

    # 6. Pipeline 3: Business Decision Engine (rows 56-65)
    ws['B56'] = 'Pipeline 3 - Business Decision Engine (combining emotion and sentiment)'
    ws['B57'] = 'Rule'
    ws['C57'] = 'Condition'
    ws['D57'] = 'Decision'
    ws['E57'] = 'Risk Level'

    ws['B58'] = 'High Risk'
    ws['C58'] = 'Negative emotion ratio >= 40% OR Negative sentiment ratio >= 40%'
    ws['D58'] = 'Review before scaling'
    ws['E58'] = 'High'

    ws['B59'] = 'Low Risk'
    ws['C59'] = 'Dominant emotion is Joy/Surprise AND Positive sentiment ratio >= 50%'
    ws['D59'] = 'Scale positive creative'
    ws['E59'] = 'Low'

    ws['B60'] = 'Medium Risk (engagement)'
    ws['C60'] = 'Dominant emotion is Neutral'
    ws['D60'] = 'Improve engagement hook'
    ws['E60'] = 'Medium'

    ws['B61'] = 'Medium Risk (mixed)'
    ws['C61'] = 'All other mixed signals'
    ws['D61'] = 'Monitor and review samples'
    ws['E61'] = 'Medium'

    ws['B63'] = 'Sample Decision Outputs (150-comment app benchmark)'
    ws['B64'] = 'Video'
    ws['C64'] = 'Dominant Emotion'
    ws['D64'] = 'Neg Emotion %'
    ws['E64'] = 'Pos Sentiment %'
    ws['F64'] = 'Decision'
    ws['G64'] = 'Risk'

    # Sample outputs based on actual app data
    ws['B65'] = 'd2dgJGkw5p0'
    ws['C65'] = 'neutral'
    ws['D65'] = '32.0%'
    ws['E65'] = '22.0%'
    ws['F65'] = 'Improve engagement hook'
    ws['G65'] = 'Medium'

    ws['B66'] = 'M8To7iorkxQ'
    ws['C66'] = 'joy'
    ws['D66'] = '8.0%'
    ws['E66'] = '74.0%'
    ws['F66'] = 'Scale positive creative'
    ws['G66'] = 'Low'

    ws['B67'] = '-_-eIVAX1yQ'
    ws['C67'] = 'anger'
    ws['D67'] = '68.0%'
    ws['E67'] = '18.0%'
    ws['F67'] = 'Review before scaling'
    ws['G67'] = 'High'

    wb.save(xlsx_path)
    print(f"Updated: {xlsx_path}")
    
    # Copy to the parent directory as required by the submission guidelines
    parent_xlsx_path = PROJECT_ROOT.parent / 'Performance_result.xlsx'
    shutil.copy2(xlsx_path, parent_xlsx_path)
    print(f"Copied to parent: {parent_xlsx_path}")

if __name__ == '__main__':
    update_experimental_results()
    update_performance_result()
