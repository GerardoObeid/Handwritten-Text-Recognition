import os
import cv2
import random
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PATH CONFIGURATION (Relative to project root)
# ==========================================
xml_dir = 'iam/xml'
forms_dir = 'iam/forms'

out_lines_dir = 'iam/lines'
out_semi_dir = 'iam/semi_paragraphs'
out_full_dir = 'iam/paragraphs'
labels_dir = 'iam/labels'

# Create directories if they don't exist
for d in [out_lines_dir, out_semi_dir, out_full_dir, labels_dir]:
    os.makedirs(d, exist_ok=True)

# Lists to store the data (we include form_id to make the splits later)
data_lines = []
data_semi = []
data_full = []

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_crop_coords(elements, w_img, h_img, pad=20):
    """Calculates the coordinates (x1, y1, x2, y2) to crop a text block."""
    all_cmps = []
    for el in elements:
        all_cmps.extend(el.findall('.//cmp'))
        
    if not all_cmps: return None
    
    try:
        xs = [int(c.attrib['x']) for c in all_cmps]
        ys = [int(c.attrib['y']) for c in all_cmps]
        ws = [int(c.attrib['width']) for c in all_cmps]
        hs = [int(c.attrib['height']) for c in all_cmps]
        
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w_img, max([x+w for x, w in zip(xs, ws)]) + pad)
        y2 = min(h_img, max([y+h for y, h in zip(ys, hs)]) + pad)
        
        return (x1, y1, x2, y2)
    except (KeyError, ValueError):
        return None

# ==========================================
# 3. IMAGE AND TEXT EXTRACTION
# ==========================================
print("Processing XMLs and generating image crops (Lines, Semi, Full)...")
archivos_xml = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

for xml_file in tqdm(archivos_xml):
    form_id = xml_file.replace('.xml', '')
    img_path = os.path.join(forms_dir, f"{form_id}.png")
    
    if not os.path.exists(img_path): continue
    
    try:
        root = ET.parse(os.path.join(xml_dir, xml_file)).getroot()
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
    except Exception as e: 
        print(f"Error processing {xml_file}: {e}")
        continue

    handwritten_part = root.find('.//handwritten-part')
    if handwritten_part is None: continue
    
    lines_elements = handwritten_part.findall('.//line')
    if not lines_elements: continue

    # --- PHASE 1: INDIVIDUAL LINES ---
    for i, line in enumerate(lines_elements):
        coords = get_crop_coords([line], w_img, h_img)
        if coords:
            x1, y1, x2, y2 = coords
            if y2 > y1 and x2 > x1:
                line_img = img[y1:y2, x1:x2]
                line_name = f"{form_id}_line_{i}.png"
                cv2.imwrite(os.path.join(out_lines_dir, line_name), line_img)
                
                text = line.attrib.get('text', '').replace('|', ' ')
                data_lines.append({'image_id': line_name, 'text': text, 'form_id': form_id})

    # --- PHASE 2: SEMI-PARAGRAPHS (Strictly 2 or 3 lines) ---
    if len(lines_elements) >= 2:
        num_to_pick = random.choice([2, 3]) if len(lines_elements) >= 3 else 2
        start_idx = random.randint(0, len(lines_elements) - num_to_pick)
        semi_slice = lines_elements[start_idx : start_idx + num_to_pick]
        
        coords_semi = get_crop_coords(semi_slice, w_img, h_img)
        if coords_semi:
            x1, y1, x2, y2 = coords_semi
            if y2 > y1 and x2 > x1:
                semi_img = img[y1:y2, x1:x2]
                semi_name = f"{form_id}_semi.png"
                cv2.imwrite(os.path.join(out_semi_dir, semi_name), semi_img)
                
                text_semi = '\n'.join([l.attrib.get('text', '').replace('|', ' ') for l in semi_slice])
                data_semi.append({'image_id': semi_name, 'text': text_semi, 'form_id': form_id})

    # --- PHASE 3: FULL PARAGRAPH ---
    coords_full = get_crop_coords(lines_elements, w_img, h_img)
    if coords_full:
        x1, y1, x2, y2 = coords_full
        if y2 > y1 and x2 > x1:
            full_img = img[y1:y2, x1:x2]
            full_name = f"{form_id}_full.png"
            cv2.imwrite(os.path.join(out_full_dir, full_name), full_img)
            
            text_full = '\n'.join([l.attrib.get('text', '').replace('|', ' ') for l in lines_elements])
            data_full.append({'image_id': full_name, 'text': text_full, 'form_id': form_id})

print(f"\nExtraction complete: {len(data_lines)} lines, {len(data_semi)} semi-paras, {len(data_full)} full paras.")

# ==========================================
# 4. CSV CREATION AND SPLITS (Train/Val/Test)
# ==========================================
print("\nGenerating splits (80% Train, 10% Val, 10% Test) based on form_id...")

df_lines = pd.DataFrame(data_lines).dropna(subset=['text'])
df_semi = pd.DataFrame(data_semi).dropna(subset=['text'])
df_full = pd.DataFrame(data_full).dropna(subset=['text'])

# Extract all unique form_ids (to ensure the same writer is not in train and test)
unique_forms = df_lines['form_id'].unique()

# Perform the split using form_ids
train_forms, temp_forms = train_test_split(unique_forms, test_size=0.20, random_state=42)
val_forms, test_forms = train_test_split(temp_forms, test_size=0.50, random_state=42)

def save_splits(df, prefix):
    """Filters the DataFrame by form_id, drops the form_id column, and saves to CSV"""
    train = df[df['form_id'].isin(train_forms)].drop(columns=['form_id'])
    val = df[df['form_id'].isin(val_forms)].drop(columns=['form_id'])
    test = df[df['form_id'].isin(test_forms)].drop(columns=['form_id'])
    
    train.to_csv(os.path.join(labels_dir, f'train_{prefix}.csv'), index=False)
    val.to_csv(os.path.join(labels_dir, f'val_{prefix}.csv'), index=False)
    test.to_csv(os.path.join(labels_dir, f'test_{prefix}.csv'), index=False)
    
    print(f"[{prefix.upper()}] -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

save_splits(df_lines, 'lines')
save_splits(df_semi, 'semi')
save_splits(df_full, 'full')

print("\nData preparation process completed successfully! ")