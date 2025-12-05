import random
import time
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DB_NAME = "vectordb"
DB_USER = "pguser"
DB_PASS = "pgpass"
DB_HOST = "localhost"

MODELS = [
    {"name": "all-MiniLM-L6-v2", "dim": 384, "table": "bench_minilm"},
    {"name": "all-mpnet-base-v2", "dim": 768, "table": "bench_mpnet"},
    {"name": "intfloat/e5-large-v2", "dim": 1024, "table": "bench_e5"},
    {"name": "BAAI/bge-large-en-v1.5", "dim": 1024, "table": "bench_bge"},
]

# --- DATA ---
REALISTIC_EVENTS = [
    ("Koncert muzyki renesansowej w Bazylice Mariackiej",
     "Wieczorny koncert poświęcony muzyce renesansowej wykonywany przez europejską orkiestrę kameralną. Usłyszycie utwory kompozytorów XVI i XVII wieku w oryginalnych aranżacjach. Bazylika Mariacka zapewnia wyjątkową akustykę, która podkreśla piękno dawnej muzyki. Koncert rozpoczyna się o godzinie 19:00 i trwa około 90 minut. Po koncercie zaplanowano krótkie spotkanie z muzykami.",
     "kultura_wysoka, muzyka_klasyczna, renesans",
     "Bazylika Mariacka"),

    ("Barokowy wieczór skrzypcowy w Dworze Artusa",
     "Recital muzyki baroku na skrzypcach solo, wzbogacony krótkim wprowadzeniem historycznym. Solistka zaprezentuje utwory Bacha, Vivaldiego i Telemanna. Każdy utwór poprzedzony będzie opowieścią o kontekście jego powstania. Dwór Artusa to idealne miejsce na kameralny koncert w historycznym otoczeniu. Wstęp wolny, ale liczba miejsc ograniczona.",
     "kultura_wysoka, barok, muzyka_klasyczna",
     "Dwór Artusa"),

    ("Spotkanie startupów AI w Gdańskim Inkubatorze",
     "Networkingowe spotkanie dla twórców projektów opartych o sztuczną inteligencję. Prezentacje MVP, panel ekspertów z branży tech oraz sesje Q&A. To doskonała okazja do wymiany doświadczeń i nawiązania współpracy. Spotkanie skierowane jest zarówno do początkujących, jak i zaawansowanych twórców. Zapewnione są przekąski i napoje.",
     "technologia, ai, startupy",
     "Intel"),

    ("Hackathon rozwiązań smart-city w Olivia Centre",
     "24-godzinny hackathon poświęcony projektowaniu aplikacji dla inteligentnych miast. Tematyka obejmuje mobilność miejską, zarządzanie energią i bezpieczeństwo publiczne. Uczestnicy pracują w zespołach nad realnymi wyzwaniami zgłoszonymi przez miasto. Najlepsze projekty otrzymają nagrody i możliwość dalszego rozwoju. Zapewnione są posiłki, mentorzy i infrastruktura techniczna.",
     "technologia, smart_city, innowacje",
     "Olivia Centre"),

    ("Koncert rockowy na dachu",
     "Energetyczny wieczór z zespołem grającym rocka na dachu starej kamienicy. W programie klasyki rocka oraz autorskie kompozycje. Atmosfera koncertowa wzmocniona widokiem na panoramę miasta. Dostępne piwo kraftowe i przekąski z foodtrucków. Koncert rozpoczyna się o zmierzchu i trwa do późnych godzin nocnych.",
     "rozrywka, koncert, rock",
     "Klub Pod Chmurką"),

    ("Koncert rokowy w parku",
     "Nietypowa nazwa, ale za to pełen rockowy set — lokalne zespoły zapraszają. Występy trzech kapel grających różne odmiany rocka. Koncert odbywa się na świeżym powietrzu w parku miejskim. Wstęp wolny, mile widziane dobrowolne datki dla artystów. Zabierzcie koce i dobre humory na letni wieczór z muzyką.",
     "rozrywka, koncert, plener",
     "Park Miejski"),

    ("Zespół gra rocka w Piwnicy",
     "Lokalny skład prezentuje klasykę rocka i własne kompozycje w kameralnej piwnicy. Kameralna przestrzeń zapewnia bliski kontakt z muzykami. W repertuarze znajdziecie covery Led Zeppelin, Pink Floyd i Deep Purple. Koncert to także okazja do poznania młodych, utalentowanych muzyków. Rezerwacja miejsc zalecana ze względu na ograniczoną przestrzeń.",
     "rozrywka, muzyka, rock",
     "Piwnica 13"),

    ("Maraton filmów retro",
     "Przegląd starych filmów 35mm — kino klasyczne, dyskusje po seansach. Projekcje obejmują klasyki kina europejskiego i amerykańskiego z lat 50-70. Każdy film poprzedzony jest krótkim wprowadzeniem filmoznawcy. Po seansach odbywają się dyskusje przy kawie i ciastkach. Maraton trwa cały weekend, można kupić bilety na pojedyncze seanse lub karnet.",
     "kultura, kino, retro",
     "Kino Retro"),

    ("Wieczór poezji i improwizacji",
     "Spotkanie poetów i muzyków — poezja czytana przy improwizowanej gitarze. Artyści prezentują własną twórczość oraz interpretacje klasyków. Publiczność może włączyć się do dyskusji o poezji współczesnej. Kameralna atmosfera galerii sprzyja intymnym przeżyciom artystycznym. Wstęp za symboliczną opłatą, zbiórka na rozwój galerii.",
     "kultura_wysoka, poezja, improwizacja",
     "Galeria Słów"),

    ("Turniej piłkarski młodzieży U-16",
     "Dwudniowy turniej drużyn piłkarskich z całego regionu, mecze i strefa kibica. Udział bierze 16 zespołów walczących o puchar i nagrody indywidualne. Organizatorzy przygotowali strefę dla kibiców z atrakcjami i gastronomią. Finał turnieju odbędzie się w niedzielę o godzinie 15:00. Wstęp wolny, zapraszamy rodziny i sympatyków piłki nożnej.",
     "sport, pilka_nozna, mlodziez",
     "Miejski Ośrodek Sportu"),

    ("Bieg nocą — latarnie i latwy dystans",
     "5km bieg miejski po zmroku z neonami i muzyką na trasie. Trasa wiedzie przez centrum miasta, oświetlona kolorowymi lampionami. Na mecie czeka ciepły posiłek i pamiątkowy medal dla wszystkich uczestników. Bieg jest otwarty dla każdego, niezależnie od poziomu zaawansowania. Zapisy online lub na miejscu godzinę przed startem.",
     "sport, bieg, nocny",
     "Fundacja Biegamy Razem"),

    ("Warsztaty wprowadzające do Pythona dla dorosłych",
     "Dwa spotkania po 90 minut — podstawy Pythona, instalacja i pierwsze skrypty. Nauczysz się pisać proste programy i zrozumiesz podstawowe koncepcje programowania. Warsztaty prowadzone są w małych grupach, każdy uczestnik ma dostęp do komputera. Materiały i zadania dostępne są online po zakończeniu kursu. Nie wymagamy wcześniejszego doświadczenia w programowaniu.",
     "technologia, python, warsztaty",
     "Biblioteka Uniwersytecka"),

    ("Warsztaty Pythona — workshop intro",
     "Krótkie wprowadzenie praktyczne: zmienne, pętle, prosty projekt. W ciągu trzech godzin stworzymy razem działającą aplikację konsolową. Warsztat jest interaktywny, uczestnicy kodują na bieżąco wraz z prowadzącym. Idealne dla osób, które chcą sprawdzić, czy programowanie jest dla nich. Zapewnione laptopy i dostęp do internetu.",
     "technologia, python, programowanie",
     "Hackerspace"),

    ("Meetup Data Science w coworku",
     "Prezentacje projektów data science, case study, networking. Eksperci z branży dzielą się swoimi doświadczeniami w analizie danych. Omówimy najnowsze narzędzia i techniki uczenia maszynowego. Po prezentacjach czas na networking przy pizzy i napojach. Spotkanie otwarte dla wszystkich zainteresowanych data science.",
     "technologia, data_science, meetup",
     "Cowork Gdansk"),

    ("Degustacja piw rzemieślniczych",
     "Spotkanie z lokalnymi browarami, opowieści o stylach i pairing z przekąskami. Spróbujecie sześciu różnych piw kraftowych z regionu. Każde piwo omówione przez browara — składniki, proces warzenia, charakterystyka smakowa. Do degustacji serwowane są specjalnie dobrane przekąski. Wydarzenie dla osób pełnoletnich, liczba miejsc ograniczona.",
     "gastronomia, degustacja, piwo",
     "Browar Lokalny"),

    ("Kurs pierwszej pomocy dla rodziców",
     "Praktyczny kurs z fantomami — resuscytacja, opatrywanie ran, bezpieczeństwo domowe. Nauczysz się reagować w sytuacjach zagrożenia życia dziecka. Kurs prowadzony przez ratowników medycznych z wieloletnim doświadczeniem. Każdy uczestnik otrzyma certyfikat ukończenia kursu. Zajęcia trwają 6 godzin z przerwą na lunch.",
     "zdrowie, pierwsza_pomoc, kurs",
     "Centrum Zdrowia"),

    ("Szkolenie z zarządzania projektami Agile",
     "Jednodniowe szkolenie z podstaw Agile, role, artefakty i praktyczne ćwiczenia. Poznasz metodyki Scrum i Kanban oraz nauczysz się je stosować w praktyce. Szkolenie obejmuje warsztaty grupowe i symulacje rzeczywistych projektów. Prowadzący to certyfikowany Scrum Master z doświadczeniem w branży IT. Uczestnicy otrzymują materiały szkoleniowe i certyfikat.",
     "biznes, agile, szkolenie",
     "SzkoleniaPro"),

    ("Festyn rodzinny na plaży",
     "Dzień pełen atrakcji dla rodzin: animacje, konkursy, foodtrucki. Dla dzieci przygotowano dmuchańce, malowanie twarzy i zabawy z animatorami. Rodzice mogą odpocząć przy muzyce na żywo i dobrej kawie. Zaplanowano konkursy z nagrodami dla całych rodzin. Wstęp wolny, impreza trwa od 10:00 do 18:00.",
     "rozrywka, rodzina, plener",
     "Urzad Miasta"),

    ("Wystawa fotografii ulicznej — Miasto blisko",
     "Kolekcja zdjęć dokumentujących życie miejskie, wernisaż i spotkanie z autorami. Fotografie ukazują codzienne życie mieszkańców w różnych porach dnia. Wystawa to efekt rocznego projektu dokumentalnego młodych fotografów. Na wernisażu będzie można porozmawiać z autorami o ich pracy. Wystawa czynna przez miesiąc, wstęp wolny.",
     "kultura, fotografia, wystawa",
     "Pracownia Fotografii"),

    ("Koncert kameralny — kwartet smyczkowy",
     "Klasyczne kwartety i nowoczesne aranżacje w sali kameralnej. W programie utwory Mozarta, Beethovena oraz współczesnych kompozytorów. Muzycy to absolwenci prestiżowych akademii muzycznych. Sala kameralna zapewnia doskonałą akustykę i intymną atmosferę. Koncert rozpoczyna się o 18:00, czas trwania około 75 minut.",
     "kultura_wysoka, muzyka_klasyczna, koncert",
     "Sala Kameralna")
]

# --- DB HELPERS ---
def get_conn():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
    return conn

def setup_db(conn):
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    except Exception:
        conn.rollback()
        print("Warning: Could not create extension 'vector'. It might already exist or permissions are missing.")
    conn.commit()
    cur.close()

def create_table(conn, table_name, dim):
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    sql = f'''
    CREATE TABLE {table_name} (
        id BIGSERIAL PRIMARY KEY,
        title TEXT,
        content TEXT,
        category TEXT,
        author TEXT,
        embedding VECTOR({dim})
    );
    '''
    cur.execute(sql)
    conn.commit()
    cur.close()

def insert_events(conn, table_name, events_with_embeddings):
    # events_with_embeddings is list of (title, content, category, author, embedding)
    cur = conn.cursor()
    sql = f"INSERT INTO {table_name} (title, content, category, author, embedding) VALUES %s"
    execute_values(cur, sql, events_with_embeddings, template="(%s,%s,%s,%s,%s)")
    conn.commit()
    cur.close()

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Arial for Polish support
try:
    pdfmetrics.registerFont(TTFont('Arial', '/System/Library/Fonts/Supplemental/Arial.ttf'))
    FONT_NAME = 'Arial'
except Exception:
    print("Warning: Arial font not found. Polish characters might not render correctly.")
    FONT_NAME = 'Helvetica' # Fallback

def get_color_for_value(val):
    """
    Returns a ReportLab Color object based on the value (0.0 to 1.0).
    Interpolates between White (0.0) and Dark Blue (1.0).
    """
    # White: (1, 1, 1)
    # Dark Blue: (0, 0, 0.5) roughly
    # Let's do White -> Blue
    # R: 1 -> 0
    # G: 1 -> 0
    # B: 1 -> 1 (or stay high)
    
    # Simple heatmap: White -> Red (low sim) is bad? No, usually Blue is good.
    # Let's do White -> Blue
    
    # 0.0 -> White (1, 1, 1)
    # 1.0 -> Blue (0, 0, 1)
    
    return colors.Color(1 - val, 1 - val, 1)

# --- FORMATTING HELPERS ---
def print_side_by_side_table(query_title, query_content, model_results):
    # model_results: dict {model_name: [(title, sim), ...]}
    
    print(f"\nQUERY: {query_title}")
    print(f"Context: {query_content}")
    
    models = list(model_results.keys())
    headers = ["Rank"] + models
    
    # Prepare rows: Rank 1, Rank 2, ...
    rows = []
    for i in range(5):
        row = [f"#{i+1}"]
        for model in models:
            if i < len(model_results[model]):
                title, sim = model_results[model][i]
                row.append(f"{title[:30]}... ({sim})")
            else:
                row.append("-")
        rows.append(row)
        
    # Calculate widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    col_widths = [w + 2 for w in col_widths]
    
    # Print
    header_str = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        print("".join(str(val).ljust(w) for val, w in zip(row, col_widths)))
    print("-" * len(header_str))

    print("-" * len(header_str))

def generate_pdf(filename, all_events, all_query_results, metric_results, similarity_matrices):
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter), topMargin=30, bottomMargin=30, leftMargin=30, rightMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom Styles with Arial
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontName=FONT_NAME, fontSize=18, spaceAfter=20)
    h1_style = ParagraphStyle('CustomH1', parent=styles['Heading1'], fontName=FONT_NAME, fontSize=14, spaceAfter=10)
    h2_style = ParagraphStyle('CustomH2', parent=styles['Heading2'], fontName=FONT_NAME, fontSize=12, spaceAfter=5)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontName=FONT_NAME, fontSize=10)
    table_cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontName=FONT_NAME, fontSize=8, leading=10, alignment=1) # Center alignment
    table_cell_left_style = ParagraphStyle('TableCellLeft', parent=styles['Normal'], fontName=FONT_NAME, fontSize=8, leading=10, alignment=0) # Left alignment
    
    # Title
    elements.append(Paragraph("Embedding Model Benchmark Results", title_style))
    elements.append(Spacer(1, 20))
    
    # 0. All Events Table
    elements.append(Paragraph("Database Events (Knowledge Base)", h1_style))
    elements.append(Spacer(1, 10))
    
    headers = ["ID", "Title", "Content", "Category", "Author"]
    header_row = [Paragraph(f"<b>{h}</b>", table_cell_style) for h in headers]
    data = [header_row]
    
    for evt in all_events:
        # evt: (id, title, content, category, author)
        row = [
            Paragraph(str(evt[0]), table_cell_style),
            Paragraph(evt[1], table_cell_left_style),
            Paragraph(evt[2], table_cell_left_style),
            Paragraph(evt[3], table_cell_style),
            Paragraph(evt[4], table_cell_style),
        ]
        data.append(row)
        
    # Calculate available width
    avail_width = landscape(letter)[0] - 60
    # ID=5%, Title=25%, Content=40%, Category=15%, Author=15%
    col_widths = [avail_width*0.05, avail_width*0.25, avail_width*0.40, avail_width*0.15, avail_width*0.15]
    
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(PageBreak())

    # 1. Model Comparisons
    for q_data in all_query_results:
        q_title = q_data['title']
        q_content = q_data['content']
        results = q_data['results'] # dict {model: [(title, sim)]}
        
        elements.append(Paragraph(f"<b>Query:</b> {q_title}", h2_style))
        elements.append(Paragraph(f"<i>Context:</i> {q_content}", normal_style))
        elements.append(Spacer(1, 10))
        
        # Table Data
        models = list(results.keys())
        # Header row
        header_row = [Paragraph("<b>Rank</b>", table_cell_style)]
        for m in models:
            header_row.append(Paragraph(f"<b>{m}</b>", table_cell_style))
        data = [header_row]
        
        for i in range(5):
            row = [Paragraph(f"{i+1}", table_cell_style)]
            for model in models:
                if i < len(results[model]):
                    title, sim = results[model][i]
                    # Wrap text for PDF
                    cell_text = f"{title}<br/>(Sim: {sim})"
                    row.append(Paragraph(cell_text, table_cell_style))
                else:
                    row.append(Paragraph("-", table_cell_style))
            data.append(row)
            
        # Calculate available width
        avail_width = landscape(letter)[0] - 60 # margins
        col_width = avail_width / (len(models) + 1)
        
        t = Table(data, colWidths=[col_width/2] + [col_width * 1.12 for _ in models]) # Rank col smaller
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)
        elements.append(PageBreak()) # One query per page
        
    # 2. Metric Comparison
    if metric_results:
        elements.append(Paragraph("Metric Comparison (Model: all-mpnet-base-v2)", h1_style))
        elements.append(Paragraph(f"<b>Query:</b> {metric_results['query']}", normal_style))
        elements.append(Spacer(1, 10))
        
        headers = ["Title", "Cosine Sim", "Euclidean Sim (1/(1+d))", "Inner Product"]
        data = [headers]
        for r in metric_results['rows']:
            data.append(r)
            
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)

    # 3. Confusion Matrices (Similarity Heatmaps)
    if similarity_matrices:
        elements.append(PageBreak())
        elements.append(Paragraph("Confusion Matrices (Cross-Similarity)", h1_style))
        elements.append(Paragraph("Heatmap of cosine similarity between all events. Axes are Event IDs.", normal_style))
        elements.append(Spacer(1, 10))

        for model_name, matrix_data in similarity_matrices.items():
            # matrix_data: {'ids': [id1, id2...], 'matrix': [[val, ...], ...]}
            ids = matrix_data['ids']
            matrix = matrix_data['matrix']
            
            elements.append(Paragraph(f"<b>Model:</b> {model_name}", h2_style))
            elements.append(Spacer(1, 5))
            
            # Prepare Table Data
            # Header Row: Empty + IDs
            header_row = [Paragraph("", table_cell_style)] + [Paragraph(f"<b>{id_}</b>", table_cell_style) for id_ in ids]
            data = [header_row]
            
            # Data Rows
            for i, row_vals in enumerate(matrix):
                row_id = ids[i]
                # First col is ID
                table_row = [Paragraph(f"<b>{row_id}</b>", table_cell_style)]
                for val in row_vals:
                    # val is float 0-1
                    bg_color = get_color_for_value(val)
                    # Text color: white if dark bg, black if light bg
                    text_color = "white" if val > 0.5 else "black"
                    cell_text = f"<font color='{text_color}'>{val:.2f}</font>"
                    table_row.append(Paragraph(cell_text, table_cell_style))
                data.append(table_row)
            
            # Create Table
            # Widths: Auto or fixed?
            # Let's try to fit in page.
            avail_width = landscape(letter)[0] - 60
            col_width = avail_width / (len(ids) + 1)
            
            t = Table(data, colWidths=[col_width] * (len(ids) + 1))
            
            # Styles
            style_cmds = [
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 6), # Smaller font for matrix
            ]
            
            # Color cells
            for r_idx, row_vals in enumerate(matrix):
                for c_idx, val in enumerate(row_vals):
                    # r_idx corresponds to data row r_idx + 1
                    # c_idx corresponds to col c_idx + 1
                    bg_color = get_color_for_value(val)
                    style_cmds.append(('BACKGROUND', (c_idx + 1, r_idx + 1), (c_idx + 1, r_idx + 1), bg_color))
            
            t.setStyle(TableStyle(style_cmds))
            elements.append(t)
            elements.append(Spacer(1, 20))
            elements.append(PageBreak())

    doc.build(elements)
    print(f"\nPDF generated: {filename}")

# --- MAIN LOGIC ---

def main():
    # random.seed(42) # Not needed for fixed data
    
    conn = get_conn()
    setup_db(conn)
    
    # 1. Use Realistic Events
    print(f"Using {len(REALISTIC_EVENTS)} realistic events...")
    
    # 2. Process each model (Ensure tables exist and data is inserted)
    # Note: If tables already exist and have data, we might duplicate if we just insert again.
    # For benchmark safety, let's drop and recreate always.
    for model_cfg in MODELS:
        print(f"\nProcessing model: {model_cfg['name']}...")
        model = SentenceTransformer(model_cfg['name'])
        
        # Create table
        create_table(conn, model_cfg['table'], model_cfg['dim'])
        
        # Embed and Insert
        rows_to_insert = []
        for title, content, category, author in REALISTIC_EVENTS:
            text = f"{title} {content} {category} {author}"
            # e5 models need "passage: " prefix for docs
            if "e5" in model_cfg['name']:
                text = "passage: " + text
                
            emb = model.encode(text, convert_to_numpy=True).tolist()
            rows_to_insert.append((title, content, category, author, emb))
            
        insert_events(conn, model_cfg['table'], rows_to_insert)
        print(f"Inserted {len(rows_to_insert)} events into {model_cfg['table']}")

    # 3. Define Query Events
    queries = [
        ("Wycieczka rowerowa po Kaszubach",
         "Półdniowa wycieczka z przewodnikiem, atrakcje krajoznawcze i przerwa na ognisko.",
         "outdoor, rowery, wycieczka",
         "Stowarzyszenie Rowerowe"),

        ("Wieczór stand-up — nowe talenty",
         "Kabaret i stand-up z udziałem debiutujących komików.",
         "rozrywka, standup, komedia",
         "Klub Komediowy"),

        ("Warsztaty kulinarne: Sushi od podstaw",
         "Naucz się robić sushi: gotowanie ryżu, krojenie ryb i zwijanie rolek.",
         "kulinaria, warsztaty, jedzenie",
         "Szkoła Gotowania"),

        ("Festiwal gier planszowych",
         "Całodniowe granie w planszówki, turnieje z nagrodami i strefa prototypów.",
         "rozrywka, gry, hobby",
         "Klub Planszówkowy"),

        ("Zajęcia jogi w parku o świcie",
         "Poranna sesja jogi na trawie, relaks i ćwiczenia oddechowe.",
         "sport, zdrowie, joga",
         "Studio Jogi")
    ]
    
    # 4. Run Queries and Compare Models
    print("\n\n" + "="*50)
    print("RESULTS COMPARISON: TOP 5 SIMILAR EVENTS PER MODEL")
    print("="*50)

    cur = conn.cursor()
    
    # Calculate Similarity Matrices
    similarity_matrices = {} # {model_name: {'ids': [], 'matrix': []}}
    
    for model_cfg in MODELS:
        # Fetch all embeddings for this model
        cur.execute(f"SELECT id, embedding FROM {model_cfg['table']} ORDER BY id ASC")
        rows = cur.fetchall()
        if not rows:
            continue
            
        ids = [r[0] for r in rows]
        # Parse vector string "[1,2,3]" -> np array
        embeddings = []
        for r in rows:
            # embedding is returned as string by psycopg2 vector extension usually, or list if casted?
            # It seems execute_values inserts lists, but fetchall returns strings like '[...]' or numpy arrays if adapter registered?
            # Let's check type or just parse safely.
            emb_val = r[1]
            if isinstance(emb_val, str):
                emb_val = np.array(eval(emb_val)) # eval is safe enough here for local vector string
            else:
                emb_val = np.array(emb_val)
            embeddings.append(emb_val)
            
        embeddings = np.array(embeddings)
        
        # Calculate Cosine Similarity
        # embeddings shape: (N, dim)
        sim_matrix = cosine_similarity(embeddings)
        
        similarity_matrices[model_cfg['name']] = {
            'ids': ids,
            'matrix': sim_matrix
        }

    all_query_results = [] # Store for PDF
    
    # Fetch all events for the first page (using first model's table)
    cur.execute(f"SELECT id, title, content, category, author FROM {MODELS[0]['table']} ORDER BY id ASC")
    all_events = cur.fetchall()
    
    for q_title, q_content, q_cat, q_auth in queries:
        query_data = {
            'title': q_title,
            'content': q_content,
            'results': {}
        }
        
        for model_cfg in MODELS:
            model = SentenceTransformer(model_cfg['name'])
            
            q_text = f"{q_title} {q_content} {q_cat} {q_auth}"
            if "e5" in model_cfg['name']:
                q_text = "query: " + q_text
                
            q_emb = model.encode(q_text, convert_to_numpy=True).tolist()
            q_emb_str = '[' + ','.join(map(str, q_emb)) + ']'
            
            # Query Top 5
            sql = f'''
            SELECT title, 1 - (embedding <=> %s::vector) as sim
            FROM {model_cfg['table']}
            ORDER BY sim DESC
            LIMIT 5;
            '''
            cur.execute(sql, (q_emb_str,))
            rows = cur.fetchall()
            
            model_res = []
            for r in rows:
                model_res.append((r[0], f"{r[1]:.4f}"))
            
            query_data['results'][model_cfg['name']] = model_res
            
        all_query_results.append(query_data)
        print_side_by_side_table(q_title, q_content, query_data['results'])

    # 5. Distance Metric Comparison (Cosine vs Euclidean vs Inner Product)
    print("\n\n" + "="*50)
    print("METRIC COMPARISON (Model: all-mpnet-base-v2)")
    print("="*50)
    
    demo_model = MODELS[1] # mpnet
    model = SentenceTransformer(demo_model['name'])
    q_title, q_content, q_cat, q_auth = queries[0] # Bike trip
    q_text = f"{q_title} {q_content} {q_cat} {q_auth}"
    q_emb = model.encode(q_text, convert_to_numpy=True).tolist()
    q_emb_str = '[' + ','.join(map(str, q_emb)) + ']'
    
    print(f"Query: {q_title}")
    
    # Cosine: <=> (distance) -> 1 - dist = sim
    # Euclidean: <-> (distance) -> 1 / (1 + dist) = sim (Higher is better)
    # Inner Product: <#> (negative ip) -> -1 * neg_ip = ip
    
    sql_metrics = f'''
    SELECT title, 1 - (embedding <=> %s::vector) as cosine_sim,
           1.0 / (1.0 + (embedding <-> %s::vector)) as euclidean_sim,
           (embedding <#> %s::vector) * -1 as inner_product
    FROM {demo_model['table']}
    ORDER BY cosine_sim DESC
    LIMIT 5;
    '''
    
    cur.execute(sql_metrics, (q_emb_str, q_emb_str, q_emb_str))
    rows = cur.fetchall()
    
    headers = ["Title", "Cosine Sim", "Euclidean Sim (1/(1+d))", "Inner Product"]
    formatted_rows = []
    for r in rows:
        formatted_rows.append([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}"])
    
    # Print table
    col_widths = [len(h) for h in headers]
    for row in formatted_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    col_widths = [w + 2 for w in col_widths]
    header_str = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    for row in formatted_rows:
        print("".join(str(val).ljust(w) for val, w in zip(row, col_widths)))
    print("-" * len(header_str))
    
    # Store for PDF
    metric_results = {
        'query': q_title,
        'rows': formatted_rows
    }

    cur.close()
    conn.close()
    
    # Generate PDF
    generate_pdf("benchmark_results.pdf", all_events, all_query_results, metric_results, similarity_matrices)
    print("\nDone.")

if __name__ == "__main__":
    main()
