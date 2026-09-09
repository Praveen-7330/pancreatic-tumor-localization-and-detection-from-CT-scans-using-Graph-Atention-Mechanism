import os

def convert_md(file_path):
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    base = os.path.splitext(file_path)[0]
    
    # Write .txt version
    txt_path = base + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Created TXT: {txt_path}")

    # Write simple styled .html version
    html_path = base + ".html"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{os.path.basename(base)}</title>
    <style>
        body {{
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            margin: 40px auto;
            max-width: 900px;
            padding: 0 20px;
            color: #1a1a1a;
            background-color: #f8fafc;
        }}
        h1, h2, h3 {{ color: #0f172a; }}
        h1 {{ border-bottom: 2px solid #0284c7; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; margin-top: 30px; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background: #ffffff;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 10px 14px;
            text-align: left;
        }}
        th {{ background-color: #0284c7; color: #ffffff; }}
        tr:nth-child(even) {{ background-color: #f1f5f9; }}
        code, pre {{
            font-family: Consolas, Monaco, monospace;
            background-color: #e2e8f0;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        pre {{ padding: 14px; overflow-x: auto; display: block; }}
        blockquote {{
            background: #e0f2fe;
            border-left: 5px solid #0284c7;
            margin: 20px 0;
            padding: 14px 20px;
        }}
    </style>
</head>
<body>
"""
    # Simple markdown to HTML replacement for basic headers, tables, bold, and blockquotes
    lines = content.splitlines()
    in_table = False
    in_code = False
    
    for line in lines:
        if line.startswith("```"):
            if in_code:
                html_content += "</pre>\n"
                in_code = False
            else:
                html_content += "<pre>\n"
                in_code = True
            continue
            
        if in_code:
            html_content += line + "\n"
            continue
            
        if line.startswith("|") and "|" in line[1:]:
            if not in_table:
                html_content += "<table>\n"
                in_table = True
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if "---" in parts[0]:
                continue
            is_header = "Evaluation Metric" in line or "Feature Component" in line or "Target Class" in line or "Model Architecture" in line
            tag = "th" if is_header else "td"
            row = "".join([f"<{tag}>{p}</{tag}>" for p in parts])
            html_content += f"<tr>{row}</tr>\n"
            continue
        else:
            if in_table:
                html_content += "</table>\n"
                in_table = False
                
        if line.startswith("# "):
            html_content += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith("## "):
            html_content += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith("### "):
            html_content += f"<h3>{line[4:]}</h3>\n"
        elif line.startswith("> "):
            html_content += f"<blockquote>{line[2:]}</blockquote>\n"
        elif line.startswith("- "):
            html_content += f"<ul><li>{line[2:]}</li></ul>\n"
        elif line.strip() == "---":
            html_content += "<hr>\n"
        elif line.strip() == "":
            html_content += "<br>\n"
        else:
            # Replace **bold**
            import re
            l_fmt = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            html_content += f"<p>{l_fmt}</p>\n"
            
    if in_table:
        html_content += "</table>\n"
        
    html_content += "</body>\n</html>"
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Created HTML: {html_path}")

convert_md("RESULTS_AND_METRICS_REPORT.md")
convert_md("PRESENTATION_STUDY_GUIDE.md")
