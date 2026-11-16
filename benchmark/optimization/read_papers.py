"""Read supplied PDFs into page-indexed searchable text; preserve originals unchanged.

Run with bundled artifact Python, not the codec environment. Optional page renders
use PyMuPDF if present, or Poppler's pdftoppm executable. No OCR is silently guessed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--papers',type=Path,default=ROOT/'paper')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'literature')
    p.add_argument('--render',type=str,help='Exact PDF filename for selected pages; also saves first-page previews of other indexed PDFs')
    p.add_argument('--pages',type=int,nargs='+',default=[1],help='1-based page numbers')
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    fitz_available = importlib.util.find_spec('fitz') is not None
    if fitz_available:
        import fitz
    else:
        from pypdf import PdfReader
    entries = []
    for path in sorted(args.papers.glob('*.pdf')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if fitz_available:
            doc = fitz.open(path)
            pages = [page.get_text(sort=True) for page in doc]
            metadata = doc.metadata
        else:
            doc = PdfReader(path)
            pages = [page.extract_text(extraction_mode='layout') or '' for page in doc.pages]
            metadata = {str(k):str(v) for k,v in (doc.metadata or {}).items()}
        text_file = args.output/(path.stem+'.txt')
        text_file.write_text('\n\n'.join(f'=== PAGE {i+1} ===\n{text}' for i,text in enumerate(pages)),encoding='utf-8')
        entries.append({'file':str(path.relative_to(ROOT)), 'sha256':digest, 'pages':len(pages),
                        'text':text_file.name,'metadata':metadata,'page_character_counts':list(map(len,pages)),
                        'needs_ocr_pages':[i+1 for i,page in enumerate(pages) if len(page.strip()) < 50]})
        render_pages = args.pages if args.render == path.name else ([1] if args.render else [])
        if render_pages:
            image_dir = args.output/'pages'/path.stem
            image_dir.mkdir(parents=True,exist_ok=True)
            for number in render_pages:
                if not 1 <= number <= len(pages):
                    raise ValueError(f'Page out of range: {number}')
                output = image_dir/f'page_{number:03d}.png'
                if fitz_available:
                    doc[number-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(output)
                else:
                    poppler = shutil.which('pdftoppm')
                    if not poppler:
                        raise RuntimeError('PDF rendering requires PyMuPDF or Poppler; extraction already saved')
                    subprocess.run([poppler,'-f',str(number),'-l',str(number),'-scale-to','1800','-singlefile','-png',str(path),str(output.with_suffix(''))],check=True)
            print(f'Rendered {path.name} pages {render_pages}',flush=True)
        if fitz_available:
            doc.close()
    (args.output/'index.json').write_text(json.dumps({'papers':entries,'extractor':'PyMuPDF' if fitz_available else 'pypdf','source_policy':'Original PDF files unchanged; text is indexed by physical PDF page number'},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps([{'file':e['file'],'pages':e['pages'],'needs_ocr_pages':e['needs_ocr_pages']} for e in entries],indent=2),flush=True)


if __name__ == '__main__':
    main()
