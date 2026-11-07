"""Empaqueta el HTML de R con sus dependencias locales y exporta su tabla.

Uso: python herramientas/autocontener_dashboard.py entrada.html salida.html
No ajusta modelos ni modifica métricas.
"""
import base64, csv, hashlib, json, mimetypes, re, sys
from html.parser import HTMLParser
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Inspector(HTMLParser):
    def __init__(self):
        super().__init__(); self.refs=[]; self.payloads=[]; self.script=None; self.buf=[]
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        for attr in ['src','href']:
            if attr in attrs: self.refs.append((tag,attrs[attr]))
        if tag=='script': self.script=attrs; self.buf=[]
    def handle_data(self,data):
        if self.script is not None: self.buf.append(data)
    def handle_endtag(self,tag):
        if tag=='script':
            if self.script and self.script.get('type')=='application/json':
                self.payloads.append(json.loads(''.join(self.buf)))
            self.script=None

def package(source, destination):
    source=Path(source).resolve(); destination=Path(destination).resolve()
    root=source.parent; text=source.read_text(encoding='utf-8'); dependencies={}
    original=Inspector(); original.feed(text)
    def asset(ref, relative_to=root):
        path=(relative_to/ref.split('?',1)[0].split('#',1)[0]).resolve()
        if not path.is_relative_to(root): raise ValueError(f'Dependencia fuera de la carpeta: {ref}')
        if not path.is_file(): raise FileNotFoundError(path)
        dependencies[path.relative_to(root).as_posix()]=sha(path)
        return path
    def data_uri(ref, relative_to=root):
        path=asset(ref,relative_to)
        mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
    def css(path):
        content=path.read_text(encoding='utf-8-sig')
        def url(match):
            ref=match.group(1).strip().strip('\"\'')
            if ref.startswith(('data:','#')): return match.group(0)
            if re.match(r'https?://|//',ref): raise ValueError('CSS requiere conexión: '+ref)
            return 'url("'+data_uri(ref,path.parent)+'")'
        if re.search(r'@import\s',content): raise ValueError('CSS contiene @import pendiente')
        return re.sub(r'url\(([^)]+)\)',url,content)
    def script(match):
        ref=match.group(2); path=asset(ref)
        code=path.read_text(encoding='utf-8-sig')
        code=re.sub(r'</script',r'<\\/script',code,flags=re.I)
        return '<script data-inlined-from="'+ref+'">\n'+code+'\n</script>'
    text=re.sub(r'<script\b[^>]*\bsrc=([\"\'])([^\"\']+)\1[^>]*>\s*</script>',script,text,flags=re.I)
    def link(match):
        tag=match.group(0)
        href=re.search(r'\bhref=([\"\'])(.*?)\1',tag,flags=re.I)
        if not href: return tag
        return '<style data-inlined-from="'+href.group(2)+'">\n'+css(asset(href.group(2)))+'\n</style>'
    text=re.sub(r'<link\b[^>]*\brel=[\"\']stylesheet[\"\'][^>]*>|<link\b[^>]*\bhref=[^>]*\brel=[\"\']stylesheet[\"\'][^>]*>',link,text,flags=re.I)
    text=re.sub(r'(<img\b[^>]*\bsrc=)([\"\'])(.*?)\2',lambda m:m.group(1)+m.group(2)+data_uri(m.group(3))+m.group(2),text,flags=re.I)
    # El HTML recibido de indicadores tenía pageLength=5 y dom='t': ocultaba
    # dos métodos sin ofrecer controles de navegación. Mostrar las siete filas.
    def json_widget(match):
        value=json.loads(match.group(2)); x=value.get('x',{})
        if isinstance(x.get('data'),list) and x.get('container','').startswith('<table'):
            x.setdefault('options',{})['paging']=False
            x['options'].pop('pageLength',None)
        return match.group(1)+json.dumps(value,ensure_ascii=True,separators=(',',':')).replace('</',r'<\/')+match.group(3)
    text=re.sub(r'(<script\b[^>]*\btype="application/json"[^>]*>)(.*?)(</script>)',json_widget,text,flags=re.S)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(text,encoding='utf-8')
    packaged=Inspector(); packaged.feed(text)
    unresolved=[(tag,ref) for tag,ref in packaged.refs if tag in ['script','link','img','iframe'] and not ref.startswith(('data:','#'))]
    assert not unresolved, unresolved
    original_data=[v['x']['data'] for v in original.payloads if 'data' in v.get('x',{})]
    packaged_data=[v['x']['data'] for v in packaged.payloads if 'data' in v.get('x',{})]
    assert original_data==packaged_data, 'Cambió un dato del dashboard'
    table=next(v['x'] for v in original.payloads if v.get('x',{}).get('container','').startswith('<table'))
    headers=re.findall(r'<th>(.*?)</th>',table['container'])
    with destination.with_name('resumen_dashboard.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(headers); writer.writerows(zip(*table['data']))
    assert len(table['data'][0])==7
    return {'original_html_sha256':sha(source),'published_html_sha256':sha(destination),
        'dependencies_sha256':dependencies,'widgets':len(original.payloads),'methods':table['data'][0],
        'all_widget_data_preserved':True,'unresolved_dependencies':unresolved,
        'transformations':['JavaScript, CSS y recursos CSS locales embebidos',
                           'Tabla sin paginación: siete métodos visibles; mismas métricas'],
        'summary_csv_source':'Tabla embebida en el HTML original; no es una nueva corrida'}

if __name__=='__main__':
    report=package(sys.argv[1],sys.argv[2]); print(json.dumps(report,ensure_ascii=True,indent=2))
